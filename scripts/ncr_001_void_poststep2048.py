#!/usr/bin/env python3
"""Post-update void.1 2048 cache/mask/suffix/generate diagnostic.

Exactly one public synthetic NCA-only diagnostic SGD step is produced, saved
temporarily, then reloaded by a second Python interpreter on the same Actions
runner. The updated adapter is tested at the native 2048-unit context. This is
NOT comparative NCA/MLP training, held-out evaluation, model-quality admission,
or permission to change any #72/global training gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime/ncr001-void-poststep2048"
ADAPTER = OUT / "nca-after-one-public-synthetic-step"
PRODUCER = OUT / "producer.json"
CONSUMER = OUT / "consumer.json"

CONTEXT = 2048
INCREMENTAL_POSITIONS = 2048
BATCH_SIZE = 2
BATCH_SHORT_UNITS = 1536
PAD_ID = 256
EOS_ID = 258
SUFFIX_MUTATION_UNITS = 64
GENERATE_PREFIX = 2044
GENERATE_UNITS = 4
ATOL = RTOL = 0.0005
LONG_SOURCE = (
    "🔧 café 日本語 πΩ — the workshop robot inspects a rail, records "
    "reversible evidence and leaves the frozen model unchanged. "
) * 64

MODEL = "dotlabs/void.1"
REV = "9c2be5087772867d45961abd11f6b35110617563"
WEIGHTS = "ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3"
TOKENIZER = "279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03"
FROZEN = "7673957ba36e8eb5427e5ecc94dd52de71fbf9e6718bb0c2a94e4d7361b3f904"
PRODUCER_STATUS = "POSTSTEP2048_UPDATED_ADAPTER_READY__SCIENTIFIC_72_UNADMITTED"
CONSUMER_STATUS = "PASS_POSTSTEP2048_CACHE_MASK_SUFFIX_GENERATE__SCIENTIFIC_72_UNADMITTED"

EXPECTED_BLOBS = {
    "experiments/ncr-001/protocol.json": "14ce5fbaa38f8806c8c8738f19c01b91385eebaf",
    "experiments/ncr-001/source-pins.json": "59fff1aed872b5563aa6898171983973e0cdb02c",
    "scripts/ncr_001_core.py": "0427ad0b0f1fc68199369d5094952017bc70f503",
    "scripts/ncr_001_void1.py": "7e6abcbf1b3738094e2b460089babed150c72a9b",
    "scripts/ncr_001f_frozen_hooks.py": "5efe8f0690bc1c2793de323d5b972b26bbd7bccb",
    "scripts/ncr_001_void_eos_postlease.py": "45d170472f3b37c6e13088212deab60bf2b223d7",
    "scripts/ncr_001_void_nca_learning_gate.py": "672fede7cb488bb33315b067bba5292678435644",
    "scripts/ncr_001_void_poststep_cold_replay.py": "8af2c53bf86f1ef8f98944ae058ef4f29585b9ff",
    "scripts/ncr_001_void_postlease_maxctx.py": "b9ade1ad56591707b291eaa31dcb3a99ff428a8f",
}


def require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError("NCR void poststep2048 REFUSED: " + why)


def exact_int(value, expected: int) -> bool:
    return type(value) is int and value == expected


def finite_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def hex64(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(tensor) -> str:
    return hashlib.sha256(
        tensor.detach().contiguous().cpu().numpy().tobytes()
    ).hexdigest()


def save(row: dict, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    row["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (OUT / name).write_text(
        json.dumps(row, indent=2, sort_keys=True, ensure_ascii=False,
                   allow_nan=False) + "\n", encoding="utf-8")


def check_plan(protocol: dict) -> None:
    require(protocol["train_authorized"] is False
            and set(protocol["models"]) ==
            {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"},
            "four-lane identity/training policy changed")
    v = protocol["models"]["void1"]
    require(v["repo_id"] == MODEL and v["revision"] == REV
            and v["weight_file"] == "model.safetensors"
            and v["taps_after_block_zero_based"] == [7, 15, 23]
            and v["expected_config"]["hidden_size"] == 640
            and v["expected_config"]["max_position_embeddings"] == CONTEXT
            and v["tokenizer_reference"]["ids"] ==
                {"pad": PAD_ID, "bos": 257, "eos": EOS_ID},
            "pinned void.1 source/tokenizer/geometry changed")
    conf = protocol["conformance"]
    require(conf["full_vs_cached_prefill_atol"] == ATOL
            and conf["full_vs_cached_prefill_rtol"] == RTOL,
            "cache numeric gate changed")
    require((CONTEXT, INCREMENTAL_POSITIONS, BATCH_SIZE, BATCH_SHORT_UNITS,
             SUFFIX_MUTATION_UNITS, GENERATE_PREFIX, GENERATE_UNITS)
            == (2048, 2048, 2, 1536, 64, 2044, 4),
            "authorized long-context diagnostic schedule changed")
    raw = LONG_SOURCE.encode("utf-8")
    require(len(raw) >= CONTEXT + BATCH_SHORT_UNITS
            and any(x >= 128 for x in raw)
            and "fixtures/" not in LONG_SOURCE,
            "public synthetic long source invalid/too short")


def source_gate() -> dict:
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    require(head == os.getenv("NCR_EXPECTED_HEAD") and len(head) == 40,
            "nonliteral source checkout")
    for path, expected in EXPECTED_BLOBS.items():
        observed = subprocess.check_output(
            ["git", "-C", str(ROOT), "hash-object", str(ROOT / path)],
            text=True
        ).strip()
        require(observed == expected, "immutable dependency Git blob drift: " + path)
    policy = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
    check_plan(policy)
    require(os.getenv("GITHUB_EVENT_NAME") == "pull_request",
            "physical diagnostic requires source-matched draft PR")
    require(os.getenv("HF_HUB_OFFLINE") == "1"
            and os.getenv("TRANSFORMERS_OFFLINE") == "1",
            "physical diagnostic must use prefetched offline pinned source")
    return {"head": head, "immutable_dependency_blobs": EXPECTED_BLOBS.copy()}


def validate_producer(row: dict) -> None:
    sample = row["public_learning_sample"]
    learning = row["learning_summary"]
    require(row["schema"] == "ncr001-void-poststep2048-v1"
            and row["status"] == PRODUCER_STATUS and row["phase"] == "producer"
            and row["model_repo"] == MODEL and row["revision"] == REV
            and row["weights_sha256"] == WEIGHTS
            and row["tokenizer_sha256"] == TOKENIZER
            and row["frozen_backbone_sha256"] == FROZEN
            and exact_int(row["diagnostic_optimizer_steps"], 1)
            and row["train_authorized"] is False
            and row["comparative_training_performed"] is False
            and row["heldouts_opened"] is False
            and row["full_72_conformance_pass"] is False
            and row["independent_scientific_review"] is False,
            "producer source/scope/admission boundary changed")
    require(exact_int(row["threads_effective_after_load"], 4)
            and exact_int(row["producer_pid"], row["producer_pid"])
            and row["producer_pid"] > 0
            and hex64(row["adapter_sha256"])
            and hex64(row["adapter_metadata_sha256"])
            and hex64(row["short_active_logits_sha256"])
            and finite_number(row["short_active_vs_disabled_max_abs"])
            and row["short_active_vs_disabled_max_abs"] > 0,
            "producer process/adapter/effect invalid")
    require(sample["sample_kind"] == "synthetic_public_diagnostic_only"
            and sample["heldout"] is False
            and exact_int(sample["total_units"], 31)
            and exact_int(sample["supervised_after_shift"], 21)
            and exact_int(sample["real_eos_id"], EOS_ID)
            and hex64(sample["ids_sha256"])
            and hex64(sample["labels_sha256"])
            and hex64(sample["mask_sha256"]),
            "public shifted-EOS learning sample changed")
    require(exact_int(learning["diagnostic_optimizer_steps"], 1)
            and learning["checks"] == {
                "native_shifted_byte_eos_loss": True,
                "initial_zero_adapter_parity": True,
                "readout_gradient_first_backward": True,
                "post_step_upstream_gradients": True,
                "three_distinct_depth_states": True,
                "backbone_unchanged_after_one_step": True,
                "disabled_source_logits_unchanged": True,
                "no_cross_model_comparison": True,
            }
            and finite_number(learning["initial_loss"])
            and finite_number(learning["post_warmup_loss"])
            and learning["initial_loss"] > 0
            and learning["post_warmup_loss"] > 0
            and all(finite_number(v) and v > 0
                    for v in learning["post_warmup_required_gradient_l1"].values())
            and learning["backbone_before_sha256"] ==
                learning["backbone_after_sha256"] == FROZEN,
            "one-step source diagnostic mechanics changed")


def validate_consumer(row: dict, producer: dict) -> None:
    require(row["schema"] == "ncr001-void-poststep2048-v1"
            and row["status"] == CONSUMER_STATUS and row["phase"] == "consumer"
            and row["source"] == producer["source"]
            and row["model_repo"] == MODEL and row["revision"] == REV
            and row["weights_sha256"] == WEIGHTS
            and row["tokenizer_sha256"] == TOKENIZER
            and row["frozen_backbone_sha256"] == producer["frozen_backbone_sha256"] == FROZEN
            and exact_int(row["producer_pid"], producer["producer_pid"])
            and exact_int(row["consumer_pid"], row["consumer_pid"])
            and row["consumer_pid"] > 0
            and row["producer_pid"] != row["consumer_pid"]
            and exact_int(row["threads_effective_after_load"], 4)
            and row["same_runner_second_python"] is True
            and row["poststep_adapter_reload_exact"] is True,
            "consumer source/cold-process boundary changed")
    cache = row["poststep_2048_all_position_cache"]
    require(exact_int(cache["context_units"], CONTEXT)
            and exact_int(cache["incremental_positions"], INCREMENTAL_POSITIONS)
            and finite_number(cache["max_abs"]) and 0 <= cache["max_abs"] <= ATOL
            and cache["within_protocol_tolerance"] is True
            and cache["all_positions_compared"] is True
            and hex64(cache["full_logits_sha256"])
            and hex64(cache["incremental_logits_sha256"]),
            "poststep 2048 all-position KV evidence invalid")
    masked = row["poststep_masked_batch"]
    require(set(masked) == {"left", "right"}, "both mask directions required")
    for side, result in masked.items():
        require(result["direction"] == side
                and exact_int(result["batch_size"], BATCH_SIZE)
                and exact_int(result["width"], CONTEXT)
                and exact_int(result["short_valid_units"], BATCH_SHORT_UNITS)
                and exact_int(result["synthetic_pad_units"], CONTEXT - BATCH_SHORT_UNITS)
                and exact_int(result["pad_id"], PAD_ID)
                and result["explicit_attention_mask"] is True
                and result["batch_vs_identical_single_within_tolerance"] is True
                and result["row_order_swap_within_tolerance"] is True
                and finite_number(result["max_batch_vs_single_abs"])
                and result["max_batch_vs_single_abs"] <= ATOL
                and finite_number(result["max_row_swap_abs"])
                and result["max_row_swap_abs"] <= ATOL,
                "masked batch/padding isolation invalid: " + side)
    suffix = row["poststep_2048_suffix"]
    require(exact_int(suffix["context_units"], CONTEXT)
            and exact_int(suffix["mutated_suffix_units"], SUFFIX_MUTATION_UNITS)
            and suffix["earlier_logits_exact"] is True
            and suffix["earlier_max_abs"] == 0.0,
            "strict poststep future-suffix causality invalid")
    gen = row["poststep_native_generation"]
    require(exact_int(gen["prefix_units"], GENERATE_PREFIX)
            and exact_int(gen["new_units"], GENERATE_UNITS)
            and exact_int(gen["total_units"], CONTEXT)
            and exact_int(gen["eos_id"], EOS_ID)
            and gen["cache_on_off_token_ids_equal"] is True
            and gen["cache_on_off_scores_within_tolerance"] is True
            and finite_number(gen["max_score_abs"]) and gen["max_score_abs"] <= ATOL
            and isinstance(gen["generated_ids"], list)
            and len(gen["generated_ids"]) == GENERATE_UNITS
            and all(type(x) is int and 0 <= x <= EOS_ID for x in gen["generated_ids"]),
            "native long-context generation cache parity invalid")
    require(row["frozen_backbone_unchanged_after_long_diagnostic"] is True
            and row["train_authorized"] is False
            and row["comparative_training_performed"] is False
            and row["heldouts_opened"] is False
            and row["full_72_conformance_pass"] is False
            and row["independent_scientific_review"] is False
            and row["actual_compute_budget_qualified"] is False
            and row["generated_eos_skill_qualified"] is False
            and row["historical_train002b_eos_effect_measured"] is False,
            "scientific/training boundary falsely promoted")


def load_original():
    import ncr_001_void_poststep_cold_replay as cold
    return cold.load_original()


def producer() -> None:
    started = time.monotonic()
    row = {
        "schema": "ncr001-void-poststep2048-v1",
        "phase": "producer", "status": "STARTED",
        "model_repo": MODEL, "revision": REV,
        "weights_sha256": WEIGHTS, "tokenizer_sha256": TOKENIZER,
        "diagnostic_optimizer_steps": 0,
        "train_authorized": False, "comparative_training_performed": False,
        "heldouts_opened": False, "full_72_conformance_pass": False,
        "independent_scientific_review": False,
        "actual_compute_budget_qualified": False,
        "producer_pid": os.getpid(),
        "runner": {"run_id": os.getenv("GITHUB_RUN_ID"),
                   "attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
                   "name": os.getenv("RUNNER_NAME"),
                   "python": platform.python_version(),
                   "platform": platform.platform()},
    }
    save(row, "producer.json")
    try:
        row["source"] = source_gate()
        torch, original, backbone, tok, wrapper, serialize = load_original()
        import ncr_001_void_nca_learning_gate as learning
        row["runner"]["torch"] = torch.__version__
        row["threads_effective_after_load"] = torch.get_num_threads()
        row["frozen_backbone_sha256"] = original.frozen_sha(backbone)
        require(row["frozen_backbone_sha256"] == FROZEN,
                "producer frozen backbone fingerprint changed")
        ids, labels, mask, sample = learning.make_sample(serialize, tok)
        row["public_learning_sample"] = sample
        learning.REPORT.clear()
        learning.REPORT.update({"_started": time.monotonic(),
                                "status": "NEW_POSTSTEP2048_DIAGNOSTIC_NOT_ORIGINAL_172"})
        learning.one_step_diagnostic(backbone, wrapper, tok, serialize)
        require(exact_int(learning.REPORT["diagnostic_optimizer_steps"], 1)
                and learning.REPORT["checks"]["backbone_unchanged_after_one_step"],
                "one public diagnostic step did not complete")
        row["diagnostic_optimizer_steps"] = 1
        row["learning_receipt_sha256"] = digest(learning.OUT / "receipt.json")
        row["learning_summary"] = {
            "diagnostic_optimizer_steps": 1,
            "checks": dict(learning.REPORT["checks"]),
            "initial_loss": learning.REPORT["warmup"]["native_causal_eos_loss"],
            "post_warmup_loss":
                learning.REPORT["post_warmup"]["native_causal_eos_loss"],
            "post_warmup_required_gradient_l1": {
                key: learning.REPORT["post_warmup"]["gradient_l1"][key]
                for key in ("readout.weight", "observe.weight",
                            "perception.weight", "rule.weight",
                            "input_norm.weight")
            },
            "backbone_before_sha256": learning.REPORT["backbone_before_sha256"],
            "backbone_after_sha256": learning.REPORT["backbone_after_sha256"],
        }
        wrapper.eval()
        with torch.inference_mode():
            active = wrapper(input_ids=ids, attention_mask=mask,
                             use_cache=False).logits
            disabled = wrapper(input_ids=ids, attention_mask=mask,
                               use_cache=False, enabled=False).logits
        row["short_active_vs_disabled_max_abs"] = float(
            (active - disabled).abs().max())
        require(row["short_active_vs_disabled_max_abs"] > 0
                and bool(torch.isfinite(active).all()),
                "one-step adapter has no finite active effect")
        row["short_active_logits_sha256"] = tensor_sha(active)
        wrapper.save_adapter(ADAPTER)
        row["adapter_sha256"] = digest(ADAPTER / "adapter.pt")
        row["adapter_metadata_sha256"] = digest(ADAPTER / "metadata.json")
        require(original.frozen_sha(backbone) == FROZEN,
                "producer changed frozen Qwen3")
        row["status"] = PRODUCER_STATUS
        row["elapsed_seconds"] = round(time.monotonic() - started, 5)
        validate_producer(row)
    except BaseException:
        row["status"] = "FAIL_POSTSTEP2048_PRODUCER__72_BLOCKED"
        row["traceback"] = traceback.format_exc()
        raise
    finally:
        save(row, "producer.json")
        print(json.dumps({"phase": "producer", "status": row["status"],
                          "diagnostic_optimizer_steps":
                              row["diagnostic_optimizer_steps"],
                          "train_authorized": False}, sort_keys=True),
              flush=True)


def compare_logits(torch, left, right, label: str) -> dict:
    require(tuple(left.shape) == tuple(right.shape), label + " shape mismatch")
    require(bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
            label + " nonfinite logits")
    maximum = float((left - right).abs().max())
    within = bool(torch.allclose(left, right, atol=ATOL, rtol=RTOL))
    require(within, label + " exceeds protocol cache tolerance: " + str(maximum))
    return {"max_abs": maximum, "within_protocol_tolerance": True,
            "atol": ATOL, "rtol": RTOL}


def masked_batch_check(torch, wrapper, first, second, direction: str) -> dict:
    require(direction in ("left", "right"), "unknown padding direction")
    width = CONTEXT
    short = BATCH_SHORT_UNITS
    ids = torch.full((BATCH_SIZE, width), PAD_ID, dtype=torch.long)
    mask = torch.zeros((BATCH_SIZE, width), dtype=torch.long)
    ids[0] = first[0]
    mask[0] = 1
    if direction == "left":
        ids[1, width-short:] = second[0, :short]
        mask[1, width-short:] = 1
    else:
        ids[1, :short] = second[0, :short]
        mask[1, :short] = 1
    require(bool((ids[mask == 1] < 256).all())
            and bool((ids[mask == 0] == PAD_ID).all())
            and int((mask[1] == 0).sum()) == width-short,
            "explicit masked batch source/pad construction invalid")
    batch = wrapper(input_ids=ids, attention_mask=mask,
                    use_cache=False).logits
    trace = wrapper.last_hook_trace
    require(len(trace) == 1 and trace[0]["tap_count"] == 3
            and trace[0]["state_shape"][:2] == [BATCH_SIZE, width]
            and trace[0]["finite"] is True,
            "B=2 NCA persistent field trace invalid")
    singles = []
    maxima = []
    for row in range(BATCH_SIZE):
        solo = wrapper(input_ids=ids[row:row+1],
                       attention_mask=mask[row:row+1],
                       use_cache=False).logits
        valid = mask[row].bool()
        check = compare_logits(torch, batch[row, valid], solo[0, valid],
                               direction + "_batch_row_" + str(row))
        maxima.append(check["max_abs"])
        singles.append(tensor_sha(solo[0, valid]))
    swapped = wrapper(input_ids=ids.flip(0), attention_mask=mask.flip(0),
                      use_cache=False).logits.flip(0)
    valid_parts = []
    swapped_parts = []
    for row in range(BATCH_SIZE):
        valid = mask[row].bool()
        valid_parts.append(batch[row, valid])
        swapped_parts.append(swapped[row, valid])
    row_swap_max = 0.0
    for row, (left, right) in enumerate(zip(valid_parts, swapped_parts)):
        check = compare_logits(torch, left, right,
                               direction + "_row_order_" + str(row))
        row_swap_max = max(row_swap_max, check["max_abs"])
    return {
        "direction": direction, "batch_size": BATCH_SIZE, "width": width,
        "short_valid_units": short, "synthetic_pad_units": width-short,
        "pad_id": PAD_ID, "explicit_attention_mask": True,
        "input_ids_sha256": tensor_sha(ids), "attention_mask_sha256": tensor_sha(mask),
        "valid_single_logits_sha256": singles,
        "batch_vs_identical_single_within_tolerance": True,
        "max_batch_vs_single_abs": max(maxima),
        "row_order_swap_within_tolerance": True,
        "max_row_swap_abs": row_swap_max,
    }


def generation_check(torch, wrapper, sequence, mask) -> dict:
    prefix = sequence[:, :GENERATE_PREFIX]
    attention = mask[:, :GENERATE_PREFIX]
    kwargs = dict(
        input_ids=prefix, attention_mask=attention, do_sample=False,
        max_new_tokens=GENERATE_UNITS, min_new_tokens=GENERATE_UNITS,
        eos_token_id=EOS_ID, pad_token_id=PAD_ID,
        return_dict_in_generate=True, output_scores=True,
    )
    results = {}
    traces = {}
    for cache in (False, True):
        response = wrapper.generate(**kwargs, use_cache=cache)
        generated = response.sequences[0, GENERATE_PREFIX:].tolist()
        require(len(generated) == GENERATE_UNITS
                and len(response.scores) == GENERATE_UNITS,
                "native GenerationMixin output length changed")
        results[cache] = (generated, list(response.scores))
        traces[str(cache)] = wrapper.last_hook_trace
    require(results[False][0] == results[True][0],
            "poststep generation token IDs changed with cache")
    max_score = 0.0
    for step, (left, right) in enumerate(zip(results[False][1], results[True][1])):
        require(tuple(left.shape) == tuple(right.shape), "generation score shape")
        finite = torch.isfinite(left)
        require(bool(torch.equal(finite, torch.isfinite(right)))
                and bool(torch.equal(left[~finite], right[~finite]))
                and bool(finite.any()), "generation nonfinite mask differs")
        maximum = float((left[finite] - right[finite]).abs().max())
        require(bool(torch.allclose(left[finite], right[finite],
                                    atol=ATOL, rtol=RTOL)),
                "generation score cache mismatch step " + str(step))
        max_score = max(max_score, maximum)
    for cache, trace in traces.items():
        require(len(trace) == GENERATE_UNITS
                and all(t["tap_count"] == 3 and t["finite"] is True for t in trace),
                "GenerationMixin NCA hook trace incomplete for cache=" + cache)
    ids = results[True][0]
    return {
        "prefix_units": GENERATE_PREFIX, "new_units": GENERATE_UNITS,
        "total_units": GENERATE_PREFIX + GENERATE_UNITS,
        "eos_id": EOS_ID, "generated_ids": ids,
        "actual_eos_id_emitted": EOS_ID in ids,
        "generated_eos_skill_qualified": False,
        "cache_on_off_token_ids_equal": True,
        "cache_on_off_scores_within_tolerance": True,
        "max_score_abs": max_score,
        "uncached_forward_calls": len(traces["False"]),
        "cached_forward_calls": len(traces["True"]),
    }


def consumer() -> None:
    started = time.monotonic()
    producer_row = json.loads(PRODUCER.read_text(encoding="utf-8"))
    validate_producer(producer_row)
    row = {
        "schema": "ncr001-void-poststep2048-v1",
        "phase": "consumer", "status": "STARTED",
        "model_repo": MODEL, "revision": REV,
        "weights_sha256": WEIGHTS, "tokenizer_sha256": TOKENIZER,
        "producer_pid": producer_row["producer_pid"],
        "consumer_pid": os.getpid(),
        "train_authorized": False, "comparative_training_performed": False,
        "heldouts_opened": False, "full_72_conformance_pass": False,
        "independent_scientific_review": False,
        "actual_compute_budget_qualified": False,
        "generated_eos_skill_qualified": False,
        "historical_train002b_eos_effect_measured": False,
        "runner": {"run_id": os.getenv("GITHUB_RUN_ID"),
                   "attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
                   "name": os.getenv("RUNNER_NAME"),
                   "python": platform.python_version(),
                   "platform": platform.platform()},
    }
    save(row, "consumer.json")
    try:
        row["source"] = source_gate()
        require(row["source"] == producer_row["source"]
                and row["runner"]["run_id"] == producer_row["runner"]["run_id"]
                and row["producer_pid"] != row["consumer_pid"],
                "producer/consumer are not same source/run different interpreters")
        torch, original, backbone, tok, wrapper, serialize = load_original()
        row["runner"]["torch"] = torch.__version__
        row["threads_effective_after_load"] = torch.get_num_threads()
        row["frozen_backbone_sha256"] = original.frozen_sha(backbone)
        require(row["frozen_backbone_sha256"] == FROZEN,
                "consumer frozen Qwen3 fingerprint changed")
        require(digest(ADAPTER / "adapter.pt") == producer_row["adapter_sha256"]
                and digest(ADAPTER / "metadata.json") ==
                    producer_row["adapter_metadata_sha256"],
                "one-step adapter bytes changed before cold reload")
        wrapper.load_adapter(ADAPTER)
        wrapper.eval()
        ids, _labels, short_mask, sample = (
            __import__("ncr_001_void_nca_learning_gate").make_sample(serialize, tok)
        )
        require(sample == producer_row["public_learning_sample"],
                "consumer public learning sample source changed")
        with torch.inference_mode():
            short = wrapper(input_ids=ids, attention_mask=short_mask,
                            use_cache=False).logits
        require(tensor_sha(short) == producer_row["short_active_logits_sha256"],
                "poststep adapter cold reload changed short active logits")
        row["same_runner_second_python"] = True
        row["poststep_adapter_reload_exact"] = True

        byte_ids = tok.encode(LONG_SOURCE, add_special_tokens=True)
        require(len(byte_ids) >= CONTEXT + BATCH_SHORT_UNITS
                and all(type(x) is int and 0 <= x < 256
                        for x in byte_ids[:CONTEXT + BATCH_SHORT_UNITS]),
                "reviewed ByteTokenizer long public source changed")
        sequence = torch.tensor([byte_ids[:CONTEXT]], dtype=torch.long)
        second = torch.tensor(
            [byte_ids[CONTEXT:CONTEXT+BATCH_SHORT_UNITS]], dtype=torch.long)
        mask = torch.ones_like(sequence)
        row["long_public_source"] = {
            "utf8_sha256": hashlib.sha256(
                LONG_SOURCE.encode("utf-8")).hexdigest(),
            "utf8_bytes": len(LONG_SOURCE.encode("utf-8")),
            "characters": len(LONG_SOURCE),
            "available_tokenizer_units": len(byte_ids),
            "selected_context_units": CONTEXT,
            "selected_ids_sha256": tensor_sha(sequence),
            "secondary_valid_units": BATCH_SHORT_UNITS,
            "secondary_ids_sha256": tensor_sha(second),
            "ordinary_byte_ids_only": True, "heldout": False,
        }
        save(row, "consumer.json")

        with torch.inference_mode():
            full = wrapper(input_ids=sequence, attention_mask=mask,
                           use_cache=False).logits
            disabled = wrapper(input_ids=sequence, attention_mask=mask,
                               use_cache=False, enabled=False).logits
            effect = float((full - disabled).abs().max())
            require(effect > 0 and bool(torch.isfinite(full).all()),
                    "updated NCA has no finite long-context effect")
            row["poststep_2048_active_vs_disabled_max_abs"] = effect
            past = None
            pieces = []
            trace_checkpoints = []
            checkpoints = {0, 31, 127, 511, 1023, 1535, 2047}
            for pos in range(CONTEXT):
                one = wrapper(
                    input_ids=sequence[:, pos:pos+1],
                    attention_mask=mask[:, :pos+1],
                    past_key_values=past, use_cache=True)
                pieces.append(one.logits[:, -1:])
                past = one.past_key_values
                if pos in checkpoints:
                    trace = wrapper.last_hook_trace
                    require(len(trace) == 1
                            and trace[0]["tap_count"] == 3
                            and trace[0]["state_shape"][:2] == [1, 1]
                            and trace[0]["finite"] is True,
                            "cached per-token NCA state trace invalid")
                    trace_checkpoints.append({
                        "position": pos, "trace": trace[0]})
            incremental = torch.cat(pieces, dim=1)
            cache_check = compare_logits(
                torch, full, incremental, "poststep all-position 2048 cache")
            row["poststep_2048_all_position_cache"] = {
                "context_units": CONTEXT,
                "incremental_positions": INCREMENTAL_POSITIONS,
                "all_positions_compared": True,
                **cache_check,
                "full_logits_sha256": tensor_sha(full),
                "incremental_logits_sha256": tensor_sha(incremental),
                "trace_checkpoints": trace_checkpoints,
            }
            save(row, "consumer.json")

            altered = sequence.clone()
            altered[:, -SUFFIX_MUTATION_UNITS:] = (
                altered[:, -SUFFIX_MUTATION_UNITS:] + 17) % 256
            require(not torch.equal(
                sequence[:, -SUFFIX_MUTATION_UNITS:],
                altered[:, -SUFFIX_MUTATION_UNITS:]),
                "future suffix negative mutation ineffective")
            changed = wrapper(input_ids=altered, attention_mask=mask,
                              use_cache=False).logits
            earlier = full[:, :-SUFFIX_MUTATION_UNITS]
            changed_earlier = changed[:, :-SUFFIX_MUTATION_UNITS]
            exact = bool(torch.equal(earlier, changed_earlier))
            max_abs = float((earlier - changed_earlier).abs().max())
            require(exact and max_abs == 0.0,
                    "poststep future suffix changed earlier logits")
            row["poststep_2048_suffix"] = {
                "context_units": CONTEXT,
                "mutated_suffix_units": SUFFIX_MUTATION_UNITS,
                "earlier_logits_exact": True, "earlier_max_abs": 0.0,
                "original_prefix_sha256": tensor_sha(earlier),
                "mutated_prefix_sha256": tensor_sha(changed_earlier),
            }
            save(row, "consumer.json")

            row["poststep_masked_batch"] = {
                side: masked_batch_check(torch, wrapper, sequence, second, side)
                for side in ("left", "right")
            }
            save(row, "consumer.json")

            row["poststep_native_generation"] = generation_check(
                torch, wrapper, sequence, mask)
            save(row, "consumer.json")

        after = original.frozen_sha(backbone)
        require(after == row["frozen_backbone_sha256"] == FROZEN
                and all(p.grad is None for p in backbone.parameters()),
                "long diagnostic changed or accumulated frozen gradients")
        require(not backbone.model._forward_hooks
                and not backbone.model._forward_pre_hooks,
                "updated NCA long diagnostic leaked Qwen3 hooks")
        row["frozen_backbone_unchanged_after_long_diagnostic"] = True
        row["elapsed_seconds"] = round(time.monotonic() - started, 5)
        require(row["elapsed_seconds"] <= 3600,
                "void.1 preregistered 3600-second CPU lane watchdog breached")
        row["status"] = CONSUMER_STATUS
        validate_consumer(row, producer_row)
    except BaseException:
        row["status"] = "FAIL_POSTSTEP2048_LONG_CONTEXT__72_BLOCKED"
        row["traceback"] = traceback.format_exc()
        raise
    finally:
        save(row, "consumer.json")
        print(json.dumps({
            "phase": "consumer", "status": row["status"],
            "context_units": row.get("poststep_2048_all_position_cache", {}).get(
                "context_units"),
            "incremental_positions":
                row.get("poststep_2048_all_position_cache", {}).get(
                    "incremental_positions"),
            "train_authorized": False,
            "full_72_conformance_pass": False,
        }, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("producer", "consumer"))
    args = parser.parse_args()
    if args.phase == "producer":
        producer()
    else:
        consumer()


if __name__ == "__main__":
    main()
