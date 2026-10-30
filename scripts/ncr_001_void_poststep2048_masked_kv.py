#!/usr/bin/env python3
"""Post-update void.1 B=2 explicit-mask incremental-KV diagnostic.

Producer is the already reviewed one-public-example NCA-only step from
ncr_001_void_poststep2048.py. This consumer runs in a second Python interpreter
on the same Actions runner, reloads that temporary updated adapter and measures
B=2 masked cached continuation under both left- and right-gap layouts.

NO comparative NCA/MLP training, heldouts, quality admission, protocol change,
or final #72 promotion.
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

import ncr_001_void_poststep2048 as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime/ncr001-void-poststep2048-masked-kv"
CONSUMER = OUT / "consumer.json"

LONG = 2048
SHORT = 1536
TAIL = 64
BATCH = 2
PAD = 256
EOS = 258
TOL = 5e-4

STATUS = "PASS_POSTSTEP2048_B2_MASKED_INCREMENTAL_KV__SCIENTIFIC_72_UNADMITTED"

EXPECTED_BLOBS = {
    "experiments/ncr-001/protocol.json":
        "14ce5fbaa38f8806c8c8738f19c01b91385eebaf",
    "experiments/ncr-001/source-pins.json":
        "59fff1aed872b5563aa6898171983973e0cdb02c",
    "scripts/ncr_001_void_poststep2048.py":
        "9d89e5c34c4a161b2c60a724b91a9a8889773010",
    "scripts/ncr_001_void1.py":
        "7e6abcbf1b3738094e2b460089babed150c72a9b",
    "scripts/ncr_001_void_eos_postlease.py":
        "45d170472f3b37c6e13088212deab60bf2b223d7",
    "scripts/ncr_001_void_nca_learning_gate.py":
        "672fede7cb488bb33315b067bba5292678435644",
}


def require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError("VOID_POSTSTEP_MASKED_KV_REFUSED: " + why)


def finite_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def tensor_sha(tensor) -> str:
    return hashlib.sha256(
        tensor.detach().contiguous().cpu().numpy().tobytes()
    ).hexdigest()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(row: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    row["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    CONSUMER.write_text(
        json.dumps(row, indent=2, sort_keys=True, ensure_ascii=False,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )


def check_plan(protocol: dict) -> None:
    require(
        protocol["train_authorized"] is False
        and set(protocol["models"]) ==
            {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"},
        "four-lane protocol/training state changed",
    )
    lane = protocol["models"]["void1"]
    require(
        lane["repo_id"] == base.MODEL
        and lane["revision"] == base.REV
        and lane["weight_file"] == "model.safetensors"
        and lane["taps_after_block_zero_based"] == [7, 15, 23]
        and lane["expected_config"]["hidden_size"] == 640
        and lane["expected_config"]["max_position_embeddings"] == LONG
        and lane["tokenizer_reference"]["ids"] ==
            {"pad": PAD, "bos": 257, "eos": EOS},
        "void.1 source/geometry/tokenizer contract changed",
    )
    conf = protocol["conformance"]
    require(
        conf["full_vs_cached_prefill_atol"] == TOL
        and conf["full_vs_cached_prefill_rtol"] == TOL,
        "cache tolerance changed",
    )
    require(
        (LONG, SHORT, TAIL, BATCH, PAD, EOS) ==
            (2048, 1536, 64, 2, 256, 258),
        "masked-cache diagnostic schedule changed",
    )


def source_gate() -> dict:
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    require(head == os.getenv("NCR_EXPECTED_HEAD") and len(head) == 40,
            "nonliteral PR checkout")
    for path, expected in EXPECTED_BLOBS.items():
        actual = subprocess.check_output(
            ["git", "-C", str(ROOT), "hash-object", str(ROOT / path)],
            text=True,
        ).strip()
        require(actual == expected, "immutable dependency drift: " + path)
    check_plan(json.loads(
        (ROOT / "experiments/ncr-001/protocol.json").read_text()))
    require(
        os.getenv("GITHUB_EVENT_NAME") == "pull_request"
        and os.getenv("HF_HUB_OFFLINE") == "1"
        and os.getenv("TRANSFORMERS_OFFLINE") == "1",
        "physical masked-cache execution must be draft-PR and offline",
    )
    return {"head": head, "immutable_dependency_blobs": EXPECTED_BLOBS.copy()}


def geometry(torch, long_ids, short_ids, side: str):
    """2x2048 with 64 real continuation units after a padded prefix.

    Left:  [PAD*512][short real prefix*1472][real tail*64]
    Right: [short real prefix*1472][PAD*512][real tail*64]

    The right-gap layout is a strict synthetic attention-mask sentinel, not a
    normal autoregressive publisher sequence. Explicit mask-derived position
    IDs preserve the short row's real-token positions across the artificial gap.
    """
    require(side in ("left", "right"), "unknown padding side")
    require(long_ids.ndim == short_ids.ndim == 1, "expected token vectors")
    require(
        int(long_ids.numel()) == LONG and int(short_ids.numel()) == SHORT,
        "long/short source length changed",
    )
    width = LONG - TAIL
    prefixes, tails, masks = [], [], []
    for seq in (long_ids, short_ids):
        real_prefix, tail = seq[:-TAIL], seq[-TAIL:]
        artificial = width - int(real_prefix.numel())
        require(artificial >= 0, "negative padding width")
        pads = torch.full((artificial,), PAD, dtype=torch.long)
        zero = torch.zeros((artificial,), dtype=torch.long)
        ones = torch.ones((int(real_prefix.numel()),), dtype=torch.long)
        if side == "left":
            prefixes.append(torch.cat((pads, real_prefix)))
            masks.append(torch.cat((zero, ones)))
        else:
            prefixes.append(torch.cat((real_prefix, pads)))
            masks.append(torch.cat((ones, zero)))
        tails.append(tail)
    ids = torch.stack([torch.cat((p, t)) for p, t in zip(prefixes, tails)])
    mask = torch.stack([
        torch.cat((m, torch.ones(TAIL, dtype=torch.long))) for m in masks
    ])
    positions = torch.clamp(mask.cumsum(-1) - 1, min=0)
    require(
        tuple(ids.shape) == tuple(mask.shape) == tuple(positions.shape)
            == (BATCH, LONG)
        and bool((mask[:, -TAIL:] == 1).all())
        and int((mask[0] == 0).sum()) == 0
        and int((mask[1] == 0).sum()) == LONG - SHORT
        and bool((ids[mask == 0] == PAD).all())
        and bool((ids[mask == 1] < 256).all()),
        "masked B=2 geometry invalid",
    )
    return ids, mask, positions


def score(torch, expected, actual, label: str, *, exact: bool = False) -> dict:
    require(tuple(expected.shape) == tuple(actual.shape),
            label + ": shape mismatch")
    require(bool(torch.isfinite(expected).all()
                 and torch.isfinite(actual).all()),
            label + ": nonfinite logits")
    maximum = float((expected - actual).abs().max())
    within = bool(torch.allclose(
        expected, actual, atol=TOL, rtol=TOL))
    require(within and (not exact or maximum == 0.0),
            label + ": mismatch " + str(maximum))
    return {
        "max_abs": maximum, "atol": TOL, "rtol": TOL,
        "within_protocol_tolerance": True, "exact": maximum == 0.0,
    }


def cached_tail_batch(torch, wrapper, ids, mask, positions, label: str):
    with torch.inference_mode():
        full = wrapper(
            input_ids=ids, attention_mask=mask,
            position_ids=positions, use_cache=False,
        ).logits[:, -TAIL:].clone()
        prefix = wrapper(
            input_ids=ids[:, :-TAIL],
            attention_mask=mask[:, :-TAIL],
            position_ids=positions[:, :-TAIL],
            use_cache=True,
        )
        past = prefix.past_key_values
        require(past is not None, label + ": missing genuine prefill cache")
        pretrace = wrapper.last_hook_trace
        require(
            len(pretrace) == 1
            and pretrace[0]["tap_count"] == 3
            and pretrace[0]["state_shape"][:2] == [BATCH, LONG - TAIL]
            and pretrace[0]["finite"] is True,
            label + ": prefill B=2 NCA state trace invalid",
        )
        pieces = []
        checkpoints = []
        for step, index in enumerate(range(LONG - TAIL, LONG)):
            out = wrapper(
                input_ids=ids[:, index:index+1],
                attention_mask=mask[:, :index+1],
                position_ids=positions[:, index:index+1],
                past_key_values=past,
                use_cache=True,
            )
            require(out.past_key_values is not None,
                    label + ": incremental cache lost")
            past = out.past_key_values
            pieces.append(out.logits[:, -1:].clone())
            trace = wrapper.last_hook_trace
            require(
                len(trace) == 1
                and trace[0]["tap_count"] == 3
                and trace[0]["state_shape"][:2] == [BATCH, 1]
                and trace[0]["finite"] is True,
                label + ": cached B=2 field leaked/reused",
            )
            if step in (0, 31, 63):
                checkpoints.append({
                    "step": step, "absolute_batch_position": index,
                    "trace": trace[0],
                })
        incremental = torch.cat(pieces, dim=1)
    return full, incremental, checkpoints


def single_tail(torch, wrapper, seq, label: str):
    seq = seq.view(1, -1)
    length = int(seq.shape[1])
    mask = torch.ones_like(seq)
    positions = torch.arange(length, dtype=torch.long).view(1, -1)
    with torch.inference_mode():
        full = wrapper(
            input_ids=seq, attention_mask=mask,
            position_ids=positions, use_cache=False,
        ).logits[:, -TAIL:].clone()
        prefix = wrapper(
            input_ids=seq[:, :-TAIL],
            attention_mask=mask[:, :-TAIL],
            position_ids=positions[:, :-TAIL],
            use_cache=True,
        )
        past = prefix.past_key_values
        pieces = []
        for index in range(length - TAIL, length):
            out = wrapper(
                input_ids=seq[:, index:index+1],
                attention_mask=mask[:, :index+1],
                position_ids=positions[:, index:index+1],
                past_key_values=past, use_cache=True,
            )
            past = out.past_key_values
            pieces.append(out.logits[:, -1:].clone())
        cached = torch.cat(pieces, dim=1)
    score(torch, full, cached, label + "_single_full_vs_cached")
    return full, cached


def side_check(torch, wrapper, long_ids, short_ids, side: str) -> dict:
    ids, mask, positions = geometry(torch, long_ids, short_ids, side)
    full, cached, checkpoints = cached_tail_batch(
        torch, wrapper, ids, mask, positions, side)
    full_vs_cached = score(
        torch, full, cached, side + "_B2_full_vs_cached")

    long_full, long_cached = single_tail(
        torch, wrapper, long_ids, side + "_long")
    short_full, short_cached = single_tail(
        torch, wrapper, short_ids, side + "_short")

    batch_vs_single_full = max(
        score(torch, full[0:1], long_full,
              side + "_B2row0_vs_B1_full")["max_abs"],
        score(torch, full[1:2], short_full,
              side + "_B2row1_vs_B1_full")["max_abs"],
    )
    batch_vs_single_cached = max(
        score(torch, cached[0:1], long_cached,
              side + "_B2row0_vs_B1_cached")["max_abs"],
        score(torch, cached[1:2], short_cached,
              side + "_B2row1_vs_B1_cached")["max_abs"],
    )

    swapped_full, swapped_cached, _ = cached_tail_batch(
        torch, wrapper, ids.flip(0), mask.flip(0), positions.flip(0),
        side + "_swapped")
    swapped_full = swapped_full.flip(0)
    swapped_cached = swapped_cached.flip(0)
    row_swap_full = score(
        torch, full, swapped_full, side + "_row_swap_full")
    row_swap_cached = score(
        torch, cached, swapped_cached, side + "_row_swap_cached")

    return {
        "direction": side,
        "batch_size": BATCH,
        "width": LONG,
        "long_valid_units": LONG,
        "short_valid_units": SHORT,
        "synthetic_pad_units": LONG - SHORT,
        "tail_real_incremental_units_per_row": TAIL,
        "genuine_B2_incremental_steps": TAIL,
        "pad_id": PAD,
        "explicit_attention_mask": True,
        "explicit_mask_derived_position_ids": True,
        "right_layout_is_synthetic_mask_gap":
            side == "right",
        "full_vs_cached": full_vs_cached,
        "batch_vs_single_full_max_abs": batch_vs_single_full,
        "batch_vs_single_cached_max_abs": batch_vs_single_cached,
        "row_swap_full": row_swap_full,
        "row_swap_cached": row_swap_cached,
        "input_ids_sha256": tensor_sha(ids),
        "attention_mask_sha256": tensor_sha(mask),
        "position_ids_sha256": tensor_sha(positions),
        "full_tail_sha256": tensor_sha(full),
        "cached_tail_sha256": tensor_sha(cached),
        "trace_checkpoints": checkpoints,
    }


def validate_consumer(row: dict, producer: dict) -> None:
    require(
        row["schema"] == "ncr001-void-poststep2048-masked-kv-v1"
        and row["status"] == STATUS
        and row["model_repo"] == base.MODEL
        and row["revision"] == base.REV
        and row["weights_sha256"] == base.WEIGHTS
        and row["tokenizer_sha256"] == base.TOKENIZER
        and row["frozen_backbone_sha256"] == base.FROZEN
        and row["source"] == producer["source"]
        and row["masked_consumer_source"]["head"] == producer["source"]["head"]
        and row["producer_pid"] == producer["producer_pid"]
        and type(row["consumer_pid"]) is int and row["consumer_pid"] > 0
        and row["consumer_pid"] != row["producer_pid"]
        and row["same_runner_second_python"] is True
        and row["poststep_adapter_reload_exact"] is True
        and row["threads_effective_after_load"] == 4,
        "consumer source/process/cold boundary changed",
    )
    require(set(row["masked_incremental_kv"]) == {"left", "right"},
            "both mask directions required")
    for side in ("left", "right"):
        result = row["masked_incremental_kv"][side]
        require(
            result["direction"] == side
            and result["batch_size"] == BATCH
            and result["width"] == LONG
            and result["long_valid_units"] == LONG
            and result["short_valid_units"] == SHORT
            and result["synthetic_pad_units"] == LONG - SHORT
            and result["tail_real_incremental_units_per_row"] == TAIL
            and result["genuine_B2_incremental_steps"] == TAIL
            and result["pad_id"] == PAD
            and result["explicit_attention_mask"] is True
            and result["explicit_mask_derived_position_ids"] is True
            and result["right_layout_is_synthetic_mask_gap"] is (side == "right")
            and result["full_vs_cached"]["within_protocol_tolerance"] is True
            and finite_number(result["full_vs_cached"]["max_abs"])
            and result["full_vs_cached"]["max_abs"] <= TOL
            and finite_number(result["batch_vs_single_full_max_abs"])
            and result["batch_vs_single_full_max_abs"] <= TOL
            and finite_number(result["batch_vs_single_cached_max_abs"])
            and result["batch_vs_single_cached_max_abs"] <= TOL
            and result["row_swap_full"]["within_protocol_tolerance"] is True
            and result["row_swap_cached"]["within_protocol_tolerance"] is True
            and result["row_swap_cached"]["max_abs"] <= TOL
            and len(result["trace_checkpoints"]) == 3
            and [x["step"] for x in result["trace_checkpoints"]] == [0,31,63],
            "masked incremental KV evidence invalid: " + side,
        )
    require(
        row["frozen_backbone_unchanged"] is True
        and row["train_authorized"] is False
        and row["comparative_training_performed"] is False
        and row["heldouts_opened"] is False
        and row["full_72_conformance_pass"] is False
        and row["independent_scientific_review"] is False
        and row["actual_compute_budget_qualified"] is False
        and row["generated_eos_skill_qualified"] is False
        and row["historical_train002b_eos_effect_measured"] is False,
        "science/training/quality boundary falsely promoted",
    )


def consumer() -> None:
    started = time.monotonic()
    producer = json.loads(base.PRODUCER.read_text(encoding="utf-8"))
    base.validate_producer(producer)
    row = {
        "schema": "ncr001-void-poststep2048-masked-kv-v1",
        "status": "STARTED",
        "model_repo": base.MODEL,
        "revision": base.REV,
        "weights_sha256": base.WEIGHTS,
        "tokenizer_sha256": base.TOKENIZER,
        "producer_pid": producer["producer_pid"],
        "consumer_pid": os.getpid(),
        "train_authorized": False,
        "comparative_training_performed": False,
        "heldouts_opened": False,
        "full_72_conformance_pass": False,
        "independent_scientific_review": False,
        "actual_compute_budget_qualified": False,
        "generated_eos_skill_qualified": False,
        "historical_train002b_eos_effect_measured": False,
        "runner": {
            "run_id": os.getenv("GITHUB_RUN_ID"),
            "attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
            "name": os.getenv("RUNNER_NAME"),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    save(row)
    try:
        row["masked_consumer_source"] = source_gate()
        row["source"] = producer["source"]
        require(
            producer["source"]["head"] == row["masked_consumer_source"]["head"]
            and producer["runner"]["run_id"] == row["runner"]["run_id"]
            and producer["producer_pid"] != row["consumer_pid"],
            "producer/consumer are not same source/run distinct interpreters",
        )
        torch, original, backbone, tok, wrapper, _serialize = base.load_original()
        row["runner"]["torch"] = torch.__version__
        row["threads_effective_after_load"] = torch.get_num_threads()
        row["frozen_backbone_sha256"] = original.frozen_sha(backbone)
        require(row["frozen_backbone_sha256"] == base.FROZEN,
                "frozen Qwen3 fingerprint changed")
        require(
            digest(base.ADAPTER / "adapter.pt") == producer["adapter_sha256"]
            and digest(base.ADAPTER / "metadata.json") ==
                producer["adapter_metadata_sha256"],
            "updated adapter bytes changed before masked-cache cold reload",
        )
        wrapper.load_adapter(base.ADAPTER)
        wrapper.eval()

        # Bind the cold-loaded state to the producer's short active logits.
        import ncr_001_void_nca_learning_gate as learning
        ids, _labels, mask, sample = learning.make_sample(_serialize, tok)
        require(sample == producer["public_learning_sample"],
                "public one-step sample identity changed")
        with torch.inference_mode():
            active = wrapper(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).logits
        require(tensor_sha(active) == producer["short_active_logits_sha256"],
                "updated adapter cold reload does not match producer")
        row["same_runner_second_python"] = True
        row["poststep_adapter_reload_exact"] = True

        byte_ids = tok.encode(base.LONG_SOURCE, add_special_tokens=True)
        require(
            len(byte_ids) >= LONG + SHORT
            and all(type(x) is int and 0 <= x < 256
                    for x in byte_ids[:LONG + SHORT]),
            "public long source/ByteTokenizer identity changed",
        )
        long_ids = torch.tensor(byte_ids[:LONG], dtype=torch.long)
        short_ids = torch.tensor(
            byte_ids[LONG:LONG + SHORT], dtype=torch.long)
        row["public_long_source"] = {
            "utf8_sha256": hashlib.sha256(
                base.LONG_SOURCE.encode("utf-8")).hexdigest(),
            "available_tokenizer_units": len(byte_ids),
            "long_units": LONG,
            "short_units": SHORT,
            "long_ids_sha256": tensor_sha(long_ids),
            "short_ids_sha256": tensor_sha(short_ids),
            "heldout": False,
            "ordinary_byte_ids_only": True,
        }
        save(row)

        row["masked_incremental_kv"] = {
            side: side_check(torch, wrapper, long_ids, short_ids, side)
            for side in ("left", "right")
        }
        after = original.frozen_sha(backbone)
        require(
            after == row["frozen_backbone_sha256"] == base.FROZEN
            and all(p.grad is None for p in backbone.parameters())
            and not backbone.model._forward_hooks
            and not backbone.model._forward_pre_hooks,
            "masked cached diagnostic changed frozen source or leaked hooks",
        )
        row["frozen_backbone_unchanged"] = True
        row["elapsed_seconds"] = round(time.monotonic() - started, 5)
        require(row["elapsed_seconds"] <= 3600,
                "3600-second physical watchdog exceeded")
        row["status"] = STATUS
        validate_consumer(row, producer)
    except BaseException:
        row["status"] = "FAIL_POSTSTEP2048_B2_MASKED_INCREMENTAL_KV__72_BLOCKED"
        row["traceback"] = traceback.format_exc()
        raise
    finally:
        save(row)
        print(json.dumps({
            "status": row["status"],
            "left_max_abs": row.get("masked_incremental_kv", {})
                .get("left", {}).get("full_vs_cached", {}).get("max_abs"),
            "right_max_abs": row.get("masked_incremental_kv", {})
                .get("right", {}).get("full_vs_cached", {}).get("max_abs"),
            "tail_units": TAIL,
            "train_authorized": False,
            "full_72_conformance_pass": False,
        }, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("consumer",), default="consumer")
    parser.parse_args()
    consumer()
