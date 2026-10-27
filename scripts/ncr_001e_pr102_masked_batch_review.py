#!/usr/bin/env python3
"""Independent PR #102 reviewer: original frozen Instruct masked-batch/cache check.

Source-bound, synthetic wire only. Exact candidate and original hook are
checked out separately; no optimizer, behavioral scoring, heldouts or training.
This is new evidence, not a reclassification of PR #102's historical receipts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runtime/ncr-001/pr102-independent-batch-review.json"
TARGET = "3592841802d85c442ca2521479b5d8795bee7233"
SOURCE_IMPL = "b95c010f9b2ba3cbd9f81317981316bb9c8acd19"
PUBLISHER = "a003767e70d766bc14b40c89792543ee8641c8bc"
UPSTREAM_SFT_SHA = "e5568c15989f3987fec07545640f2f271ffe967c27cbe9d9883c0dd7b5c4ec69"
UPSTREAM_WIRE_BLOB = "4e558ae530a1b79050e22bee587b3caf056e4563"
HF_REPO = "kandivault/sprocket-500m"
HF_REV = "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
ROOT_SHA = "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864"
CHAT_SHA = "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97"
CONTROL_IDS = {
    "<|system|>": 2, "<|user|>": 3, "<|assistant|>": 4, "<|end|>": 5,
    "<|tool_call|>": 6, "<|tool_result|>": 7,
    "<|memory_read|>": 10, "<|memory_write|>": 11,
}
ROLE_IDS = {"system": 2, "user": 3, "assistant": 4, "tool": 7, "memory": 10}
END = 5
TAIL = 4
# Fresh synthetic reviewer test inputs, not the sealed original evaluation suite.
PRIMARY = (
    ("system", "Read the meter only after an explicit instrument result."),
    ("memory", "Calibration label is amber. No live telemetry is implied."),
    ("user", "Read the synthetic channel A."),
    ("assistant", '<|tool_call|>{"name":"read_meter","arguments":{"channel":"A"}}'),
    ("tool", '{"reading":42,"unit":"C"}'),
    ("assistant", "The synthetic returned reading is 42 C."),
    ("user", "Store my preferred temperature unit."),
    ("assistant", "<|memory_write|>Preference is Celsius."),
    ("user", "Search for the stored unit."),
    ("assistant", "<|memory_read|>temperature unit"),
    ("memory", "Stored temperature unit: Celsius."),
    ("user", "Report the calibration state.")
)
SECONDARY = (
    ("user", "Inspect a different bearing and record the run number."),
    ("assistant", "The run number is not supplied."),
    ("user", "What evidence would you need?")
)


def require(ok, why):
    if not ok:
        raise RuntimeError("NCR independent reviewer refusal: " + why)


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as src:
        for part in iter(lambda: src.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def git_rev(directory: Path):
    return subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
    ).strip()


def independent_wire(turns, tokenizer):
    """Build publisher-listed turns without importing candidate serializer.

    Returns (ids, assistant-training-mask, explicit raw prompt) to compare to
    exact PR #102 candidate separately. Never builds a real external tool call.
    """
    require(isinstance(turns, tuple) and 1 <= len(turns) <= 12,
            "Unexpected reviewer synthetic message count")
    ids, mask, text_parts = [], [], []
    for role, body in turns:
        require(role in ROLE_IDS and isinstance(body, str) and body,
                "Unexpected role or empty synthetic message")
        prefix = ROLE_IDS[role]
        chunks = tokenizer.encode(body, add_special_tokens=False)
        require(bool(chunks) and all(type(x) is int and 0 <= x < 32000
                                     for x in chunks), "Unexpected source tokenizer body")
        trained = role == "assistant"
        ids.extend([prefix, *chunks, END])
        mask.extend([False, *([trained] * len(chunks)), trained])
        marker = next(k for k, v in CONTROL_IDS.items() if v == prefix)
        text_parts.append(marker + body + "<|end|>")
    require(turns[-1][0] != "assistant", "Prompt must cue next assistant turn")
    ids.append(ROLE_IDS["assistant"])
    mask.append(False)
    text = "".join(text_parts) + "<|assistant|>"
    require(ids == tokenizer.encode(text, add_special_tokens=False),
            "Publisher segmented tokenization differs from whole prompt")
    return ids, mask, text


def guard_report_claims(report):
    require(report["train_authorized"] is False
            and report["optimizer_steps"] == 0
            and report["heldout_evaluation"] is False
            and report["semantic_tool_or_memory_quality_qualified"] is False
            and report["final_four_model_72_pass"] is False
            and report["model_variant"] == "sprocket_instruct",
            "False reviewer admission")


def compare(report, key, actual, expected, *, atol, rtol, exact=False):
    import torch
    require(actual.shape == expected.shape, key + " tensor shape drift")
    require(bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
            key + " nonfinite logits")
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(diff.max())
    # Do not use maximum absolute value alone as a replacement for elementwise
    # torch.allclose: preserve both the fixed source atols/rtols.
    passed = bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))
    if exact:
        passed = passed and bool(torch.equal(actual, expected))
    report["checks"][key] = {"max_abs": max_abs, "atol": atol,
                              "rtol": rtol, "exact_required": exact,
                              "pass": passed}
    require(passed, key + " causal/frozen/masked mismatch")
    return max_abs


def batch_layout(sequences, side, pad_id=1):
    """Pure-Python padded stream and mask, independently of Transformers."""
    require(side in ("left", "right") and len(sequences) == 2
            and all(isinstance(x, list) and x for x in sequences),
            "Reviewer source batch shape")
    n = max(len(x) for x in sequences)
    ids, masks, spans = [], [], []
    for seq in sequences:
        pad = n - len(seq)
        start = pad if side == "left" else 0
        ids.append(([pad_id] * pad + seq) if side == "left"
                   else (seq + [pad_id] * pad))
        masks.append(([0] * pad + [1] * len(seq)) if side == "left"
                     else ([1] * len(seq) + [0] * pad))
        spans.append((start, start + len(seq)))
    return ids, masks, spans


def check_synthetic_plan():
    """Strictly new synthetic probe: verify source-marker role positions."""
    require([role for role, _ in PRIMARY].count("tool") == 1
            and [role for role, _ in PRIMARY].count("memory") == 2
            and [role for role, _ in PRIMARY].count("assistant") == 4
            and len(PRIMARY) == 12 and len(SECONDARY) == 3
            and TAIL == 4, "Unexpected fixed reviewer case geometry")
    require(PRIMARY[3][1].startswith("<|tool_call|>")
            and PRIMARY[7][1].startswith("<|memory_write|>")
            and PRIMARY[9][1].startswith("<|memory_read|>"),
            "Reviewer control message ordering changed")
    return True


def run():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import hf_hub_download
    report = {
        "research": "NCR-001E",
        "stage": "independent_pr102_original_host_masked_batch_conformance_prepass",
        "status": "RUNNING",
        "source_reviewed_pr": 102,
        "reviewed_exact_head": TARGET,
        "model_variant": "sprocket_instruct",
        "train_authorized": False,
        "optimizer_steps": 0,
        "heldout_evaluation": False,
        "semantic_tool_or_memory_quality_qualified": False,
        "final_four_model_72_pass": False,
        "comparative_training": False,
        "checks": {},
        "runner": {
            "date": datetime.now(timezone.utc).isoformat(),
            "github_run_id": os.getenv("GITHUB_RUN_ID"),
            "github_sha": os.getenv("GITHUB_SHA"),
            "run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
            "runner_name": os.getenv("RUNNER_NAME"),
            "runner_os": os.getenv("RUNNER_OS"),
            "pid": os.getpid(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "torch": torch.__version__,
        }
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    def save():
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        OUTPUT.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    save()
    try:
        require(os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
                "Frozen-model execution must be network-offline")
        require(check_synthetic_plan(), "Synthetic plan changed")
        candidate = ROOT / "candidate"
        impl = ROOT / "implementation"
        publisher = ROOT / "publisher"
        require(git_rev(candidate) == TARGET and git_rev(impl) == SOURCE_IMPL
                and git_rev(publisher) == PUBLISHER,
                "Exact independent source checkout mismatch")
        serializer = candidate / "scripts/ncr_001e_sprocket_publisher_wire.py"
        sft = publisher / "src/train/sft_data.py"
        require(sha(sft) == UPSTREAM_SFT_SHA,
                "Publisher SFT renderer source bytes drifted")
        result = subprocess.check_output(
            ["git", "-C", str(candidate), "hash-object", str(serializer.relative_to(candidate))],
            text=True).strip()
        require(result == UPSTREAM_WIRE_BLOB,
                "Reviewed PR #102 serializer source changed")
        protocol_file = candidate / "experiments/ncr-001/protocol.json"
        pins_file = candidate / "experiments/ncr-001/source-pins.json"
        protocol, pins = json.loads(protocol_file.read_text()), json.loads(pins_file.read_text())
        require(protocol["train_authorized"] is False
                and set(protocol["models"]) ==
                {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"}
                and protocol["conformance"]["full_vs_cached_prefill_atol"] ==
                protocol["conformance"]["full_vs_cached_prefill_rtol"] == 0.0005,
                "Frozen four-lane protocol or numeric gate changed")
        spec = protocol["models"]["sprocket_instruct"]
        pin = pins["models"]["sprocket_instruct"]
        require(spec["weight_file"] == pin["hf_weight_file"] == "model.safetensors"
                and spec["repo_id"] == pin["repo_id"] == HF_REPO
                and spec["revision"] == pin["revision"] == HF_REV
                and pin["weight_declared_lfs_sha256_not_downloaded"] == ROOT_SHA
                and pins["models"]["sprocket_chat"][
                    "weight_declared_lfs_sha256_not_downloaded"] == CHAT_SHA
                and ROOT_SHA != CHAT_SHA, "Wrong Instruct/Chat source variants")
        report["source"] = {
            "model_repo": HF_REPO, "model_revision": HF_REV,
            "checkpoint_file": "model.safetensors",
            "checkpoint_sha256": ROOT_SHA, "weight_bytes": 1002207984,
            "candidate_commit": TARGET, "candidate_wire_blob": UPSTREAM_WIRE_BLOB,
            "original_hook_commit": SOURCE_IMPL, "publisher_commit": PUBLISHER,
            "publisher_sft_sha256": UPSTREAM_SFT_SHA,
            "publisher_not_training_build_attested": True,
            "protocol_sha256": sha(protocol_file), "pins_sha256": sha(pins_file),
        }
        save()

        # This reviewer independently re-hashes complete root weights and source
        # metadata; the publisher and model code never comes from a floating ref.
        filenames = {
            "model.safetensors": ("weight_declared_lfs_sha256_not_downloaded", 1002207984),
            "config.json": ("config_sha256", None),
            "tokenizer.json": ("tokenizer_sha256", None),
            "tokenizer_config.json": ("tokenizer_config_sha256", None),
            "special_tokens_map.json": ("special_tokens_map_sha256", None),
        }
        for filename, (key, expected_len) in filenames.items():
            f = Path(hf_hub_download(HF_REPO, filename, revision=HF_REV,
                                     local_files_only=True))
            actual = sha(f)
            require(actual == pin[key] and (expected_len is None or f.stat().st_size == expected_len),
                    "Pinned complete source mismatch: " + filename)
            report.setdefault("actual_files", {})[filename] = {
                "sha256": actual, "bytes": f.stat().st_size}
            save()
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(20261027)
        opts = dict(revision=HF_REV, local_files_only=True,
                    torch_dtype=torch.float32, attn_implementation="eager",
                    trust_remote_code=False)
        original = AutoModelForCausalLM.from_pretrained(HF_REPO, **opts).eval()
        sys.path.insert(0, str(impl / "scripts"))
        import ncr_001_sprocket_instruct as original_impl
        require(original_impl.MODEL_ID == HF_REPO
                and original_impl.MODEL_REVISION == HF_REV
                and tuple(original_impl.POST_BLOCK_TAPS) == (7, 16, 25),
                "Frozen original geometry changed")
        wrapper = original_impl.FrozenSprocketInstructDepthNCA(
            original, verified_weight_sha256=ROOT_SHA).eval()
        wrapper.verify_frozen_base()
        require(all(not p.requires_grad for p in original.parameters()),
                "Frozen original parameter requires grad")
        require(wrapper.nca.trainable_parameters() == 658904,
                "NCA geometry altered")
        tokenizer = AutoTokenizer.from_pretrained(
            HF_REPO, revision=HF_REV, local_files_only=True)
        for marker, idx in CONTROL_IDS.items():
            require(tokenizer.convert_tokens_to_ids(marker) == idx
                    and tokenizer.encode(marker, add_special_tokens=False) == [idx],
                    "Sprocket dedicated publisher role token ID mismatch " + marker)
        seqs = []
        masks = []
        sys.path.insert(0, str(candidate / "scripts"))
        from ncr_001e_sprocket_publisher_wire import render_ids_mask
        for label, turns in (("primary", PRIMARY), ("secondary", SECONDARY)):
            ids, mask, raw = independent_wire(turns, tokenizer)
            model_rows = [{"role": role, "content": value} for role, value in turns]
            c_ids, c_mask = render_ids_mask(model_rows, tokenizer, variant="instruct")
            require(c_ids == ids and c_mask == mask,
                    "Independent publisher wire disagrees with reviewed PR #102")
            require(len(ids) <= 512 and ids[0] in (2, 3),
                    "Synthetic wire outside bounded source budget")
            report.setdefault("wire", {})[label] = {
                "id_count": len(ids), "trained_units": sum(mask),
                "text_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "ids_sha256": hashlib.sha256(
                    torch.tensor(ids, dtype=torch.int64).numpy().tobytes()).hexdigest(),
                "literal_message_roles": [r for r, _ in turns],
                "not_heldout": True,
            }
            seqs.append(ids)
            masks.append(mask)
        require(len(seqs[0]) > len(seqs[1]) + 12,
                "Insufficient bounded synthetic padding difference")
        report["adapter_parameters"] = wrapper.nca.trainable_parameters()
        report["frozen_base_parameters"] = wrapper.frozen_parameters()
        save()

        def tensor_hash():
            h = hashlib.sha256()
            for name, tensor in original.state_dict().items():
                h.update(name.encode() + str(tensor.shape).encode() +
                         str(tensor.dtype).encode())
                h.update(tensor.detach().contiguous().cpu().numpy().tobytes())
            return h.hexdigest()

        report["backbone_before_sha256"] = tensor_hash()
        with torch.inference_mode():
            # The first independent reviewer pass checks the original frozen
            # base and candidate zero NCA at BOTH padding orientations.
            for side in ("left", "right"):
                ids, masks, spans = batch_layout(seqs, side)
                x = torch.tensor(ids, dtype=torch.long)
                m = torch.tensor(masks, dtype=torch.long)
                positions = torch.clamp(m.cumsum(-1) - 1, min=0)
                kw = dict(input_ids=x, attention_mask=m,
                          position_ids=positions, use_cache=False)
                bare = wrapper(**kw, enabled=False).logits
                zero = wrapper(**kw, enabled=True).logits
                compare(report, side + "_frozen_zero_exact",
                        bare, zero, atol=0, rtol=0, exact=True)
                with torch.no_grad():
                    wrapper.nca.readout.weight[0, 0] = 0.03
                active = wrapper(**kw, enabled=True).logits
                effect = float((active - bare).abs().max())
                require(effect > 0 and bool(torch.isfinite(active).all()),
                        "Untrained NCA intervention invisible/invalid")
                report["checks"][side + "_active_untrained_effect"] = {
                    "max_abs": effect, "trained": False, "pass": True}
                # This is the absent #102 check: both padded and standalone
                # source-structured messages must agree on non-padding logits,
                # without accidental cross-batch field state.
                for row, original_ids in enumerate(seqs):
                    standalone = wrapper(
                        input_ids=torch.tensor([original_ids], dtype=torch.long),
                        position_ids=torch.arange(len(original_ids)).unsqueeze(0),
                        use_cache=False).logits
                    start, end = spans[row]
                    compare(report, side + "_batched_vs_single_" + str(row),
                            active[row, start:end], standalone[0],
                            atol=0.0005, rtol=0.0005)
                # Deliberately corrupt a masked PAD token to the *real*
                # tool-result control ID 7. Valid positions must NOT change.
                mutated = x.clone()
                short = 1
                invalid = torch.where(m[short] == 0)[0]
                require(len(invalid) >= 12, "No masked negative positions")
                mutated[short, invalid] = 7
                altered = wrapper(
                    input_ids=mutated, attention_mask=m,
                    position_ids=positions, use_cache=False).logits
                for row, (start, end) in enumerate(spans):
                    compare(report, side + "_masked_pad_control_no_effect_" + str(row),
                            active[row, start:end], altered[row, start:end],
                            atol=0, rtol=0, exact=True)
                # Whole other batch member must not change when a real token
                # in secondary sample is varied; causality is sample-local.
                changed = x.clone()
                i = spans[short][1] - 1
                changed[short, i] = (changed[short, i] + 11) % 32000
                other = wrapper(input_ids=changed, attention_mask=m,
                                position_ids=positions, use_cache=False).logits
                first_start, first_end = spans[0]
                compare(report, side + "_other_request_isolation",
                        active[0, first_start:first_end],
                        other[0, first_start:first_end],
                        atol=0, rtol=0, exact=True)
                # Fail-open sentinel: if invalid pads are mistakenly included
                # in attention on LEFT, their nonzero effect must be detected.
                if side == "left":
                    broken_mask = m.clone()
                    broken_mask[short, invalid] = 1
                    broken_positions = torch.clamp(broken_mask.cumsum(-1) - 1, min=0)
                    broken = wrapper(
                        input_ids=x, attention_mask=broken_mask,
                        position_ids=broken_positions,
                        use_cache=False).logits
                    start, end = spans[short]
                    discrepancy = float((broken[short, start:end] -
                                         active[short, start:end]).abs().max())
                    require(discrepancy > 0,
                            "Negative sentinel cannot detect unmasked pad leakage")
                    report["checks"]["unmasked_pad_injection_detected"] = {
                        "max_abs": discrepancy, "expected_to_change": True,
                        "pass": True}
                    # Prefill plus four *real* one-token continuation forwards,
                    # with explicit batched masks and per-example position IDs.
                    prefill = wrapper(
                        input_ids=x[:, :-TAIL],
                        attention_mask=m[:, :-TAIL],
                        position_ids=positions[:, :-TAIL],
                        use_cache=True)
                    past = prefill.past_key_values
                    pieces = []
                    for index in range(x.shape[-1] - TAIL, x.shape[-1]):
                        output = wrapper(
                            input_ids=x[:, index:index + 1],
                            attention_mask=m[:, :index + 1],
                            position_ids=positions[:, index:index + 1],
                            past_key_values=past, use_cache=True)
                        pieces.append(output.logits)
                        past = output.past_key_values
                        require(len(wrapper.last_hook_trace) == 1
                                and wrapper.last_hook_trace[0]["state_shape"][:2] == [2, 1],
                                "Depth NCA cached state not scoped to batched single token")
                    compare(report, "left_masked_batch_full_vs_4_cached",
                            active[:, -TAIL:], torch.cat(pieces, dim=1),
                            atol=0.0005, rtol=0.0005)
                with torch.no_grad():
                    wrapper.nca.readout.weight.zero_()
            # Strict request reset after multiple masked source/control messages
            ref = wrapper(input_ids=torch.tensor([seqs[0][:16]]),
                          use_cache=False).logits
            wrapper(input_ids=torch.tensor([seqs[1]]),
                    use_cache=False)
            replay = wrapper(input_ids=torch.tensor([seqs[0][:16]]),
                             use_cache=False).logits
            compare(report, "fresh_request_A_B_A_zero_exact",
                    ref, replay, atol=0, rtol=0, exact=True)
        report["backbone_after_sha256"] = tensor_hash()
        require(report["backbone_before_sha256"] == report["backbone_after_sha256"],
                "Original frozen tensors changed without optimizer")
        require(all(p.grad is None for p in original.parameters()),
                "Frozen model accumulated gradient unexpectedly")
        require(all(not layer._forward_hooks and not layer._forward_pre_hooks
                    for layer in original.modules()),
                "Original post-block NCA hook leaked between requests")
        guard_report_claims(report)
        report["status"] = "PASS_REVIEWER_MASKED_BATCH_SOURCE_BOUND_PREPASS"
        report["qualification_limit"] = [
            "Independent synthetic source-structured Instruct masked batch reviewer test only",
            "No held-out fixture, behavioral tool/memory PASS, optimizer or training",
            "Does not grant final independent four-model #72 acceptance",
            "Publisher public SFT renderer not cryptographically attested to released model build"
        ]
    except BaseException:
        report["status"] = "FAIL_REVIEWER_MASKED_BATCH_SOURCE_BOUND_PREPASS"
        report["failure_traceback"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-test", action="store_true",
                        help="Stdlib-only synthetic plan check; no frozen model")
    args = parser.parse_args()
    if args.source_test:
        print(json.dumps({"source_synthetic_plan": bool(check_synthetic_plan()),
                          "target": TARGET, "train_authorized": False}, sort_keys=True))
    else:
        run()
        print(json.dumps({"status": "PASS_REVIEWER_MASKED_BATCH_SOURCE_BOUND_PREPASS",
                          "target": TARGET, "train_authorized": False}, sort_keys=True))
