#!/usr/bin/env python3
"""NCR-001E post-lease *source-structured* Sprocket Instruct wire witness.

Exactly one original frozen root Instruct HF checkpoint; no optimizer, held-outs,
external tools, persistent memory, learned competence or training admission.
Import the publisher-derived renderer from the separate immutable #102 tree.
This is intentionally NOT a semantic tool/memory performance experiment.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import traceback

import torch
import transformers

from ncr_001_postlease_sprocket import (
    ROOT, digest, fingerprint_backbone, load_actual,
    load_formatter, require, verify_contract,
)

WIRE_TREE = ROOT / "wire-authority"
WIRE_HEAD = "3592841802d85c442ca2521479b5d8795bee7233"
WIRE_FILE = "scripts/ncr_001e_sprocket_publisher_wire.py"
WIRE_BLOB = "4e558ae530a1b79050e22bee587b3caf056e4563"
SOURCE_HEAD = "2271218d625e1462b2cf218445da975efe17fe32"
OUTPUT = ROOT / "runtime/ncr001-postlease-publisher-wire"
REPORT: dict = {}


def save() -> None:
    path = OUTPUT / "instruct" / "receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    REPORT["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    path.write_text(json.dumps(REPORT, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def checked(key: str, value) -> None:
    REPORT.setdefault("checks", {})[key] = value
    save()


def compare(key: str, left, right, *, atol: float, rtol: float,
            exact: bool = False) -> None:
    """Report numeric observations, including failures, before refusal."""
    require(left.shape == right.shape, key + ": tensor shape mismatch")
    delta = float((left - right).abs().max().item())
    finite = bool(torch.isfinite(left).all().item()
                  and torch.isfinite(right).all().item())
    good = bool(finite and torch.allclose(left, right, atol=atol, rtol=rtol)
                and (not exact or delta == 0))
    checked(key, {
        "pass": good, "max_abs": delta, "atol": atol, "rtol": rtol,
        "exact_required": exact, "shape": list(left.shape),
    })
    require(good, key + ": pinned frozen parity or cache mismatch")


def wire_authority():
    """Reject changed, absent or shadow-imported publisher renderer bytes."""
    head = subprocess.check_output(
        ["git", "-C", str(WIRE_TREE), "rev-parse", "HEAD"], text=True).strip()
    require(head == WIRE_HEAD, "Publisher wire reviewer tree head mismatch")
    blob = subprocess.check_output(
        ["git", "-C", str(WIRE_TREE), "hash-object", WIRE_FILE],
        text=True).strip()
    require(blob == WIRE_BLOB, "Publisher wire serializer blob mismatch")
    sys.path.insert(0, str(WIRE_TREE / "scripts"))
    from ncr_001e_sprocket_publisher_wire import (
        IDS, UPSTREAM_COMMIT, SFT_BLOB, HARVEST_BLOB,
        render_ids_mask, render_text,
    )
    require(UPSTREAM_COMMIT == "a003767e70d766bc14b40c89792543ee8641c8bc",
            "Publisher source commit changed")
    require(SFT_BLOB == "90bddf8976b9d6410fe7c5c1f084708e1cebab73"
            and HARVEST_BLOB == "97811aa4245ad3f05f66ce0c2d8aa8c353619267",
            "Publisher SFT/harvest source references changed")
    require(IDS["<|tool_call|>"] == 6 and IDS["<|tool_result|>"] == 7
            and IDS["<|memory_read|>"] == 10
            and IDS["<|memory_write|>"] == 11
            and IDS["<|end|>"] == 5, "Publisher control IDs changed")
    return IDS, render_ids_mask, render_text


def synthetic_turns() -> list[dict[str, str]]:
    """Invented diagnostic observations; NO preregistered evaluation samples."""
    return [
        {"role": "system", "content": "Only quote explicit instrument results."},
        {"role": "memory", "content": "The unit preference is Celsius."},
        {"role": "user", "content": "Read the synthetic relay channel A."},
        {"role": "assistant", "content":
         '<|tool_call|>{"name":"read_meter","arguments":{"channel":"A"}}'},
        {"role": "tool", "content": '{"reading":42,"unit":"C"}'},
        {"role": "assistant", "content": "The returned reading was 42 C."},
        {"role": "user", "content": "Remember that the unit is Celsius."},
        {"role": "assistant", "content": "<|memory_write|>Preferred unit: Celsius."},
        {"role": "user", "content": "Recall the unit preference."},
        {"role": "assistant", "content": "<|memory_read|>temperature unit"},
        {"role": "memory", "content": "Preferred unit: Celsius."},
        {"role": "user", "content": "What did the synthetic relay say?"},
    ]


def assert_source_ownership(ids: list[int], loss_mask: list[bool], markers) -> dict:
    """SFT loss mask is NOT the runtime attention mask."""
    require(len(ids) == len(loss_mask) and 10 < len(ids) <= 512,
            "Wire sequence exceeds synthetic bounds or mask differs")
    require(all(type(n) is int and 0 <= n < 32000 for n in ids)
            and all(type(v) is bool for v in loss_mask),
            "Wire IDs or loss mask have invalid types")
    require(ids[0] == markers["<|system|>"]
            and ids[-1] == markers["<|assistant|>"] and not loss_mask[-1],
            "Source wire host header/dangling assistant cue invalid")
    for label, expected in (("<|tool_call|>", True),
                            ("<|tool_result|>", False),
                            ("<|memory_write|>", True)):
        locations = [i for i, token in enumerate(ids) if token == markers[label]]
        require(len(locations) == 1 and loss_mask[locations[0]] is expected,
                "Source role/label ownership mismatch at " + label)
    read = [i for i, token in enumerate(ids)
            if token == markers["<|memory_read|>"]]
    require(len(read) == 3 and [loss_mask[i] for i in read] ==
            [False, True, False],
            "Host/modeled memory-read ownership collapsed")
    end_positions = [i for i, token in enumerate(ids)
                     if token == markers["<|end|>"]]
    require(len(end_positions) == 12 and
            sum(bool(loss_mask[i]) for i in end_positions) == 4,
            "Publisher end-marker host/model supervision altered")
    require(sum(loss_mask) > 0, "Empty trained assistant action supervision")
    return {
        "sequence_units": len(ids), "assistant_loss_units": sum(loss_mask),
        "source_end_id5_count": len(end_positions),
        "source_end_id5_assistant_label_count": 4,
        "memory_read_host_model_host": [False, True, False],
        "tool_call_labelled": True, "host_tool_result_unlabelled": True,
        "assistant_memory_write_labelled": True,
    }


def explicit_left_pad(short: list[int], long: list[int], *, pad_id: int):
    """Synthetic pad ID 5 must not hide actual role-ending ID 5."""
    require(len(long) > len(short) > 5 and pad_id == 5,
            "Left-pad witness needs real distinct Sprocket sequences")
    offset = len(long) - len(short)
    ids = torch.tensor([[pad_id] * offset + short, long], dtype=torch.long)
    mask = torch.tensor([[0] * offset + [1] * len(short),
                         [1] * len(long)], dtype=torch.long)
    naive = (ids != pad_id).long()
    require(not torch.equal(naive, mask),
            "Naive token-equality attention-mask sentinel ineffective")
    require(bool((ids[0, offset:] == pad_id).any())
            and bool((ids[1] == pad_id).any()),
            "Actual <|end|> markers absent from content")
    require(bool(torch.all(mask[ids == pad_id][0:offset] == 0)),
            "Synthetic padding was not masked")
    require(bool(((ids == pad_id) & (mask == 1)).any()),
            "A genuine end delimiter was hidden")
    return ids, mask, offset


def run() -> None:
    protocol_path = ROOT / "experiments/ncr-001/protocol.json"
    pin_path = ROOT / "experiments/ncr-001/source-pins.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    pins = json.loads(pin_path.read_text(encoding="utf-8"))
    spec = verify_contract("instruct", protocol, pins)
    require(protocol["train_authorized"] is False,
            "Training admission changed")
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    require(os.getenv("NCR_EXPECTED_HEAD") == head and len(head) == 40,
            "Actual source head differs from pull request")
    require(os.getenv("GITHUB_EVENT_NAME") == "pull_request",
            "Only exact-head PR qualification allowed")
    require(os.getenv("HF_HUB_OFFLINE") == "1"
            and os.getenv("TRANSFORMERS_OFFLINE") == "1",
            "Exact frozen host must be loaded offline")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    torch.manual_seed(20261028)
    REPORT.update({
        "experiment": "NCR-001", "lane": "sprocket_instruct",
        "status": "STARTED", "original_frozen_model_executed": False,
        "source_structural_wire_qualified": False,
        "model_generated_tool_memory_semantics_qualified": False,
        "publisher_source_checkpoint_build_attested": False,
        "train_authorized": False, "optimizer_steps": 0,
        "comparative_training_performed": False,
        "heldout_opened": False, "full_72_conformance_pass": False,
        "source_identity": spec,
        "runner": {
            "head_sha": head, "run_id": os.getenv("GITHUB_RUN_ID"),
            "attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
            "runner_name": os.getenv("RUNNER_NAME"),
            "runner_os": os.getenv("RUNNER_OS"),
            "host": platform.node(), "pid": os.getpid(),
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "source_sha256": {
            name: digest(ROOT / name) for name in (
                "scripts/ncr_001_postlease_sprocket_wire.py",
                "scripts/ncr_001_postlease_sprocket.py",
                "scripts/ncr_001_backbone_lease.py",
                "experiments/ncr-001/protocol.json",
                "experiments/ncr-001/source-pins.json",
            )
        },
        "wire_authority": {
            "reviewed_tree": WIRE_HEAD, "reviewed_renderer_blob": WIRE_BLOB,
            "format_source_tree": SOURCE_HEAD,
        },
        "checks": {},
    })
    save()
    markers, render_ids_mask, render_text = wire_authority()
    REPORT["wire_authority"]["renderer_sha256"] = digest(WIRE_TREE / WIRE_FILE)
    save()
    formatter = load_formatter()
    model, tokenizer, nca, mlp, hashes, serialize = load_actual(
        "instruct", spec, pins["models"]["sprocket_instruct"], formatter)
    REPORT["original_frozen_model_executed"] = True
    REPORT["verified_source_files"] = hashes
    before = fingerprint_backbone(model)
    REPORT["in_memory_frozen_before_sha256"] = before
    save()
    turns = synthetic_turns()
    long_ids, loss = render_ids_mask(turns, tokenizer, variant="instruct")
    summary = assert_source_ownership(long_ids, loss, markers)
    raw_text = render_text(turns, variant="instruct")
    require(long_ids == tokenizer.encode(raw_text, add_special_tokens=False),
            "Whole-source wire serialization mismatch")
    short_ids, short_loss = render_ids_mask(
        turns[:3], tokenizer, variant="instruct")
    require(not any(short_loss), "Host-only short wire labels unexpectedly trained")
    ids, attention, offset = explicit_left_pad(
        short_ids, long_ids, pad_id=tokenizer.pad_token_id)
    summary.update({
        "prompt_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "ids_sha256": hashlib.sha256(torch.tensor(long_ids).numpy().tobytes()).hexdigest(),
        "assistant_mask_sha256": hashlib.sha256(
            bytes(int(v) for v in loss)).hexdigest(),
        "attention_mask_sha256": hashlib.sha256(
            attention.numpy().tobytes()).hexdigest(),
        "padded_shape": list(ids.shape), "synthetic_pad_positions": offset,
        "real_end_marker_id_5_visible_with_attention": True,
        "naive_attention_mask_detected_invalid": True,
        "source_is_diagnostic_not_heldout": True,
    })
    REPORT["wire"] = summary
    REPORT["source_structural_wire_qualified"] = True
    save()
    atol = float(protocol["conformance"]["full_vs_cached_prefill_atol"])
    rtol = float(protocol["conformance"]["full_vs_cached_prefill_rtol"])
    require(atol == rtol == 0.0005,
            "Preregistered Sprocket causal tolerance changed")
    full = torch.tensor([long_ids], dtype=torch.long)
    full_mask = torch.ones_like(full)
    with torch.inference_mode():
        for kind, wrapper in (("nca", nca), ("mlp", mlp)):
            raw = model(input_ids=full, attention_mask=full_mask,
                        use_cache=False).logits
            zero = wrapper(input_ids=full, attention_mask=full_mask,
                           use_cache=False).logits
            disabled = wrapper(input_ids=full, attention_mask=full_mask,
                               use_cache=False, enabled=False).logits
            compare(kind+"_zero_vs_original",
                         raw, zero, atol=0, rtol=0, exact=True)
            compare(kind+"_disabled_vs_original",
                         raw, disabled, atol=0, rtol=0, exact=True)
            checked(kind+"_zero_disabled_parity", {"pass": True, "exact": True})
            if kind == "nca":
                wrapper.nca.readout.weight[0, 0] = 0.03
            else:
                wrapper.control.down.weight[0, 0] = 0.05
            active = wrapper(input_ids=full, attention_mask=full_mask,
                             use_cache=False).logits
            delta = float((active - raw).abs().max())
            require(delta > 0 and torch.isfinite(active).all().item(),
                    kind + ": manual untrained adapter has no finite effect")
            checked(kind+"_untrained_manual_effect",
                    {"pass": True, "max_abs": delta, "trained": False})
            # Source-bound left padding: genuine END=5 tokens must stay visible
            # while synthetic END=5 padding must be masked.
            batch = wrapper(input_ids=ids, attention_mask=attention,
                            use_cache=False).logits
            a = wrapper(input_ids=torch.tensor([short_ids]),
                        attention_mask=torch.ones((1,len(short_ids)),dtype=torch.long),
                        use_cache=False).logits
            b = wrapper(input_ids=full, attention_mask=full_mask,
                        use_cache=False).logits
            compare(kind+"_left_pad_short",
                         batch[0, offset:], a[0], atol=atol, rtol=rtol)
            compare(kind+"_left_pad_long",
                         batch[1], b[0], atol=atol, rtol=rtol)
            checked(kind+"_explicit_left_pad", {"pass": True, "batch":list(ids.shape)})
            # Authentic tool and memory positions, not arbitrary raw control IDs.
            boundaries = {
                "assistant_tool_call": long_ids.index(markers["<|tool_call|>"]),
                "host_tool_result": long_ids.index(markers["<|tool_result|>"]),
                "assistant_memory_write": long_ids.index(markers["<|memory_write|>"]),
                "host_memory_read": long_ids.index(markers["<|memory_read|>"]),
            }
            for label, start in boundaries.items():
                tail = min(4, len(long_ids) - start)
                require(start > 0 and tail >= 2, "Missing source wire cache boundary")
                pre = wrapper(input_ids=full[:, :start],
                              attention_mask=full_mask[:, :start],
                              use_cache=True)
                past = pre.past_key_values
                parts = []
                for at in range(start, start+tail):
                    one = wrapper(input_ids=full[:, at:at+1],
                                  attention_mask=full_mask[:, :at+1],
                                  past_key_values=past, use_cache=True)
                    parts.append(one.logits[:, -1:, :])
                    past = one.past_key_values
                cached = torch.cat(parts, dim=1)
                # Delegating the numeric check directly into the persistent
                # receipt ensures failure cannot be re-labelled as PASS.
                error = float((active[:, start:start+tail] - cached).abs().max())
                good = bool(torch.allclose(
                    active[:, start:start+tail], cached, atol=atol, rtol=rtol))
                checked(kind+"_cache_"+label, {
                    "pass": good, "max_abs": error, "atol": atol, "rtol": rtol,
                    "boundary_position": start, "steps": tail})
                require(good, kind + ": cached source wire boundary drift " + label)
            # Record raw generated IDs as diagnostics, NEVER semantic scores.
            gen_kw = dict(attention_mask=full_mask, do_sample=False,
                          use_cache=True, max_new_tokens=6, min_new_tokens=6,
                          pad_token_id=tokenizer.pad_token_id)
            generated_disabled = wrapper.generate(
                input_ids=full, enabled=False, **gen_kw)[:, -6:]
            # Check a genuinely independent zero adapter, not manually active.
            # Original raw tokens are intentionally retained; not judged valid.
            generated_active = wrapper.generate(input_ids=full, **gen_kw)[:, -6:]
            checked(kind+"_raw_untrained_generation", {
                "source_only_diagnostic": True, "semantic_score": None,
                "disabled_token_ids": generated_disabled.tolist()[0],
                "active_token_ids": generated_active.tolist()[0],
                "disabled_text": tokenizer.decode(
                    generated_disabled[0], skip_special_tokens=False),
                "active_text": tokenizer.decode(
                    generated_active[0], skip_special_tokens=False),
            })
        after = fingerprint_backbone(model)
        REPORT["in_memory_frozen_after_sha256"] = after
        require(after == before, "Frozen root Instruct model bytes changed")
        require(all(p.grad is None for p in model.parameters()),
                "Frozen original backbone accumulated gradients")
    REPORT["status"] = "PASS_SOURCE_STRUCTURED_POSTLEASE_WIRE_CACHE_ONLY"
    REPORT["model_generated_tool_memory_semantics_qualified"] = False
    REPORT["full_72_conformance_pass"] = False
    REPORT["limitations"] = [
        "Publisher SFT Git source is not cryptographic historical model-build attestation",
        "Synthetic host tool/memory turns and assistant control labels are not useful model-generated tool/memory semantics",
        "Real frozen model and manually untrained NCA/MLP only; zero optimizer and no held-out fixture",
        "No runtime tool execution, stateful memory service, NCA/MLP training or competence admission",
        "Independent scientific review and all four #72 model admissions remain blocked",
    ]
    save()


def main() -> None:
    try:
        run()
    except BaseException as exc:
        REPORT["status"] = "FAIL_SOURCE_STRUCTURED_POSTLEASE_WIRE_CACHE"
        REPORT["exception_type"] = type(exc).__name__
        REPORT["traceback"] = traceback.format_exc()
        raise
    finally:
        save()
        print(json.dumps({
            "lane":"sprocket_instruct", "status":REPORT.get("status"),
            "source_head":REPORT.get("runner",{}).get("head_sha"),
            "train_authorized":False, "optimizer_steps":0,
            "model_semantic_admission":False,
        }, sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
