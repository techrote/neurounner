#!/usr/bin/env python3
"""Independent Sprocket Chat HF.generate/cache conformance; no training.

Fills the explicit #72 gap left by earlier Chat evidence: the original Chat
checkpoint had source formatting, padding, cache, causal isolation and
separate-process cold replay evidence, but only HF generation with cache
enabled. This reviewer compares original GenerationMixin with cache on/off and
a right-padded masked KV gap using ordinary source-documented Chat roles.

Synthetic prompts only. No tool/memory grammar is inherited from Instruct.
No optimizer, backward pass, held-out fixture, training or quality scoring.
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
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime/ncr-001/sprocket-chat-generate-review.json"

HF_REPO = "kandivault/sprocket-500m"
HF_REV = "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
CHAT_SHA = "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97"
INSTRUCT_SHA = "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864"
HOOK_SHA = "83a56835a7476f17a17980d28dd1a7c14bb6f2f6"
FORMAT_BLOB = "39cf015735e74a1d1e08682b623404bc1e600b9b"
SOURCE_CARD_SHA = "09aec879dc5b387bdc031dc3c74d77a9ba44362a55163f9a8b456761b68142c9"

PAD = 1
END = 5
TAIL = 4
ATOL = 0.0005
RTOL = 0.0005
MODEL_KEYS = {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"}

PROMPT_A = (
    {"role": "user",
     "content": "A conveyor bearing warms slowly. What should be inspected first?"},
    {"role": "assistant",
     "content": "Inspect the housing, lubrication, alignment, load, and nearby vibration."},
    {"role": "user",
     "content": "List the observations to record before replacing any part."},
)
PROMPT_B = (
    {"role": "user",
     "content": "A latch clicks twice. What should be noted before touching it?"},
)


def require(ok, reason):
    if not ok:
        raise RuntimeError("NCR Sprocket Chat reviewer REFUSED: " + reason)


def file_sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev(path: Path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def git_blob(path: Path, relative: str):
    return subprocess.check_output(
        ["git", "-C", str(path), "hash-object", relative], text=True
    ).strip()


def compare(report, name, x, y, *, atol=ATOL, rtol=RTOL, exact=False):
    import torch
    require(x.shape == y.shape, name + " shape changed")
    require(bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
            name + " nonfinite logits")
    max_abs = float((x.float() - y.float()).abs().max())
    passed = bool(torch.allclose(x, y, atol=atol, rtol=rtol))
    if exact:
        passed &= bool(torch.equal(x, y))
    report["checks"][name] = {
        "pass": passed, "max_abs": max_abs, "atol": atol, "rtol": rtol,
        "exact_required": exact,
    }
    require(passed, name + " numerical/source mismatch")
    return max_abs


def compare_generation_scores(report, name, a, b, *,
                              atol=ATOL, rtol=RTOL, exact=False):
    """Compare processed generate scores without NaN from identical -inf masks."""
    import torch
    require(a.shape == b.shape, name + " score shape changed")
    finite = torch.isfinite(a)
    require(bool(torch.equal(finite, torch.isfinite(b))),
            name + " nonfinite score masks differ")
    require(bool(torch.equal(a[~finite], b[~finite])),
            name + " masked source logits differ")
    require(bool(finite.any()), name + " no finite generation scores")
    max_abs = float((a[finite] - b[finite]).abs().max())
    passed = bool(torch.allclose(a[finite], b[finite], atol=atol, rtol=rtol))
    if exact:
        passed &= bool(torch.equal(a, b))
    report["checks"][name] = {
        "pass": passed, "max_abs_finite": max_abs,
        "atol": atol, "rtol": rtol, "exact_required": exact,
        "finite_logits": int(finite.sum()), "nonfinite_logits": int((~finite).sum()),
    }
    require(passed, name + " generation cache/source mismatch")
    return max_abs


def right_gap(seqs):
    require(isinstance(seqs, list) and len(seqs) == 2
            and all(isinstance(s, list) and len(s) > TAIL + 8 for s in seqs),
            "two sufficiently long source role sequences required")
    prefixes = [s[:-TAIL] for s in seqs]
    width = max(map(len, prefixes))
    require(len(prefixes[0]) > len(prefixes[1]) + 8,
            "reviewer ordinary Chat prompts do not create useful padding gap")
    ids, masks, pos = [], [], []
    for s in prefixes:
        n = len(s)
        ids.append(s + [PAD] * (width - n))
        masks.append([1] * n + [0] * (width - n))
        pos.append(list(range(n)) + [0] * (width - n))
    return {
        "ids": ids, "mask": masks, "positions": pos,
        "valid_prefix_lengths": [len(s) for s in prefixes],
        "width": width,
        "tail_tokens": [[seqs[row][-TAIL + i] for row in range(2)]
                        for i in range(TAIL)],
        "tail_positions": [[len(prefixes[row]) + i for row in range(2)]
                           for i in range(TAIL)],
    }


def source_only_plan():
    require(len(PROMPT_A) == 3 and len(PROMPT_B) == 1
            and all(row["role"] in {"user", "assistant"}
                    for row in PROMPT_A + PROMPT_B)
            and PROMPT_A[0]["role"] == PROMPT_A[-1]["role"] ==
            PROMPT_B[0]["role"] == "user"
            and TAIL == 4 and PAD == 1 and END == 5,
            "fixed synthetic ordinary Chat plan changed")
    require(CHAT_SHA != INSTRUCT_SHA,
            "Chat and root Instruct checkpoint identity collapsed")
    return True


def guard_claims(report):
    require(report["variant"] == "chat"
            and report["train_authorized"] is False
            and report["optimizer_steps"] == 0
            and report["heldout_evaluation"] is False
            and report["actual_training"] is False
            and report["tool_memory_semantics_qualified"] is False
            and report["final_four_model_72_conformance"] is False,
            "false Chat semantic/training/final admission claim")


def execute():
    import torch
    import transformers
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()
    report = {
        "schema_version": 1,
        "experiment": "NCR-001E",
        "stage": "independent_sprocket_chat_actual_HF_generate_cache_on_off_and_right_gap",
        "status": "RUNNING",
        "variant": "chat",
        "train_authorized": False, "optimizer_steps": 0,
        "heldout_evaluation": False, "actual_training": False,
        "tool_memory_semantics_qualified": False,
        "final_four_model_72_conformance": False,
        "checks": {}, "generations": [],
        "runner": {
            "github_run_id": os.getenv("GITHUB_RUN_ID"),
            "github_sha": os.getenv("GITHUB_SHA"),
            "run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
            "runner_name": os.getenv("RUNNER_NAME"),
            "pid": os.getpid(), "cpu_count": os.cpu_count(),
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "transformers": transformers.__version__,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report["wall_seconds"] = time.monotonic() - started
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        OUT.write_text(json.dumps(report, indent=2, sort_keys=True,
                                  allow_nan=False) + "\n", encoding="utf-8")

    def tensor_hash(model):
        h = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            h.update((name + str(tensor.shape) + str(tensor.dtype)).encode())
            h.update(tensor.detach().contiguous().cpu().numpy().tobytes())
        return h.hexdigest()

    save()
    try:
        require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1",
                "actual frozen Chat execution must be offline")
        require(source_only_plan(), "source-only plan changed")
        impl = ROOT / "implementation"
        require(git_rev(impl) == HOOK_SHA,
                "original Chat NCA hook checkout changed")
        require(git_blob(ROOT, "scripts/ncr_001e_sprocket_format.py") == FORMAT_BLOB,
                "ordinary source formatter changed")
        protocol_path = ROOT / "experiments/ncr-001/protocol.json"
        pins_path = ROOT / "experiments/ncr-001/source-pins.json"
        protocol, pins = json.loads(protocol_path.read_text()), json.loads(pins_path.read_text())
        require(protocol["train_authorized"] is False
                and set(protocol["models"]) == set(pins["models"]) == MODEL_KEYS
                and protocol["conformance"]["full_vs_cached_prefill_atol"] ==
                protocol["conformance"]["full_vs_cached_prefill_rtol"] == ATOL == RTOL,
                "four-model protocol or cache gate changed")
        pin = pins["models"]["sprocket_chat"]
        root = pins["models"]["sprocket_instruct"]
        require(pin["variant"] == "chat" and pin["hf_weight_file"] ==
                protocol["models"]["sprocket_chat"]["weight_file"] ==
                "chat/model.safetensors"
                and pin["weight_declared_lfs_sha256_not_downloaded"] == CHAT_SHA
                and root["hf_weight_file"] == "model.safetensors"
                and root["weight_declared_lfs_sha256_not_downloaded"] == INSTRUCT_SHA,
                "Chat/root checkpoint source identity changed")
        report["source"] = {
            "repo_id": HF_REPO, "revision": HF_REV,
            "checkpoint_file": "chat/model.safetensors",
            "checkpoint_sha256": CHAT_SHA, "checkpoint_bytes": 1002207984,
            "distinct_root_instruct_sha256": INSTRUCT_SHA,
            "original_hook_commit": HOOK_SHA, "formatter_git_blob": FORMAT_BLOB,
            "model_card_sha256": SOURCE_CARD_SHA,
            "protocol_sha256": file_sha(protocol_path),
            "source_pins_sha256": file_sha(pins_path),
            "ordinary_roles_only": True,
            "tool_memory_semantics_inherited_from_instruct": False,
        }
        save()

        # The workflow prefetched the full Chat subfolder and all metadata.
        source_files = {
            "README.md": (None, None, None),
            "chat/model.safetensors": ("weight_declared_lfs_sha256_not_downloaded",
                                      1002207984, "chat"),
            "chat/config.json": ("chat_config_sha256", 689, "chat"),
            "chat/tokenizer.json": ("chat_tokenizer_sha256", 2269469, "chat"),
            "chat/tokenizer_config.json": ("chat_tokenizer_config_sha256", 3325, "chat"),
        }
        for name, (key, expected_size, subfolder) in source_files.items():
            filename = name.split("/", 1)[-1] if subfolder else name
            kwargs = {"revision": HF_REV, "local_files_only": True}
            if subfolder:
                kwargs["subfolder"] = subfolder
            path = Path(hf_hub_download(HF_REPO, filename, **kwargs))
            digest = file_sha(path)
            if key is not None:
                require(digest == pin[key], "pinned Chat source SHA changed: " + name)
            if expected_size is not None:
                require(path.stat().st_size == expected_size,
                        "pinned Chat source byte length changed: " + name)
            report.setdefault("actual_source_files", {})[name] = {
                "sha256": digest, "bytes": path.stat().st_size,
            }
            save()
        require(report["actual_source_files"]["README.md"]["sha256"] == SOURCE_CARD_SHA,
                "pinned model card bytes changed")

        sys.path.insert(0, str(ROOT / "scripts"))
        from ncr_001e_sprocket_format import (
            serialize_dialogue, validate_pinned_card, validate_pinned_tokenizer
        )
        card = Path(hf_hub_download(HF_REPO, "README.md",
                                    revision=HF_REV, local_files_only=True)).read_text()
        validate_pinned_card(card)
        tokenizer = AutoTokenizer.from_pretrained(
            HF_REPO, revision=HF_REV, subfolder="chat", local_files_only=True)
        inventory = validate_pinned_tokenizer(tokenizer, variant="chat")
        require(inventory["tool_and_memory_semantics_qualified"] is False,
                "Chat ordinary formatter cannot claim Instruct tool/memory semantics")
        require(tokenizer.convert_tokens_to_ids("<|user|>") == 3
                and tokenizer.convert_tokens_to_ids("<|assistant|>") == 4
                and tokenizer.convert_tokens_to_ids("<|end|>") == END
                and tokenizer.convert_tokens_to_ids("<|pad|>") == PAD,
                "source Chat role/padding token IDs changed")

        seqs = []
        for label, rows in (("long_chat", PROMPT_A), ("short_chat", PROMPT_B)):
            rendered = serialize_dialogue(rows, variant="chat")
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            require(ids and ids[0] == 3 and ids[-1] == 4
                    and all(type(x) is int and 0 <= x < 32000 for x in ids),
                    "ordinary role source IDs invalid")
            seqs.append(ids)
            report.setdefault("wire", {})[label] = {
                "tokenizer_units": len(ids),
                "ids_sha256": hashlib.sha256(
                    torch.tensor(ids, dtype=torch.int64).numpy().tobytes()).hexdigest(),
                "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "source_role_format_only": True,
            }
        gap = right_gap(seqs)
        report["right_gap"] = {
            "prefill_width": gap["width"],
            "valid_prefix_lengths": gap["valid_prefix_lengths"],
            "tail_units": TAIL, "pad_id": PAD,
        }
        save()

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(20261027)
        base = AutoModelForCausalLM.from_pretrained(
            HF_REPO, revision=HF_REV, subfolder="chat", local_files_only=True,
            torch_dtype=torch.float32, attn_implementation="eager",
            trust_remote_code=False).eval()
        sys.path.insert(0, str(impl / "scripts"))
        from ncr_001_sprocket_chat import FrozenSprocketChatDepthNCA, POST_BLOCK_TAPS
        require(tuple(POST_BLOCK_TAPS) == (7, 16, 25),
                "original Chat depth taps changed")
        wrapper = FrozenSprocketChatDepthNCA(
            base, verified_weight_sha256=CHAT_SHA).eval()
        wrapper.verify_frozen_base()
        require(wrapper.nca.trainable_parameters() == 658904
                and not any(p.requires_grad for p in base.parameters()),
                "frozen Chat/adapter parameter geometry changed")
        report["frozen_base_parameters"] = wrapper.frozen_parameters()
        report["adapter_parameters"] = wrapper.nca.trainable_parameters()
        report["backbone_sha256_before"] = tensor_hash(base)
        save()

        with torch.inference_mode():
            frozen = []
            for label, seq in zip(("long_chat", "short_chat"), seqs):
                x = torch.tensor([seq], dtype=torch.long)
                direct = wrapper(input_ids=x, enabled=False, use_cache=False).logits
                zero = wrapper(input_ids=x, enabled=True, use_cache=False).logits
                compare(report, "frozen_zero_exact_" + label, direct, zero,
                        atol=0, rtol=0, exact=True)
                frozen.append(direct)
            with torch.no_grad():
                wrapper.nca.readout.weight[0, 0] = 0.03
            active_refs = [
                wrapper(input_ids=torch.tensor([seq], dtype=torch.long),
                        use_cache=False).logits[:, -TAIL:]
                for seq in seqs
            ]
            effect = max(float((active_refs[i] - frozen[i][:, -TAIL:]).abs().max())
                         for i in range(2))
            require(effect > 0, "manually activated untrained Chat NCA had no effect")
            report["checks"]["manual_untrained_effect"] = {
                "max_abs": effect, "trained": False, "pass": True}

            def cached_gap(*, replacement_pad=PAD, unmask=False):
                ids = torch.tensor(gap["ids"], dtype=torch.long).clone()
                mask = torch.tensor(gap["mask"], dtype=torch.long).clone()
                pos = torch.tensor(gap["positions"], dtype=torch.long).clone()
                if replacement_pad != PAD:
                    ids[mask == 0] = replacement_pad
                if unmask:
                    mask[mask == 0] = 1
                prefill = wrapper(input_ids=ids, attention_mask=mask,
                                  position_ids=pos, use_cache=True)
                past = prefill.past_key_values
                require(len(wrapper.last_hook_trace) == 1
                        and wrapper.last_hook_trace[0]["state_shape"][:2] ==
                        [2, gap["width"]],
                        "Chat right-gap prefill field shape/state invalid")
                pieces = []
                for step in range(TAIL):
                    token = torch.tensor([[v] for v in gap["tail_tokens"][step]],
                                         dtype=torch.long)
                    positions = torch.tensor(
                        [[v] for v in gap["tail_positions"][step]], dtype=torch.long)
                    full_mask = torch.cat(
                        [mask, torch.ones((2, step + 1), dtype=torch.long)], dim=1)
                    out = wrapper(input_ids=token, attention_mask=full_mask,
                                  position_ids=positions,
                                  past_key_values=past, use_cache=True)
                    pieces.append(out.logits)
                    past = out.past_key_values
                    require(len(wrapper.last_hook_trace) == 1
                            and wrapper.last_hook_trace[0]["state_shape"][:2] == [2, 1],
                            "Chat cached per-token NCA state did not reset")
                return torch.cat(pieces, dim=1)

            cached = cached_gap()
            for row, label in enumerate(("long_chat", "short_chat")):
                compare(report, "rightgap_vs_unpadded_" + label,
                        cached[row:row+1], active_refs[row])
            swapped = cached_gap(replacement_pad=7)
            compare(report, "masked_padding_control_id7_exact",
                    cached, swapped, atol=0, rtol=0, exact=True)
            bad = cached_gap(unmask=True)
            bad_delta = float((bad[1] - cached[1]).abs().max())
            require(bad_delta > 0, "deliberately invalid Chat attention mask undetected")
            report["checks"]["unmasked_padding_negative_detected"] = {
                "max_abs": bad_delta, "expected_difference": True, "pass": True}
            save()

            gen_kwargs = dict(
                input_ids=torch.tensor([seqs[0]], dtype=torch.long),
                attention_mask=torch.ones((1, len(seqs[0])), dtype=torch.long),
                do_sample=False, max_new_tokens=4, min_new_tokens=4,
                eos_token_id=END, pad_token_id=PAD,
                return_dict_in_generate=True, output_scores=True,
            )
            runs = {}
            for mode in ("disabled", "zero", "active_untrained"):
                with torch.no_grad():
                    wrapper.nca.readout.weight.zero_()
                    if mode == "active_untrained":
                        wrapper.nca.readout.weight[0, 0] = 0.03
                for use_cache in (False, True):
                    response = wrapper.generate(
                        **gen_kwargs, use_cache=use_cache,
                        enabled=(mode != "disabled"))
                    ids = response.sequences[0].tolist()
                    scores = list(response.scores)
                    require(len(ids) == len(seqs[0]) + 4 and len(scores) == 4,
                            "Chat source generation did not produce fixed 4 units")
                    if mode != "disabled":
                        traces = wrapper.last_hook_trace
                        require(len(traces) == 4
                                and all(t["tap_count"] == 3 and t["finite"]
                                        for t in traces),
                                "Chat generate hook path did not complete three taps")
                        lengths = [t["state_shape"][1] for t in traces]
                        expected = ([len(seqs[0]), 1, 1, 1] if use_cache else
                                    [len(seqs[0]), len(seqs[0])+1,
                                     len(seqs[0])+2, len(seqs[0])+3])
                        require(lengths == expected,
                                "Chat source state did not reset per generation forward")
                    runs[(mode, use_cache)] = (ids, scores)
                    report["generations"].append({
                        "mode": mode, "use_cache": use_cache,
                        "prompt_units": len(seqs[0]),
                        "generated_ids": ids[len(seqs[0]):],
                        "score_sha256": [
                            hashlib.sha256(s.contiguous().cpu().numpy().tobytes()).hexdigest()
                            for s in scores],
                        "tool_memory_semantics_qualified": False,
                        "optimizer_steps": 0,
                    })
                    save()

            for use_cache in (False, True):
                disabled_ids, disabled_scores = runs[("disabled", use_cache)]
                zero_ids, zero_scores = runs[("zero", use_cache)]
                require(disabled_ids == zero_ids,
                        "zero Chat NCA changed original generation IDs")
                for step in range(4):
                    compare_generation_scores(
                        report,
                        f"generate_zero_exact_cache_{use_cache}_step_{step}",
                        disabled_scores[step], zero_scores[step],
                        atol=0, rtol=0, exact=True)

            for mode in ("disabled", "zero", "active_untrained"):
                ids_no, scores_no = runs[(mode, False)]
                ids_cache, scores_cache = runs[(mode, True)]
                require(ids_no == ids_cache,
                        "Chat generation token IDs depend on cache mode: " + mode)
                for step in range(4):
                    compare_generation_scores(
                        report, f"generate_full_vs_cache_{mode}_step_{step}",
                        scores_no[step], scores_cache[step])

            with torch.no_grad():
                wrapper.nca.readout.weight.zero_()
            a = wrapper(input_ids=torch.tensor([seqs[0][:16]], dtype=torch.long),
                        use_cache=False).logits
            wrapper(input_ids=torch.tensor([seqs[1]], dtype=torch.long),
                    use_cache=False)
            b = wrapper(input_ids=torch.tensor([seqs[0][:16]], dtype=torch.long),
                        use_cache=False).logits
            compare(report, "post_generation_A_B_A_exact", a, b,
                    atol=0, rtol=0, exact=True)

        report["backbone_sha256_after"] = tensor_hash(base)
        require(report["backbone_sha256_before"] == report["backbone_sha256_after"],
                "frozen Chat base tensor bytes changed")
        require(all(p.grad is None for p in base.parameters()),
                "frozen Chat base accumulated gradients")
        require(all(not m._forward_hooks and not m._forward_pre_hooks
                    for m in base.modules()), "Chat generation hook leaked")
        guard_claims(report)
        report["status"] = "PASS_CHAT_RIGHTGAP_AND_HF_GENERATE_PRETRAIN_ONLY"
        report["limits"] = [
            "Ordinary source-documented Chat roles only; no Instruct tool/memory semantics",
            "Synthetic source prompts, no heldout quality or behavioral scoring",
            "Manual untrained readout intervention, zero optimizer/backward steps",
            "Not final four-model #72 admission or training authorization",
        ]
    except BaseException:
        report["status"] = "FAIL_CHAT_RIGHTGAP_OR_HF_GENERATE_PRETRAIN"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        save()

    print(json.dumps({"status": report["status"],
                      "checks": report["checks"],
                      "train_authorized": False}, sort_keys=True))


if __name__ == "__main__":
    if "--source-test" in sys.argv:
        print(json.dumps({"source_plan": bool(source_only_plan()),
                          "variant": "chat", "train_authorized": False}))
    else:
        execute()
