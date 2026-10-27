#!/usr/bin/env python3
"""NCR-001E PR #113: finite original Instruct right-padding/cache/generate review.

Reviewer-owned source-matched NO-TRAINING test, distinct from original 133/31
left-padded receipt. Synthetic messages only; do not open held-out suites or
claim behavioral tool/memory competence. All numerical failures are retained.
"""
from __future__ import annotations

import argparse
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

from ncr_001e_pr102_masked_batch_review import (
    PRIMARY, SECONDARY, ROOT_SHA, CHAT_SHA, HF_REPO, HF_REV,
    SOURCE_IMPL, PUBLISHER, TARGET, UPSTREAM_SFT_SHA, UPSTREAM_WIRE_BLOB,
    independent_wire, batch_layout, check_synthetic_plan,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime/ncr-001/pr113-rightpad-generate.json"
REVIEWER_SOURCE_BLOB = "6ade653f0452e58f82c8c78e79d326018c30bcf1"
GENERATE_TOKENS = 4
PAD = 1
END = 5
CACHED_ATOL = 0.0005
CACHED_RTOL = 0.0005
MODELS = ("smollm2", "sprocket_chat", "sprocket_instruct", "void1")


def require(value, message):
    if not value:
        raise RuntimeError("NCR finite rightpad/generate source refusal: " + message)


def hash_bytes(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_rev(path: Path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                   text=True).strip()


def git_blob(root: Path, relative: str):
    return subprocess.check_output(["git", "-C", str(root), "hash-object", relative],
                                   text=True).strip()


def check_plan():
    require(check_synthetic_plan(), "original independent synthetic dialogue changed")
    require(GENERATE_TOKENS == 4 and PAD == 1 and END == 5,
            "Generation/cache envelope changed")
    require(PRIMARY[-1][0] == SECONDARY[-1][0] == "user",
            "source prompts must end with host turn")
    require(len(PRIMARY) == 12 and len(SECONDARY) == 3,
            "fixed reviewer source message count changed")
    return True


def gapped_right_prefix(sequences, tail=GENERATE_TOKENS, pad=PAD):
    """Model actual right-padding KV gap instead of left-padding from #113.

    For variable-length requests, prefill all but their final `tail` true
    tokens, right-pad shorter prefixes, and APPEND each real final token after
    the KV gap with an attention mask excluding the holes. Position IDs count
    only valid source tokens; full-reference logits are per request.
    """
    require(isinstance(sequences, (list, tuple)) and len(sequences) == 2
            and all(isinstance(s, list) and len(s) > tail + 8
                    for s in sequences), "exactly two nonempty sources required")
    require(tail == GENERATE_TOKENS and pad == PAD,
            "fixed gapped source geometry drift")
    prefixes = [s[:-tail] for s in sequences]
    width = max(map(len, prefixes))
    require(len(prefixes[0]) > len(prefixes[1]) + 8,
            "right-padding source gap must be substantial")
    input_rows, masks, positions = [], [], []
    for s in prefixes:
        n = len(s)
        input_rows.append(s + [pad] * (width - n))
        masks.append([1] * n + [0] * (width - n))
        positions.append(list(range(n)) + [0] * (width - n))
    tails = [[s[-tail + i] for s in sequences] for i in range(tail)]
    pos_tails = [[len(prefixes[j]) + i for j in range(2)]
                 for i in range(tail)]
    require(all(len(row) == width for row in input_rows + masks + positions),
            "invalid right-padding review dimensions")
    return {
        "prefix_ids": input_rows,
        "prefix_mask": masks,
        "prefix_position_ids": positions,
        "incremental_tokens": tails,
        "incremental_position_ids": pos_tails,
        "valid_prefix_lengths": list(map(len, prefixes)),
        "width": width,
    }


def masked_score_comparison(report, name, reference, candidate,
                            *, exact=False, atol=CACHED_ATOL,
                            rtol=CACHED_RTOL):
    """Compare HF generation postprocessor scores including masked -Inf.

    Generated EOS may be masked by `min_new_tokens`. Check identical inf masks,
    then compare only finite logits elementwise; never subtract -Inf from -Inf.
    """
    import torch
    require(reference.shape == candidate.shape,
            name + " score shape drift")
    fm = torch.isfinite(reference)
    require(bool(torch.equal(fm, torch.isfinite(candidate))),
            name + " generation finite masks diverged")
    require(bool(torch.equal(reference[~fm], candidate[~fm])),
            name + " nonfinite generation logits differ")
    require(bool(fm.any()), name + " all original logits masked")
    error = float((reference[fm] - candidate[fm]).abs().max())
    passed = bool(torch.allclose(reference[fm], candidate[fm],
                                 atol=atol, rtol=rtol))
    if exact:
        passed &= bool(torch.equal(reference, candidate))
    report["checks"][name] = {
        "max_abs_finite": error, "pass": passed, "exact_required": exact,
        "atol": atol, "rtol": rtol,
        "finite_logits": int(fm.sum()),
        "nonfinite_logits": int((~fm).sum()),
    }
    require(passed, name + " source generation score/causal gate failed")
    return error


def compare(report, name, x, y, *, exact=False,
            atol=CACHED_ATOL, rtol=CACHED_RTOL):
    import torch
    require(x.shape == y.shape, name + " shape mismatch")
    require(bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
            name + " nonfinite output")
    max_error = float((x.float() - y.float()).abs().max())
    passed = bool(torch.allclose(x, y, atol=atol, rtol=rtol))
    if exact:
        passed &= bool(torch.equal(x, y))
    report["checks"][name] = {
        "pass": passed, "max_abs": max_error,
        "exact_required": exact, "atol": atol, "rtol": rtol,
    }
    require(passed, name + " source/causal parity failed")
    return max_error


def guard_claims(report):
    require(report["train_authorized"] is False
            and report["optimizer_steps"] == 0
            and report["heldout_evaluation"] is False
            and report["any_model_training"] is False
            and report["semantic_tool_memory_qualified"] is False
            and report["final_four_model_72_conformance"] is False
            and report["variant"] == "sprocket_instruct",
            "training, tool semantics or final admission falsely promoted")


def execute():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download
    report = {
        "schema": 1,
        "experiment": "NCR-001E",
        "status": "RUNNING",
        "stage": "independent_right_padded_gapped_KV_and_real_HF_generate_source_review",
        "variant": "sprocket_instruct",
        "reviewed_pr": 102, "reviewed_head": TARGET,
        "reviewer_original_pr": 113,
        "train_authorized": False, "optimizer_steps": 0,
        "any_model_training": False,
        "heldout_evaluation": False,
        "semantic_tool_memory_qualified": False,
        "final_four_model_72_conformance": False,
        "checks": {}, "generations": [], "runner": {
            "github_run_id": os.getenv("GITHUB_RUN_ID"),
            "attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
            "github_sha": os.getenv("GITHUB_SHA"),
            "runner_name": os.getenv("RUNNER_NAME"),
            "pid": os.getpid(), "python": platform.python_version(),
            "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "torch": torch.__version__, "transformers": transformers.__version__,
        }
    }
    started = time.monotonic()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report["wall_seconds"] = time.monotonic() - started
        report["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        OUT.write_text(json.dumps(report, indent=2, sort_keys=True,
                                  allow_nan=False) + "\n", encoding="utf-8")
    save()

    def hash_tensors(model):
        h = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            h.update((name + str(tensor.shape) + str(tensor.dtype)).encode())
            h.update(tensor.detach().contiguous().cpu().numpy().tobytes())
        return h.hexdigest()

    try:
        require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1",
                "All actual source runtime must be offline")
        require(check_plan(), "source fixture review envelope changed")
        for directory, revision in (
            ("candidate", TARGET), ("implementation", SOURCE_IMPL),
            ("publisher", PUBLISHER),
        ):
            require(git_rev(ROOT / directory) == revision,
                    "original source revision drift: " + directory)
        require(git_blob(ROOT / "candidate",
                         "scripts/ncr_001e_sprocket_publisher_wire.py") ==
                UPSTREAM_WIRE_BLOB, "PR #102 candidate source changed")
        require(git_blob(ROOT,
                         "scripts/ncr_001e_pr102_masked_batch_review.py") ==
                REVIEWER_SOURCE_BLOB,
                "Inherited independent-reviewer source changed")
        require(hash_bytes(ROOT / "publisher/src/train/sft_data.py") ==
                UPSTREAM_SFT_SHA, "Original publisher ChatTemplate bytes drifted")
        protocol = json.loads((ROOT / "candidate/experiments/ncr-001/protocol.json").read_text())
        pins = json.loads((ROOT / "candidate/experiments/ncr-001/source-pins.json").read_text())
        require(protocol["train_authorized"] is False
                and set(protocol["models"]) == set(MODELS)
                and protocol["conformance"]["full_vs_cached_prefill_atol"] ==
                protocol["conformance"]["full_vs_cached_prefill_rtol"] ==
                CACHED_ATOL == CACHED_RTOL,
                "source protocol/causal tolerances altered")
        pin = pins["models"]["sprocket_instruct"]
        require(pin["hf_weight_file"] == "model.safetensors"
                and protocol["models"]["sprocket_instruct"]["variant"] == "instruct"
                and pin["repo_id"] == HF_REPO and pin["revision"] == HF_REV
                and pin["weight_declared_lfs_sha256_not_downloaded"] == ROOT_SHA
                and pins["models"]["sprocket_chat"]["weight_declared_lfs_sha256_not_downloaded"]
                == CHAT_SHA and ROOT_SHA != CHAT_SHA,
                "source checkpoint/Chat Instruct variant confused")
        report["source"] = {
            "candidate_commit": TARGET, "reviewer_dependency_blob": REVIEWER_SOURCE_BLOB,
            "original_hook_commit": SOURCE_IMPL, "publisher_commit": PUBLISHER,
            "publisher_sft_sha256": UPSTREAM_SFT_SHA,
            "checkpoint_sha256": ROOT_SHA, "checkpoint_file": "model.safetensors",
            "source_revision": HF_REV, "model_repo": HF_REPO,
            "original_publisher_is_training_build_attestation": False,
            "protocol_sha256": hash_bytes(ROOT / "candidate/experiments/ncr-001/protocol.json"),
            "pins_sha256": hash_bytes(ROOT / "candidate/experiments/ncr-001/source-pins.json"),
        }
        save()
        for file, key in (
            ("model.safetensors", "weight_declared_lfs_sha256_not_downloaded"),
            ("config.json", "config_sha256"),
            ("tokenizer.json", "tokenizer_sha256"),
            ("tokenizer_config.json", "tokenizer_config_sha256"),
            ("special_tokens_map.json", "special_tokens_map_sha256"),
        ):
            path = Path(hf_hub_download(HF_REPO, file, revision=HF_REV,
                                        local_files_only=True))
            digest = hash_bytes(path)
            require(digest == pin[key], "actual downloaded immutable source drift: " + file)
            if file == "model.safetensors":
                require(path.stat().st_size == 1002207984,
                        "complete original model bytes mismatch")
            report.setdefault("original_source_files", {})[file] = {
                "sha256": digest, "bytes": path.stat().st_size}
            save()

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(20261027)
        base = AutoModelForCausalLM.from_pretrained(
            HF_REPO, revision=HF_REV, local_files_only=True,
            attn_implementation="eager", torch_dtype=torch.float32,
            trust_remote_code=False).eval()
        sys.path.insert(0, str(ROOT / "implementation/scripts"))
        from ncr_001_sprocket_instruct import (  # noqa:E402
            FrozenSprocketInstructDepthNCA, POST_BLOCK_TAPS,
        )
        require(tuple(POST_BLOCK_TAPS) == (7, 16, 25),
                "Original required post-block taps drifted")
        wrapper = FrozenSprocketInstructDepthNCA(
            base, verified_weight_sha256=ROOT_SHA).eval()
        wrapper.verify_frozen_base()
        require(wrapper.nca.trainable_parameters() == 658904
                and not any(p.requires_grad for p in base.parameters()),
                "Frozen parameters/adapter source geometry changed")
        tokenizer = AutoTokenizer.from_pretrained(
            HF_REPO, revision=HF_REV, local_files_only=True)
        require(tokenizer.convert_tokens_to_ids("<|pad|>") == PAD and
                tokenizer.convert_tokens_to_ids("<|end|>") == END,
                "Explicit source pad/end ID mismatch")
        sys.path.insert(0, str(ROOT / "candidate/scripts"))
        from ncr_001e_sprocket_publisher_wire import render_ids_mask
        seqs = []
        for name, rows in (("control_context", PRIMARY), ("unrelated_request", SECONDARY)):
            ids, mask, text = independent_wire(rows, tokenizer)
            candidate_ids, candidate_mask = render_ids_mask(
                [{"role": role, "content": value} for role, value in rows],
                tokenizer, variant="instruct")
            require(ids == candidate_ids and mask == candidate_mask,
                    "Reviewer input and exact pinned candidate source disagree")
            seqs.append(ids)
            report.setdefault("wire", {})[name] = {
                "source_units": len(ids), "assistant_mask_units": sum(mask),
                "token_id_sha256": hashlib.sha256(
                    torch.tensor(ids, dtype=torch.int64).numpy().tobytes()).hexdigest(),
                "rendered_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "non_heldout": True,
            }
        require([len(seq) for seq in seqs] == [133, 31],
                "Unexpected synthetic source tokenizer units")
        geometry = gapped_right_prefix(seqs)
        report["right_padding"] = {
            "cached_prefix_width": geometry["width"],
            "true_prefix_units": geometry["valid_prefix_lengths"],
            "source_padding_id": PAD, "masked_cache_gap": True,
            "tail_units": GENERATE_TOKENS,
        }
        save()

        report["backbone_sha256_before"] = hash_tensors(base)
        with torch.inference_mode():
            # Finite synthetic source-structured no-op checks and independent
            # original per-request full-sequence references.
            frozen_refs = []
            for label, seq in zip(("control_context", "unrelated_request"), seqs):
                ids = torch.tensor([seq], dtype=torch.long)
                disabled = wrapper(input_ids=ids, use_cache=False, enabled=False).logits
                zero = wrapper(input_ids=ids, use_cache=False, enabled=True).logits
                compare(report, "frozen_original_zero_" + label,
                        disabled, zero, exact=True, atol=0, rtol=0)
                frozen_refs.append(disabled)
            # One manually activated *untrained* residual is enough to detect
            # actual adapter hooking without any fitting or optimizer.
            wrapper.nca.readout.weight[0, 0] = 0.03
            active_refs = [
                wrapper(input_ids=torch.tensor([seq], dtype=torch.long),
                        use_cache=False, enabled=True).logits[:, -GENERATE_TOKENS:]
                for seq in seqs
            ]
            report["checks"]["manual_nontrained_effect"] = {
                "max_abs": max(float((active_refs[i] - frozen_refs[i][:, -GENERATE_TOKENS:]).abs().max())
                               for i in range(2)),
                "trained": False, "pass": True,
            }
            require(report["checks"]["manual_nontrained_effect"]["max_abs"] > 0,
                    "Manually nontrained NCA effect is absent")
            save()

            def cached_from_right_gap(pad_value=PAD, *, corrupt_mask=False):
                pids = torch.tensor(geometry["prefix_ids"], dtype=torch.long).clone()
                pmask = torch.tensor(geometry["prefix_mask"], dtype=torch.long).clone()
                ppos = torch.tensor(geometry["prefix_position_ids"], dtype=torch.long).clone()
                if pad_value != PAD:
                    pids[pmask == 0] = pad_value
                if corrupt_mask:
                    pmask[pmask == 0] = 1
                prefill = wrapper(
                    input_ids=pids, attention_mask=pmask,
                    position_ids=ppos, use_cache=True, enabled=True)
                past = prefill.past_key_values
                require(len(wrapper.last_hook_trace) == 1 and
                        wrapper.last_hook_trace[0]["state_shape"][:2] ==
                        [2, geometry["width"]],
                        "Right-pad prefill depth field leaked across requests")
                pieces = []
                for step in range(GENERATE_TOKENS):
                    values = torch.tensor(
                        [[v] for v in geometry["incremental_tokens"][step]],
                        dtype=torch.long)
                    positions = torch.tensor(
                        [[v] for v in geometry["incremental_position_ids"][step]],
                        dtype=torch.long)
                    full_mask = torch.cat(
                        [pmask, torch.ones((2, step + 1), dtype=torch.long)],
                        dim=1)
                    output = wrapper(
                        input_ids=values, position_ids=positions,
                        attention_mask=full_mask, past_key_values=past,
                        use_cache=True, enabled=True)
                    past = output.past_key_values
                    pieces.append(output.logits)
                    require(len(wrapper.last_hook_trace) == 1 and
                            wrapper.last_hook_trace[0]["state_shape"][:2] == [2, 1],
                            "Right-pad cached one-token NCA state contaminated")
                return torch.cat(pieces, dim=1)

            cached = cached_from_right_gap()
            for row, label in enumerate(("control_context", "unrelated_request")):
                compare(report, "rightpad_KV_gap_vs_unpadded_" + label,
                        cached[row:row+1], active_refs[row])
            # Masked right-pad IDs may be replaced by actual host-control token
            # ID 7 without affecting any *valid* cached continuation logits.
            swapped = cached_from_right_gap(7)
            compare(report, "rightpad_masked_tool_result_no_effect",
                    cached, swapped, exact=True, atol=0, rtol=0)
            # Explicit negative: a faulty host that unmasks those padding slots
            # must be distinguishable. The erroneous output must NOT be accepted
            # as equivalent to the correct cached result.
            broken = cached_from_right_gap(corrupt_mask=True)
            difference = float((broken[1] - cached[1]).abs().max())
            report["checks"]["unmasked_rightpad_negative_detected"] = {
                "max_abs": difference, "expected_difference": True,
                "pass": difference > 0,
            }
            require(difference > 0,
                    "Host bad-mask sentinel did not detect source contamination")
            save()
            # True original HF GenerationMixin, not handwritten greedy. Compare
            # disabled/zero on both use_cache modes and manual *untrained*
            # adapter cache vs no-cache. Four new unit cap is a conformance probe,
            # not a model competence score.
            kwargs = dict(input_ids=torch.tensor([seqs[0]], dtype=torch.long),
                          attention_mask=torch.ones((1, len(seqs[0])), dtype=torch.long),
                          do_sample=False, max_new_tokens=GENERATE_TOKENS,
                          min_new_tokens=GENERATE_TOKENS,
                          eos_token_id=END, pad_token_id=PAD,
                          return_dict_in_generate=True, output_scores=True)
            runs = {}
            for mode in ("disabled", "zero", "active_untrained"):
                if mode == "active_untrained":
                    wrapper.nca.readout.weight[0, 0] = 0.03
                else:
                    wrapper.nca.readout.weight.zero_()
                for cache in (False, True):
                    response = wrapper.generate(
                        **kwargs, use_cache=cache, enabled=(mode != "disabled"))
                    output = response.sequences[0].tolist()
                    scores = list(response.scores)
                    require(len(output) == len(seqs[0]) + GENERATE_TOKENS and
                            len(scores) == GENERATE_TOKENS,
                            "Actual frozen HF generate did not produce four units")
                    if mode != "disabled":
                        traces = wrapper.last_hook_trace
                        require(len(traces) == GENERATE_TOKENS,
                                "Original hooked HF.generate decoder forward count mismatch")
                        require(all(t["tap_count"] == 3 and t["finite"] for t in traces),
                                "Original depth taps not complete/finite in generation")
                        actual_lengths = [t["state_shape"][1] for t in traces]
                        expected_lengths = (
                            [len(seqs[0])] + ([1] * 3 if cache
                                               else [len(seqs[0]) + i for i in (1, 2, 3)]
                                               )
                        )
                        require(actual_lengths == expected_lengths,
                                "HF.generate source NCA field did not reset on each forward")
                    score_hashes = [
                        hashlib.sha256(s.contiguous().cpu().numpy().tobytes()).hexdigest()
                        for s in scores
                    ]
                    runs[(mode, cache)] = (output, scores)
                    report["generations"].append({
                        "mode": mode, "use_cache": cache,
                        "source_prompt_unit_count": len(seqs[0]),
                        "original_generated_ids": output[len(seqs[0]):],
                        "scores_sha256": score_hashes,
                        "not_behavioral_tool_or_memory_qualification": True,
                        "optimizer_steps": 0,
                    })
                    save()
            for cache in (False, True):
                a, asc = runs[("disabled", cache)]
                b, bsc = runs[("zero", cache)]
                require(a == b, "Zero NCA changed actual frozen HF generation IDs")
                for step in range(GENERATE_TOKENS):
                    masked_score_comparison(
                        report, "generate_zero_original_exact_cache_" +
                        str(cache) + "_step_" + str(step),
                        asc[step], bsc[step],
                        exact=True, atol=0, rtol=0)
            for mode in ("disabled", "zero", "active_untrained"):
                a, asc = runs[(mode, False)]
                b, bsc = runs[(mode, True)]
                require(a == b, "Actual HF generation cache changed token IDs: " + mode)
                for step in range(GENERATE_TOKENS):
                    masked_score_comparison(
                        report, "generate_full_vs_cache_" + mode + "_step_" + str(step),
                        asc[step], bsc[step])
            # All scores are from the original source frozen base under
            # teacher-free greedy decoding. No tool dispatch or memory storage.
            wrapper.nca.readout.weight.zero_()
            replay = wrapper(
                input_ids=torch.tensor([seqs[0][:16]], dtype=torch.long),
                enabled=True, use_cache=False).logits
            wrapper(input_ids=torch.tensor([seqs[1]], dtype=torch.long),
                    enabled=True, use_cache=False)
            after = wrapper(
                input_ids=torch.tensor([seqs[0][:16]], dtype=torch.long),
                enabled=True, use_cache=False).logits
            compare(report, "generation_followed_by_fresh_A_B_A_exact",
                    replay, after, exact=True, atol=0, rtol=0)
        report["backbone_sha256_after"] = hash_tensors(base)
        require(report["backbone_sha256_before"] == report["backbone_sha256_after"],
                "Frozen backbone tensors changed")
        require(all(p.grad is None for p in base.parameters()),
                "Frozen parameters accumulated training gradients")
        require(all(not m._forward_hooks and not m._forward_pre_hooks
                    for m in base.modules()), "HF.generate original source hook leaked")
        guard_claims(report)
        report["status"] = "PASS_RIGHTPAD_GAPPED_KV_AND_ORIGINAL_HF_GENERATE_ONLY"
        report["limits"] = [
            "Original frozen Sprocket Instruct synthetic right-pad and HF.generate source test only",
            "No preregistered heldout fixture or evaluated tool/memory quality",
            "Manual nontrained NCA scalar; zero optimizer steps; no comparative NCA vs MLP training",
            "Not independent final #72 four-model acceptance or training admission",
        ]
    except BaseException:
        report["status"] = "FAIL_RIGHTPAD_GAPPED_KV_OR_ORIGINAL_HF_GENERATE"
        report["error_traceback"] = traceback.format_exc()
        raise
    finally:
        save()

    print(json.dumps({"status": report["status"],
                      "checks": report["checks"],
                      "train_authorized": False}, sort_keys=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-test", action="store_true",
                    help="Pure standard-library synthetic source geometry validation")
    args = ap.parse_args()
    if args.source_test:
        g = gapped_right_prefix([
            [3, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 4],
            [3, 17, 18, 19, 20, 21, 22, 23, 24, 4],
        ])
        print(json.dumps({"geometry_valid": bool(check_plan()),
                          "right_gap_width": g["width"],
                          "train_authorized": False}))
    else:
        execute()
