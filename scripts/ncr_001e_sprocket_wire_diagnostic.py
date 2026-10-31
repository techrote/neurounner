#!/usr/bin/env python3
"""Fixed no-training source-oracle and generated-output diagnostic for PR #102.

Seven NEW synthetic cases, locked in code before their first model invocation.
They are not the sealed evaluation suite and cannot authorize tuning/training.
Model failures are observations, never grounds for deleting/retrying a case.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time
import traceback

from ncr_001e_sprocket_publisher_wire import (
    IDS, ROLE, SFT_BLOB, UPSTREAM_COMMIT, render_ids_mask, render_text,
    _valid_assistant,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runtime/ncr-001/wire-diagnostic"
SFT_SHA = "e5568c15989f3987fec07545640f2f271ffe967c27cbe9d9883c0dd7b5c4ec69"
LEGACY_BLOB = "0abf73a8877cad61392517baa1a8487276ba5507"
MAX_NEW = 64
STOP_IDS = (0, 5)  # source document separator or assistant turn end, not pad
MODES = ("disabled", "zero", "active_untrained")


def turn(role, content):
    return {"role": role, "content": content}


CALL = '<|tool_call|>{"name":"read_meter","arguments":{"channel":"A"}}'
CASES = [
    {"id": "tool-request", "kind": "tool_request", "turns": [
        turn("system", "Available tool: read_meter. Its arguments object has a channel string. Use the tool when asked for a reading."),
        turn("user", "Read meter channel A using the available tool.")]},
    *[{"id": "tool-result-" + value, "kind": "tool_result_literal", "literal": value,
       "turns": [turn("system", "Report only the returned meter reading."),
                 turn("user", "Read meter channel A."), turn("assistant", CALL),
                 turn("tool", '{"reading":' + value + ',"unit":"C"}')]} for value in ("42", "17")],
    *[{"id": "memory-context-" + value, "kind": "memory_literal", "literal": value,
       "turns": [turn("memory", "Stored calibration label: " + value + "."),
                 turn("user", "What calibration label is stored? Reply with that label only.")]}
      for value in ("amber-17", "violet-28")],
    {"id": "memory-write", "kind": "memory_write", "turns": [
        turn("user", "Remember for next time: my preferred temperature unit is Celsius.")]},
    {"id": "memory-request", "kind": "memory_read", "turns": [
        turn("user", "Search your stored memory for my preferred temperature unit.")]},
]
RICH_TURNS = [
    turn("system", "Report meter readings from explicit results."),
    turn("memory", "The user prefers Celsius."),
    turn("user", "Read channel A."), turn("assistant", CALL),
    turn("tool", '{"reading":42,"unit":"C"}'),
    turn("assistant", "Channel A reads 42 C."),
    turn("user", "Remember the unit."),
    turn("assistant", "<|memory_write|>Celsius is preferred."),
    turn("user", "Recall that preference."),
    turn("assistant", "<|memory_read|>preferred temperature unit"),
    turn("memory", "Temperature units: Celsius."),
]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob_sha(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                              allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def diagnostic_plan():
    return {"schema": 1, "max_new_units": MAX_NEW, "stop_ids": list(STOP_IDS),
            "modes": list(MODES), "cases": CASES, "rich_turns": RICH_TURNS,
            "heldout_evaluation": False, "quality_acceptance_threshold": None,
            "sampling": False, "optimizer_steps": 0, "train_authorized": False}


def load_publisher_oracle(path, tokenizer_path, tokenizer_class):
    """Execute only the reviewed, hash-exact ChatTemplate AST, not training code."""
    raw = Path(path).read_bytes()
    require(sha(raw) == SFT_SHA and blob_sha(raw) == SFT_BLOB,
            "Publisher SFT source differs from reviewed immutable bytes")
    tree = ast.parse(raw.decode("utf-8"))
    wanted = {"DEFAULT_TOKENIZER", "ROLE_TOKEN", "TRAINED_ROLES"}
    selected, found = [], set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                ast.literal_eval(node.value)  # only declarative source constants
                selected.append(node)
                found.add(target.id)
        if isinstance(node, ast.ClassDef) and node.name == "ChatTemplate":
            methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
            require(methods == {"__init__", "encode_text", "render", "render_prompt"},
                    "Publisher renderer interface changed")
            selected.append(node)
            found.add("ChatTemplate")
    require(found == wanted | {"ChatTemplate"}, "Missing publisher renderer nodes")
    namespace = {"Tokenizer": tokenizer_class}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    require(namespace["ROLE_TOKEN"] == ROLE and namespace["TRAINED_ROLES"] == {"assistant"},
            "Publisher source role or ownership contract changed")
    return namespace["ChatTemplate"](str(tokenizer_path))


def parity(oracle, tokenizer, turns):
    observed, mask = render_ids_mask(turns, tokenizer, variant="instruct")
    upstream_ids, upstream_mask = oracle.render(turns)
    require(observed == oracle.render_prompt(turns) == upstream_ids + [IDS["<|assistant|>"]],
            "Candidate IDs differ from executed publisher renderer")
    require(mask == upstream_mask + [False], "Candidate masks differ from publisher renderer")
    return observed, mask


def legacy_reproduction(path):
    raw = Path(path).read_bytes()
    require(blob_sha(raw) == LEGACY_BLOB, "Historical serializer source not exact")
    spec = importlib.util.spec_from_file_location("historical_wire", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bad_cases = {
        "host_document_separator": [turn("user", "x<|endoftext|>y")],
        "host_padding_marker": [turn("user", "x<|pad|>y")],
        "assistant_padding_marker": [turn("assistant", "x<|pad|>y"), turn("user", "next")],
        "overflow_json_number": [turn("assistant", '<|tool_call|>{"name":"meter","arguments":{"n":1e309}}'), turn("tool", "done")],
    }
    records = []
    for name, rows in bad_cases.items():
        module.render_text(rows, variant="instruct")  # old source accepts defect
        try:
            render_text(rows, variant="instruct")
        except ValueError:
            records.append({"id": name, "turns": rows,
                            "legacy_accepted": True, "repaired_rejected": True})
        else:
            raise RuntimeError("Repair failed to reject " + name)
    return {"legacy_blob": LEGACY_BLOB, "legacy_sha256": sha(raw), "cases": records,
            "model_execution": False, "train_authorized": False}


def observe(case, text):
    """Descriptive literal/format observations, NOT claims of semantic success."""
    result = {"strict_tool_envelope": False,
              "memory_write_at_start": text.startswith("<|memory_write|>"),
              "memory_read_at_start": text.startswith("<|memory_read|>")}
    if text.startswith("<|tool_call|>"):
        try:
            _valid_assistant(text)
            payload = json.loads(text[len("<|tool_call|>"):])
            result.update(strict_tool_envelope=True,
                          requested_tool=payload["name"], arguments=payload["arguments"])
        except (ValueError, TypeError):
            pass
    if "literal" in case:
        needle = re.escape(case["literal"].casefold())
        result["expected_literal_present"] = bool(re.search(r"(?<!\w)" + needle + r"(?!\w)", text.casefold()))
    return result


def main():
    import torch
    import transformers
    from importlib.metadata import version
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    import ncr_001e_sprocket_cross_process as source

    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"schema": 1, "status": "RUNNING", "variant": "sprocket_instruct",
              "plan": diagnostic_plan(), "plan_sha256": sha(json.dumps(diagnostic_plan(), sort_keys=True).encode()),
              "actual_model_training": False, "optimizer_steps": 0,
              "heldout_evaluation": False, "train_authorized": False,
              "final_72_admission": False, "tool_memory_semantics_qualified": False,
              "publisher_commit": UPSTREAM_COMMIT, "observations": [], "checks": {},
              "runner": {"run_id": os.getenv("GITHUB_RUN_ID"), "attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
                         "github_sha": os.getenv("GITHUB_SHA"),
                         "checkout_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                         "name": os.getenv("RUNNER_NAME"),
                         "platform": platform.platform(), "python": platform.python_version(),
                         "cpu_count": os.cpu_count(), "pid": os.getpid(),
                         "cpu_model": next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')), 'unknown'),
                         "torch": torch.__version__, "transformers": transformers.__version__}}
    report["libraries"] = {name: version(name) for name in ("torch", "transformers", "tokenizers", "huggingface_hub", "safetensors", "numpy", "sentencepiece")}
    source.REPORT = report
    source.OUT = OUTPUT / "receipt.json"

    def save():
        report["wall_seconds"] = time.monotonic() - started
        source.save_receipt(source.OUT)

    def weights_hash():
        h = hashlib.sha256()
        for name, parameter in wrapper.backbone.state_dict().items():
            h.update(name.encode() + str(parameter.shape).encode() + str(parameter.dtype).encode())
            h.update(parameter.detach().contiguous().numpy().tobytes())
        return h.hexdigest()

    def greedy(ids, enabled):
        # Fresh cache per request. No mutable cache or cellular state escapes.
        current = torch.tensor([ids], dtype=torch.long)
        past, generated = None, []
        for _ in range(MAX_NEW):
            output = wrapper(input_ids=current, past_key_values=past, use_cache=True, enabled=enabled)
            require(bool(torch.isfinite(output.logits).all()), "Non-finite generated logits")
            token = int(output.logits[0, -1].argmax())
            generated.append(token)
            if token in STOP_IDS:
                break
            current = torch.tensor([[token]], dtype=torch.long)
            past = output.past_key_values
        return generated

    save()
    try:
        require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1", "Offline runtime required")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(20261027)
        spec, pin, protocol, _, wrapper, _ = source.verify_sources("instruct")
        require(protocol["conformance"]["full_vs_cached_prefill_atol"] ==
                protocol["conformance"]["full_vs_cached_prefill_rtol"] == 0.0005,
                "Fixed cache gate changed")
        require(subprocess.check_output(["git", "-C", str(ROOT / "publisher"), "rev-parse", "HEAD"], text=True).strip() == UPSTREAM_COMMIT,
                "Publisher checkout moved")
        tokenizer = AutoTokenizer.from_pretrained(source.SOURCE_REPO, revision=source.SOURCE_REV, local_files_only=True)
        tokenizer_path = hf_hub_download(source.SOURCE_REPO, "tokenizer.json", revision=source.SOURCE_REV, local_files_only=True)
        oracle = load_publisher_oracle(ROOT / "publisher/src/train/sft_data.py", tokenizer_path, Tokenizer)
        report["source_verifier_initial_probe"] = {k: report.pop(k) for k in ("prompt_tokenizer_units", "token_ids_sha256") if k in report}
        report["publisher_sft_sha256"] = SFT_SHA
        report["checkpoint_sha256"] = pin["weight_declared_lfs_sha256_not_downloaded"]
        report["hook_commit"] = spec["implementation_commit"]
        report["candidate_script_sha256"] = sha(Path(__file__).read_bytes())
        report["serializer_sha256"] = sha((ROOT / "scripts/ncr_001e_sprocket_publisher_wire.py").read_bytes())
        report["protocol_sha256"] = sha((ROOT / "experiments/ncr-001/protocol.json").read_bytes())
        report["eos_token_id_from_metadata"] = tokenizer.eos_token_id
        report["backbone_before_sha256"] = weights_hash()
        report["legacy_regressions"] = legacy_reproduction(ROOT / "previous/scripts/ncr_001e_sprocket_publisher_wire.py")
        prepared = []
        for case in CASES:
            ids, mask = parity(oracle, tokenizer, case["turns"])
            require(len(ids) + MAX_NEW <= 2048 and len(ids) <= 512, "Diagnostic sequence budget exceeded")
            prepared.append((case, ids, mask))
        rich_ids, rich_mask = parity(oracle, tokenizer, RICH_TURNS)
        report["source_oracle"] = {"cases_exact": len(CASES) + 1,
                                  "rich_units": len(rich_ids), "rich_mask_true": sum(rich_mask),
                                  "ids_and_masks_equal_executed_publisher": True}
        tensor = torch.tensor([rich_ids], dtype=torch.long)
        with torch.inference_mode():
            reference = wrapper.backbone(input_ids=tensor, use_cache=False).logits
            zero = wrapper(input_ids=tensor, use_cache=False).logits
            source.compare("direct_frozen_vs_zero", reference, zero, atol=0, rtol=0, exact=True)
            wrapper.nca.readout.weight[0, 0] = 0.03
            full = wrapper(input_ids=tensor, use_cache=False).logits
            report["untrained_effect"] = float((full - reference).abs().max())
            require(report["untrained_effect"] > 0, "Inactive diagnostic adapter")
            pieces, past = [], None
            for index in range(len(rich_ids)):
                one = wrapper(input_ids=tensor[:, index:index+1], past_key_values=past, use_cache=True)
                pieces.append(one.logits)
                past = one.past_key_values
                require(len(wrapper.last_hook_trace) == 1 and wrapper.last_hook_trace[0]["state_shape"][1] == 1,
                        "Incremental token cellular isolation failed")
            source.compare("all_rich_positions_full_vs_incremental", full, torch.cat(pieces, dim=1),
                           atol=0.0005, rtol=0.0005)
            request_before = wrapper(input_ids=tensor[:, :16], use_cache=False).logits.clone()
            for case, ids, mask in prepared:
                outputs = {}
                for mode in MODES:
                    wrapper.nca.readout.weight.zero_()
                    if mode == "active_untrained":
                        wrapper.nca.readout.weight[0, 0] = 0.03
                    tick = time.monotonic()
                    generated = greedy(ids, enabled=mode != "disabled")
                    outputs[mode] = generated
                    raw_text = tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    ended = generated[-1] in STOP_IDS
                    body = tokenizer.decode(generated[:-1] if ended else generated,
                                            skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    report["observations"].append({
                        "id": case["id"] + "/" + mode, "case_id": case["id"], "kind": case["kind"],
                        "mode": mode, "messages": case["turns"],
                        "serialized_prompt": render_text(case["turns"], variant="instruct"),
                        "input_ids": ids, "assistant_mask": mask, "generated_ids": generated,
                        "generated_control_ids": [v for v in generated if v < 12],
                        "raw_text": raw_text, "body_excluding_final_stop_only": body,
                        "finish_reason": "turn_end" if ended and generated[-1] == 5 else "document_end" if ended else "budget_exhausted",
                        "generated_units": len(generated), "wall_seconds": time.monotonic() - tick,
                        "observed": observe(case, body)})
                    save()  # raw generations precede any derived assertion
                require(outputs["disabled"] == outputs["zero"], "Zero adapter changed actual greedy output: " + case["id"])
            wrapper.nca.readout.weight.zero_()
            wrapper.nca.readout.weight[0, 0] = 0.03
            after = wrapper(input_ids=tensor[:, :16], use_cache=False).logits
            source.compare("request_after_all_generation", request_before, after, atol=0, rtol=0, exact=True)
        report["backbone_after_sha256"] = weights_hash()
        require(report["backbone_before_sha256"] == report["backbone_after_sha256"], "Frozen tensor bytes changed")
        wrapper.verify_frozen_base()
        require(all(p.grad is None for p in wrapper.backbone.parameters()), "Unexpected backbone gradients")
        require(all(not m._forward_hooks and not m._forward_pre_hooks for m in wrapper.backbone.modules()), "Leaked model hooks")
        report["status"] = "PASS_SOURCE_PARITY_AND_DIAGNOSTIC_EXECUTION_ONLY"
        report["limitations"] = [
            "21 bounded generations from 7 synthetic cases are not a benchmark or a qualified tool/memory ability claim",
            "Literal presence is not semantic correctness; raw negatives and budget exhaustion are retained",
            "Manual nontrained NCA intervention, no learned comparison, optimizer, external tool or heldout access",
            "Publisher source is linked and byte-pinned, not historical checkpoint training-build attestation",
            "Independent final #72 reviewer acceptance and protocol training admission remain absent"]
    except BaseException:
        report["status"] = "FAIL_DIAGNOSTIC_EXECUTION"
        report["failure_traceback"] = traceback.format_exc()
        raise
    finally:
        save()
    print(json.dumps({"status": report["status"], "observations": len(report["observations"]),
                      "source_oracle": report["source_oracle"], "train_authorized": False}))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Archive fixed plan and reproduce old-source refusals without loading any model")
    args = parser.parse_args()
    if args.prepare:
        dump(OUTPUT / "plan.json", diagnostic_plan())
        dump(OUTPUT / "legacy-reproduction.json", legacy_reproduction(ROOT / "previous/scripts/ncr_001e_sprocket_publisher_wire.py"))
    else:
        main()
