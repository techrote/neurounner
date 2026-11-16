#!/usr/bin/env python3
"""Actual hosted Smol first-pass DEVELOPMENT inference, with every refusal retained."""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import signal
import sys
import time

import contracts
from contracts import common
from numerics import configure_cpu
from train import PACKAGES, SYSTEM, load_base

STOP = False


def request_stop(_signal, _frame):
    global STOP
    STOP = True


class TokenRecorder:
    """Retain generated IDs even when generation raises after a partial answer."""
    def __init__(self, prompt_ids):
        self.prompt_ids = prompt_ids
        self.started = False
        self.tokens = []

    def put(self, value):
        values = value.tolist()
        if values and isinstance(values[0], list):
            if len(values) != 1:
                raise ValueError("only one generation per process is permitted")
            values = values[0]
        if not self.started:
            if values != self.prompt_ids:
                raise ValueError("streamed prompt differs from recorded input")
            self.started = True
        else:
            self.tokens.extend(values)

    def end(self):
        pass


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-seconds", type=int, default=1200)
    args = parser.parse_args(argv)
    if (args.arm == "candidate") != (args.checkpoint is not None):
        parser.error("candidate requires a checkpoint; baseline must use the untuned parent")
    if not 1 <= args.max_new_tokens <= 1024 or not 60 <= args.max_seconds <= 1200:
        parser.error("evaluation is bounded to 1–1024 new tokens and 60–1200 seconds")
    return args


def emit_case(output, request, generation, verdict, program, compiled, identity):
    identifier = request["id"]
    relative = f"cases/{identifier}"
    directory = output / relative
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "completion.txt").write_text(verdict["raw_text"], encoding="utf-8")
    common.write_json(directory / "raw-output.json", verdict)
    paths = {"completion": relative + "/completion.txt", "raw_output": relative + "/raw-output.json",
             "receipt": relative + "/receipt.json"}
    if verdict["accepted"]:
        common.write_json(directory / "program.json", program)
        common.write_json(directory / "compiled.json", compiled)
        paths.update(program=relative + "/program.json", compiled=relative + "/compiled.json")
    hashes = {name: common.sha256_file(output / path) for name, path in paths.items() if name != "receipt"}
    case = {**request, "accepted": verdict["accepted"],
            "generated_family": verdict.get("generated_family"),
            "requested_family_matches": verdict["requested_family_matches"],
            "generation_id": generation["id"],
            "generation_reused": identifier != generation["request_ids"][0],
            "generation_attempted": generation["attempted"],
            "generation_failure": generation["failure"], "verdict": verdict,
            "evaluation_identity_sha256": identity, "paths": paths, "sha256": hashes,
            "native_execution": False, "visual_quality_claim": False}
    common.write_json(directory / "receipt.json", case)
    case["sha256"]["receipt"] = common.sha256_file(directory / "receipt.json")
    return case


def run(args):
    started = time.monotonic()
    source = contracts.hosted_source(args.source_sha)
    if args.output.exists():
        raise ValueError("refusing to overwrite evaluation evidence")
    rows, admission = contracts.load_development(args.corpus)
    requests = [contracts.request_fields(row) for row in rows]
    groups = contracts.generation_groups(requests)
    if len(groups) != 6:
        raise ValueError("expected six distinct frozen DEVELOPMENT prompts")
    selected = common.profile("smol", require_supported=True)
    if (selected["model_id"], selected["revision"]) != (contracts.MODEL_ID, contracts.MODEL_REVISION):
        raise ValueError("Smol model pin changed")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    numerics, numerical_runtime = configure_cpu(1)
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class Deadline(StoppingCriteria):
        def __call__(self, _input_ids, _scores, **_kwargs):
            return STOP or time.monotonic() - started >= args.max_seconds

    base, tokenizer = load_base(args.model, selected)
    model_files = common.read_json(args.model / "download_receipt.json")["files"]
    packages = {name: importlib.metadata.version(name) for name in PACKAGES}
    python_version = platform.python_version()
    checkpoint = None
    model = base
    if args.checkpoint:
        receipt = common.verify_checkpoint(args.checkpoint)
        contracts.validate_candidate(receipt["config"], selected, source, numerics,
                                     model_files, SYSTEM, tokenizer.chat_template, packages, python_version)
        if not 1 <= receipt["global_step"] <= contracts.MAX_STEPS:
            raise ValueError("candidate checkpoint is outside this bounded training iteration")
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, args.checkpoint / "adapter", is_trainable=False, local_files_only=True)
        checkpoint = {"global_step": receipt["global_step"], "source_sha": receipt["config"]["source_sha"],
                      "inventory_sha256": common.sha256_file(args.checkpoint / "inventory.json"),
                      "adapter_sha256": common.sha256_file(args.checkpoint / "adapter" / "adapter_model.safetensors"),
                      "receipt_sha256": common.sha256_file(args.checkpoint / "receipt.json")}
    model.eval()
    compiler = common.compiler_module()
    corpus = {"manifest_sha256": contracts.MANIFEST_SHA256,
              "development_sha256": contracts.DEVELOPMENT_SHA256, "train_sha256": contracts.TRAIN_SHA256,
              "native_source_sha": admission["source_sha"], "runtime_sha256": admission["runtime_sha256"]}
    decoding = {"do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new_tokens,
                "max_seconds": args.max_seconds, "use_cache": True, "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id, "repair": False, "retry": False, "fallback": False,
                "prompt_reuse": "one first-pass decode per identical request, separately compiled with each host seed"}
    comparison = {"source_sha": source, "model": selected, "model_files": model_files,
                  "corpus": corpus, "compiler_sha256": contracts.COMPILER_SHA256,
                  "system": SYSTEM, "system_sha256": common.digest_json(SYSTEM),
                  "chat_template_sha256": common.digest_json(tokenizer.chat_template),
                  "adapter_contract": contracts.adapter_identity(selected),
                  "numerics": numerics, "decoding": decoding,
                  "evaluator_sha256": common.sha256_file(__file__),
                  "contracts_sha256": common.sha256_file(contracts.__file__),
                  "packages": packages, "python": python_version}
    identity = common.digest_json({"comparison": comparison, "arm": args.arm, "checkpoint": checkpoint})
    args.output.mkdir(parents=True, exist_ok=False)
    args._output_owned = True
    report = {"schema_version": 1, "kind": "smol-development-first-pass", "arm": args.arm,
              "source_sha": source, "model": selected, "model_id": selected["model_id"],
              "model_revision": selected["revision"], "checkpoint": checkpoint,
              "compiler_sha256": contracts.COMPILER_SHA256, "corpus": corpus,
              "comparison_identity": comparison, "comparison_identity_sha256": common.digest_json(comparison),
              "evaluation_identity_sha256": identity, "system": SYSTEM, "decoding": decoding,
              "numerics": numerics, "numerical_runtime": numerical_runtime,
              "runner": {"github_run_id": os.environ.get("GITHUB_RUN_ID"),
                         "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT")},
              "cases": [], "generations": [], "native_execution": False,
              "visual_quality_claim": False, "final_targets_opened": False}
    common.write_json(args.output / "evaluation-start.json", report)
    candidates = []
    for group in groups:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": group["prompt"]}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        generation = {"id": group["id"], "prompt": group["prompt"], "rendered_prompt": rendered,
                      "input_token_ids": ids, "request_ids": [row["id"] for row in group["requests"]],
                      "attempted": False, "failure": None, "seconds": 0.0, "output_tokens": 0}
        tokens = []
        if STOP or time.monotonic() - started >= args.max_seconds:
            generation["failure"] = "not attempted: Stop" if STOP else "not attempted: total evaluation deadline"
        elif len(ids) + args.max_new_tokens > selected["max_sequence_tokens"]:
            generation["failure"] = "prompt plus output budget exceeds context; no truncation"
        else:
            recorder = TokenRecorder(ids)
            generation["attempted"] = True
            generation_started = time.monotonic()
            try:
                inputs = torch.tensor([ids], dtype=torch.long)
                with torch.inference_mode():
                    output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                            max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                                            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                            use_cache=True, streamer=recorder,
                                            stopping_criteria=StoppingCriteriaList([Deadline()]))
                tokens = output[0, len(ids):].tolist()
                if tokens != recorder.tokens:
                    raise ValueError("returned token IDs differ from recorded first-pass stream")
            except Exception as error:
                tokens = recorder.tokens
                generation["failure"] = f"{type(error).__name__}: {error}"
            generation["seconds"] = time.monotonic() - generation_started
            if not tokens or tokens[-1] != tokenizer.eos_token_id:
                if STOP or time.monotonic() - started >= args.max_seconds:
                    generation["failure"] = generation["failure"] or "generation stopped at total deadline or Stop"
        generation["output_tokens"] = len(tokens)
        generation["token_ids_sha256"] = common.digest_json(tokens)
        report["generations"].append(generation)
        for request in group["requests"]:
            verdict, program, compiled = contracts.classify_output(tokenizer, tokens, request, compiler,
                                                                   generation["failure"])
            case = emit_case(args.output, request, generation, verdict, program, compiled, identity)
            report["cases"].append(case)
            if case["accepted"]:
                candidates.append({**request, "family": program["family"], "request_family": request["family"],
                                   "requested_family_matches": verdict["requested_family_matches"],
                                   "program": program, "compiled": compiled,
                                   "generation_id": generation["id"], "evaluation_identity_sha256": identity})
        common.write_json(args.output / "evaluation-progress.json", report)
        print(common.canonical({"event": "development-generation", "arm": args.arm,
                                "id": generation["id"], "attempted": generation["attempted"],
                                "output_tokens": len(tokens), "failure": generation["failure"]}), flush=True)
    cases = report["cases"]
    report["counts"] = {"case_views": len(cases), "structural_groups": len({c["group_id"] for c in cases}),
                        "distinct_prompts": len(groups), "model_generations_attempted": sum(g["attempted"] for g in report["generations"]),
                        "attempted_case_views": sum(c["generation_attempted"] for c in cases),
                        "accepted_case_views": sum(c["accepted"] for c in cases),
                        "accepted_generations": len({c["generation_id"] for c in cases if c["accepted"]}),
                        "family_matching_accepted_case_views": sum(c["accepted"] and c["requested_family_matches"] for c in cases)}
    report["seconds"] = time.monotonic() - started
    report["status"] = "completed" if all(g["attempted"] and g["failure"] is None for g in report["generations"]) else "completed-with-execution-refusals"
    common.write_json(args.output / "accepted-candidates.json", {"schema_version": 1, "source_sha": source,
                      "compiler_sha256": contracts.COMPILER_SHA256, "arm": args.arm,
                      "evaluation_identity_sha256": identity, "examples": candidates,
                      "native_execution": False, "visual_quality_claim": False})
    report["accepted_candidates_sha256"] = common.sha256_file(args.output / "accepted-candidates.json")
    common.write_json(args.output / "evaluation.json", report)
    print(common.canonical({"event": "development-finished", "arm": args.arm, "counts": report["counts"],
                            "status": report["status"], "native_execution": False}), flush=True)
    return 0


def main(argv=None):
    args = arguments(argv)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        return run(args)
    except Exception as error:
        if getattr(args, "_output_owned", False):
            common.write_json(args.output / "failure.json", {"type": type(error).__name__, "error": str(error),
                              "native_execution": False, "model_quality_claim": False})
        raise


if __name__ == "__main__":
    sys.exit(main())
