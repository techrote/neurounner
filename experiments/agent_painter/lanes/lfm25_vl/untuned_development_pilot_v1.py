#!/usr/bin/env python3
"""Original DEVELOPMENT: three fixed untuned LFM first passes; no TRAIN, FINAL, or native writes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

LANE = Path(__file__).resolve().parent
ROOT = LANE.parents[1]
sys.path.insert(0, str(ROOT / "training"))
from common import (compiler_module, profile, relative_file, sha256_file, strict_completion,
                    strict_json, verify_file, write_json)

CORPUS_SHA256 = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
NATIVE_SOURCE = "633920373a3d406a2ede61ce4137c324f76be23b"
FAMILIES = ("architecture", "greebles", "pipework")
SEEDS = (104729, 130363)
LIMIT = 256

def choose_development(corpus):
    """Read manifest metadata and the DEVELOPMENT file only. Never open FINAL contents."""
    corpus = Path(corpus)
    manifest_path = relative_file(corpus, "admitted-corpus.json")
    if sha256_file(manifest_path) != CORPUS_SHA256:
        raise ValueError("original corpus manifest bytes changed")
    manifest = strict_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("state") != "native-admitted" or manifest.get("source_sha") != NATIVE_SOURCE:
        raise ValueError("original source/native admission identity mismatch")
    partitions = manifest.get("structural_groups", {})
    if set(partitions) != {"TRAIN", "DEVELOPMENT", "FINAL"}:
        raise ValueError("three frozen structural partitions required")
    if any(set(partitions[a]) & set(partitions[b]) for a,b in
           (("TRAIN","DEVELOPMENT"),("TRAIN","FINAL"),("DEVELOPMENT","FINAL"))):
        raise ValueError("structural split leakage")
    dev_groups = partitions["DEVELOPMENT"]
    if len(dev_groups) != 6 or len(set(dev_groups)) != 6:
        raise ValueError("expected six original DEVELOPMENT whole groups")
    entry = manifest["split_files"]["DEVELOPMENT"]
    if entry.get("count") != 12:
        raise ValueError("unexpected original DEVELOPMENT case count")
    path = verify_file(corpus, entry["path"], entry["sha256"])
    rows = [strict_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 12 or len({row["id"] for row in rows}) != 12:
        raise ValueError("missing/duplicate DEVELOPMENT rows")
    grouped = {}
    for row in rows:
        group = row.get("group_id", row.get("structural_group"))
        if row.get("split") != "DEVELOPMENT" or group not in dev_groups:
            raise ValueError("non-DEVELOPMENT row/structural identity")
        if row.get("structural_group", group) != group or row.get("family") not in FAMILIES:
            raise ValueError("family/structural identity mismatch")
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError("empty DEVELOPMENT prompt")
        if row.get("seed") not in SEEDS:
            raise ValueError("noncanonical host seed")
        grouped.setdefault(group, []).append(row)
    if set(grouped) != set(dev_groups) or any(
            len(cases)!=2 or {x["seed"] for x in cases} != set(SEEDS)
            or len({x["family"] for x in cases})!=1 for cases in grouped.values()):
        raise ValueError("missing two-seed whole-group coverage")
    chosen = {}
    for family in FAMILIES:
        candidates = sorted(g for g, cases in grouped.items() if cases[0]["family"] == family)
        if len(candidates) != 2:
            raise ValueError("expected two original DEVELOPMENT groups per family")
        # Precommitted deterministic order. No model output or teacher content drives selection.
        group = candidates[0]
        row = next(x for x in grouped[group] if x["seed"] == SEEDS[0])
        chosen[family] = {"id": row["id"], "group": group, "prompt": row["prompt"]}
    return chosen, {"native_source": NATIVE_SOURCE, "corpus_manifest_sha256": CORPUS_SHA256,
                    "development_sha256": entry["sha256"], "development_groups": len(grouped),
                    "development_rows": len(rows), "selected_groups": 3,
                    "final_targets_opened": False}

def execute(args):
    selection, custody = choose_development(args.corpus)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "INCOMPLETE.json",
               {"status":"FAIL_OR_INCOMPLETE_NO_NATIVE","optimizer_updates":0,
                "native_executions":0,"final_targets_opened":False})
    write_json(args.output / "selection.json",
               {"custody": custody, "family_order": list(FAMILIES),
                "rows": [{"family": family, "id": x["id"], "group":x["group"],
                          "prompt_sha256":hashlib.sha256(x["prompt"].encode()).hexdigest()}
                         for family,x in selection.items()],
                "max_new_tokens":LIMIT, "status":"untuned_model_only"})
    if args.preflight_only:
        write_json(args.output / "preflight.json", {"status":"PASS", **custody})
        (args.output / "INCOMPLETE.json").unlink()
        return
    # Configure strict portable CPU profile before importing torch or loading model.
    from numerics import configure_cpu
    numerics, runtime = configure_cpu(1)
    import torch
    from lane_adapters import encode_prompt
    from train import SYSTEM, load_base
    selected = profile("lfm25_vl", require_supported=True)
    if selected["revision"] != "fc6221ca597f3315e4f82fc2df606783267b34ba":
        raise ValueError("pinned LFM model differs")
    started = time.monotonic()
    base, tokenizer = load_base(args.model, selected)
    base.eval()
    cold_seconds = time.monotonic() - started
    compiler = compiler_module()
    completed = []
    for family in FAMILIES:
        item = selection[family]
        ids = encode_prompt(tokenizer, item["prompt"], SYSTEM, selected)
        if not ids or len(ids)+LIMIT>selected["max_sequence_tokens"]:
            raise ValueError("unaltered official framing exceeds declared context")
        begin = time.monotonic()
        cpu_begin = time.process_time()
        with torch.inference_mode():
            prompt_tensor = torch.tensor([ids],dtype=torch.long)
            generated = base.generate(
                input_ids=prompt_tensor,
                attention_mask=torch.ones_like(prompt_tensor),
                max_new_tokens=LIMIT, do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id, use_cache=True
            )[0,len(ids):].tolist()
        verdict = strict_completion(tokenizer, generated, compiler.validate_program)
        accepted = verdict["first_pass_accepted"]
        compile_error = None
        operations = 0
        if accepted:
            try:
                program = strict_json(verdict["raw_text"])
                for seed in SEEDS:
                    built = compiler.compile_program(
                        program, seed=seed, focus=(0,0,96,96), material=58,
                        allowed_materials=(58,))
                    operations += len(built["operations"])
            except (ValueError,TypeError,KeyError) as error:
                accepted = False
                compile_error = str(error)
                operations = 0
        destination = args.output / family
        destination.mkdir()
        (destination / "completion.txt").write_text(verdict["raw_text"], encoding="utf-8")
        write_json(destination / "raw-output.json", verdict)
        receipt = {
            "status":"COMPLETE_FIRST_PASS", "family":family, "id":item["id"], "group":item["group"],
            "prompt":item["prompt"], "prompt_sha256":hashlib.sha256(item["prompt"].encode()).hexdigest(),
            "model_id":selected["model_id"],"revision":selected["revision"],
            "base_unmodified":True,"adapter_loaded":False,"optimizer_updates":0,
            "max_new_tokens":LIMIT,"input_tokens":len(ids),"output_tokens":len(generated),
            "eos_terminated":verdict["eos_terminated"],"json_valid":verdict["json_valid"],
            "program_valid":verdict["program_valid"],"compiler_accepted":accepted,
            "compile_failure":compile_error,"compiler_only_operations_across_two_seeds":operations,
            "native_executions":0,"final_targets_opened":False,
            "completion_sha256":sha256_file(destination/"completion.txt"),
            "generation_seconds":time.monotonic()-begin,
            "generation_cpu_seconds":time.process_time()-cpu_begin,
            "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "decoding":"one greedy pass, full tokens retained, no EOS/JSON repair/retry/teacher",
            "model_native_painting":False
        }
        write_json(destination / "receipt.json",receipt)
        completed.append(receipt)
        write_json(args.output/"partial.json",
                   {"completed":len(completed),"requested":len(FAMILIES),
                    "families":[x["family"] for x in completed],
                    "native_executions":0,"optimizer_updates":0})
        print(json.dumps({"family":family,"eos":receipt["eos_terminated"],
                          "json":receipt["json_valid"],"compile":accepted,
                          "seconds":receipt["generation_seconds"]}),flush=True)
    write_json(args.output/"summary.json",{
        "status":"PASS_RAW_BASELINE_EXECUTED", "model_quality_certified":False,
        "source_kind":"untuned pinned official multimodal LFM2.5-VL",
        "model_revision":selected["revision"],"numerics":numerics,"numerical_runtime":runtime,
        "cold_load_seconds":cold_seconds, "elapsed_seconds":time.monotonic()-started,
        "corpus":custody, "requests":len(completed),
        "terminal_eos":sum(x["eos_terminated"] for x in completed),
        "json_valid":sum(x["json_valid"] for x in completed),
        "compiler_accepted":sum(x["compiler_accepted"] for x in completed),
        "native_executions":0,"optimizer_updates":0,"model_native_paintings":0,
        "final_targets_opened":False})
    (args.output / "INCOMPLETE.json").unlink()

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus",type=Path,required=True)
    ap.add_argument("--model",type=Path)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--preflight-only",action="store_true")
    args=ap.parse_args()
    if not args.preflight_only and args.model is None:
        ap.error("--model required except for corpus preflight")
    execute(args)
