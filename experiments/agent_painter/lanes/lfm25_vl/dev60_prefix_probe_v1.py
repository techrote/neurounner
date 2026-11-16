#!/usr/bin/env python3
"""Provisional LFM DEV60 static-prefix custody probe. NO TRAIN, MODEL INFERENCE OR DEV60 FREEZE.

All 60 prompts here are explicitly synthetic placeholders, not native-admitted
DEVELOPMENT requests. The actual official Lfm2VlProcessor is loaded from
hash-checked *small files only*; no substitute tokenizer, model or teacher JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

LANE = Path(__file__).resolve().parent
ROOT = LANE.parents[3]
MODEL_ID = "LiquidAI/LFM2.5-VL-450M"
REVISION = "fc6221ca597f3315e4f82fc2df606783267b34ba"
# Original immutable qualified source 2a565df...; refuse adapter/profile drift
# BEFORE importing the local adapter.
QUALIFIED_BLOBS = {
    "experiments/agent_painter/lanes/lfm25_vl/adapter.py":
        "c41061aaf3e6296a75d63893ac4d4c6ae65d3a32",
    "experiments/agent_painter/lanes/lfm25_vl/profile.json":
        "547ab00bed6f90aad0a09c4a4aa84217918bb605",
    "experiments/agent_painter/training/lane_adapters.py":
        "503b799626da3c8193c9392963d0cbedb100d279",
    "experiments/agent_painter/training/train.py":
        "7278ba991cbe7709aa5a12c3fcc3090a090676e4",
}
FAMILIES = ("Greebles", "Pipework", "Architecture")
FORMS = ("F0", "F1", "F2", "F3", "F4")
MAX_CONTEXT = 2048
PROVISIONAL_NEW_TOKENS = 512  # Not a proposed/frozen model budget.
SYSTEM = ("Return only one valid CyberSand painter program JSON object. "
          "Follow the requested structure. Do not set host seeds, material permissions or focus.")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def synthetic_rows():
    """Only disposable static-encoder fixtures; no original DEV or teacher data."""
    rows = []
    for family in FAMILIES:
        for slot in range(1, 5):
            for form in FORMS:
                rows.append({
                    "id": f"SYN-{family[0]}-{slot:02d}-{form}",
                    "family": family,
                    "group_id": f"SYNTHETIC-{family[0]}-{slot:02d}",
                    "formulation_id": form,
                    "prompt": f"SYNTHETIC FORMAT TEST ONLY, NOT A PAINTING REQUEST. "
                              f"Family {family}; imaginary assignment {slot}; wording {form}.",
                })
    return rows


def check_rows(rows):
    if len(rows) != 60:
        raise ValueError("exactly 60 synthetic rows required")
    ids, texts, keys, groups = set(), set(), set(), set()
    for row in rows:
        if type(row) is not dict or set(row) != {"id", "family", "group_id", "formulation_id", "prompt"}:
            raise ValueError("unexpected row keys")
        if any(type(row[k]) is not str or not row[k] for k in row):
            raise ValueError("empty or non-string request metadata")
        if row["family"] not in FAMILIES or row["formulation_id"] not in FORMS:
            raise ValueError("unregistered family/form")
        key = (row["family"], row["group_id"], row["formulation_id"])
        if row["id"] in ids or row["prompt"] in texts or key in keys:
            raise ValueError("duplicate ID, literal prompt, or group/form")
        ids.add(row["id"]); texts.add(row["prompt"]); keys.add(key)
        groups.add((row["family"], row["group_id"]))
    if len(groups) != 12 or any(sum(k[0] == f for k in groups) != 4 for f in FAMILIES):
        raise ValueError("not four whole groups per family")
    if any({form for family, group, form in keys if (family, group) == item} != set(FORMS)
           for item in groups):
        raise ValueError("missing one of five formulations")
    return sha256(canonical(sorted(rows, key=lambda item: item["id"])))


def verify_local_source():
    """Pin byte-identical qualified adapter/profile and their common framing."""
    checked = {}
    for rel, expected in QUALIFIED_BLOBS.items():
        path = ROOT / rel
        if not path.is_file() or path.is_symlink():
            raise ValueError("missing/symlinked qualified source: " + rel)
        actual = subprocess.check_output(
            ["git", "hash-object", "--", str(path)], cwd=ROOT, text=True).strip()
        if actual != expected:
            raise ValueError("qualified framing/source drift: " + rel)
        checked[rel] = actual
    return checked


def load_qualified_adapter():
    """Load exact path after checking original Git blobs, ignoring sys.modules.

    An ambient import adapter could resolve a different pre-cached or earlier
    sys.path module even when the on-disk qualified adapter hash is correct.
    This source-custody gate is not a defense against a compromised runner.
    """
    verify_local_source()
    path = LANE / "adapter.py"
    spec = importlib.util.spec_from_file_location("_lfm25_vl_verified_adapter", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load exact qualified LFM adapter path")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve():
        raise ValueError("loaded adapter is not the qualified local file")
    return module


def verify_model_files(snapshot, profile):
    files = {k: v for k, v in profile["required_hashes"].items() if k != "model.safetensors"}
    for name, expected in files.items():
        path = snapshot / name
        if not path.is_file() or path.is_symlink() or sha256(path.read_bytes()) != expected:
            raise ValueError("pinned LFM processor file mismatch: " + name)
    return files


def obtain_small_files(destination, profile):
    from huggingface_hub import HfApi, snapshot_download
    if destination.exists():
        raise ValueError("snapshot destination already exists; refuse overwrite")
    info = HfApi(token=False).model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION or info.private or info.gated:
        raise ValueError("not original public pinned model")
    card = info.card_data or {}
    if card.get("license") != "other" or card.get("license_name") != "lfm1.0" or card.get("license_link") != "LICENSE":
        raise ValueError("original LFM 1.0 model-card license mismatch")
    names = sorted(k for k in profile["required_hashes"] if k != "model.safetensors")
    snapshot_download(repo_id=MODEL_ID, revision=REVISION, allow_patterns=names,
                      local_dir=destination, max_workers=2, token=False)
    return verify_model_files(destination, profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--skip-processor", action="store_true", help="unit-test/custody-only; never pass as processor qualification")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "INCOMPLETE.json").write_text('{"status":"INCOMPLETE_NO_QUALIFICATION"}\n')
    started = time.monotonic()
    rows = synthetic_rows()
    request_set = check_rows(rows)
    blobs = verify_local_source()
    profile = json.loads((LANE / "profile.json").read_text(encoding="utf-8"))
    if profile["model_id"] != MODEL_ID or profile["revision"] != REVISION or profile["framing"]["eos_token_id"] != 7:
        raise ValueError("unqualified model/profile/EOS")
    if args.skip_processor:
        summary = {"status": "SOURCE_ONLY_NOT_PROCESSOR_QUALIFIED", "rows": len(rows),
                   "request_set_sha256": request_set, "qualified_source_blobs": blobs,
                   "target_representability_checked": False, "dev60_frozen": False,
                   "model_inference": 0, "optimizer_updates": 0, "native_executions": 0}
    else:
        files = obtain_small_files(args.processor_dir, profile)
        adapter = load_qualified_adapter()
        tokenizer = adapter.load_processor(args.processor_dir, profile)
        receipts = []
        for row in rows:
            ids = adapter.encode_prompt(tokenizer, row["prompt"], SYSTEM)
            if not ids or any(type(x) is not int for x in ids):
                raise ValueError("empty/invalid genuine processor prefix")
            if len(ids) + PROVISIONAL_NEW_TOKENS > MAX_CONTEXT:
                raise ValueError("provisional context overflow; no truncation")
            receipts.append({
                "id": row["id"], "family": row["family"], "group_id": row["group_id"],
                "formulation_id": row["formulation_id"],
                "request_sha256": sha256(row["prompt"].encode("utf-8")),
                "prefix_tokens": len(ids), "prefix_ids_sha256": sha256(canonical(ids)),
            })
        (args.output / "prefixes.json").write_bytes(
            json.dumps(receipts, indent=2, sort_keys=True).encode("utf-8") + b"\n")
        summary = {
            "status": "PASS_PROVISIONAL_SYNTHETIC_PREFIX_ONLY_NOT_DEV60_READY",
            "model_id": MODEL_ID, "model_revision": REVISION,
            "processor_class": type(tokenizer.processor).__name__,
            "eos_id": tokenizer.eos_token_id, "model_weights_downloaded": False,
            "model_file_sha256": files,
            "qualified_source_blobs": blobs, "rows": len(receipts),
            "request_set_sha256": request_set,
            "prefix_min": min(x["prefix_tokens"] for x in receipts),
            "prefix_max": max(x["prefix_tokens"] for x in receipts),
            "provisional_max_new_tokens": PROVISIONAL_NEW_TOKENS,
            "context_tokens": MAX_CONTEXT,
            "target_representability_checked": False, "dev60_frozen": False,
            "original_development_accessed": False, "final_accessed": False,
            "model_inference": 0, "optimizer_updates": 0, "native_executions": 0,
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
    (args.output / "summary.json").write_bytes(
        json.dumps(summary, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    (args.output / "INCOMPLETE.json").unlink()
    print(json.dumps({"status": summary["status"], "rows": summary["rows"],
                      "request_set_sha256": request_set}), flush=True)


if __name__ == "__main__":
    main()
