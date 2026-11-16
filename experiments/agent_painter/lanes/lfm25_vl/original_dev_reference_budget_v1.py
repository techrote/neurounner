#!/usr/bin/env python3
"""Original six-group DEVELOPMENT teacher token-budget audit for pinned LFM processor.

Reference-only, no TRAIN, model generation, native application, DEV60 freeze or FINAL reads.
Original corpus and producer SHA are fixed independently from this source commit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

LANE = Path(__file__).resolve().parent
sys.path.insert(0, str(LANE.parents[1] / "training"))
from common import (canonical, digest_json, relative_file, sha256_file,
                    strict_json, verify_file, write_json)
from dev60_prefix_probe_v1 import (MODEL_ID, REVISION, MAX_CONTEXT, SYSTEM,
                                   obtain_small_files, verify_local_source,
                                   load_qualified_adapter)

CORPUS_MANIFEST_SHA256 = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
ORIGINAL_NATIVE_SOURCE = "633920373a3d406a2ede61ce4137c324f76be23b"
SEEDS = {104729, 130363}
FAMILIES = {"greebles", "pipework", "architecture"}
PROVISIONAL_RESERVE = 512  # diagnostic, NOT approved common DEV60 budget


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def select_visible_development(corpus: Path, expected_manifest_sha256=CORPUS_MANIFEST_SHA256):
    """Read manifest metadata and ONLY original development.jsonl; do not open FINAL."""
    corpus = Path(corpus)
    manifest_path = relative_file(corpus, "admitted-corpus.json")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("original native corpus manifest bytes changed")
    manifest = strict_json(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1
            or manifest.get("state") != "native-admitted"
            or manifest.get("source_sha") != ORIGINAL_NATIVE_SOURCE):
        raise ValueError("original native source/admission mismatch")
    groups = manifest.get("structural_groups", {})
    if set(groups) != {"TRAIN", "DEVELOPMENT", "FINAL"}:
        raise ValueError("expected three complete partition identities")
    if any(type(v) is not list or not v or len(v) != len(set(v)) for v in groups.values()):
        raise ValueError("invalid structural groups")
    if set(groups["TRAIN"]) & set(groups["DEVELOPMENT"]) or set(groups["TRAIN"]) & set(groups["FINAL"]) or set(groups["DEVELOPMENT"]) & set(groups["FINAL"]):
        raise ValueError("structural split leakage")
    if len(groups["TRAIN"]) != 6 or len(groups["DEVELOPMENT"]) != 6 or len(groups["FINAL"]) != 3:
        raise ValueError("original structural allocation differs")
    files = manifest.get("split_files", {})
    if set(files) != {"TRAIN", "DEVELOPMENT", "FINAL"}:
        raise ValueError("missing partition file metadata")
    entry = files["DEVELOPMENT"]
    if entry.get("count") != 12:
        raise ValueError("expected 12 original DEVELOPMENT seed rows")
    path = verify_file(corpus, entry["path"], entry["sha256"])
    rows = [strict_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 12 or len({r.get("id") for r in rows}) != 12:
        raise ValueError("duplicate/missing DEVELOPMENT row IDs")
    selected = defaultdict(list)
    for row in rows:
        group = row.get("group_id", row.get("structural_group"))
        if (row.get("split") != "DEVELOPMENT" or group not in groups["DEVELOPMENT"]
                or row.get("structural_group", group) != group or row.get("family") not in FAMILIES
                or type(row.get("prompt")) is not str or not row["prompt"].strip()
                or type(row.get("program")) is not dict or row.get("seed") not in SEEDS):
            raise ValueError("invalid original DEVELOPMENT row")
        target = canonical(row["program"])
        if row.get("completion", target) != target or row.get("program_sha256", digest_json(row["program"])) != digest_json(row["program"]):
            raise ValueError("original DEVELOPMENT teacher bytes/hash mismatch")
        admission = row.get("native_admission", {})
        if (row.get("native_qualified") is not True
                or admission.get("state") != "PASS" or admission.get("native_execution") is not True
                or admission.get("verified") is not True
                or admission.get("source_sha") != ORIGINAL_NATIVE_SOURCE
                or admission.get("runtime_sha256") != manifest.get("runtime_sha256")
                or admission.get("program_sha256") != digest_json(row["program"])):
            raise ValueError("missing genuine original native admission receipt")
        selected[group].append((row, target))
    if set(selected) != set(groups["DEVELOPMENT"]):
        raise ValueError("missing independent original DEVELOPMENT structures")
    family_count = Counter()
    for group, cases in selected.items():
        if (len(cases) != 2 or {row["seed"] for row, _ in cases} != SEEDS
                or len({row["family"] for row, _ in cases}) != 1):
            raise ValueError("not one independent structure with two known seeds")
        family_count[cases[0][0]["family"]] += 1
    if family_count != Counter({"greebles": 2, "pipework": 2, "architecture": 2}):
        raise ValueError("expected 2 groups per original DEVELOPMENT family")
    ordered = [pair for group in sorted(selected) for pair in sorted(selected[group], key=lambda x: x[0]["seed"])]
    custody = {
        "source_sha": ORIGINAL_NATIVE_SOURCE,
        "corpus_manifest_sha256": expected_manifest_sha256,
        "development_file_sha256": entry["sha256"],
        "development_seed_rows": len(rows),
        "independent_development_groups": len(selected),
        "families": dict(sorted(family_count.items())),
        "sealed_final_content_opened": False,
    }
    return ordered, custody


def encode_references(rows, processor, adapter):
    receipts = []
    for row, target in rows:
        encoded = adapter.encode_row(
            processor, {"id": row["id"], "prompt": row["prompt"], "completion": target},
            SYSTEM, MAX_CONTEXT)
        ids, labels = encoded["input_ids"], encoded["labels"]
        boundary = encoded["boundary"]
        if (not ids or len(ids) != len(labels) or len(ids) > MAX_CONTEXT
                or ids[-1] != labels[-1] or ids[-1] != 7
                or any(x != -100 for x in labels[:boundary])
                or any(x < 0 for x in labels[boundary:])
                or labels[boundary:].count(7) != 1):
            raise ValueError("incomplete official processor EOS/label framing")
        if (encoded["prompt_bytes"] != len(row["prompt"].encode("utf-8"))
                or encoded["completion_bytes"] != len(target.encode("utf-8"))
                or encoded["target_tokens"] != len(ids) - boundary):
            raise ValueError("original teacher bytes or complete target miscounted")
        receipts.append({
            "id": row["id"], "group_id": row.get("group_id", row["structural_group"]),
            "family": row["family"], "seed": row["seed"],
            "prompt_sha256": sha(row["prompt"].encode("utf-8")),
            "teacher_sha256": sha(target.encode("utf-8")),
            "prefix_tokens": boundary, "sequence_tokens": len(ids),
            "supervised_target_tokens_including_eos": encoded["target_tokens"],
            "fits_provisional_512": encoded["target_tokens"] <= PROVISIONAL_RESERVE,
            "input_ids_sha256": sha(json.dumps(ids, separators=(",", ":")).encode("utf-8")),
            "eos_token_id": 7,
        })
    return receipts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--processor-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "INCOMPLETE.json", {"status": "FAIL_OR_PARTIAL_NOT_QUALIFIED",
               "optimizer_updates": 0, "model_generations": 0, "native_executions": 0})
    started = time.monotonic()
    blobs = verify_local_source()  # pin exact qualified language/processor boundaries before importing
    cases, custody = select_visible_development(args.corpus)
    profile = strict_json((LANE / "profile.json").read_text(encoding="utf-8"))
    if profile.get("model_id") != MODEL_ID or profile.get("revision") != REVISION or profile["framing"]["eos_token_id"] != 7:
        raise ValueError("profile/model identity mismatch")
    files = obtain_small_files(args.processor_dir, profile)
    adapter = load_qualified_adapter()
    processor = adapter.load_processor(args.processor_dir, profile)
    receipt = encode_references(cases, processor, adapter)
    if len(receipt) != 12 or len({x["group_id"] for x in receipt}) != 6:
        raise ValueError("incomplete source-group coverage")
    write_json(args.output / "original-reference-tokens.json", receipt)
    group_rows = defaultdict(list)
    for x in receipt:
        group_rows[x["group_id"]].append(x)
    groups = [{
        "group_id": g, "family": items[0]["family"],
        "seeds": sorted(x["seed"] for x in items),
        "target_tokens_min": min(x["supervised_target_tokens_including_eos"] for x in items),
        "target_tokens_max": max(x["supervised_target_tokens_including_eos"] for x in items),
        "all_two_seed_targets_fit_provisional_512": all(x["fits_provisional_512"] for x in items),
        "seed_target_bytes_identical": len({x["teacher_sha256"] for x in items}) == 1,
    } for g, items in sorted(group_rows.items())]
    write_json(args.output / "original-six-groups.json", groups)
    summary = {
        "status": "PASS_ORIGINAL_VISIBLE_DEV_REFERENCE_ONLY_NOT_DEV60",
        "model_id": MODEL_ID, "model_revision": REVISION,
        "official_processor": type(processor.processor).__name__,
        "model_weights_downloaded": False,
        "model_file_sha256": files, "qualified_source_blobs": blobs,
        "original_corpus": custody,
        "independent_development_groups": len(groups), "native_seed_rows": len(receipt),
        "supervised_target_tokens_min": min(x["supervised_target_tokens_including_eos"] for x in receipt),
        "supervised_target_tokens_max": max(x["supervised_target_tokens_including_eos"] for x in receipt),
        "full_sequence_tokens_max": max(x["sequence_tokens"] for x in receipt),
        "provisional_512_target_fit_seed_rows": sum(x["fits_provisional_512"] for x in receipt),
        "provisional_512_target_fit_whole_groups": sum(x["all_two_seed_targets_fit_provisional_512"] for x in groups),
        "provisional_512_is_approved_budget": False,
        "future_dev60_requests_tested": 0, "future_dev60_frozen": False,
        "newly_admitted_structures": 0, "model_generations": 0,
        "optimizer_updates": 0, "native_executions": 0,
        "sealed_final_content_opened": False, "elapsed_seconds": time.monotonic() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    write_json(args.output / "summary.json", summary)
    (args.output / "INCOMPLETE.json").unlink()
    print(json.dumps({k: summary[k] for k in ("status", "independent_development_groups",
            "native_seed_rows", "supervised_target_tokens_max",
            "provisional_512_target_fit_whole_groups")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
