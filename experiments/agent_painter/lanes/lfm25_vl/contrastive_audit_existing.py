#!/usr/bin/env python3
"""Read-only retrospective audit of one immutable LFM native research artifact.

The original Godot job executed six positive physical Art cases but its verifier
failed globally because process-result.json used inputs_unchanged rather than
the common verifier's input_unchanged. This tool never rewrites that original
archive: it verifies its ZIP hash, copies into an ephemeral directory, applies
one explicitly recorded metadata-alias correction, then runs the ORIGINAL common
native verifier against every receipt, actual material buffer and PNG.

This does not replay the engine, admit TRAIN, train a model or clear semantic
split quarantine. Keep the old failed job as failure evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

LANE = Path(__file__).resolve().parent
sys.path.insert(0, str(LANE.parents[1] / "native"))
from verify import verify as verify_native

SOURCE = "abef683ceaa0fb2782e2d4b65db2c4c3c475e52d"
ZIP_SHA256 = "e90014c8883bb89e1caaebfdb00cf28f2078604553a6b8de803e6caf9305aebf"
ORIGINAL_RUN_ID = 43587733967
ARTIFACT_ID = 14111581483
PREFIX = "lfm-contrastive-native/"
C3 = "lfm-c3-gantry-room-and-underdeck-stairs"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def repaired_alias(original: dict) -> dict:
    if (type(original) is not dict or "input_unchanged" in original
            or original.get("inputs_unchanged") is not True
            or original.get("runtime_unchanged") is not True
            or original.get("harness_unchanged") is not True
            or original.get("exit_code") != 0 or original.get("timed_out") is not False):
        raise ValueError("Not the exact eligible failed-custody metadata contract")
    fixed = dict(original)
    del fixed["inputs_unchanged"]
    fixed["input_unchanged"] = True
    if {k: v for k, v in fixed.items() if k != "input_unchanged"} != {
            k: v for k, v in original.items() if k != "inputs_unchanged"}:
        raise ValueError("Unrelated process evidence changed")
    return fixed


def verify_disposition(path: Path, ids: set[str]) -> list[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if (obj.get("schema") != "cybersand.lfm25_vl.contrastive-review-disposition/1"
            or obj.get("source_screen_producer_sha") != "44721f3b116cb0eedfea36965be4c20274d4a5e3"
            or obj.get("native_research_producer_sha") != SOURCE
            or obj.get("not_common_train") is not True
            or obj.get("final_targets_opened") is not False):
        raise ValueError("Missing approved research-only disposition")
    entries = obj["entries"]
    if (len(entries) != 3 or {r["id"] for r in entries} != ids
            or any(r.get("train_admitted") is not False
                   or r.get("global_independence_confirmed") is not False for r in entries)):
        raise ValueError("Proposal registry changed or was implicitly admitted")
    quarantined = [r["id"] for r in entries if r["disposition"].startswith("QUARANTINED_")]
    if quarantined != [C3]:
        raise ValueError("Architecture C3 near-alias quarantine must remain active")
    return quarantined


def audit(archive: Path, disposition: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Output must be a new path; no artifact overwrite")
    zip_bytes = archive.read_bytes()
    if sha(zip_bytes) != ZIP_SHA256:
        raise ValueError("Original outer ZIP hash differs from published immutable receipt")
    with tempfile.TemporaryDirectory(prefix="lfm-rehydrate-") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(archive) as z:
            infos = z.infolist()
            names = [i.filename for i in infos]
            if (len(infos) != 62 or len(set(names)) != len(names)
                    or z.testzip() is not None or sum(i.file_size for i in infos) > 24000000
                    or any(not i.filename.startswith(PREFIX) and
                           not i.filename.startswith(("build/", "native-build.json"))
                           for i in infos)
                    or any(i.filename.startswith("/") or ".." in Path(i.filename).parts
                           or (i.external_attr >> 16) & 0o170000 == 0o120000 for i in infos)):
                raise ValueError("Original artifact entry inventory unsafe or unexpected")
            z.extractall(root)
        original = root / "lfm-contrastive-native"
        process_path = original / "process-result.json"
        process_bytes = process_path.read_bytes()
        process = json.loads(process_bytes)
        corrected = repaired_alias(process)
        old_verification = json.loads((original / "native-verification.json").read_text())
        if (old_verification.get("passed") is not False
                or old_verification.get("failures") != ["input_unchanged"]):
            raise ValueError("Old failed native verifier was not the single known alias")
        custody = json.loads((original / "runtime-custody.json").read_text())
        if custody.get("source_sha") != SOURCE:
            raise ValueError("Original native source identity mismatch")
        original_report = json.loads((original / "lfm-contrastive-native-summary.json").read_text())
        if (original_report.get("actual_native_source_sha") != SOURCE
                or original_report.get("actual_native_verified_views") != 0
                or original_report.get("new_common_train_admissions") != 0
                or original_report.get("new_model_completions") != 0
                or original_report.get("optimizer_updates") != 0):
            raise ValueError("Original failed summary changed")
        original_file_hashes = {p.relative_to(original).as_posix(): sha(p.read_bytes())
                                for p in original.rglob("*") if p.is_file()}
        corrected_path = root / "verified-copy"
        shutil.copytree(original, corrected_path)
        (corrected_path / "process-result.json").write_text(
            json.dumps(corrected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        verified = verify_native(corrected_path)
        cases = verified.get("cases", [])
        if (verified.get("passed") is not True
                or len(cases) != 6 or not all(c.get("success") is True for c in cases)
                or len({c["id"] for c in cases}) != 6
                or sorted(c["seed"] for c in cases) != [104729]*3+[130363]*3
                or {c["family"] for c in cases} != {"greebles", "pipework", "architecture"}
                or not all(c.get("affected_cells", 0) > 0 and
                           c.get("structure", {}).get("passed") is True for c in cases)):
            raise ValueError("Rehydrated original common native verifier did not pass all cases")
        # Every original file, including the old failed report, remains unchanged.
        after = {p.relative_to(corrected_path).as_posix(): sha(p.read_bytes())
                 for p in corrected_path.rglob("*") if p.is_file()}
        if (set(after) != set(original_file_hashes)
                or [p for p in after if after[p] != original_file_hashes[p]]
                   != ["process-result.json"]):
            raise ValueError("Retrospective audit modified anything beyond process field alias")
        proposals = {c["id"].rsplit("-s", 1)[0] for c in cases}
        quarantine = verify_disposition(disposition, proposals)
        # These are NOT three new TRAIN structural groups.
        report = {
            "schema": "cybersand.lfm25_vl.retrospective-native-audit/1",
            "original_failed_run_id": ORIGINAL_RUN_ID,
            "original_artifact_id": ARTIFACT_ID,
            "original_artifact_zip_sha256": ZIP_SHA256,
            "original_native_source_sha": SOURCE,
            "original_failed_verifier": "input_unchanged missing; inputs_unchanged=true",
            "original_process_sha256": sha(process_bytes),
            "repaired_ephemeral_process_sha256": sha(
                (corrected_path / "process-result.json").read_bytes()),
            "repair_scope": "exact metadata key rename, only in ephemeral verification copy",
            "original_failed_report_preserved": True,
            "source_native_verifier_passed_after_custody_alias": True,
            "verification_checks": verified.get("checks"),
            "original_engine_assertions": verified.get("engine_checks"),
            "physical_verified_seed_views": len(cases),
            "independent_hand_authored_compositions": 3,
            "case_results": [{"id": c["id"], "family": c["family"], "seed": c["seed"],
                              "affected_cells": c["affected_cells"],
                              "raw_sha256": c["readback_sha256"],
                              "png_sha256": c["image_sha256"],
                              "history_reopen": c["history_and_reopen_checked"]}
                             for c in cases],
            "quarantined_near_alias_ids": quarantine,
            "unquarantined_proposals_pending_global_review": 2,
            "new_shared_train_admissions": 0,
            "model_generated_paintings": 0,
            "optimizer_updates": 0,
            "new_native_execution": False,
            "final_targets_opened": False,
            "useful_vN_freeze": False,
            "candidate_status": "RESEARCH_ONLY_HOLD",
        }
        output.mkdir(parents=True)
        (output / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
        (output / "reverified-native.json").write_text(json.dumps(verified, indent=2, sort_keys=True)+"\n")
        return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--disposition", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = audit(args.archive, args.disposition, args.output)
    print(json.dumps({"pass": True, "cases": result["physical_verified_seed_views"],
                      "checks": result["verification_checks"],
                      "new_shared_train_admissions": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
