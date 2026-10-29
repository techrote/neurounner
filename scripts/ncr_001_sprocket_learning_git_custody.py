#!/usr/bin/env python3
"""NCR-001E: offline, stdlib-only readback of six byte-exact original ZIP members.

Archive is ASCII base64(gzip(JSON(base64(original ZIP member bytes)))), split
into three ordered Git files. This never accesses Actions/network/model/optimizer
or held-out fixtures. It is evidence custody, not scientific reviewer admission.
"""
from __future__ import annotations

import base64
import copy
import gzip
import json
from pathlib import Path

from ncr_001_sprocket_learning_artifact_readback import (
    ARTIFACTS, HEAD, ROOT, RUN_ID, WEIGHTS, check, negatives, receipt_gate, sha,
)

EVIDENCE = ROOT / "experiments/ncr-001/controls-evidence/sprocket-learning-20261028"
EXPECTED_GZIP_SHA = "dc3aef656ab5b3cc51fc169c93892d9bed9826d0b9846946d1dad1604ecb961a"
EXPECTED_JSON_SHA = "68982cc3654c458b91a9355d1c8692cc56e19b534375c6379e61de5fe9d1596d"


def load_archive() -> tuple[dict, dict[str, dict[str, bytes]]]:
    m = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    parts = []
    for index, length in enumerate((3200, 3200, 3084), start=1):
        p = EVIDENCE / ("original-receipt-bundle.part%03d.b64" % index)
        part = p.read_text(encoding="ascii").strip()
        check(len(part) == length, "Truncated/reordered Git archive part: " + str(index))
        parts.append(part)
    encoded = "".join(parts)
    check(len(encoded) == 9484, "Original encoded archive length differs")
    zipped = base64.b64decode(encoded, validate=True)
    check(sha(zipped) == m["archive_gzip_sha256"] == EXPECTED_GZIP_SHA,
          "Original Git gzip bytes digest differs")
    original_json = gzip.decompress(zipped)
    check(sha(original_json) == m["archive_json_sha256"] == EXPECTED_JSON_SHA,
          "Original uncompressed JSON bytes digest differs")
    packed = json.loads(original_json)
    check(m["original_source_head"] == HEAD and str(m["original_run_id"]) == RUN_ID,
          "Original physical source SHA/run changed")
    check(m["train_authorized"] is False and
          m["full_72_conformance_pass"] is False and
          m["independent_scientific_review"] is False and
          m["comparative_training_performed"] is False,
          "Manifest wrongly admits training or scientific review")
    expected = m["original_actions_artifacts"]
    check(set(packed) == set(expected) == set(ARTIFACTS) ==
          {"source", "chat", "instruct"}, "Four-model parent source lanes changed")
    decoded = {}
    for lane in packed:
        members = packed[lane]
        orig_zip_sha, authoritative = ARTIFACTS[lane]
        meta = expected[lane]
        check(meta["zip_sha256"] == orig_zip_sha and
              set(members) == set(meta["members"]) == set(authoritative),
              "Original ZIP metadata or member set wrong: " + lane)
        decoded[lane] = {}
        for name, b64 in members.items():
            check(name in ("source-tests.log", "lease-tests.log", "host.log",
                           "receipt.json"), "Unexpected ZIP member name")
            raw = base64.b64decode(b64, validate=True)
            a_sha, a_size = authoritative[name]
            check(sha(raw) == a_sha == meta["members"][name]["sha256"] and
                  len(raw) == a_size == meta["members"][name]["bytes"],
                  "Original Git ZIP member tampered: " + lane + "/" + name)
            decoded[lane][name] = raw
    return m, decoded


def main() -> None:
    m, d = load_archive()
    check(b"Ran 10 tests" in d["source"]["source-tests.log"] and
          b"Ran 10 tests" in d["source"]["lease-tests.log"] and
          b"\nOK\n" in d["source"]["source-tests.log"] and
          b"\nOK\n" in d["source"]["lease-tests.log"],
          "Original synthetic/lease source tests incomplete")
    results = {variant: receipt_gate(
        json.loads(d[variant]["receipt.json"]), variant, source_tree=True)
        for variant in WEIGHTS}
    check(results["chat"]["backbone_fingerprint"] !=
          results["instruct"]["backbone_fingerprint"],
          "Original distinct model identity collapsed")
    for variant in WEIGHTS:
        check(b"PASS_DIAGNOSTIC_NCA_LEARNING_GATES__FINAL_72_BLOCKED" in
              d[variant]["host.log"], "Original physical producer log failed")
    receipts = {variant: json.loads(d[variant]["receipt.json"])
                for variant in WEIGHTS}
    # The twelve semantic mutations call the same receipt validator, not mere value comparisons.
    report = {
        "classification": "ORIGINAL_SPROCKET_NCA_DIAGNOSTIC_GIT_BYTE_CUSTODY_PASS__FINAL_72_BLOCKED",
        "original_source_head": HEAD,
        "original_run_id": RUN_ID,
        "original_member_count": sum(map(len, d.values())),
        "archive_promoted_to_git": True,
        "original_gzip_sha256": EXPECTED_GZIP_SHA,
        "original_json_sha256": EXPECTED_JSON_SHA,
        "adversarial_semantic_mutations_rejected": negatives(receipts),
        "train_authorized": False,
        "comparative_training_performed": False,
        "full_72_conformance_pass": False,
        "independent_scientific_review": False,
        "original_physical_results": results,
    }
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
