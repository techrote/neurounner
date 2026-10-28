"""Offline original-job custody verifier for NCR-001 cross-adapter lease.

No model downloads, optimizer, original-host execution, held-out data or
training admission. The original SHA identities are deliberately hard-coded:
changing a provenance JSON alone cannot bless altered run logs/receipts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/ncr-001/controls-evidence/cross-decoder-lease"
LOG_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z ")
CASE = re.compile(r"^test_[a-z_]+ \(test_ncr_001_cross_decoder_lease\.CrossAdapterLeaseTests\.test_[a-z_]+\) \.\.\. ok$")

EXPECTED = {
    "success": {
        "run_id": 40673634722,
        "job_id": 122766480553,
        "head_sha": "4096aa1238ba65851eae78106417131b377d8d82",
        "status": "completed",
        "conclusion": "success",
        "runner": "GitHub Actions 1000046432",
        "log_git_blob": "cc64b38294243ca567e7453e4bfb96dceca1db88",
        "receipt_git_blob": "ce6682d91441d1d3d0f3edafac7469194a2a8fb7",
        "artifact_id": 12808873895,
        "artifact_digest": "sha256:6ad8c0a33e229c02019b203a9e82e743a8c720a4621f8a7ddedc396f039f82b5"
    },
    "cancelled": {
        "run_id": 40673587828,
        "job_id": 122766223560,
        "head_sha": "ec9327c892e0655a8ce9464ff34a4ae22ebbbb11",
        "status": "completed",
        "conclusion": "cancelled",
        "runner": "GitHub Actions 1000046431",
        "log_git_blob": "31230f27860b4adace3c3d6bdafc672eb81520fc",
        "receipt_git_blob": "f56919b54be347bc0c04e0ece1cb51d5d957ca1a",
        "artifact_id": 12809188078,
        "artifact_digest": "sha256:929f00f849c25c07f58eb55eda3d8a316a0584a1aa8cb432171b8b0d2bf3c404"
    }
}

def git_blob_sha(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()

def unstamp(log: str) -> list[str]:
    return [LOG_PREFIX.sub("", line) for line in log.lstrip("\ufeff").splitlines()]

def original_receipt_from_log(log: str) -> dict:
    lines = unstamp(log)
    for i, line in enumerate(lines):
        if line != "{" or i + 1 >= len(lines) or '"execution": "synthetic_cross_adapter_only"' not in lines[i + 1]:
            continue
        for stop in range(i + 3, min(len(lines), i + 100)):
            if lines[stop] == "}":
                try:
                    return json.loads("\n".join(lines[i:stop + 1]))
                except json.JSONDecodeError:
                    continue
    raise AssertionError("Original complete producer JSON is absent from job log")

def verify_run(tag: str, meta: dict, *, log_data: bytes | None = None, receipt_data: bytes | None = None,
               check_sources: bool = True) -> dict:
    fixed = EXPECTED[tag]
    for key in ("run_id","job_id","head_sha","status","conclusion","runner","log_git_blob",
                "receipt_git_blob","artifact_id","artifact_digest"):
        assert meta[key] == fixed[key], f"{tag} original {key} changed"
    assert meta["log_path"].startswith("experiments/ncr-001/controls-evidence/cross-decoder-lease/")
    assert meta["receipt_path"].startswith("experiments/ncr-001/controls-evidence/cross-decoder-lease/")
    log_raw = log_data if log_data is not None else (ROOT / meta["log_path"]).read_bytes()
    receipt_raw = receipt_data if receipt_data is not None else (ROOT / meta["receipt_path"]).read_bytes()
    assert git_blob_sha(log_raw) == fixed["log_git_blob"], f"{tag} original log bytes changed"
    assert git_blob_sha(receipt_raw) == fixed["receipt_git_blob"], f"{tag} source-emitted receipt changed"
    log_text = log_raw.decode("utf-8-sig")
    parsed = original_receipt_from_log(log_text)
    assert receipt_raw == (json.dumps(parsed, sort_keys=True, indent=2) + "\n").encode(), f"{tag} receipt not verbatim source-emitted JSON"
    assert parsed["run_id"] == str(fixed["run_id"])
    assert parsed["head_sha"] == fixed["head_sha"]
    assert parsed["runner_name"] == fixed["runner"]
    assert parsed["runner_os"] == "Linux" and parsed["runner_arch"] == "X64"
    assert parsed["execution"] == "synthetic_cross_adapter_only"
    assert parsed["original_model_executed"] is False
    assert parsed["optimizer_steps"] == 0 and parsed["training_authorized"] is False
    assert parsed["test_log_has_OK"] is True
    clean = unstamp(log_text)
    cases = [m.group() for row in clean if (m := CASE.fullmatch(row))]
    assert len(cases) == 10 and len(set(cases)) == 10, f"{tag} negative test coverage incomplete"
    assert any(re.fullmatch(r"Ran 10 tests in [0-9.]+s", row) for row in clean), f"{tag} missing unittest total"
    assert "OK" in clean, f"{tag} missing source-test OK"
    assert any(f'\"head_sha\": \"{fixed["head_sha"]}\"' in row for row in clean), f"{tag} wrong head in physical log"
    if check_sources:
        for path,sha in parsed["source_sha256"].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha, path
        assert len(parsed["source_sha256"]) == 9
        assert parsed["source_sha256"]["experiments/ncr-001/protocol.json"] == "a76d38f80a57012ca8e62fba183d1cf2e91fd0986032f1adcdfb0aef843bf0b4"
        p = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
        assert p["train_authorized"] is False
        assert set(p["models"]) == {"smollm2","sprocket_chat","sprocket_instruct","void1"}
    return {"tag":tag,"source_head":parsed["head_sha"],"tests":len(cases),"verdict":fixed["conclusion"],"runner":fixed["runner"]}

def verify_archive() -> dict:
    p = json.loads((EVIDENCE / "original-ci-provenance.json").read_text())
    assert p["schema_version"] == 1 and p["experiment"] == "NCR-001"
    assert p["workflow_original_git_blob"] == "306cb2bb6330e7e56cfe2b3a6f31c869c694ffb0"
    assert git_blob_sha((EVIDENCE / "original-tested-workflow.yml").read_bytes()) == p["workflow_original_git_blob"]
    assert p["authorisation"] == {"train_authorized":False,"optimizer_steps":0,"original_model_executed":False}
    results = {}
    for tag in ("success","cancelled"):
        results[tag] = verify_run(tag, p["logs"][tag])
    assert results["success"]["verdict"] == "success"
    assert results["cancelled"]["verdict"] == "cancelled"
    return results

if __name__ == "__main__":
    print(json.dumps(verify_archive(), indent=2, sort_keys=True))
