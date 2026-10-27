#!/usr/bin/env python3
"""Byte-exact readback of TWO original frozen-host Sprocket 2048-unit JSON receipts.

No models executed, no training/fixture access, no source-invented tool grammar.
This independently checks preserved bytes and fail-closed numeric/source claims.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "experiments/ncr-001/evidence-review/sprocket-extended-context"
ARCHIVE = FOLDER / "original-jsons.tar.gz.b64"
PROVENANCE = FOLDER / "provenance.json"
ARCHIVE_SHA = "60446ea2f60e5f0b7cc22333dabe45f6fa6f53d4099fda86fa9480c3987bb4af"
RUN_ID = "40544384293"
MERGE_SHA = "2fd70b173a56742239cfd31327d63ac2cefb981f"
SOURCE_REV = "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
SOURCE_CARD_SHA = "09aec879dc5b387bdc031dc3c74d77a9ba44362a55163f9a8b456761b68142c9"
EXPECTED = {
    "chat": {
        "bytes": 4678, "sha": "12423aee3dfb9cd493698388309c5b36749db051f074f3d48f23215906a10d24",
        "weight": "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97",
        "file": "chat/model.safetensors", "hook": "83a56835a7476f17a17980d28dd1a7c14bb6f2f6",
        "runner": "GitHub Actions 1000045238", "cache_2048": 2.193450927734375e-05,
        "raw_controls": 2.5033950805664062e-05,
        "artifact": 12753100555, "zip_sha": "ef5fd03c53818136fde2ab34c8b935d311628354b7dcfd7ba872d79fdd602482",
    },
    "instruct": {
        "bytes": 4801, "sha": "4f47f690270ef58e051960e23a187573b04461ae2ed9e6ef6e68347c2a366a59",
        "weight": "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864",
        "file": "model.safetensors", "hook": "b95c010f9b2ba3cbd9f81317981316bb9c8acd19",
        "runner": "GitHub Actions 1000045239", "cache_2048": 2.0503997802734375e-05,
        "raw_controls": 6.198883056640625e-05,
        "artifact": 12752646313, "zip_sha": "c9fdcf5a64d6463c2aa5b84b4ba7f2aad261cfdc0f52208ccb8d620568e87816",
    },
}
STATUS = "PASS_ORIGINAL_TWO_VARIANT_2048_FROZEN_RECEIPT_BYTE_READBACK__FINAL_NCR72_BLOCKED"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(ok, why: str) -> None:
    if not ok:
        raise RuntimeError(why)


def originals() -> dict[str, dict]:
    encoded = ARCHIVE.read_text(encoding="ascii").strip()
    raw = base64.b64decode(encoded, validate=True)
    require(sha(raw) == ARCHIVE_SHA, "Original compressed source archive SHA changed")
    uncompressed = gzip.decompress(raw)
    require(len(uncompressed) < 30000, "Unexpected original receipt payload volume")
    with tarfile.open(fileobj=io.BytesIO(uncompressed), mode="r:") as tar:
        entries = tar.getmembers()
        require({e.name for e in entries} == {v + "/original.json" for v in EXPECTED},
                "Missing, extra or renamed frozen source receipts")
        require(len(entries) == 2, "Duplicate or extraneous original JSON members")
        reports = {}
        for e in entries:
            require(e.isfile() and not e.issym() and not e.islnk(),
                    "Unsafe original archive member")
            variant = e.name.split("/")[0]
            data = tar.extractfile(e).read()
            spec = EXPECTED[variant]
            require(len(data) == spec["bytes"] and sha(data) == spec["sha"],
                    "Original " + variant + " JSON bytes do not match source CI")
            reports[variant] = json.loads(data)
    return reports


def check(variant: str, report: dict, spec: dict, provenance: dict) -> dict:
    require(report["experiment"] == "NCR-001E" and report["variant"] == variant,
            "Source experiment/variant swapped")
    require(report["status"] == "PASS_SOURCE_BOUND_EXTENDED_CONTEXT_PREPASS",
            "Original actual-host report did not pass")
    for key in ("train_authorized", "actual_model_training", "heldout_evaluation",
                "tool_and_memory_behavior_qualified", "full_four_model_conformance_pass",
                "control_probe_is_valid_tool_memory_dialogue",
                "original_short_256_process_results_reused_as_new_model_run"):
        require(report[key] is False, "False admission/qualification in " + key)
    require(report["optimizer_steps"] == 0,
            "Original actual-host optimizer steps were not zero")
    require(report["base_parameters"] == 501090560 and
            report["adapter_parameters"] == 658904,
            "Frozen original Sprocket source parameter geometry changed")
    require(report["context_limit"] == 2048 and
            report["context_lengths"] == [512, 1024, 2048] and
            report["prompt_tokenizer_units"] == 2048 and
            report["source_verifier_initial_prefix_units"] == 256 and
            report["cache_tail_units"] == 8,
            "Actual source context/length/cached-step claim changed")
    require(report["source_checkpoint_sha256"] == spec["weight"] and
            report["source_hook_commit"] == spec["hook"] and
            report["source_files"][spec["file"]] == {
                "bytes": 1002207984, "sha256": spec["weight"]},
            "Original actual frozen checkpoint or hook source differs")
    require(report["source_files"]["README.md"]["sha256"] == SOURCE_CARD_SHA,
            "Original source-role card checksum changed")
    require(report["role_ids"] == {
        "<|user|>": [3], "<|end|>": [5], "<|assistant|>": [4]},
        "Publisher ordinary-role tokenizer IDs changed")
    require(report["tokens_sha256"] ==
            "0f4094d5e98a558601c281297191bd1bd0bc00acfa5a14472811758f4ae6970c" and
            report["token_ids_sha256"] ==
            "845ece67eb9e4930cdc9f5bd72c18c234f7b34ce48d9189151fecfa7ba69af82",
            "Pinned original source-ordinary-role token sequence changed")
    require(report["control_tokens"] == {
        "<|tool_call|>": 6, "<|tool_result|>": 7,
        "<|memory_read|>": 10, "<|memory_write|>": 11},
        "Original tool/memory control marker identity changed")
    runner = report["runner"]
    require(runner["github_run_id"] == RUN_ID and
            runner["github_run_attempt"] == "1" and
            runner["github_sha"] == MERGE_SHA and
            runner["runner_name"] == spec["runner"] and
            runner["cpu_count"] == 4 and
            runner["python"] == "3.12.3" and
            runner["torch"] == "2.14.1+cpu" and
            runner["transformers"] == "4.57.6",
            "Original hosted runner/source process identity differs")
    require(report["peak_rss_kib"] > 0 and
            report["peak_rss_kib"] == provenance["variants"][variant]["peak_rss_kib"],
            "Peak memory measurement missing or unbound")

    checks = report["checks"]
    names = set()
    for n in (512, 1024, 2048):
        names.add(f"{n}_full_vs_prefill_plus_8_incremental")
        names.add(f"{n}_untrained_effect")
    for n in (512, 2048):
        names.add(f"{n}_frozen_zero_adapter_exact")
    names.update({"512_strict_future_suffix", "cross_request_A_B_A_exact",
                  "raw_control_ids_transport_only_cache"})
    require(set(checks) == names, "Actual runtime required checks missing or altered")
    for key in names:
        actual = checks[key]
        value = actual["max_abs"]
        require(isinstance(value, (int, float)) and math.isfinite(value),
                "Missing finite source runtime numerical metric")
        if key.endswith("_untrained_effect"):
            require(actual["trained"] is False and actual["pass"] is True and value > 0,
                    "No genuine manually activated untrained NCA intervention")
        elif key.endswith("_exact") or key == "512_strict_future_suffix":
            require(actual["pass"] is True and actual["exact_required"] is True and
                    actual["atol"] == actual["rtol"] == 0 and value == 0,
                    "Strict zero drift or causal suffix invariant failed")
        else:
            require(actual["pass"] is True and
                    actual["atol"] == actual["rtol"] == 0.0005 and
                    actual["exact_required"] is False and 0 <= value <= 0.0005,
                    "Full vs genuine cached token sequence exceeded protocol gate")
    require(checks["2048_full_vs_prefill_plus_8_incremental"]["max_abs"] ==
            spec["cache_2048"] and
            checks["raw_control_ids_transport_only_cache"]["max_abs"] ==
            spec["raw_controls"], "Original precise 2048/control cache metric changed")
    require(provenance["variants"][variant]["artifact_id"] == spec["artifact"] and
            provenance["variants"][variant]["artifact_zip_sha256"] == spec["zip_sha"] and
            provenance["variants"][variant]["checkpoint_sha256"] == spec["weight"] and
            provenance["original_originals"][variant]["sha256"] == spec["sha"],
            "Original artifact provenance no longer matches immutable source")
    return {
        "variant": variant, "full_checkpoint_sha256": spec["weight"],
        "original_hook_commit": spec["hook"],
        "original_artifact_zip_sha256": spec["zip_sha"],
        "original_receipt_sha256": spec["sha"],
        "max_abs_2048_cached": spec["cache_2048"],
        "max_abs_raw_control_ID_transport": spec["raw_controls"],
        "source_receipt_readback_pass": True,
        "tool_memory_semantics_qualified": False,
        "final_ncr72_admitted": False, "training_authorized": False,
    }


def audit(records: dict[str, dict] | None = None) -> dict:
    if records is None:
        records = originals()
    require(set(records) == set(EXPECTED), "Both source variants required")
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    require(provenance["original_run_id"] == int(RUN_ID) and
            provenance["original_actions_pr_merge_sha"] == MERGE_SHA and
            provenance["original_archive_gzip_sha256"] == ARCHIVE_SHA and
            provenance["source_revision"] == SOURCE_REV and
            provenance["source_card_sha256"] == SOURCE_CARD_SHA,
            "Original run/revision/source archive provenance mismatch")
    for key in ("train_authorized", "tool_memory_semantics_qualified",
                "final_four_model_72_admission", "heldout_evaluation",
                "comparative_adapter_training"):
        require(provenance["evidence_limit"][key] is False,
                "Original provenance falsely claims completed qualification")
    result = {name: check(name, records[name], spec, provenance)
              for name, spec in EXPECTED.items()}
    require(result["chat"]["full_checkpoint_sha256"] !=
            result["instruct"]["full_checkpoint_sha256"],
            "Chat/root Instruct model identities collapsed")
    return {
        "status": STATUS,
        "original_real_model_run_id": int(RUN_ID),
        "original_receipt_count": 2,
        "original_archive_gzip_sha256": ARCHIVE_SHA,
        "this_auditor_executed_models": False,
        "training_authorized": False,
        "tool_memory_semantics_qualified": False,
        "final_four_model_conformance": False,
        "variants": result,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
