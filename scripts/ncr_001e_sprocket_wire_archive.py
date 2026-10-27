#!/usr/bin/env python3
"""Independent, stdlib-only audit of ORIGINAL 21-generation NCR-001E receipts.

Reads losslessly archived Actions text members. Never executes/downloads a model,
opens sealed fixtures, trains, authorizes #72, or replaces original source evidence.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import lzma
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "datasets/ncr-001-sprocket-wire-diagnostic"
ARCHIVE = FOLDER / "original-text-members.tar.xz.b64"
MANIFEST = FOLDER / "provenance.json"
ORIGINAL_RUN = 40573188117
ORIGINAL_HEAD = "905e237136e4ea75208a8ed92392489a9f62849d"
ORIGINAL_ZIP_SHA = "7e38a9c8438701c460597fc87016f3a26ece2d12b97482cfbcb44b0e6974a2ab"
ORIGINAL_ARCHIVE_SHA = "982dfaee356bbcdce705194413a71ace1ce7787c4421ea017dd8b95e8aec2921"
ORIGINAL_TAR_SHA = "8524f88a3b81a0365aec65386d3b9c0ff4000bc4e1e91a4cf0ab15d8f71e3f0a"
ORIGINAL_RECEIPT_SHA = "94b1ff4f3ee4cb19d3e1e8cd57f8e9e7eee9534503b87a86d1aa6c5951de3b45"
ROOT_WEIGHT_SHA = "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864"
ORIGINAL_HOOK = "b95c010f9b2ba3cbd9f81317981316bb9c8acd19"
PUBLISHER_PIN = "a003767e70d766bc14b40c89792543ee8641c8bc"
ORIGINAL_SOURCE_SNAPSHOT_SHA = "bb6109524362c8be1b19b420e7c646de4cc9a685ca7157e6afa55e0c203ae961"
EXPECTED_MEMBERS = {
    "checkout-sha.txt": (41, "1dae2f72293ef2421228c72da575fd272aeafcece49872686b6113445b3ed9d5"),
    "legacy-reproduction.json": (1317, "3b9e12425d260c526d2aeffe14498422863830aefbf2f11e7276f26811957430"),
    "packages.txt": (1714, "506849ee72ac85b225d9f7ede1acb412b95df9233e5947c4eebab2d988bf6cb9"),
    "plan.json": (4084, "642ff0367d172e54fce2a9a1464a8033ff6a31a58c5116fa4d5484dc681a822c"),
    "prefetch.json": (783, "a54e21d669ce8994adf66441c666d0618d79b38c0abab5d8b99c6f3283e01127"),
    "receipt.json": (75464, ORIGINAL_RECEIPT_SHA),
    "unit-tests.log": (1594, "11db10d07b22011cb65c95061cb2aade0f9a573c40a8f0864a085b24e6bcaeb2"),
}
EXPECTED_CASES = (
    "tool-request", "tool-result-42", "tool-result-17",
    "memory-context-amber-17", "memory-context-violet-28",
    "memory-write", "memory-request",
)
MODES = ("disabled", "zero", "active_untrained")
EXHAUSTED = {
    "tool-request", "tool-result-42", "tool-result-17",
    "memory-context-violet-28", "memory-request",
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError("NCR original diagnostic audit REFUSED: " + reason)


def load_originals(archive_path=ARCHIVE, manifest_path=MANIFEST):
    """Verify anchored compressed bytes, exact tar members and original file bytes."""
    provenance = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    metadata = provenance["archived_text_members"]
    encoded = Path(archive_path).read_text(encoding="ascii").strip()
    require(len(encoded) == 10124 and encoded.isascii(), "original base64 envelope")
    compressed = base64.b64decode(encoded, validate=True)
    require(len(compressed) == metadata["xz_bytes"] == 7592, "xz envelope length")
    require(sha(compressed) == metadata["xz_sha256"] == ORIGINAL_ARCHIVE_SHA,
            "original xz digest: raw bytes replaced")
    raw_tar = lzma.decompress(compressed, format=lzma.FORMAT_XZ)
    require(len(raw_tar) == 92160 and sha(raw_tar) ==
            metadata["tar_sha256"] == ORIGINAL_TAR_SHA, "original tar digest")
    file_bytes = {}
    with tarfile.open(fileobj=io.BytesIO(raw_tar), mode="r:") as source:
        members = source.getmembers()
        require([m.name for m in members] == sorted(EXPECTED_MEMBERS),
                "missing, reordered or extra archival text member")
        for item in members:
            require(item.isfile() and item.uid == item.gid == item.mtime == 0
                    and item.mode == 0o644 and item.name in EXPECTED_MEMBERS,
                    "archival member metadata or type differs")
            raw = source.extractfile(item).read()
            expected_size, expected_sha = EXPECTED_MEMBERS[item.name]
            require(item.size == len(raw) == expected_size
                    and sha(raw) == expected_sha, "original member bytes: " + item.name)
            file_bytes[item.name] = raw
    expected_manifest = {row["name"]: (row["bytes"], row["sha256"])
                         for row in metadata["members"]}
    require(expected_manifest == EXPECTED_MEMBERS, "manifest member inventory altered")
    return file_bytes, provenance


def verify_original(file_bytes: dict[str, bytes], provenance: dict) -> None:
    """Semantic overclaim/tamper guards are independent of compression SHA checks."""
    action = provenance["original_actions"]
    require(provenance["schema_version"] == 1
            and provenance["parent_issue"] == 67
            and provenance["conformance_issue"] == 72
            and provenance["draft_pr"] == 102
            and action["run_id"] == ORIGINAL_RUN and action["run_attempt"] == 1
            and action["artifact_id"] == 12766071456
            and action["original_zip_bytes"] == 79486
            and action["original_zip_sha256"] == ORIGINAL_ZIP_SHA
            and action["original_zip_archived_in_git"] is False,
            "incorrect Actions/PR/ZIP provenance")
    snapshot = provenance["original_binary_source_snapshot"]
    require(snapshot["filename"] == "source-snapshot.tar.gz"
            and snapshot["original_bytes"] == 68602
            and snapshot["original_sha256"] == ORIGINAL_SOURCE_SNAPSHOT_SHA
            and snapshot["original_bytes_archived_in_git"] is False
            and snapshot["source_git_commit"] == ORIGINAL_HEAD,
            "binary source snapshot misrepresented as archived")
    require(file_bytes["checkout-sha.txt"].decode("utf-8").strip() == ORIGINAL_HEAD,
            "actual source checkout mismatch")

    original = json.loads(file_bytes["receipt.json"])
    plan = json.loads(file_bytes["plan.json"])
    legacy = json.loads(file_bytes["legacy-reproduction.json"])
    require(original["schema"] == 1
            and original["status"] == "PASS_SOURCE_PARITY_AND_DIAGNOSTIC_EXECUTION_ONLY"
            and provenance["original_results"]["status"] == original["status"]
            and original["variant"] == "sprocket_instruct",
            "model, receipt or result status substituted")
    for field in ("actual_model_training", "heldout_evaluation", "train_authorized",
                  "final_72_admission", "tool_memory_semantics_qualified"):
        require(original[field] is False, "false admission or training claim: " + field)
    require(original["optimizer_steps"] == provenance["original_results"]["optimizer_steps"] == 0,
            "optimizer steps not zero")
    source = provenance["source"]
    require(source["model_key"] == "sprocket_instruct"
            and source["model_revision"] ==
            "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
            and source["root_checkpoint"] == "model.safetensors"
            and original["checkpoint_sha256"] ==
            source["root_checkpoint_sha256"] ==
            original["source_files"]["model.safetensors"]["sha256"] == ROOT_WEIGHT_SHA
            and original["source_files"]["model.safetensors"]["bytes"] ==
            source["root_checkpoint_bytes"] == 1002207984,
            "root Instruct source model replaced")
    require(original["hook_commit"] == source["original_hook_commit"] ==
            ORIGINAL_HOOK and original["publisher_commit"] ==
            source["publisher_commit"] == PUBLISHER_PIN
            and source["publisher_source_not_training_build_attested"] is True,
            "original pinned source or publisher authority changed")
    for prop, source_field, path in (
        ("candidate_script_sha256", "diagnostic_script_sha256",
         "scripts/ncr_001e_sprocket_wire_diagnostic.py"),
        ("serializer_sha256", "serializer_sha256",
         "scripts/ncr_001e_sprocket_publisher_wire.py"),
        ("protocol_sha256", "protocol_sha256",
         "experiments/ncr-001/protocol.json"),
    ):
        require(original[prop] == source[source_field] ==
                sha((ROOT / path).read_bytes()), "original source bytes drift: " + path)
    require(original["publisher_sft_sha256"] == source["publisher_sft_sha256"] ==
            "e5568c15989f3987fec07545640f2f271ffe967c27cbe9d9883c0dd7b5c4ec69",
            "reviewed upstream SFT SHA mismatch")
    policy = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
    require(policy["train_authorized"] is False
            and set(policy["models"]) ==
            {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"}
            and policy["conformance"]["full_vs_cached_prefill_atol"] ==
            policy["conformance"]["full_vs_cached_prefill_rtol"] == 0.0005,
            "four-lane protocol/causal gate drift")

    require(original["plan"] == plan
            and original["plan_sha256"] == source["plan_sha256"] ==
            sha(json.dumps(plan, sort_keys=True).encode("utf-8")),
            "fixed original synthetic plan changed")
    require(plan["optimizer_steps"] == 0 and plan["train_authorized"] is False
            and plan["heldout_evaluation"] is False
            and plan["quality_acceptance_threshold"] is None
            and plan["max_new_units"] == 64
            and plan["stop_ids"] == [0, 5]
            and tuple(plan["modes"]) == MODES
            and tuple(c["id"] for c in plan["cases"]) == EXPECTED_CASES,
            "bounded seven-case negative-friendly protocol changed")
    require(original["legacy_regressions"] == legacy
            and legacy["legacy_blob"] == "0abf73a8877cad61392517baa1a8487276ba5507"
            and legacy["train_authorized"] is False
            and legacy["model_execution"] is False
            and {v["id"] for v in legacy["cases"]} ==
            {"host_document_separator", "host_padding_marker",
             "assistant_padding_marker", "overflow_json_number"}
            and all(v["legacy_accepted"] is True and
                    v["repaired_rejected"] is True for v in legacy["cases"]),
            "historical four negative failure witnesses altered")

    runner = original["runner"]
    provenance_runner = provenance["original_runner"]
    require(runner["run_id"] == str(ORIGINAL_RUN)
            and runner["attempt"] == "1"
            and runner["checkout_sha"] == ORIGINAL_HEAD
            and runner["github_sha"] ==
            provenance_runner["github_event_merge_sha"]
            and runner["pid"] == provenance_runner["pid"] == 2926
            and runner["name"] == provenance_runner["name"]
            and runner["cpu_count"] == provenance_runner["cpu_count"] == 4
            and runner["torch"] == provenance_runner["torch"] == "2.14.1+cpu"
            and runner["transformers"] == provenance_runner["transformers"] ==
            "4.57.6" and runner["cpu_model"] == provenance_runner["cpu_model"],
            "actual hosted runner/source identity mismatch")
    require(original["backbone_before_sha256"] == original["backbone_after_sha256"]
            and len(original["backbone_before_sha256"]) == 64,
            "frozen backbone changed")
    require(original["source_oracle"] ==
            {"cases_exact": 8, "ids_and_masks_equal_executed_publisher": True,
             "rich_mask_true": 46, "rich_units": 108},
            "published renderer direct token/mask parity claim changed")

    metrics = original["checks"]
    for key in ("direct_frozen_vs_zero", "request_after_all_generation"):
        check = metrics[key]
        require(check["pass"] is True and check["exact_required"] is True
                and check["atol"] == check["rtol"] == check["max_abs"] == 0,
                "frozen/no-op or cross-request leak: " + key)
    cache = metrics["all_rich_positions_full_vs_incremental"]
    require(cache["pass"] is True and cache["exact_required"] is False
            and cache["atol"] == cache["rtol"] ==
            provenance["original_results"]["cache_tolerance_atol"] ==
            provenance["original_results"]["cache_tolerance_rtol"] == 0.0005
            and cache["max_abs"] ==
            provenance["original_results"]["all_positions_cache_max_abs"] ==
            0.0003590583801269531
            and cache["max_abs"] <= 0.0005,
            "causal cache gate weakened or original numerical result changed")
    require(original["untrained_effect"] ==
            provenance["original_results"]["manual_untrained_effect"] ==
            0.06742143630981445, "untrained readout measurement changed")

    observations = original["observations"]
    require(len(observations) ==
            provenance["original_results"]["raw_generations"] == 21,
            "original 21 raw observations lost")
    mapped = {}
    for case in EXPECTED_CASES:
        group = [o for o in observations if o["case_id"] == case]
        require([o["mode"] for o in group] == list(MODES),
                "missing or duplicate case/mode:" + case)
        ids = [o["generated_ids"] for o in group]
        require(ids[0] == ids[1] == ids[2], "unexpected mode generation delta: " + case)
        for o in group:
            gen = o["generated_ids"]
            require(o["id"] == case + "/" + o["mode"]
                    and o["generated_units"] == len(gen)
                    and 0 < len(gen) <= 64
                    and all(type(token) is int and 0 <= token < 32000 for token in gen)
                    and o["messages"] == next(c for c in plan["cases"] if c["id"] == case)["turns"],
                    "raw case/source tokens altered: " + o["id"])
            end = gen[-1] in (0, 5)
            expected_stop = ("turn_end" if gen[-1] == 5 else "document_end") if end else "budget_exhausted"
            require(o["finish_reason"] == expected_stop
                    and (end or len(gen) == 64)
                    and isinstance(o["raw_text"], str)
                    and isinstance(o["body_excluding_final_stop_only"], str),
                    "missing original stop/negative output: " + o["id"])
            observed = o["observed"]
            require(observed["strict_tool_envelope"] is False
                    and observed["memory_write_at_start"] is False
                    and observed["memory_read_at_start"] is False,
                    "false tool/memory generation success: " + o["id"])
            if case == "tool-result-42":
                require(observed["expected_literal_present"] is False,
                        "negative missing tool-result literal suppressed")
            elif case in {"tool-result-17",
                          "memory-context-amber-17",
                          "memory-context-violet-28"}:
                require(observed["expected_literal_present"] is True,
                        "original literal-present descriptive observation suppressed")
        mapped[case] = group[0]["finish_reason"]
    require({case for case, status in mapped.items() if status == "budget_exhausted"}
            == EXHAUSTED
            and set(provenance["original_results"]["budget_exhausted_cases"]) == EXHAUSTED
            and provenance["original_results"]["strict_tool_envelopes_seen"] == 0
            and provenance["original_results"]["assistant_memory_write_markers_seen"] == 0,
            "raw negative/termination census falsely improved")
    require(all(provenance["original_results"][name] is expected for name, expected in (
        ("expected_literal_42_present", False),
        ("expected_literal_17_present", True),
        ("memory_amber_literal_present", True),
        ("memory_violet_literal_present", True))),
        "literal observations must remain descriptive and source-matched")


def main() -> None:
    files, manifest = load_originals()
    verify_original(files, manifest)
    print(json.dumps({
        "status": "PASS_ORIGINAL_TEXT_ARCHIVE__NO_FINAL_72_ADMISSION",
        "original_run_id": ORIGINAL_RUN,
        "original_zip_sha256": ORIGINAL_ZIP_SHA,
        "original_receipt_sha256": ORIGINAL_RECEIPT_SHA,
        "raw_generations_preserved": 21,
        "generation_budget_exhausted_cases": sorted(EXHAUSTED),
        "tool_memory_behaviour_qualified": False,
        "frozen_model_executed_during_readback": False,
        "train_authorized": False,
        "optimizer_steps": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
