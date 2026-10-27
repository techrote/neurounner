#!/usr/bin/env python3
"""Independent stdlib-only byte/causal reviewer of PR #102 masked-batch receipts.

Two ORIGINAL GitHub Actions JSONs preserved unmodified. No model downloads,
no fixture access, no training, no final #72 acceptance.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/ncr-001/evidence-review/pr102-masked-batch"
ARCHIVE = EVIDENCE / "original-jsons.tar.gz.b64"
PROVENANCE = EVIDENCE / "provenance.json"
ORIGINAL_RUN = 40590487839
ORIGINAL_PR_HEAD = "07eaac38a857a8d67285166944c91b7bc4543e9c"
REVIEWED_102_HEAD = "3592841802d85c442ca2521479b5d8795bee7233"
ORIGINAL_ZIP_SHA = "d0328b54347b952534ef1321953c5c94f6b66c48c791aa72e67bf2cf51d56b42"
GZIP_SHA = "ee01823ebccb19f7d417096db7b9819dc40d0b86ad690d4436dce9765de72460"
TAR_SHA = "a516ebb491b44406858158c2b0d2fe8ae13bc745b21339ec992e2f9e49a959d2"
MODEL_SHA = "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864"
SOURCE_HOOK = "b95c010f9b2ba3cbd9f81317981316bb9c8acd19"
EXPECTED = {
    "pr102-independent-batch-review.json":
        (6721, "1757dc6836b974d74d32acf6c45ccece4d1f69f980030b3bdffc8fad216f403e"),
    "pr102-independent-prefetch.json":
        (874, "0272d25c42331841ce62ec26156b0805adb38f4a52e5258bac5c6272cbad538a"),
}
EXACT_CHECKS = (
    "left_frozen_zero_exact",
    "right_frozen_zero_exact",
    "left_masked_pad_control_no_effect_0",
    "left_masked_pad_control_no_effect_1",
    "right_masked_pad_control_no_effect_0",
    "right_masked_pad_control_no_effect_1",
    "left_other_request_isolation",
    "right_other_request_isolation",
    "fresh_request_A_B_A_zero_exact",
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError("PR102 independent source audit refusal: " + message)


def load_originals(archive_path=ARCHIVE, provenance_path=PROVENANCE):
    """Enforce original ZIP/JSON/tar source and exact unmodified bytes."""
    p = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    record = p["archived_originals"]
    data = Path(archive_path).read_text(encoding="ascii").strip()
    require(len(data) == 3224 and data.isascii(), "ASCII base64 envelope length")
    comp = base64.b64decode(data, validate=True)
    require(len(comp) == record["gzip_bytes"] == 2417
            and sha(comp) == record["gzip_sha256"] == GZIP_SHA,
            "original compressed archive mismatch")
    tarbytes = gzip.decompress(comp)
    require(len(tarbytes) == record["tar_bytes"] == 10240
            and sha(tarbytes) == record["tar_sha256"] == TAR_SHA,
            "original USTAR payload mismatch")
    files = {}
    with tarfile.open(fileobj=io.BytesIO(tarbytes), mode="r:") as t:
        members = t.getmembers()
        require([m.name for m in members] == sorted(EXPECTED),
                "source member list, count or ordering changed")
        for m in members:
            require(m.isfile() and m.mtime == m.uid == m.gid == 0
                    and m.mode == 0o644 and m.name in EXPECTED,
                    "unexpected source member type/metadata")
            data = t.extractfile(m).read()
            expected_bytes, expected_hash = EXPECTED[m.name]
            source = p["original_actions"]["original_originals"][m.name]
            require(len(data) == m.size == source["bytes"] == expected_bytes
                    and sha(data) == source["sha256"] == expected_hash,
                    "original source receipt bytes drifted: " + m.name)
            files[m.name] = json.loads(data)
    return files, p


def verify_originals(files, provenance) -> None:
    """Separate semantic fail-closed audit after raw JSON SHA check."""
    actions = provenance["original_actions"]
    source = provenance["source"]
    runner = provenance["runner"]
    numeric = provenance["numerical"]
    limits = provenance["evidence_limit"]
    j = files["pr102-independent-batch-review.json"]
    prefetch = files["pr102-independent-prefetch.json"]
    require(provenance["schema_version"] == 1
            and provenance["source_issue"] == 72
            and provenance["parent_issue"] == 67
            and provenance["reviewed_pr"] == 102
            and provenance["reviewer_draft_pr"] == 113,
            "PR/research identity changed")
    require(actions["run_id"] == ORIGINAL_RUN
            and actions["run_attempt"] == 1
            and actions["offline_source_conclusion"] == "success"
            and actions["original_model_conclusion"] == "success"
            and actions["offline_source_job_id"] == 122493135745
            and actions["original_model_job_id"] == 122493234132
            and actions["reviewer_pr_head_during_original_execution"] == ORIGINAL_PR_HEAD
            and actions["original_artifact_id"] == 12773497808
            and actions["original_artifact_zip_bytes"] == 2975
            and actions["original_artifact_zip_sha256"] == ORIGINAL_ZIP_SHA
            and actions["original_artifact_zip_archived_in_git"] is False,
            "original Actions/ZIP provenance falsely changed")
    require(j["status"] == provenance["status"] .replace(
                "PASS_SOURCE_BOUND_REVIEWER_MASKED_BATCH_PREPASS__FINAL_72_BLOCKED",
                "PASS_REVIEWER_MASKED_BATCH_SOURCE_BOUND_PREPASS")
            and prefetch["status"] == "PASS_PREFETCH_SOURCES",
            "original source/root runtime conclusion changed")
    require(j["source_reviewed_pr"] == 102
            and j["reviewed_exact_head"] == source["reviewed_candidate_commit"] ==
            REVIEWED_102_HEAD
            and j["model_variant"] == source["model_variant"] == "sprocket_instruct"
            and j["source"]["candidate_wire_blob"] ==
            source["reviewed_candidate_wire_git_blob"] ==
            "4e558ae530a1b79050e22bee587b3caf056e4563",
            "candidate PR #102 immutable code/head identity changed")
    require(j["source"]["original_hook_commit"] ==
            source["original_frozen_instruct_hook_commit"] == SOURCE_HOOK
            and j["source"]["publisher_commit"] ==
            source["original_publisher_commit"] ==
            "a003767e70d766bc14b40c89792543ee8641c8bc"
            and j["source"]["publisher_sft_sha256"] ==
            source["original_publisher_sft_sha256"] ==
            "e5568c15989f3987fec07545640f2f271ffe967c27cbe9d9883c0dd7b5c4ec69",
            "model-hook/publisher source identity changed")
    require(j["source"]["checkpoint_file"] == source["model_weight_file"] ==
            "model.safetensors"
            and j["source"]["checkpoint_sha256"] ==
            source["model_weight_sha256"] == MODEL_SHA
            and j["source"]["weight_bytes"] ==
            source["model_weight_bytes"] == 1002207984
            and j["actual_files"]["model.safetensors"]["sha256"] ==
            prefetch["files"]["model.safetensors"]["sha256"] == MODEL_SHA,
            "complete original root checkpoint digest/size changed")
    for name, key in (("config.json","config_sha256"),):
        require(j["actual_files"][name]["sha256"] ==
                prefetch["files"][name]["sha256"] ==
                "557aacbcb7e23a322221ab91b62c4343dd22130388efc3eb4dcdfe1edfd8d1b7",
                "checkpoint config source mutated")
    require(prefetch["model_variant"] == "sprocket_instruct"
            and prefetch["train_authorized"] is False
            and prefetch["optimizer_steps"] == 0
            and len(prefetch["files"]) == 5,
            "prefetch wrongly substituted or trained")
    require(source["publisher_code_is_training_build_attestation"] is False
            and source["model_repository"] == "kandivault/sprocket-500m"
            and source["model_revision"] ==
            "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
            and source["distinct_chat_checkpoint_sha256"] != MODEL_SHA,
            "publisher checkpoint build claim or Chat/Instruct collision")
    require(j["source"]["protocol_sha256"] == source["protocol_sha256"]
            and j["source"]["pins_sha256"] == source["source_pins_sha256"]
            and sha((ROOT / "experiments/ncr-001/protocol.json").read_bytes()) ==
            source["protocol_sha256"], "protocol/pinned source SHA mismatch")
    protocol = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
    require(protocol["train_authorized"] is False
            and set(protocol["models"]) ==
            {"smollm2","sprocket_chat","sprocket_instruct","void1"}
            and protocol["conformance"]["full_vs_cached_prefill_atol"] ==
            protocol["conformance"]["full_vs_cached_prefill_rtol"] == .0005,
            "fixed original four-model conformance policy altered")
    for label in ("training_performed","heldout_evaluation","train_authorized",
                  "model_generated_tool_semantics_qualified",
                  "final_four_model_72_accepted",
                  "cross_process_cold_reload_in_this_new_review"):
        require(limits[label] is False,
                "false claim of independent final admission/training: " + label)
    require(limits["only_synthetic_host_injected_source_wire"] is True
            and limits["reviewed_pr_102_draft_unchanged"] is True
            and j["train_authorized"] is False
            and j["heldout_evaluation"] is False
            and j["semantic_tool_or_memory_quality_qualified"] is False
            and j["final_four_model_72_pass"] is False
            and j["comparative_training"] is False
            and j["optimizer_steps"] == limits["optimizer_steps"] == 0,
            "fake semantic evidence/training pass in actual original receipt")
    r = j["runner"]
    require(r["github_run_id"] == runner["github_run_id"] == str(ORIGINAL_RUN)
            and r["github_sha"] == runner["github_event_merge_sha"]
            and r["run_attempt"] == "1"
            and r["pid"] == runner["pid"] == 2909
            and r["runner_name"] == runner["runner_name"]
            and r["cpu_count"] == runner["cpu_count"] == 4
            and r["torch"] == runner["torch"] == "2.14.1+cpu"
            and r["python"] == runner["python"] == "3.12.3"
            and r["platform"] == runner["platform"],
            "actual independent source reviewer runner/provenance changed")
    require(j["frozen_base_parameters"] == numeric["root_backbone_parameters"] ==
            501090560 and j["adapter_parameters"] ==
            numeric["untrained_NCA_parameters"] == 658904
            and j["backbone_before_sha256"] == j["backbone_after_sha256"] ==
            numeric["frozen_tensors_before_sha256"] ==
            numeric["frozen_tensors_after_sha256"] ==
            "b73d854127c43266d123bdb266db63a15b4189db9bb3398be035463cbb4fac4d",
            "frozen base mutation or falsely changed source/parameter count")
    wire = j["wire"]
    for label, length, count, expected in (
        ("primary", 133, 45, "c1ade84d732b9b900860483bf0de1da9828c632a955b5e930981dfc0bec8380d"),
        ("secondary",31,8,"893ef9ea95f70bcff2e34b07d8403d9979747670e78e65a3a8757f0d579b3fb0")
    ):
        require(wire[label]["id_count"] == length
                and wire[label]["trained_units"] == count
                and wire[label]["ids_sha256"] == expected
                and wire[label]["not_heldout"] is True,
                "original synthetic reviewer scenario or mask claims changed")
    require(numeric["primary_synthetic_tokenizer_units"] == 133
            and numeric["secondary_synthetic_tokenizer_units"] == 31,
            "reviewer synthetic size changed")
    checks = j["checks"]
    for key in EXACT_CHECKS:
        c = checks[key]
        require(c["pass"] is True and c["exact_required"] is True
                and c["max_abs"] == c["atol"] == c["rtol"] == 0,
                "exact source/mask/isolation invariant violated: " + key)
    for side in ("left","right"):
        for row in range(2):
            c = checks[side + "_batched_vs_single_" + str(row)]
            require(c["pass"] is True and c["exact_required"] is False
                    and c["max_abs"] == 0
                    and c["atol"] == c["rtol"] == .0005,
                    "batch vs standalone drift: " + side + str(row))
        active = checks[side + "_active_untrained_effect"]
        require(active["trained"] is False and active["pass"] is True
                and active["max_abs"] ==
                numeric["untrained_manual_readout_effect"] ==
                0.10165786743164062,
                "nontrained adapter readout effect altered")
    cache = checks["left_masked_batch_full_vs_4_cached"]
    require(cache["pass"] is True and cache["exact_required"] is False
            and cache["atol"] == numeric["fixed_cache_atol"] ==
            cache["rtol"] == numeric["fixed_cache_rtol"] == 0.0005
            and cache["max_abs"] ==
            numeric["left_padded_batch_cached_four_max_abs"] ==
            3.218650817871094e-05,
            "masked batch real one-token cache evidence modified or gate weakened")
    negative = checks["unmasked_pad_injection_detected"]
    require(negative["expected_to_change"] is True
            and negative["pass"] is True and
            negative["max_abs"] ==
            numeric["positive_bad_unmasked_pad_sentinel_max_abs"] ==
            14.123714447021484
            and negative["max_abs"] > 0,
            "negative sentinel cannot detect wrongly unmasked padding")
    require(provenance["status"] ==
            "PASS_SOURCE_BOUND_REVIEWER_MASKED_BATCH_PREPASS__FINAL_72_BLOCKED",
            "false final source qualification promotion")


def main():
    files, source = load_originals()
    verify_originals(files, source)
    print(json.dumps({
        "status":"PASS_INDEPENDENT_ORIGINAL_REVIEWER_BYTES__FINAL_72_BLOCKED",
        "original_model_run":ORIGINAL_RUN,
        "original_model_json_sha256":EXPECTED["pr102-independent-batch-review.json"][1],
        "original_actions_zip_sha256":ORIGINAL_ZIP_SHA,
        "real_frozen_model_reexecuted_during_archive_audit":False,
        "independent_final_four_model_admission":False,
        "optimizer_steps":0,
        "train_authorized":False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
