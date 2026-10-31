#!/usr/bin/env python3
"""Immutable original right-padded KV/HF.generate receipt audit for PR #113.

Standard library only. Replays byte custody and numerical/semantic guards,
NOT the original frozen model, sealed evaluations, gradients, or training.
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
DIR = ROOT / "experiments/ncr-001/evidence-review/pr113-rightpad-generation"
ARCHIVE = DIR / "original-text-evidence.tar.xz.b64"
PROVENANCE = DIR / "provenance.json"
# Pin the Git-stored representation, not just parsed data or decoded source bytes.
ORIGINAL_ARCHIVE_GIT_BLOB = "3ca13858e4a878de259406a01f1a0baa1c97282d"
ORIGINAL_PROVENANCE_GIT_BLOB = "e944ca2655729e60c587865a9168b34698031114"

ORIGINAL_RUN = 40598034739
ORIGINAL_HEAD = "d727640e36ec9f7915aa208d69d9b78e4448f0c6"
ORIGINAL_MERGE = "4470f3070abdee1a6b31d82b27663a8bfe0fdd41"
ORIGINAL_ZIP_SHA = "30bc7fa3231047561b5ab0142b59b7012e7e1ee4e72329198f7d778b65be9a1e"
ARCHIVE_XZ_SHA = "3d2081f1494af69eb7cf8a9e90eea326ffc60ff1845483a4d02be34951728751"
ARCHIVE_TAR_SHA = "40098d8a7453fde67ff7e7c7dd5bd7578a97a2e4eb4581e0e0fa2f4dc71e1eef"
RECEIPT_SHA = "c50bc096161a2b0cfe40c19278adada101cb7314bff822064117ae347f232224"
ROOT_SHA = "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864"
CHAT_SHA = "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97"
REVIEWED_102 = "3592841802d85c442ca2521479b5d8795bee7233"
SOURCE_HOOK = "b95c010f9b2ba3cbd9f81317981316bb9c8acd19"
PUBLISHER = "a003767e70d766bc14b40c89792543ee8641c8bc"
SOURCE_SCRIPT_SHA = "42f08dd8c2f96d5e60edad97d2972ce47bafb96b7e22e2c78b8a84b9a4d04b7e"
HELPER_BLOB = "6ade653f0452e58f82c8c78e79d326018c30bcf1"
SOURCE_HF = "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
GEN_IDS = [8184, 746, 1066, 1525]

EXPECTED = {
    "pr113-rightpad-generate.json": (13952, RECEIPT_SHA),
    "pr113-rightpad-packages.txt": (
        1714, "38376d0be1f9ec1d5b922c7c5fa493dd969f01ddc2453cff6cc94ffbe4f58f9f"),
    "pr113-rightpad-prefetch.json": (
        1056, "e121751fb3299154830b452e9d01e1ced59ccb1928d7b8230441663d0ff41918"),
    "pr113-rightpad-source-commit.txt": (
        41, "30196dc5793cd6f3e727168110e4fd5a2c82950dc1e74bcef972416eee8ada3a"),
    "pr113-rightpad-source-hashes.sha256": (
        445, "f9902886d0487f1bb9c79b932b5b2b198092f2f0f9e743a440f59dbfefd481bc"),
}

PIN_FILES = {
    "model.safetensors": (
        "weight_declared_lfs_sha256_not_downloaded", 1002207984),
    "config.json": ("config_sha256", 689),
    "tokenizer.json": ("tokenizer_sha256", 2269469),
    "tokenizer_config.json": ("tokenizer_config_sha256", 3325),
    "special_tokens_map.json": ("special_tokens_map_sha256", 333),
}

EXACT_CHECKS = (
    "frozen_original_zero_control_context",
    "frozen_original_zero_unrelated_request",
    "rightpad_masked_tool_result_no_effect",
    "generation_followed_by_fresh_A_B_A_exact",
)


def require(ok, why):
    if not ok:
        raise RuntimeError("NCR original rightgap archive REFUSED: " + why)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def gitblob(raw):
    """Compute the Git blob object ID from exact committed bytes."""
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\x00" + raw).hexdigest()


def load_originals(archive_path=ARCHIVE, provenance_path=PROVENANCE):
    provenance_raw = Path(provenance_path).read_bytes()
    require(gitblob(provenance_raw) == ORIGINAL_PROVENANCE_GIT_BLOB,
            "committed provenance JSON Git blob changed")
    p = json.loads(provenance_raw)
    a = p["archived_text"]
    # Whitespace-normalized Base64 and reserialized JSON are NOT an acceptable
    # substitute for exact Git custody of the historical representations.
    archive_raw = Path(archive_path).read_bytes()
    require(gitblob(archive_raw) == ORIGINAL_ARCHIVE_GIT_BLOB,
            "committed encoded original-text archive Git blob changed")
    encoded = "".join(archive_raw.decode("ascii").split())
    require(len(encoded) == 5848 and encoded.isascii(), "base64 envelope altered")
    compressed = base64.b64decode(encoded, validate=True)
    require(len(compressed) == a["original_xz_bytes"] == 4384
            and digest(compressed) == a["original_xz_sha256"] == ARCHIVE_XZ_SHA,
            "original xz archive bytes changed")
    raw = lzma.decompress(compressed, format=lzma.FORMAT_XZ)
    require(len(raw) == a["original_tar_bytes"] == 30720
            and digest(raw) == a["original_tar_sha256"] == ARCHIVE_TAR_SHA,
            "original tar source bytes changed")
    out = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
        members = tar.getmembers()
        require([m.name for m in members] == sorted(EXPECTED),
                "original ZIP members missing, duplicated or altered")
        for member in members:
            require(member.isfile() and member.name in EXPECTED
                    and member.mode == 0o644
                    and member.uid == member.gid == member.mtime == 0,
                    "noncanonical archived source member metadata")
            content = tar.extractfile(member).read()
            size, sha = EXPECTED[member.name]
            provenance_member = a["expected_files"][member.name]
            require(member.size == len(content) == size ==
                    provenance_member["bytes"]
                    and digest(content) == sha == provenance_member["sha256"],
                    "original member changed: " + member.name)
            out[member.name] = content
    require(set(a["expected_files"]) == set(EXPECTED)
            and a["original_files_lossless"] is True,
            "source archive inventory/qualification misrepresented")
    return out, p


def verify_originals(members, provenance):
    p = provenance
    actions, source, runner = p["original_actions"], p["original_model"], p["original_runner"]
    results, scope = p["results"], p["scope"]
    main = json.loads(members["pr113-rightpad-generate.json"])
    prefetch = json.loads(members["pr113-rightpad-prefetch.json"])

    require(p["schema_version"] == 1 and p["reviewer_pr"] == 113
            and p["reviewed_pr"] == 102 and p["source_issue"] == 72
            and p["parent_issue"] == 67,
            "wrong issue, PR or experiment identity")
    require(p["status"] ==
            "PASS_RIGHTPAD_GAPPED_KV_AND_ORIGINAL_HF_GENERATE_ONLY__FINAL_72_BLOCKED"
            and main["status"] ==
            "PASS_RIGHTPAD_GAPPED_KV_AND_ORIGINAL_HF_GENERATE_ONLY"
            and prefetch["status"] == "PASS_EXACT_FULL_SOURCE_PREFETCH",
            "original model/source conclusion changed")
    require(actions["run_id"] == ORIGINAL_RUN and actions["run_attempt"] == 1
            and actions["model_result"] == actions["source_result"] == "success"
            and actions["original_pr_branch_head"] == ORIGINAL_HEAD
            and actions["github_checkout_merge_sha"] == ORIGINAL_MERGE
            and actions["offline_source_job_id"] == 122518832102
            and actions["original_frozen_host_job_id"] == 122518920068
            and actions["original_artifact_id"] == 12777685787
            and actions["original_zip_sha256"] == ORIGINAL_ZIP_SHA
            and actions["original_zip_bytes"] == 5754
            and actions["original_zip_bytes_committed"] is False,
            "original ZIP, workflow or published result misattributed")
    errors = p["pre_model_failures"]
    require(len(errors) == 2 and [e["run_id"] for e in errors] ==
            [40597707675, 40597891619] and all(
                e["result"] == "failure"
                and e["stage"] == "offline_source_pin"
                and e["model_executed"] is False for e in errors),
            "initial source-pin failure history removed/reclassified")
    require(members["pr113-rightpad-source-commit.txt"] ==
            (ORIGINAL_MERGE + "\n").encode("ascii"),
            "original Actions source checkout SHA tampered")
    report_hashes = {}
    for line in members["pr113-rightpad-source-hashes.sha256"].decode("ascii").splitlines():
        require(len(line) > 66 and line[64:66] == "  ",
                "original source-hashes format")
        sha, name = line[:64], line[66:]
        require(len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)
                and name not in report_hashes,
                "invalid or duplicate original source SHA inventory")
        report_hashes[name] = sha
    require(report_hashes == {
        "scripts/ncr_001e_pr102_rightpad_generate.py": SOURCE_SCRIPT_SHA,
        "scripts/ncr_001e_pr102_masked_batch_review.py":
            "d57f82eb5bcffcfeb8d0ff4d7fe5356fd509732f7a0f277b6c12bda4b7624762",
        "candidate/experiments/ncr-001/protocol.json":
            "a76d38f80a57012ca8e62fba183d1cf2e91fd0986032f1adcdfb0aef843bf0b4",
        "candidate/experiments/ncr-001/source-pins.json":
            "db794f53a49bb1bcbf8b23bd4a76d24430eb495da246639b506abdbbf11734b1",
    }, "original checkout source hash inventory replaced")
    require(digest((ROOT / "scripts/ncr_001e_pr102_rightpad_generate.py").read_bytes())
            == SOURCE_SCRIPT_SHA, "post-original reviewer runner code drift")
    protocol_path = ROOT / "experiments/ncr-001/protocol.json"
    pins_path = ROOT / "experiments/ncr-001/source-pins.json"
    require(digest(protocol_path.read_bytes()) == source["protocol_sha256"] ==
            main["source"]["protocol_sha256"] ==
            report_hashes["candidate/experiments/ncr-001/protocol.json"],
            "immutable protocol bytes differ from original model")
    require(digest(pins_path.read_bytes()) == source["source_pins_sha256"] ==
            main["source"]["pins_sha256"] ==
            report_hashes["candidate/experiments/ncr-001/source-pins.json"],
            "immutable original pinned source inventory drift")
    protocol = json.loads(protocol_path.read_text())
    pins = json.loads(pins_path.read_text())
    require(protocol["train_authorized"] is False
            and set(protocol["models"]) == set(pins["models"]) ==
            {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"}
            and protocol["conformance"]["full_vs_cached_prefill_atol"] ==
            protocol["conformance"]["full_vs_cached_prefill_rtol"] ==
            results["fixed_source_atol"] == results["fixed_source_rtol"] == 0.0005,
            "four-way protocol or cache numerical policy weakened")
    pin = pins["models"]["sprocket_instruct"]
    other = pins["models"]["sprocket_chat"]
    require(source["variant"] == main["variant"] == prefetch["variant"] ==
            "sprocket_instruct"
            and pin["hf_weight_file"] == source["model_weight_file"] ==
            main["source"]["checkpoint_file"] == "model.safetensors"
            and source["repo_id"] == main["source"]["model_repo"] ==
            "kandivault/sprocket-500m"
            and source["revision"] == main["source"]["source_revision"] ==
            pin["revision"] == SOURCE_HF
            and pin["weight_declared_lfs_sha256_not_downloaded"] ==
            source["weight_sha256"] == main["source"]["checkpoint_sha256"] == ROOT_SHA
            and other["hf_weight_file"] == "chat/model.safetensors"
            and other["weight_declared_lfs_sha256_not_downloaded"] ==
            source["chat_sibling_weight_sha256"] == CHAT_SHA,
            "Sprocket Instruct/Chat identities were substituted")
    require(source["hook_source_commit"] == main["source"]["original_hook_commit"] ==
            SOURCE_HOOK and source["reviewed_candidate_pr102_commit"] ==
            main["source"]["candidate_commit"] == main["reviewed_head"] ==
            REVIEWED_102 and source["reviewed_candidate_serializer_blob"] ==
            "4e558ae530a1b79050e22bee587b3caf056e4563",
            "original candidate/immutable hook SHA changed")
    require(source["publisher_source_commit"] ==
            main["source"]["publisher_commit"] == PUBLISHER
            and source["publisher_sft_sha256"] ==
            main["source"]["publisher_sft_sha256"] ==
            "e5568c15989f3987fec07545640f2f271ffe967c27cbe9d9883c0dd7b5c4ec69"
            and source["publisher_source_is_training_build_attestation"] is False
            and main["source"]["original_publisher_is_training_build_attestation"] is False
            and source["reviewer_helper_blob"] ==
            main["source"]["reviewer_dependency_blob"] == HELPER_BLOB,
            "false public publisher attestation or reviewer dependency substitution")
    require(set(prefetch["files"]) == set(main["original_source_files"]) ==
            set(PIN_FILES), "original complete frozen source files not accounted for")
    for name,(pin_key,expected_bytes) in PIN_FILES.items():
        f = prefetch["files"][name]
        g = main["original_source_files"][name]
        require(f["sha256"] == g["sha256"] == pin[pin_key]
                and f["bytes"] == g["bytes"] == expected_bytes,
                "original model weight, tokenizer or config bytes changed: " + name)
    require(prefetch["train_authorized"] is False and
            prefetch["optimizer_steps"] == 0 and
            main["schema"] == 1 and main["reviewed_pr"] == 102 and
            main["reviewer_original_pr"] == 113 and main["train_authorized"] is False
            and main["optimizer_steps"] == 0 and main["any_model_training"] is False
            and main["heldout_evaluation"] is False
            and main["semantic_tool_memory_qualified"] is False
            and main["final_four_model_72_conformance"] is False,
            "any original source/full-model training or semantic claim misrepresented")
    for field in ("train_authorized", "actual_training", "heldout_evaluation",
                  "model_tool_memory_semantic_pass",
                  "final_independent_four_model_72_admission",
                  "comparative_NCA_MLP_experiment",
                  "sealed_evaluation_fixture_changed", "protocol_changed",
                  "reviewed_PR102_modified"):
        require(scope[field] is False, "false training/heldout/quality admission "+field)
    require(scope["optimizer_steps"] == 0 and
            scope["reviewer_PR113_draft_unmerged"] is True,
            "draft boundary or zero-step no training was removed")
    actual_runner = main["runner"]
    require(actual_runner["github_run_id"] == runner["github_run_id"] ==
            str(ORIGINAL_RUN) and
            actual_runner["github_sha"] == runner["github_merge_sha"] == ORIGINAL_MERGE
            and actual_runner["attempt"] == runner["run_attempt"] == "1"
            and actual_runner["runner_name"] == runner["runner_name"] ==
            "GitHub Actions 1000045611"
            and actual_runner["pid"] == runner["pid"] == 3423
            and actual_runner["cpu_count"] == runner["cpu_count"] == 4
            and actual_runner["torch"] == runner["torch"] == "2.14.1+cpu"
            and actual_runner["transformers"] == runner["transformers"] == "4.57.6"
            and actual_runner["python"] == runner["python"] == "3.12.3",
            "original source/runner/PID or libraries not matched")
    wire = main["wire"]
    require(wire["control_context"]["source_units"] == results["input_lengths"][0] ==
            133 and wire["unrelated_request"]["source_units"] ==
            results["input_lengths"][1] == 31
            and wire["control_context"]["assistant_mask_units"] ==
            results["source_assistant_supervised_units"][0] == 45
            and wire["unrelated_request"]["assistant_mask_units"] ==
            results["source_assistant_supervised_units"][1] == 8
            and all(t["non_heldout"] is True for t in wire.values()),
            "source-structured synthetic prompt/mask record altered")
    gap = main["right_padding"]
    require(gap["cached_prefix_width"] == results["right_gap_prefill_width"] == 129
            and gap["true_prefix_units"] ==
            results["right_gap_true_input_tokens"] == [129,27]
            and gap["source_padding_id"] == results["source_pad_token_id"] == 1
            and gap["tail_units"] == results["generated_response_length"] == 4
            and gap["masked_cache_gap"] is True,
            "non-standard masked right-padding source generation claimed")
    checks = main["checks"]
    for name in EXACT_CHECKS:
        c = checks[name]
        require(c["pass"] is True and c["exact_required"] is True
                and c["max_abs"] == c["atol"] == c["rtol"] == 0,
                "original frozen/strict masking/request exact gate changed: "+name)
    for name,expected in (
        ("rightpad_KV_gap_vs_unpadded_control_context",
         results["rightpad_control_full_vs_cache_max_abs"]),
        ("rightpad_KV_gap_vs_unpadded_unrelated_request",
         results["rightpad_unrelated_full_vs_cache_max_abs"]),
    ):
        c=checks[name]
        require(c["pass"] is True and c["exact_required"] is False
                and c["atol"] == c["rtol"] == 0.0005
                and 0<=c["max_abs"] == expected <= 0.0005,
                "rightpad real frozen KV gap failed exact original source metric: "+name)
    require(results["rightpad_control_full_vs_cache_max_abs"] ==
            4.920363426208496e-05 and
            results["rightpad_unrelated_full_vs_cache_max_abs"] ==
            2.6702880859375e-05,
            "manually altered original right-pad numerical results")
    m=checks["manual_nontrained_effect"]
    require(m["pass"] is True and m["trained"] is False
            and m["max_abs"] == results["nontrained_readout_effect"] ==
            0.03652691841125488,
            "untrained NCA readout proof changed")
    bad=checks["unmasked_rightpad_negative_detected"]
    require(bad["expected_difference"] is True and bad["pass"] is True
            and bad["max_abs"] == results["negative_unmasked_gap_max_abs"] ==
            11.143782615661621,
            "intentionally invalid right-gap attention mask no longer detected")
    require(main["backbone_sha256_before"] == main["backbone_sha256_after"] ==
            results["frozen_base_tensor_sha256_before"] ==
            results["frozen_base_tensor_sha256_after"] ==
            "b73d854127c43266d123bdb266db63a15b4189db9bb3398be035463cbb4fac4d",
            "frozen parameter bytes not preserved")
    observed = main["generations"]
    require(len(observed) == 6 and results["mode_count"] == 3
            and results["cache_configurations"] == [False,True]
            and results["generated_response_length"] == 4
            and results["four_token_generated_ids"] == GEN_IDS,
            "original HF generation modes or 4 token budget falsified")
    require([(g["mode"],g["use_cache"]) for g in observed] == [
        (mode,use_cache) for mode in ("disabled","zero","active_untrained")
        for use_cache in (False,True)],
        "source generation on/off adapter/cache coverage missing or reordered")
    for gen in observed:
        require(gen["source_prompt_unit_count"] == 133 and
                gen["original_generated_ids"] == GEN_IDS and
                gen["not_behavioral_tool_or_memory_qualification"] is True and
                gen["optimizer_steps"] == 0 and
                len(gen["scores_sha256"]) == 4 and
                all(isinstance(h,str) and len(h)==64 for h in gen["scores_sha256"]),
                "raw generated IDs, score hashes or no-training limits drifted")
    maxima=[]
    generated_scores = {
        (g["mode"], g["use_cache"]): g["scores_sha256"] for g in observed
    }
    for use_cache in (False, True):
        require(generated_scores[("disabled", use_cache)] ==
                generated_scores[("zero", use_cache)],
                "original frozen and zero-adapter HF.generate score bytes disagree")
    for mode in ("disabled", "zero", "active_untrained"):
        require(generated_scores[(mode, False)][0] ==
                generated_scores[(mode, True)][0],
                "source-generated first-step cached and uncached scores diverged")
    for cache in ("False","True"):
        for step in range(4):
            name=f"generate_zero_original_exact_cache_{cache}_step_{step}"
            c=checks[name]
            require(c["pass"] is True and c["exact_required"] is True
                    and c["max_abs_finite"] == c["atol"] == c["rtol"] == 0
                    and c["finite_logits"] == 31999 and c["nonfinite_logits"] == 1,
                    "frozen source baseline-vs-zero HF generation parity failed")
    for mode in ("disabled","zero","active_untrained"):
        for step in range(4):
            name=f"generate_full_vs_cache_{mode}_step_{step}"
            c=checks[name]
            require(c["pass"] is True and c["exact_required"] is False
                    and c["atol"] == c["rtol"] == .0005
                    and c["finite_logits"] == 31999 and c["nonfinite_logits"] == 1
                    and 0<=c["max_abs_finite"]<=.0005,
                    "HF original cached vs uncached generate score gate: "+name)
            maxima.append(c["max_abs_finite"])
    require(max(maxima) == results["max_HF_generate_full_vs_cache_abs"] ==
            3.528594970703125e-05,
            "generated-source numeric evidence or EOS mask altered")
    require(results["expected_processed_eos_masked_logits_per_step"] == 1
            and results["rightpad_masked_tool_result_abs"] == 0
            and results["source_frozen_zero_exact"] is True
            and results["request_reset_exact"] is True
            and results["model_generated_tool_call_competence_qualified"] is False
            and results["general_memory_behavior_qualified"] is False,
            "source generation semantics/false competence promoted")


def main():
    members, manifest = load_originals()
    verify_originals(members, manifest)
    print(json.dumps({
        "status":"PASS_ORIGINAL_RIGHTPAD_AND_HF_GENERATE_BYTE_AUDIT__FINAL_72_BLOCKED",
        "original_frozen_model_run_id":ORIGINAL_RUN,
        "original_actions_zip_sha256":ORIGINAL_ZIP_SHA,
        "original_original_model_json_sha256":RECEIPT_SHA,
        "model_reexecuted_during_audit":False,
        "training_authorized":False,
        "optimizer_steps":0,
        "original_hf_generation_modes":6,
        "cached_tool_memory_competence_qualified":False,
    },sort_keys=True))


if __name__ == "__main__":
    main()
