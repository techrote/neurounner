#!/usr/bin/env python3
"""Four-cohort void.1 #72 evidence reconciliation; NO MODEL, NO TRAIN.

This is an independently checkable *source/receipt scientific interpretation*,
NOT a new physical execution, an aggregate per-model PASS or gate mutation.
Every original physical cohort retains its own source head, run and scope.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

import ncr_001_void_postdiagnostic_reconciliation as diagnostic
import ncr_001_void_nca_original_custody as old_custody
import ncr_001_void_poststep_original_custody as new_custody

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"
PINS = ROOT / "experiments/ncr-001/source-pins.json"
HISTORICAL = ROOT / "experiments/ncr-001/adjudication/void1-postlease-pretrain.json"
ADDENDUM = ROOT / "experiments/ncr-001/adjudication/void1-postdiagnostic-review-pending.json"
OUTPUT = ROOT / "experiments/ncr-001/adjudication/void1-fourcohort-scientific-boundary-20261030.json"

PARENT = "0c1a47c4d562a24c7d5a81305affb4a2501fd80e"
PROTOCOL_BLOB = "14ce5fbaa38f8806c8c8738f19c01b91385eebaf"
PINS_BLOB = "59fff1aed872b5563aa6898171983973e0cdb02c"
HISTORICAL_BLOB = "81c0c7239e64fab8d6d077978a1591f3458c0367"
ADDENDUM_BLOB = "9fdccd13aa4ec679b943e3c30b565d5cdc10061c"
OLD_MANIFEST_BLOB = "33dc325239f4518a21679dd746ce2d4b5030e489"
NEW_MANIFEST_BLOB = "3b56c4b3e05da323e65cc3a187e5d2d320a7a007"
MODEL = "dotlabs/void.1"
REV = "9c2be5087772867d45961abd11f6b35110617563"
WEIGHT = "ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3"
TOKENIZER = "279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03"
FROZEN = "7673957ba36e8eb5427e5ecc94dd52de71fbf9e6718bb0c2a94e4d7361b3f904"
EOS = 258
TOL = 5e-4
RUNS = {
    "A_pretraining_short_and_cold": (40747035876, "9aed5ca0574a6885c5d60c07d16561e0f73d5043"),
    "B_pretraining_maxcontext": (40755894558, "1438f45efc0dab8ed669a814b4897fc7a839302f"),
    "C_one_step_gradient": (40765025332, "2810e800ef890f0c99f9ea44adca29f879b5d0ea"),
    "D_poststep_cold": (41058648266, "f78d302ab6062c3739315c3e04317a9a97b97191"),
}
NINE = (
    "same_base_logits", "base_weight_unchanged_after_step",
    "causal_suffix_invariance", "full_prefix_vs_incremental_cache",
    "per_token_isolation", "per_request_reset",
    "nonzero_adapter_gradient_after_warmup", "finite_state_and_loss",
    "cold_reload_state_dict",
)
OBSERVED_IN = {
    "same_base_logits": ("A_pretraining_short_and_cold", "B_pretraining_maxcontext"),
    "base_weight_unchanged_after_step": ("C_one_step_gradient", "D_poststep_cold"),
    "causal_suffix_invariance": ("A_pretraining_short_and_cold", "B_pretraining_maxcontext"),
    "full_prefix_vs_incremental_cache": ("A_pretraining_short_and_cold", "B_pretraining_maxcontext", "D_poststep_cold"),
    "per_token_isolation": ("A_pretraining_short_and_cold", "B_pretraining_maxcontext"),
    "per_request_reset": ("A_pretraining_short_and_cold", "D_poststep_cold"),
    "nonzero_adapter_gradient_after_warmup": ("C_one_step_gradient", "D_poststep_cold"),
    "finite_state_and_loss": ("A_pretraining_short_and_cold", "C_one_step_gradient", "D_poststep_cold"),
    "cold_reload_state_dict": ("A_pretraining_short_and_cold", "D_poststep_cold"),
}
OBSERVATION_CLASS = {
    "same_base_logits": "PHYSICAL_PRESTEP_NOOP_SUPPORTED",
    "base_weight_unchanged_after_step": "SINGLE_PUBLIC_CASE_POSTSTEP_CANDIDATE",
    "causal_suffix_invariance": "PHYSICAL_PRESTEP_UP_TO_2048_SUPPORTED",
    "full_prefix_vs_incremental_cache": "PRESTEP_2048_PLUS_DISTINCT_POSTSTEP_FOUR_TAIL_POSITIONS",
    "per_token_isolation": "PHYSICAL_PRESTEP_AND_SOURCE_RUNTIME_ONLY",
    "per_request_reset": "PRESTEP_AND_POSTSTEP_A_B_A_OBSERVED",
    "nonzero_adapter_gradient_after_warmup": "SINGLE_PUBLIC_CASE_GRADIENT_CANDIDATE",
    "finite_state_and_loss": "PRESTEP_FINITE_AND_SINGLE_CASE_POSTSTEP_FINITE",
    "cold_reload_state_dict": "PRESTEP_NCA_MLP_PLUS_POSTSTEP_NCA_SECOND_INTERPRETER",
}
UNPROVEN = (
    "no_single_original_run_jointly_proves_poststep_2048_context_and_learning",
    "no_poststep_all_31_positions_incremental_kv_or_all_2048_positions",
    "no_poststep_2048_strict_future_suffix_or_masked_batched_state_isolation",
    "no_poststep_native_generate_cache_on_off_or_2048_generation",
    "no_poststep_independent_runner_or_host_replication_only_second_python_same_runner",
    "no_multi_sample_or_multi_seed_learned_quality_result",
    "no_final_independent_model_specific_scientific_72_gate_acceptance",
)
STATUS = "SOURCE_BOUND_FOUR_DISTINCT_VOID1_COHORTS__SCIENTIFIC_72_UNADMITTED"


def refuse(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("VOID_FOURCOHORT_REFUSED: " + reason)


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def require_blob(path: Path, sha: str) -> None:
    refuse(path.is_file() and not path.is_symlink() and git_blob(path.read_bytes()) == sha,
           "immutable original source or report bytes changed: " + str(path))
    rel = path.relative_to(ROOT).as_posix()
    original = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "HEAD", "--", rel],
                              capture_output=True, text=True, check=False)
    refuse(original.returncode == 0
           and original.stdout == f"100644 blob {sha}\t{rel}\n",
           "immutable committed Git tree type/mode/blob changed: " + rel)


def source_inputs() -> dict:
    """Execute all four *existing* original source/byte auditors first."""
    for path, blob in (
        (PROTOCOL, PROTOCOL_BLOB), (PINS, PINS_BLOB),
        (HISTORICAL, HISTORICAL_BLOB), (ADDENDUM, ADDENDUM_BLOB),
        (old_custody.MANIFEST, OLD_MANIFEST_BLOB),
        (new_custody.EVIDENCE / "manifest.json", NEW_MANIFEST_BLOB),
    ):
        require_blob(path, blob)
    historical = diagnostic.prior.build_report()  # Runs A/B original ZIP audits.
    addendum = diagnostic.build_report()          # Reruns A/B, full C ZIP/log audit.
    current = new_custody.audit(live=False)       # D's committed tree, ZIP/log bytes.
    old = old_custody.audit()                     # Original C host receipt, not D.
    archived, raw = new_custody.read_and_check(verify_git=True)
    refuse(archived["status"] == current["status"]
           and archived["original_run"] == current["original_run"],
           "D duplicated original source audit does not match")
    with zipfile.ZipFile(io.BytesIO(raw["host-original.zip"])) as z:
        producer = json.loads(z.read("ncr001-void-poststep-cold/producer.json"))
        consumer = json.loads(z.read("ncr001-void-poststep-cold/consumer.json"))
        intermediate = json.loads(z.read("ncr001-void-nca-learning/receipt.json"))
    return {
        "protocol": json.loads(PROTOCOL.read_bytes()),
        "pins": json.loads(PINS.read_bytes()),
        "historical": historical,
        "addendum": addendum,
        "old_receipt": old,
        "new_receipt": current,
        "new_manifest": json.loads((new_custody.EVIDENCE / "manifest.json").read_bytes()),
        "producer": producer,
        "consumer": consumer,
        "intermediate": intermediate,
    }


def interpret(s: dict) -> dict:
    """Fail closed even on intentionally recomputed or internally consistent fakes."""
    p = s["protocol"]
    pins = s["pins"]
    h = s["historical"]
    a = s["addendum"]
    c = s["old_receipt"]
    d = s["new_receipt"]
    m = s["new_manifest"]
    prod = s["producer"]
    cons = s["consumer"]
    intermediate = s["intermediate"]

    refuse(p["experiment"] == "NCR-001" and p["train_authorized"] is False
           and set(p["models"]) == set(pins["models"]) ==
           {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"}
           and tuple(p["conformance"]["required_checks"]) == NINE
           and p["conformance"]["any_failure_blocks_model_qualification"] is True
           and p["conformance"]["full_vs_cached_prefill_atol"] == TOL
           and p["conformance"]["full_vs_cached_prefill_rtol"] == TOL,
           "locked four-model preregistration or causal tolerance changed")
    v = p["models"]["void1"]
    vpin = pins["models"]["void1"]
    refuse(v["repo_id"] == vpin["repo_id"] == MODEL
           and v["revision"] == vpin["revision"] == REV
           and v["weight_file"] == vpin["hf_weight_file"] == "model.safetensors"
           and v["taps_after_block_zero_based"] == [7,15,23]
           and v["expected_config"]["hidden_size"] == 640
           and vpin["weight_declared_lfs_sha256_not_downloaded"] == WEIGHT
           and vpin["tokenization_byte_py_sha256"] == TOKENIZER
           and v["tokenizer_reference"]["ids"]["eos"] == EOS,
           "Qwen3/frozen model/tokenizer source identity changed")

    refuse(h["adjudication"] ==
           "BLOCKED_REQUIRED_LEARNING_CHECKS__POSTLEASE_SOURCE_RUNTIME_PRETRAIN_PASS"
           and h["per_model_gate"] is False
           and h["independent_scientific_review"] is False
           and h["train_authorized"] is False
           and h["optimizer_steps_observed"] == 0
           and h["source"]["weight_sha256"] == WEIGHT
           and h["source"]["tokenizer_source_sha256"] == TOKENIZER
           and h["source"]["eos_id"] == EOS
           and h["evidence"]["corrected_eos_postlease_run"] == RUNS["A_pretraining_short_and_cold"][0]
           and h["evidence"]["postlease_max_context_run"] == RUNS["B_pretraining_maxcontext"][0]
           and set(h["matrix"]) == set(NINE)
           and h["blocking_required_checks"] == [
               "base_weight_unchanged_after_step",
               "nonzero_adapter_gradient_after_warmup", "finite_state_and_loss"],
           "historical A/B blocked decision misrepresented")
    refuse(h["evidence"]["frozen_backbone_sha256"] == FROZEN
           and h["evidence"]["actual_eos_258_emitted_in_four_step_probe"] is False
           and h["evidence"]["max_context_generation_records"] == 12
           and max(h["evidence"]["max_context_cache_max_abs"].values()) <= TOL,
           "historical 2048/no-EOS bounds changed")
    refuse(a["historical_matrix_unchanged"] is True
           and a["historical_adjudication"] == h["adjudication"]
           and a["source_heads"] == {
               "original_physical_head": RUNS["C_one_step_gradient"][1],
               "original_custody_head": "d6b730083bdf2d7d213e647f9770460bc1424e42",
               "historical_adjudication_head": "1738ac2b6bc6105d57d6a5b4c35f4beaad94ff86"}
           and a["original_physical_run"] == RUNS["C_one_step_gradient"][0]
           and a["status"] == "PRELIMINARY_DIAGNOSTIC_EVIDENCE__SCIENTIFIC_ADMISSION_BLOCKED"
           and a["per_model_gate"] is False
           and a["independent_scientific_review"] is False
           and a["train_authorized"] is False
           and a["candidate_required_check_evidence"] == {
               k: diagnostic.OBSERVED
               for k in ("base_weight_unchanged_after_step",
                         "nonzero_adapter_gradient_after_warmup", "finite_state_and_loss")},
           "C original one-step diagnostic falsely promoted or conflated")
    cobs = a["diagnostic_observation"]
    refuse(c["status"] ==
           "PASS_VOID1_ONE_STEP_NCA_DIAGNOSTIC__72_SCIENTIFIC_ADMISSION_BLOCKED"
           and c["runner"]["expected_head"] == RUNS["C_one_step_gradient"][1]
           and c["diagnostic_optimizer_steps"] == cobs["optimizer_steps"] == 1
           and cobs["post_warmup_backward"] is True
           and cobs["three_distinct_ordered_depth_fields"] is True
           and cobs["post_warmup_all_required_gradients_nonzero"] is True
           and c["backbone_before_sha256"] == c["backbone_after_sha256"] ==
               cobs["frozen_backbone_sha256"] == FROZEN
           and c["synthetic_sample"]["total_units"] == 31
           and c["synthetic_sample"]["supervised_after_shift"] == 21
           and c["synthetic_sample"]["real_eos_id"] == EOS
           and c["warmup"]["native_causal_eos_loss"] == cobs["first_loss"]
           and c["post_warmup"]["native_causal_eos_loss"] == cobs["post_warmup_loss"]
           and cobs["first_loss"] > cobs["post_warmup_loss"] > 0,
           "C original update/gradient/finite-loss observations changed")
    refuse(m["scientific_boundary"] == new_custody.BOUNDARY
           and m["retention_of_original_zips_and_logs"] == "permanent_git_bytes"
           and m["physical_host_run_job_success"] is True
           and m["source_only_run_job_success"] is True,
           "D original manifest scientific/retention scope falsely promoted")
    refuse(d["status"] ==
           "PASS_ORIGINAL_POSTSTEP_COLD_BYTES__SCIENTIFIC_72_BLOCKED"
           and d["original_run"] == m["run"] == RUNS["D_poststep_cold"][0]
           and d["original_head"] == m["source_head"] == RUNS["D_poststep_cold"][1]
           and m["original"]["physical_result"]["frozen_backbone_sha256"] == FROZEN
           and d["original_zips_and_logs"] == 4
           and d["original_zip_members"] == 8
           and d["synthetic_nca_steps"] == 1
           and d["context_units"] == 31
           and d["cache_checked_tail_tokens"] == 4
           and d["poststep_cache_max_abs"] ==
               m["original"]["physical_result"]["post_step_last_four_true_kv_max_abs"]
           and 0 <= d["poststep_cache_max_abs"] <= TOL
           and d["separate_process_cold_adapter"] is True
           and d["train_authorized"] is False
           and d["full_72_conformance_pass"] is False
           and d["scientific_review_complete"] is False,
           "D original post-update cold/KV source or scope replaced")
    refuse(prod["status"] == cons["status"] ==
           "POSTSTEP_COLD_DIAGNOSTIC_ONLY__SCIENTIFIC_72_BLOCKED"
           and type(prod["diagnostic_optimizer_steps"]) is int
           and prod["diagnostic_optimizer_steps"] == 1
           and type(cons["diagnostic_optimizer_steps"]) is int
           and cons["diagnostic_optimizer_steps"] == 0
           and intermediate["status"] == "NEW_DIAGNOSTIC_NOT_ORIGINAL_172"
           and type(intermediate["diagnostic_optimizer_steps"]) is int
           and intermediate["diagnostic_optimizer_steps"] == 1
           and prod["sample"] == cons["sample"] == intermediate["synthetic_sample"]
           and all(prod["sample"][k] == c["synthetic_sample"][k]
                   for k in ("ids_sha256","labels_sha256","mask_sha256",
                             "total_units","supervised_after_shift","real_eos_id"))
           and prod["producer_pid"] != cons["cold_process_pid"]
           and cons["post_step_cold_reload_exact"] is True
           and cons["per_request_aba_exact"] is True
           and cons["incremental_tokens"] == 4
           and cons["context_units"] == 31
           and prod["frozen_backbone_sha256"] == cons["frozen_backbone_sha256"] == FROZEN,
           "C/D same-public-example identity or poststep process boundary incorrect")
    refuse(len({v[0] for v in RUNS.values()}) == 4
           and len({v[1] for v in RUNS.values()}) == 4
           and a["nine_check_original_cohort_crosswalk"]["full_prefix_vs_incremental_cache"][
               "later_172_one_step_original_run"] is None,
           "different original physical source heads collapsed into one execution")
    for k in NINE:
        refuse(a["nine_check_original_cohort_crosswalk"][k]["final_scientific_check_admitted"] is False
               and a["nine_check_original_cohort_crosswalk"][k]["historical_171_status"] ==
                   h["matrix"][k]["status"]
               and tuple(OBSERVED_IN[k]),
               "historical nine-check source attribution or admission overwritten")

    matrix = []
    for k in NINE:
        matrix.append({
            "required_check": k,
            "historical_171_status_unchanged": h["matrix"][k]["status"],
            "original_cohorts_with_bounded_observation": list(OBSERVED_IN[k]),
            "observational_class": OBSERVATION_CLASS[k],
            "final_scientific_check_admitted": False,
        })
    cohorts = {
        key: {"original_run": run, "original_source_head": sha}
        for key, (run, sha) in RUNS.items()
    }
    cohorts["A_pretraining_short_and_cold"].update({
        "optimizer_steps": 0, "poststep_context_units": None,
        "scope": "prestep_postlease_32_128_kv_causal_eos_request_reset_second_python_NCA_MLP",
    })
    cohorts["B_pretraining_maxcontext"].update({
        "optimizer_steps": 0, "max_tested_context_units": 2048,
        "nca_cache_max_abs_2048": h["evidence"]["max_context_cache_max_abs"]["nca_2048"],
        "generation_records_without_eos_258": 12,
        "scope": "prestep_512_1024_2048_cache_future_suffix_and_HF_generate_no_learned_adapter",
    })
    cohorts["C_one_step_gradient"].update({
        "optimizer_steps": 1, "context_units": 31,
        "supervised_after_shift": 21, "eos_id": EOS,
        "initial_loss": cobs["first_loss"], "post_warmup_loss": cobs["post_warmup_loss"],
        "scope": "one_public_synthetic_sample_NCA_only_SGD_post_warmup_gradient_frozen_base_no_cold_replay",
    })
    cohorts["D_poststep_cold"].update({
        "optimizer_steps": 1, "context_units": 31,
        "incremental_KV_tail_positions": 4,
        "incremental_KV_max_abs": d["poststep_cache_max_abs"],
        "same_runner_second_python": True,
        "distinct_original_run_from_C": True,
        "scope": "one_public_synthetic_NCA_only_step_cold_reload_A_B_A_last_four_KV",
        "original_artifact_zip_sha256": {
            kind: m["original"][kind]["zip_sha256"]
            for kind in ("source", "host")
        },
        "original_decoded_job_log_sha256": {
            kind: m["original"][kind]["decoded_log_sha256"]
            for kind in ("source", "host")
        },
    })
    return {
        "schema": "ncr001-void1-fourcohort-science-v1",
        "status": STATUS, "model_key": "void1", "source_only": True,
        "parent_pr": 211, "parent_head": PARENT,
        "model_repo": MODEL, "model_revision": REV,
        "checkpoint_sha256": WEIGHT, "byte_tokenizer_sha256": TOKENIZER,
        "protocol_git_blob": PROTOCOL_BLOB, "source_pins_git_blob": PINS_BLOB,
        "historical_171_adjudication_unchanged": h["adjudication"],
        "original_four_physical_cohorts": cohorts,
        "nine_required_checks": matrix,
        "scope_limits_still_unproven": list(UNPROVEN),
        "physical_run_results_are_separate_not_interchangeable": True,
        "poststep_2048_full_KV_measured": False,
        "poststep_all_31_token_KV_measured": False,
        "poststep_masked_batch_and_future_suffix_measured": False,
        "poststep_2048_generation_measured": False,
        "independent_scientific_72_admission": False,
        "per_model_conformance_pass": False,
        "full_72_conformance_pass": False,
        "train_authorized": False,
        "comparative_nca_mlp_training_performed": False,
        "new_optimizer_steps": 0,
        "fresh_physical_model_execution": False,
        "heldouts_opened": False,
        "matched_training_compute_qualified": False,
        "generated_eos_skill_qualified": False,
        "historical_train002b_eos_258_effect_measured": False,
        "additional_note": (
            "Positive physical observations across different source heads and contexts "
            "are bounded evidence, not a single trained 2048-context run or formal "
            "nine-check independent scientific admission. Historical #171 BLOCKED "
            "report, original failed CI and all four raw cohorts remain unchanged."
        ),
    }


def build_report() -> dict:
    return interpret(source_inputs())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = build_report()
    if args.check:
        refuse(OUTPUT.exists() and json.loads(OUTPUT.read_text(encoding="utf-8")) == report,
               "committed derived scientific report does not match original raw cohorts")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
