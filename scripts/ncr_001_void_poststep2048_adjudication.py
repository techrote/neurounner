#!/usr/bin/env python3
"""Independent source-only adjudication of void.1 post-update 2048 evidence.

Re-audits the prior four original physical cohorts and the new permanent
poststep2048 source/host ZIP+decoded-log custody. Historical #171 statuses are
immutable. Positive bounded physical observations are not silently converted
to a final per-model scientific PASS or training authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

import ncr_001_void_fourcohort_scientific_delta as prior
import ncr_001_void_poststep2048_original_custody as custody

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"
PINS = ROOT / "experiments/ncr-001/source-pins.json"
PRIOR_DECISION = (
    ROOT / "experiments/ncr-001/adjudication/"
    "void1-fourcohort-scientific-boundary-20261030.json"
)
OUTPUT = (
    ROOT / "experiments/ncr-001/adjudication/"
    "void1-poststep2048-scientific-boundary-20261030.json"
)

PARENT = "058ed306e9ea13395c08b8513265363da179fa4e"
PROTOCOL_BLOB = "14ce5fbaa38f8806c8c8738f19c01b91385eebaf"
PINS_BLOB = "59fff1aed872b5563aa6898171983973e0cdb02c"
PRIOR_DECISION_BLOB = "7dfa8e7e38367e2eada34897d781e0a5dc41b682"
NEW_MANIFEST_BLOB = "462a43cf3e56b5edefe9dcf14ffc6dae199a3158"
NEW_RUN = 41133450850
NEW_HEAD = "520c4ae5b3152aff8d94760dab2e3103746f7334"
MODEL = "dotlabs/void.1"
REV = "9c2be5087772867d45961abd11f6b35110617563"
WEIGHT = "ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3"
TOKENIZER = "279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03"
FROZEN = "7673957ba36e8eb5427e5ecc94dd52de71fbf9e6718bb0c2a94e4d7361b3f904"
TOL = 5e-4
NINE = prior.NINE
NEW_COHORT = "E_poststep_2048"
STATUS = (
    "VOID1_NINE_CHECK_BOUNDED_PHYSICAL_EVIDENCE_PRESENT__"
    "MASKED_CACHE_AND_FINAL_SCIENTIFIC_ADMISSION_PENDING"
)

OBSERVATIONS = {
    "same_base_logits": {
        "cohorts": ["A_pretraining_short_and_cold",
                    "B_pretraining_maxcontext",
                    "C_one_step_gradient", NEW_COHORT],
        "classification": "ZERO_BASELINE_PLUS_UPDATED_SOURCE_MECHANICS_SUPPORTED",
    },
    "base_weight_unchanged_after_step": {
        "cohorts": ["C_one_step_gradient", "D_poststep_cold", NEW_COHORT],
        "classification": "PUBLIC_ONE_STEP_FROZEN_BACKBONE_SUPPORTED",
    },
    "causal_suffix_invariance": {
        "cohorts": ["A_pretraining_short_and_cold",
                    "B_pretraining_maxcontext", NEW_COHORT],
        "classification": "POSTSTEP_2048_STRICT_SUFFIX_SUPPORTED",
    },
    "full_prefix_vs_incremental_cache": {
        "cohorts": ["A_pretraining_short_and_cold",
                    "B_pretraining_maxcontext", "D_poststep_cold", NEW_COHORT],
        "classification": "POSTSTEP_ALL_2048_POSITIONS_SUPPORTED_B1",
    },
    "per_token_isolation": {
        "cohorts": ["A_pretraining_short_and_cold",
                    "B_pretraining_maxcontext", NEW_COHORT],
        "classification":
            "POSTSTEP_B2_MASKED_FULL_FORWARD_AND_CACHED_B1_STATE_SUPPORTED",
    },
    "per_request_reset": {
        "cohorts": ["A_pretraining_short_and_cold", "D_poststep_cold"],
        "classification": "PRESTEP_AND_POSTSTEP_A_B_A_SUPPORTED",
    },
    "nonzero_adapter_gradient_after_warmup": {
        "cohorts": ["C_one_step_gradient", "D_poststep_cold", NEW_COHORT],
        "classification": "PUBLIC_ONE_STEP_POSTWARMUP_GRADIENT_SUPPORTED",
    },
    "finite_state_and_loss": {
        "cohorts": ["A_pretraining_short_and_cold", "C_one_step_gradient",
                    "D_poststep_cold", NEW_COHORT],
        "classification": "FINITE_PUBLIC_LOSS_STATE_AND_LONG_FORWARD_SUPPORTED",
    },
    "cold_reload_state_dict": {
        "cohorts": ["A_pretraining_short_and_cold",
                    "D_poststep_cold", NEW_COHORT],
        "classification": "POSTSTEP_SECOND_INTERPRETER_2048_SUPPORTED",
    },
}
REMAINING = (
    "poststep_B2_masked_incremental_KV_cache_not_measured",
    "poststep_updated_adapter_not_replicated_on_a_fresh_independent_host",
    "multi_sample_multi_seed_learned_effectiveness_not_measured",
    "matched_NCA_MLP_training_quality_and_compute_not_measured",
    "generated_EOS_258_skill_not_qualified",
    "historical_TRAIN002b_EOS_effect_remains_unmeasured",
    "explicit_final_model_specific_scientific_72_admission_not_granted",
)
NOT_RETROACTIVE_GATES = (
    "fresh_independent_host_replication",
    "multi_sample_learning_effectiveness",
    "matched_NCA_MLP_quality_compute",
)


def refuse(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("VOID_POSTSTEP2048_ADJUDICATION_REFUSED: " + reason)


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
    ).hexdigest()


def require_blob(path: Path, expected: str) -> None:
    refuse(path.is_file() and not path.is_symlink()
           and git_blob(path.read_bytes()) == expected,
           "immutable report/source bytes changed: " + str(path))
    rel = path.relative_to(ROOT).as_posix()
    row = subprocess.run(
        ["git", "-C", str(ROOT), "ls-tree", "HEAD", "--", rel],
        capture_output=True, text=True, check=False,
    )
    refuse(
        row.returncode == 0
        and row.stdout == f"100644 blob {expected}\t{rel}\n",
        "committed Git tree object/mode changed: " + rel,
    )


def source_inputs() -> dict:
    for path, blob in (
        (PROTOCOL, PROTOCOL_BLOB),
        (PINS, PINS_BLOB),
        (PRIOR_DECISION, PRIOR_DECISION_BLOB),
        (custody.MANIFEST, NEW_MANIFEST_BLOB),
    ):
        require_blob(path, blob)
    prior_report = prior.build_report()
    committed_prior = json.loads(PRIOR_DECISION.read_text(encoding="utf-8"))
    refuse(prior_report == committed_prior,
           "prior #216 four-cohort decision no longer regenerates exactly")
    new_report, originals = custody.read_and_check(verify_git=True)
    new_manifest = json.loads(custody.MANIFEST.read_text(encoding="utf-8"))
    with zipfile.ZipFile(io.BytesIO(originals["host-original.zip"])) as archive:
        producer = json.loads(archive.read(
            "ncr001-void-poststep2048/producer.json"))
        consumer = json.loads(archive.read(
            "ncr001-void-poststep2048/consumer.json"))
        learning = json.loads(archive.read(
            "ncr001-void-nca-learning/receipt.json"))
    return {
        "protocol": json.loads(PROTOCOL.read_text(encoding="utf-8")),
        "pins": json.loads(PINS.read_text(encoding="utf-8")),
        "prior": prior_report,
        "new": new_report,
        "manifest": new_manifest,
        "producer": producer,
        "consumer": consumer,
        "learning": learning,
    }


def interpret(s: dict) -> dict:
    p, pins = s["protocol"], s["pins"]
    old, new = s["prior"], s["new"]
    manifest = s["manifest"]
    producer, consumer, learning = (
        s["producer"], s["consumer"], s["learning"]
    )

    refuse(
        p["experiment"] == "NCR-001" and p["train_authorized"] is False
        and set(p["models"]) == set(pins["models"]) ==
            {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"}
        and tuple(p["conformance"]["required_checks"]) == NINE
        and p["conformance"]["any_failure_blocks_model_qualification"] is True
        and p["conformance"]["full_vs_cached_prefill_atol"] == TOL
        and p["conformance"]["full_vs_cached_prefill_rtol"] == TOL,
        "locked four-lane protocol or causal tolerance changed",
    )
    lane, pin = p["models"]["void1"], pins["models"]["void1"]
    refuse(
        lane["repo_id"] == pin["repo_id"] == MODEL
        and lane["revision"] == pin["revision"] == REV
        and lane["weight_file"] == pin["hf_weight_file"] == "model.safetensors"
        and lane["taps_after_block_zero_based"] == [7,15,23]
        and lane["expected_config"]["max_position_embeddings"] == 2048
        and lane["tokenizer_reference"]["ids"]["eos"] == 258
        and pin["weight_declared_lfs_sha256_not_downloaded"] == WEIGHT
        and pin["tokenization_byte_py_sha256"] == TOKENIZER,
        "void.1 exact source/tokenizer identity changed",
    )
    refuse(
        old["status"] ==
            "SOURCE_BOUND_FOUR_DISTINCT_VOID1_COHORTS__SCIENTIFIC_72_UNADMITTED"
        and old["parent_head"] ==
            "0c1a47c4d562a24c7d5a81305affb4a2501fd80e"
        and old["historical_171_adjudication_unchanged"] ==
            "BLOCKED_REQUIRED_LEARNING_CHECKS__POSTLEASE_SOURCE_RUNTIME_PRETRAIN_PASS"
        and old["per_model_conformance_pass"] is False
        and old["independent_scientific_72_admission"] is False
        and old["full_72_conformance_pass"] is False
        and old["train_authorized"] is False
        and len(old["original_four_physical_cohorts"]) == 4
        and len(old["nine_required_checks"]) == 9,
        "prior #216 interpretation falsely promoted or changed",
    )
    historical = {
        row["required_check"]: row["historical_171_status_unchanged"]
        for row in old["nine_required_checks"]
    }
    refuse(
        historical == {
            "same_base_logits": "PASS",
            "base_weight_unchanged_after_step": "BLOCKED_BY_TRAINING_POLICY",
            "causal_suffix_invariance": "PASS",
            "full_prefix_vs_incremental_cache": "PASS",
            "per_token_isolation": "PASS_SOURCE_RUNTIME",
            "per_request_reset": "PASS",
            "nonzero_adapter_gradient_after_warmup":
                "BLOCKED_BY_TRAINING_POLICY",
            "finite_state_and_loss": "PARTIAL_BLOCKED_WARMUP_LOSS",
            "cold_reload_state_dict": "PASS_PRETRAIN",
        },
        "historical #171 statuses changed",
    )

    physical = new["physical_result"]
    refuse(
        new["status"] ==
            "PASS_ORIGINAL_POSTSTEP2048_COMPLETE_BYTES__SCIENTIFIC_72_UNADMITTED"
        and new["original_run"] == manifest["run"] == NEW_RUN
        and new["original_head"] == manifest["source_head"] == NEW_HEAD
        and new["permanent_git_custody"] is True
        and new["original_zips_and_logs"] == 4
        and new["original_zip_members"] == 9
        and new["fresh_physical_execution"] is False
        and new["train_authorized"] is False
        and new["per_model_gate"] is False
        and new["full_72_conformance_pass"] is False
        and new["independent_scientific_review"] is False
        and new["comparative_training"] is False
        and new["heldouts_opened"] is False
        and new["actual_compute_budget_qualified"] is False
        and new["generated_eos_skill_qualified"] is False
        and new["historical_train002b_eos_effect_measured"] is False
        and manifest["scientific_boundary"] == custody.BOUNDARY,
        "new original cohort custody/admission scope changed",
    )
    refuse(
        physical["nca_only_optimizer_steps"] == 1
        and physical["public_training_units"] == 31
        and physical["supervised_after_shift"] == 21
        and physical["true_eos_id"] == 258
        and physical["frozen_backbone_sha256"] == FROZEN
        and physical["poststep_2048_incremental_positions"] == 2048
        and 0 <= physical["poststep_2048_cache_max_abs"] <= TOL
        and physical["masked_batch"]["left"] == {
            "max_batch_vs_single_abs": 0.0, "max_row_swap_abs": 0.0}
        and physical["masked_batch"]["right"] == {
            "max_batch_vs_single_abs": 0.0, "max_row_swap_abs": 0.0}
        and physical["future_suffix_units"] == 64
        and physical["future_suffix_earlier_exact"] is True
        and physical["generate_prefix_units"] == 2044
        and physical["generate_new_units"] == 4
        and physical["generated_ids"] == [105,110,115,112]
        and physical["generation_actual_eos_id_emitted"] is False
        and 0 <= physical["generation_cache_score_max_abs"] <= TOL,
        "new physical result altered or over tolerance",
    )

    refuse(
        producer["status"] ==
            "POSTSTEP2048_UPDATED_ADAPTER_READY__SCIENTIFIC_72_UNADMITTED"
        and consumer["status"] ==
            "PASS_POSTSTEP2048_CACHE_MASK_SUFFIX_GENERATE__SCIENTIFIC_72_UNADMITTED"
        and type(producer["diagnostic_optimizer_steps"]) is int
        and producer["diagnostic_optimizer_steps"] == 1
        and producer["public_learning_sample"]["total_units"] == 31
        and producer["public_learning_sample"]["supervised_after_shift"] == 21
        and producer["public_learning_sample"]["real_eos_id"] == 258
        and producer["learning_summary"]["checks"][
            "backbone_unchanged_after_one_step"] is True
        and producer["learning_summary"]["checks"][
            "post_step_upstream_gradients"] is True
        and producer["learning_summary"]["backbone_before_sha256"] ==
            producer["learning_summary"]["backbone_after_sha256"] == FROZEN
        and learning["status"] ==
            "NEW_POSTSTEP2048_DIAGNOSTIC_NOT_ORIGINAL_172"
        and type(learning["diagnostic_optimizer_steps"]) is int
        and learning["diagnostic_optimizer_steps"] == 1,
        "new one-step learning/frozen/gradient mechanics changed",
    )
    cache = consumer["poststep_2048_all_position_cache"]
    refuse(
        cache["context_units"] == 2048
        and cache["incremental_positions"] == 2048
        and cache["all_positions_compared"] is True
        and len(cache["trace_checkpoints"]) == 7
        and [x["position"] for x in cache["trace_checkpoints"]] ==
            [0,31,127,511,1023,1535,2047]
        and all(
            x["trace"]["tap_count"] == 3
            and x["trace"]["state_shape"] == [1,1,4,8,8]
            and x["trace"]["finite"] is True
            for x in cache["trace_checkpoints"]
        ),
        "new all-position cached per-token state evidence changed",
    )
    refuse(
        consumer["poststep_2048_suffix"]["earlier_logits_exact"] is True
        and consumer["poststep_2048_suffix"]["earlier_max_abs"] == 0.0
        and set(consumer["poststep_masked_batch"]) == {"left","right"}
        and all(
            consumer["poststep_masked_batch"][side][
                "batch_vs_identical_single_within_tolerance"] is True
            and consumer["poststep_masked_batch"][side][
                "row_order_swap_within_tolerance"] is True
            for side in ("left","right")
        )
        and consumer["poststep_native_generation"][
            "cache_on_off_token_ids_equal"] is True
        and consumer["poststep_native_generation"][
            "cache_on_off_scores_within_tolerance"] is True
        and consumer["frozen_backbone_unchanged_after_long_diagnostic"] is True,
        "new masked/suffix/generation/frozen evidence changed",
    )
    for row in (producer, consumer):
        refuse(
            row["train_authorized"] is False
            and row["comparative_training_performed"] is False
            and row["heldouts_opened"] is False
            and row["full_72_conformance_pass"] is False
            and row["independent_scientific_review"] is False,
            "new physical receipt falsely claims science/training",
        )

    matrix = []
    for check in NINE:
        prior_row = next(
            row for row in old["nine_required_checks"]
            if row["required_check"] == check
        )
        info = OBSERVATIONS[check]
        cohorts = list(info["cohorts"])
        refuse(len(cohorts) == len(set(cohorts)),
               "duplicate source cohort in check attribution")
        matrix.append({
            "required_check": check,
            "historical_171_status_unchanged":
                prior_row["historical_171_status_unchanged"],
            "prior_216_observational_class":
                prior_row["observational_class"],
            "bounded_physical_observation_cohorts": cohorts,
            "current_observational_class": info["classification"],
            "bounded_observation_present": True,
            "final_scientific_check_admitted": False,
        })

    e = {
        "original_run": NEW_RUN,
        "original_source_head": NEW_HEAD,
        "optimizer_steps": 1,
        "public_training_units": physical["public_training_units"],
        "supervised_after_shift": physical["supervised_after_shift"],
        "true_eos_id": physical["true_eos_id"],
        "same_runner_second_python": True,
        "effective_threads": physical["effective_threads"],
        "poststep_context_units": 2048,
        "incremental_positions": 2048,
        "all_positions_compared": True,
        "cache_max_abs": physical["poststep_2048_cache_max_abs"],
        "masked_batch_full_forward": {
            "batch_size": 2,
            "width": 2048,
            "left": physical["masked_batch"]["left"],
            "right": physical["masked_batch"]["right"],
        },
        "masked_batch_incremental_cache_measured": False,
        "future_suffix_units": 64,
        "future_suffix_earlier_exact": True,
        "generation_prefix_units": 2044,
        "generation_new_units": 4,
        "generation_ids": physical["generated_ids"],
        "generation_actual_eos_258_emitted":
            physical["generation_actual_eos_id_emitted"],
        "generation_cache_score_max_abs":
            physical["generation_cache_score_max_abs"],
        "frozen_backbone_sha256": physical["frozen_backbone_sha256"],
        "consumer_elapsed_seconds": physical["consumer_elapsed_seconds"],
        "consumer_peak_rss_kib": physical["consumer_peak_rss_kib"],
        "permanent_original_zip_log_custody": True,
        "original_artifact_zip_sha256": {
            kind: manifest["original"][kind]["zip_sha256"]
            for kind in ("source","host")
        },
        "original_decoded_job_log_sha256": {
            kind: manifest["original"][kind]["decoded_log_sha256"]
            for kind in ("source","host")
        },
    }

    return {
        "schema": "ncr001-void1-poststep2048-adjudication-v1",
        "status": STATUS,
        "model_key": "void1",
        "parent_pr": 218,
        "parent_head": PARENT,
        "protocol_git_blob": PROTOCOL_BLOB,
        "source_pins_git_blob": PINS_BLOB,
        "model_repo": MODEL,
        "model_revision": REV,
        "checkpoint_sha256": WEIGHT,
        "byte_tokenizer_sha256": TOKENIZER,
        "historical_171_adjudication_unchanged":
            old["historical_171_adjudication_unchanged"],
        "prior_216_status": old["status"],
        "prior_four_cohorts": old["original_four_physical_cohorts"],
        "new_fifth_physical_cohort": e,
        "nine_required_checks": matrix,
        "all_nine_have_bounded_source_bound_physical_observations": True,
        "largest_prior_joint_gap_poststep_2048": "CLOSED_BY_COHORT_E",
        "remaining_precise_scientific_gaps": list(REMAINING),
        "remaining_gaps_not_retroactive_protocol_requirements":
            list(NOT_RETROACTIVE_GATES),
        "masked_B2_full_forward_isolation_measured": True,
        "masked_B2_incremental_cache_measured": False,
        "fresh_independent_host_poststep_replication_measured": False,
        "generated_eos_skill_qualified": False,
        "historical_train002b_eos_258_effect_measured": False,
        "multi_sample_learning_effectiveness_measured": False,
        "matched_nca_mlp_training_quality_measured": False,
        "matched_training_compute_qualified": False,
        "source_only_adjudication": True,
        "fresh_physical_execution_in_this_adjudication": False,
        "new_optimizer_steps_in_this_adjudication": 0,
        "heldouts_opened": False,
        "independent_scientific_72_admission": False,
        "per_model_conformance_pass": False,
        "full_72_conformance_pass": False,
        "train_authorized": False,
        "decision_note": (
            "Cohort E closes the largest prior joint mechanical gap by physically "
            "testing one updated NCA at native 2048 with all-position B1 KV, "
            "B=2 explicit masked full-forward isolation, strict future suffix, "
            "native generation cache parity and frozen-source preservation. "
            "Historical #171 statuses remain archival facts. Final void1 scientific "
            "admission is deliberately not granted by this same-account source-only "
            "adjudication; masked B=2 incremental-cache coverage and explicit "
            "scientific admission remain separate."
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
        refuse(
            OUTPUT.is_file()
            and json.loads(OUTPUT.read_text(encoding="utf-8")) == report,
            "committed machine decision differs from original evidence rebuild",
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
