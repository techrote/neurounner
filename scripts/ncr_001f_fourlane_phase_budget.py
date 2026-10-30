#!/usr/bin/env python3
"""NCR-001F: four independent source-MAC accounts; missing physical TRAIN phases.

SOURCE-ONLY. Never turns forward-only samples or synthetic MACs into equal
training compute, profiler FLOPs, learned capability or permission to train.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import ncr_001f_corrected_samehost_budget_delta as original

ROOT = Path(__file__).resolve().parents[1]
PARENT = "28353d042fe479589e04fc7a85fbc56a90fb7707"
STATUS = "FOUR_LANE_SOURCE_MAC_ACCOUNT__ACTUAL_TRAINING_BUDGET_UNQUALIFIED"
WIDTHS = {"smollm2": 576, "sprocket_chat": 1280,
          "sprocket_instruct": 1280, "void1": 640}
TAPS = {"smollm2": [9,19,29], "sprocket_chat": [7,16,25],
        "sprocket_instruct": [7,16,25], "void1": [7,15,23]}
WEIGHT_FILES = {"smollm2": "model.safetensors",
                "sprocket_chat": "chat/model.safetensors",
                "sprocket_instruct": "model.safetensors",
                "void1": "model.safetensors"}
ARMS = ("frozen_base_no_adapter", "parameter_matched_gated_mlp",
        "persistent_nca_3_steps", "persistent_nca_1_step",
        "reset_each_tap_nca_3_steps", "permuted_neighborhood_nca_3_steps")
REQUIRED_PHYSICAL_PHASES = ("forward_ns", "backward_ns", "optimizer_ns")
REQUIRED_PROVENANCE = (
    "source_head", "original_run_id", "original_job_id",
    "original_artifact_zip_sha256", "original_full_job_log_sha256",
    "frozen_checkpoint_sha256", "same_runner_and_cpu_identity",
    "torch_threads_effective_after_loader", "warmup_policy",
    "seed_and_schedule_sha256", "actual_input_and_supervised_token_counts",
    "per_step_forward_backward_optimizer_samples", "peak_rss_bytes",
)

def require(condition: bool, why: str) -> None:
    if not condition:
        raise ValueError("NCR001F_FOURLANE_REFUSED: " + why)

def estimate(hidden: int) -> dict:
    """Primary and source ablations: counted MACs, NOT measured operations."""
    require(type(hidden) is int and hidden in (576, 640, 1280),
            "unregistered or bool hidden width")
    taps, grid, channels, bottleneck = 3, 8, 4, 256
    field = grid * grid * channels
    # 4->16, 3x3 Conv and 16->8, 1x1 Conv, 64 cells each.
    local_conv = grid * grid * ((channels*4)*channels*9
                              + (channels*2)*(channels*4))
    require(field == bottleneck and local_conv == 45056,
            "cellular geometry arithmetic drift")
    dense = taps * 2 * hidden * field
    ordinary = taps * 2 * hidden * bottleneck
    require(dense == ordinary, "dense transforms unexpectedly unmatched")
    nca_one = dense + taps * local_conv
    nca_three = dense + taps * 3 * local_conv
    nca_params = 514 * hidden + 984
    mlp_params = 514 * hidden + 512
    require(nca_three - ordinary == 405504 and
            nca_params - mlp_params == 472 and
            (nca_params-mlp_params)/nca_params < 0.05,
            "source arithmetic/parameter match drift")
    result = {
        "parameter_count": {"nca_all_steps": nca_params,
                            "gated_mlp": mlp_params,
                            "absolute_difference": 472},
        "counted_mac_per_token_not_flops": {
            "frozen_base_adapter_only": 0,
            "parameter_matched_gated_mlp": ordinary,
            "persistent_nca_3_steps": nca_three,
            "persistent_nca_1_step": nca_one,
            "reset_each_tap_nca_3_steps": nca_three,
            "permuted_neighborhood_nca_3_steps": nca_three,
        },
        "nca_primary_over_mlp_source_mac_ratio": nca_three/ordinary,
        "extra_primary_local_conv_mac_per_token": 405504,
        "normalization_nonlin_autograd_and_memory_traffic_counted": False,
        "measured_hardware_flops": False,
        "measured_training_compute": False,
    }
    return result

def assert_protocol(p: dict) -> None:
    require(type(p) is dict and p.get("experiment") == "NCR-001"
            and p.get("train_authorized") is False, "preregistration/training policy changed")
    require(set(p.get("models", {})) == set(WIDTHS),
            "four independent model keys required")
    require(tuple(p.get("comparisons", ())) == ARMS,
            "six preregistered control/ablation arms changed")
    arch = p["architecture"]
    require(arch["state_geometry"] ==
            {"height":8,"width":8,"channels":4,"scalars_per_token":256}
            and arch["tap_count"] == 3 and arch["nca_steps_per_tap"] == 3,
            "primary NCA geometry changed")
    training = p["training"]
    require(training["optimizer"] == "AdamW"
            and training["primary_steps_per_seed"] == 192
            and training["seed_list"] == [20261026,20261027,20261028]
            and training["torch_threads"] == 4
            and training["trainable_backbone"] is False,
            "locked schedule, optimizer or frozen base changed")
    require(p["match_constraints"]["same_examples_order_optimizer_updates_seed_and_taps"] is True
            and p["match_constraints"]["comparison_within_each_model"] is True
            and p["match_constraints"]["adapter_trainable_param_tolerance_fraction"] == 0.05,
            "matched within-model protocol changed")
    require(p["conformance"]["required_checks"] ==
            ["same_base_logits","base_weight_unchanged_after_step",
             "causal_suffix_invariance","full_prefix_vs_incremental_cache",
             "per_token_isolation","per_request_reset",
             "nonzero_adapter_gradient_after_warmup","finite_state_and_loss",
             "cold_reload_state_dict"],
            "required #72 gate changed")
    for key, hidden in WIDTHS.items():
        m=p["models"][key]
        require(type(m["expected_config"]["hidden_size"]) is int
                and m["expected_config"]["hidden_size"] == hidden
                and m["taps_after_block_zero_based"] == TAPS[key]
                and m["weight_file"] == WEIGHT_FILES[key],
                "variant/width/taps/weight path drift: " + key)

def build() -> dict:
    p=json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text(encoding="utf-8"))
    assert_protocol(p)
    original_decision=original.audit()  # Re-audits complete Git-custodied ZIP/logs.
    require(original_decision["status"] == original.STATUS
            and original_decision["custody_head"] ==
                "55026d914efd3ef4548da0644ccd27bb8de3074f"
            and original_decision["new_physical_execution"] is False
            and original_decision["new_optimizer_steps"] == 0
            and original_decision["train_authorized"] is False
            and original_decision["actual_compute_budget_qualified"] is False,
            "parent original physical/custody boundary changed")
    sprocket_proxy=original_decision["model_proxy"]
    expected_sprocket=estimate(1280)
    require(sprocket_proxy["nca_dense_conv_mac_per_token"] ==
            expected_sprocket["counted_mac_per_token_not_flops"]["persistent_nca_3_steps"]
            and sprocket_proxy["mlp_dense_mac_per_token"] ==
            expected_sprocket["counted_mac_per_token_not_flops"]["parameter_matched_gated_mlp"]
            and sprocket_proxy["measured_flops"] is False,
            "independent source arithmetic disagrees with parent source audit")
    lanes={}
    for key, width in WIDTHS.items():
        m=p["models"][key]
        physical = original_decision["lanes"].get(key)
        if physical:
            require(physical["trained_quality_measured"] is False
                    and physical["actual_compute_budget_qualified"] is False
                    and physical["effective_cpu_threads"] == 4
                    and [x["context_units"] for x in physical["cases"]] == [512,2048],
                    "untrained observed Sprocket forward sample promoted")
        lanes[key]={
            "model_key": key, "checkpoint_repo": m["repo_id"],
            "checkpoint_revision": m["revision"],
            "checkpoint_weight_file": m["weight_file"],
            "width": width, "taps_post_block_zero_based": TAPS[key],
            "source_cost_proxy": estimate(width),
            "original_untrained_complete_host_forward_samples_available": physical is not None,
            "original_untrained_forward_samples_per_arm_per_context":
                3 if physical else None,
            "original_untrained_forward_contexts": [512,2048] if physical else [],
            "original_corrected_cpu_threads": 4 if physical else None,
            "original_physical_job_id": physical["original_job_id"] if physical else None,
            "measured_forward_backward_optimizer_training_cost_ns": None,
            "actual_training_tokens_observed": None,
            "matched_training_wall_cost_ratio": None,
            "matched_training_compute_qualified": False,
        }
    require(lanes["sprocket_chat"]["original_physical_job_id"] !=
            lanes["sprocket_instruct"]["original_physical_job_id"],
            "Chat and root Instruct physical hosts collapsed")
    return {
        "schema": "ncr001f-fourlane-source-phase-budget-v1",
        "status": STATUS, "parent_pr": 209, "parent_source_head": PARENT,
        "original_corrected_physical_run": 40904839128,
        "original_physical_execution_in_this_run": False,
        "preregistered_seeds": p["training"]["seed_list"],
        "preregistered_updates_per_seed_per_arm": 192,
        "planned_updates_per_arm_not_executed": 576,
        "actual_optimizer_updates_in_this_run": 0,
        "required_physical_phase_samples": list(REQUIRED_PHYSICAL_PHASES),
        "required_original_provenance_before_scientific_budget_admission":
            list(REQUIRED_PROVENANCE),
        "lanes": lanes,
        "cross_model_performance_pooled": False,
        "original_sealed_fixture_contents_opened": False,
        "no_actual_training_phase_receipts_present": True,
        "reported_source_mac_are_measured_flops": False,
        "forward_only_is_matched_training_compute": False,
        "train_authorized": False,
        "full_72_conformance_pass": False,
        "actual_compute_budget_qualified": False,
    }

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    payload=json.dumps(build(),indent=2,sort_keys=True)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(payload,encoding="utf-8")
    print(payload,end="")
