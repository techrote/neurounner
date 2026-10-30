#!/usr/bin/env python3
"""NCR-001 void.1 post-diagnostic evidence reconciliation, NOT scientific admission.

Source-only, stdlib-only. Rebuild the historical #171 matrix from its original
post-lease archives; independently verify the later #172 original frozen-Qwen3
diagnostic via #173's byte-preserved archive; report what new evidence exists.
Never rewrite #171, promote a source-only audit to a distinct scientific review,
or enable comparative NCA/MLP training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import ncr_001e_void1_postlease_adjudication as prior
import ncr_001_void_nca_original_custody as custody

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "experiments/ncr-001/adjudication/void1-postlease-pretrain.json"
OUT = ROOT / "experiments/ncr-001/adjudication/void1-postdiagnostic-review-pending.json"
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"
MANIFEST = custody.MANIFEST

THREE = (
    "base_weight_unchanged_after_step",
    "nonzero_adapter_gradient_after_warmup",
    "finite_state_and_loss",
)
SOURCES = {
    "original_physical_head": "2810e800ef890f0c99f9ea44adca29f879b5d0ea",
    "original_custody_head": "d6b730083bdf2d7d213e647f9770460bc1424e42",
    "historical_adjudication_head": "1738ac2b6bc6105d57d6a5b4c35f4beaad94ff86",
}
OBSERVED = "DIAGNOSTIC_EVIDENCE_PRESENT__DISTINCT_SCIENTIFIC_REVIEW_PENDING"


def refuse(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError("void1 post-diagnostic reconciliation REFUSED: " + reason)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_measurement(value: object) -> bool:
    """Measured losses/gradients must be real numbers, never bool or strings."""
    return type(value) in (int, float) and math.isfinite(value)


def exact_int(value: object, expected: int) -> bool:
    """Reject bool-as-int for optimizer steps, taps, and causal-label counts."""
    return type(value) is int and value == expected


# Separate stdlib oracle for the exact public diagnostic sample in original
# immutable #172 source (no ByteTokenizer, Torch, model or heldout imported).
# Original torch.long on the recorded Linux x86_64 runner is signed LE int64.
ORIGINAL_PROMPT = "Q: π?\nA: "
ORIGINAL_COMPLETION = "🔧 café 日本語"


def original_synthetic_oracle() -> dict:
    prompt = list(ORIGINAL_PROMPT.encode("utf-8", errors="strict"))
    completion = list(ORIGINAL_COMPLETION.encode("utf-8", errors="strict"))
    ids = prompt + completion + [258]
    labels = [-100] * len(prompt) + completion + [258]
    mask = [1] * len(ids)

    def tensor_sha(values: list[int]) -> str:
        # Matches source-pinned original.tensor_sha(torch.int64_tensor),
        # whose SHA256 input was contiguous little-endian int64 bytes.
        return hashlib.sha256(struct.pack("<" + "q" * len(values), *values)).hexdigest()

    return {
        "prompt_bytes": len(prompt),
        "supervised_after_shift": len(labels[1:]) - (len(prompt) - 1),
        "total_units": len(ids),
        "real_eos_id": 258,
        "supervised_eos_id": 258,
        "source_text_utf8_bytes": len(prompt) + len(completion),
        "ids_sha256": tensor_sha(ids),
        "labels_sha256": tensor_sha(labels),
        "mask_sha256": tensor_sha(mask),
    }


def check_inputs(legacy: dict, receipt: dict, manifest: dict, policy: dict) -> None:
    """Require consistent source-bound receipts, not a self-certified final PASS."""
    refuse(policy["experiment"] == "NCR-001"
           and policy["train_authorized"] is False
           and set(policy["models"]) ==
           {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"}
           and policy["models"]["smollm2"]["repo_id"] ==
           "HuggingFaceTB/SmolLM2-135M-Instruct"
           and policy["models"]["sprocket_chat"]["weight_file"] ==
           "chat/model.safetensors"
           and policy["models"]["sprocket_chat"]["variant"] == "chat"
           and policy["models"]["sprocket_instruct"]["weight_file"] ==
           "model.safetensors"
           and policy["models"]["sprocket_instruct"]["variant"] == "instruct"
           and policy["models"]["sprocket_chat"]["repo_id"] ==
           policy["models"]["sprocket_instruct"]["repo_id"] ==
           "kandivault/sprocket-500m"
           and policy["models"]["void1"]["repo_id"] == "dotlabs/void.1"
           and policy["models"]["void1"]["variant"] == "base_byte",
           "four-lane protocol/variant/checkpoint or training authorization changed")
    refuse(legacy["model_key"] == "void1"
           and legacy["source_runtime_pretraining_subgate"] is True
           and legacy["per_model_gate"] is False
           and legacy["train_authorized"] is False
           and legacy["independent_scientific_review"] is False
           and legacy["optimizer_steps_observed"] == 0
           and legacy["backward_executed"] is False,
           "historical pretraining decision promoted")
    refuse(legacy["adjudication"] ==
           "BLOCKED_REQUIRED_LEARNING_CHECKS__POSTLEASE_SOURCE_RUNTIME_PRETRAIN_PASS"
           and set(legacy["blocking_required_checks"]) == set(THREE)
           and set(legacy["matrix"]) == set(policy["conformance"]["required_checks"]),
           "historical required-check contract changed")
    refuse(legacy["matrix"][THREE[0]]["status"] == "BLOCKED_BY_TRAINING_POLICY"
           and legacy["matrix"][THREE[1]]["status"] == "BLOCKED_BY_TRAINING_POLICY"
           and legacy["matrix"][THREE[2]]["status"] == "PARTIAL_BLOCKED_WARMUP_LOSS"
           and all(not legacy["matrix"][name]["status"].startswith("PASS")
                   for name in THREE),
           "historical learning blockers overwritten")
    refuse(legacy["evidence"]["max_context_generation_records"] == 12
           and legacy["evidence"]["actual_eos_258_emitted_in_four_step_probe"] is False
           and legacy["byte_eos_dependency"]["historical_train002b_effect_measured"] is False,
           "12 negative EOS cases or historical EOS uncertainty lost")
    refuse(manifest["original_run"] == 40765025332
           and manifest["original_head"] == SOURCES["original_physical_head"]
           and manifest["original_artifacts"] ==
           {"source": 12850046593, "host": 12850675205}
           and manifest["physical_result"]["source_test_count"] == 11
           and manifest["boundary"]["train_authorized"] is False
           and manifest["boundary"]["independent_scientific_review"] is False
           and manifest["boundary"]["full_72_conformance_pass"] is False
           and manifest["boundary"]["comparative_training_performed"] is False,
           "physical run or custodial boundary substituted")
    refuse(receipt["status"] ==
           "PASS_VOID1_ONE_STEP_NCA_DIAGNOSTIC__72_SCIENTIFIC_ADMISSION_BLOCKED"
           and receipt["runner"]["expected_head"] == SOURCES["original_physical_head"]
           and receipt["runner"]["run_id"] == str(manifest["original_run"])
           and exact_int(receipt["diagnostic_optimizer_steps"], 1)
           and receipt["model_training_experiment"] is False
           and receipt["independent_scientific_review"] is False
           and receipt["comparative_training_performed"] is False
           and receipt["heldout_opened"] is False
           and receipt["full_72_conformance_pass"] is False,
           "one synthetic diagnostic misrepresented as a training experiment")
    refuse(receipt["source"]["protocol_sha256"] == legacy["protocol"]["sha256"]
           and receipt["source"]["pins_sha256"] ==
           legacy["protocol"]["source_pins_sha256"]
           and receipt["source"]["serializer_commit"] ==
           legacy["source"]["corrected_serializer_commit"]
           and receipt["actual_source_files"]["model.safetensors"]["sha256"] ==
           legacy["source"]["weight_sha256"]
           and receipt["actual_source_files"]["tokenization_byte.py"]["sha256"] ==
           legacy["source"]["tokenizer_source_sha256"],
           "source, prior evidence, serializer or weights cross-run mismatch")
    refuse(receipt["backbone_before_sha256"] ==
           receipt["backbone_after_sha256"] ==
           legacy["evidence"]["frozen_backbone_sha256"]
           and receipt["initial_zero_readout_parity_exact"] is True
           and receipt["checks"]["disabled_source_logits_unchanged"] is True
           and receipt["checks"]["backbone_unchanged_after_one_step"] is True,
           "source weights/logits altered during single diagnostic update")
    sample = receipt["synthetic_sample"]
    refuse(sample["heldout"] is False
           and exact_int(sample["real_eos_id"], 258)
           and exact_int(sample["supervised_eos_id"], 258)
           and exact_int(sample["prompt_bytes"], 10)
           and exact_int(sample["supervised_after_shift"], 21)
           and exact_int(sample["total_units"], 31),
           "source sample is not the fixed public shifted-EOS diagnostic")
    # The historical source audited the id/label digests; independently
    # reconstruct all *three* original int64 tensor hashes, including the
    # attention mask, from pinned public Unicode/UTF-8 text and true EOS 258.
    # A readable literal "<eos>" is ordinary bytes, never terminal EOS.
    expected_sample = original_synthetic_oracle()
    refuse(all(sample.get(name) == value
               and (type(sample.get(name)) is int if name in {
                   "prompt_bytes", "supervised_after_shift", "total_units",
                   "real_eos_id", "supervised_eos_id",
                   "source_text_utf8_bytes"} else True)
               for name, value in expected_sample.items()),
           "independent UTF-8/causal shifted EOS int64 tensor oracle mismatch")
    warm, post = receipt["warmup"], receipt["post_warmup"]
    refuse(warm["optimizer"] == "NCA-only SGD"
           and warm["learning_rate"] == 0.005
           and warm["readout_before_sha256"] != warm["readout_after_sha256"]
           and all(finite_measurement(v) and v > 0 for v in
                   (warm["native_causal_eos_loss"], post["native_causal_eos_loss"]))
           and warm["native_causal_eos_loss"] ==
           manifest["physical_result"]["initial_native_loss"]
           and post["native_causal_eos_loss"] ==
           manifest["physical_result"]["post_warmup_native_loss"],
           "single NCA-only step/finite teacher-forced loss not established")
    refuse(finite_measurement(warm["gradient_l1"]["readout.weight"])
           and warm["gradient_l1"]["readout.weight"] > 0
           and all(finite_measurement(warm["gradient_l1"][key])
                   and warm["gradient_l1"][key] == 0 for key in
                   ("observe.weight", "perception.weight",
                    "rule.weight", "input_norm.weight"))
           and all(finite_measurement(post["gradient_l1"][key])
                   and post["gradient_l1"][key] > 0 for key in
                   ("readout.weight", "observe.weight",
                    "perception.weight", "rule.weight", "input_norm.weight")),
           "zero-readout warmup or upstream post-warmup gradient invalid")
    fields = post["ordered_field_trace"]
    refuse(type(fields) is list and len(fields) == 3
           and all(type(f) is dict and exact_int(f["tap_index"], i)
                   and exact_int(f["next_tap"], i + 1)
                   and type(f["shape"]) is list
                   and all(type(n) is int for n in f["shape"])
                   and f["shape"] == [1, 31, 4, 8, 8]
                   and type(f["field_sha256"]) is str
                   and len(f["field_sha256"]) == 64
                   and all(c in "0123456789abcdef" for c in f["field_sha256"])
                   for i, f in enumerate(fields))
           and len({f["field_sha256"] for f in fields}) == 3,
           "depth-field order/identity not demonstrated")
    refuse(set(receipt["checks"]) == {
        "backbone_unchanged_after_one_step", "disabled_source_logits_unchanged",
        "initial_zero_adapter_parity", "native_shifted_byte_eos_loss",
        "no_cross_model_comparison", "post_step_upstream_gradients",
        "readout_gradient_first_backward", "three_distinct_depth_states",
    } and all(v is True for v in receipt["checks"].values()),
           "original diagnostic checks missing or false")


def build_report() -> dict:
    # Rebuild the historical matrix from the two ORIGINAL post-lease archives.
    legacy = prior.build_report()
    refuse(legacy == read(LEGACY), "committed historical #171 matrix changed")
    # This auditor validates original ZIPs, all members, complete original
    # job logs, producer source, frozen hashes and runtime semantics.
    receipt = custody.audit()
    adversarial = custody.adversarial()
    refuse(adversarial["negative_mutations_refused"] == 14
           and adversarial["full_72_conformance_pass"] is False,
           "custody's original adversarial refusal suite regressed")
    manifest = read(MANIFEST)
    check_inputs(legacy, receipt, manifest, read(PROTOCOL))

    return {
        "schema_version": 1,
        "experiment": "NCR-001E",
        "issue": 72,
        "model_key": "void1",
        "status": "PRELIMINARY_DIAGNOSTIC_EVIDENCE__SCIENTIFIC_ADMISSION_BLOCKED",
        "historical_matrix_unchanged": True,
        "historical_adjudication": legacy["adjudication"],
        "source_heads": SOURCES,
        "original_physical_run": manifest["original_run"],
        "original_custody_ci_run": 40766551985,
        "original_source_zip_sha256":
            manifest["original_artifact_zip_sha256"]["source"],
        "original_host_zip_sha256":
            manifest["original_artifact_zip_sha256"]["host"],
        "source_runtime_pretraining_subgate": True,
        "diagnostic_observation": {
            "step_kind": "one_in_memory_nca_only_sgd_on_public_synthetic_case",
            "optimizer_steps": 1,
            "post_warmup_backward": True,
            "prompt_units": 10,
            "supervised_after_causal_shift_units": 21,
            "terminal_eos_id": 258,
            "first_loss": warm_loss(receipt, "warmup"),
            "post_warmup_loss": warm_loss(receipt, "post_warmup"),
            "frozen_backbone_sha256": receipt["backbone_after_sha256"],
            "three_distinct_ordered_depth_fields": True,
            "post_warmup_all_required_gradients_nonzero": True,
            "original_source_tests": 11,
            "custody_adversarial_mutations_refused": 14,
        },
        "candidate_required_check_evidence": {name: OBSERVED for name in THREE},
        "unaccepted_required_checks": list(THREE),
        "independent_scientific_review": False,
        "per_model_gate": False,
        "full_72_conformance_pass": False,
        "train_authorized": False,
        "comparative_nca_mlp_training_performed": False,
        "heldouts_opened": False,
        "trained_checkpoint_created": False,
        "byte_eos_negative_results": {
            "max_context_four_step_no_eos_records": 12,
            "learned_eos_stopping_qualified": False,
            "historical_train002b_eos_effect_measured": False,
        },
        "decision_note": (
            "All three formerly missing learning/update/loss observations have "
            "source-bound synthetic diagnostic receipts, not accepted scientific "
            "review. Retain their historical blocked statuses until a distinct "
            "reviewer accepts exact pinned model/source, original receipts, "
            "protocol dependencies and four-lane admission. No comparative "
            "training, heldout evaluation, fixture changes or merges."
        ),
    }


def warm_loss(receipt: dict, key: str) -> float:
    return receipt[key]["native_causal_eos_loss"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Refuse any drift from committed postdiagnostic report")
    args = parser.parse_args()
    result = build_report()
    if args.check:
        refuse(result == read(OUT), "committed postdiagnostic report drift")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
