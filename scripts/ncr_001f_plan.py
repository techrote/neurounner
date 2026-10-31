#!/usr/bin/env python3
"""NCR-001F: read-only, fail-closed four-model matched-control plan.

This tool has no training command, no optimizer step and no model downloads.
It validates source/fixture/split identities and writes only a prospective plan.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import random
from pathlib import Path

from ncr_001_core import PersistentDepthNCA
from ncr_001f_controls import matched_control_contract

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"
DATASET_ROOT = ROOT / "datasets/train-003"
CONTROLS = ("frozen_base_no_adapter", "parameter_matched_gated_mlp",
            "persistent_nca_3_steps", "persistent_nca_1_step",
            "reset_each_tap_nca_3_steps", "permuted_neighborhood_nca_3_steps")


# Fixed source identities are independent of the supplied protocol JSON. An
# internally self-consistent but replaced protocol must never receive a PASS.
# These values are the four distinct frozen NCR-001A/#73 model lanes.
FROZEN_MODEL_LANES = {
    "smollm2": {
        "repo_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        "variant": "instruct", "weight_file": "model.safetensors",
        "taps": (9, 19, 29), "model_type": "llama",
        "hidden_size": 576, "num_hidden_layers": 30, "token_limit": 256,
    },
    "sprocket_chat": {
        "repo_id": "kandivault/sprocket-500m",
        "revision": "eb157d4ddddc4619e661d4a2426bb2dd000fc668",
        "variant": "chat", "weight_file": "chat/model.safetensors",
        "taps": (7, 16, 25), "model_type": "llama",
        "hidden_size": 1280, "num_hidden_layers": 26, "token_limit": 256,
    },
    "sprocket_instruct": {
        "repo_id": "kandivault/sprocket-500m",
        "revision": "eb157d4ddddc4619e661d4a2426bb2dd000fc668",
        "variant": "instruct", "weight_file": "model.safetensors",
        "taps": (7, 16, 25), "model_type": "llama",
        "hidden_size": 1280, "num_hidden_layers": 26, "token_limit": 256,
    },
    "void1": {
        "repo_id": "dotlabs/void.1",
        "revision": "9c2be5087772867d45961abd11f6b35110617563",
        "variant": "base_byte", "weight_file": "model.safetensors",
        "taps": (7, 15, 23), "model_type": "qwen3",
        "hidden_size": 640, "num_hidden_layers": 24, "token_limit": 1024,
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()
            if s.strip()]


def validate_split(train: list[dict], heldout: list[dict], manifest: dict) -> dict:
    """Check both row ID and family isolation, not just manifest self-claims."""
    if len(train) != manifest["train_examples"] or len(heldout) != manifest["heldout_examples"]:
        raise RuntimeError("TRAIN-003 split size drift")
    if any(r.get("split") != "train" for r in train):
        raise RuntimeError("TRAIN-003 training row split mismatch")
    if any(r.get("split") != "eval" for r in heldout):
        raise RuntimeError("TRAIN-003 held-out split mismatch")
    train_ids = [r["id"] for r in train]
    heldout_ids = [r["id"] for r in heldout]
    if len(set(train_ids)) != len(train_ids) or len(set(heldout_ids)) != len(heldout_ids):
        raise RuntimeError("Duplicate row identity")
    if set(train_ids) & set(heldout_ids):
        raise RuntimeError("Held-out row leaked into train")
    families_train = {r["family"] for r in train}
    families_eval = {r["family"] for r in heldout}
    if families_train & families_eval:
        raise RuntimeError("TRAIN-003 train/eval families overlap")
    if len(families_train) != manifest["train_family_count"] or len(families_eval) != manifest["heldout_family_count"]:
        raise RuntimeError("TRAIN-003 family count mismatch")
    labels = tuple(manifest["labels"])
    for split, rows in (("train", train), ("heldout", heldout)):
        counts = Counter(r["output"]["decision"] for r in rows)
        if set(counts) != set(labels):
            raise RuntimeError(f"{split} decision labels drift")
        if dict(counts) != manifest["label_counts"][split]:
            raise RuntimeError(f"{split} label count drift")
    return {
        "train_rows": len(train), "heldout_rows": len(heldout),
        "train_family_count": len(families_train), "heldout_family_count": len(families_eval),
        "id_overlap": 0, "family_overlap": 0,
    }


def fixed_schedule(count: int, steps: int, seed: int) -> list[int]:
    """Match the deterministic epoch shuffle convention of TRAIN-002b."""
    if min(count, steps) <= 0:
        raise ValueError("positive train count and steps required")
    schedule = []
    epoch = 0
    while len(schedule) < steps:
        indices = list(range(count))
        random.Random(seed + epoch * 1_000_003).shuffle(indices)
        schedule.extend(indices)
        epoch += 1
    return schedule[:steps]


def build_plan(protocol_path: Path = PROTOCOL, dataset_root: Path = DATASET_ROOT,
               root: Path = ROOT) -> dict:
    p = load_json(protocol_path)
    if p["train_authorized"] is not False:
        raise RuntimeError("Source-gated preflight refuses protocol with training enabled")
    if p["experiment"] != "NCR-001":
        raise RuntimeError("wrong experiment")
    if tuple(p["comparisons"]) != CONTROLS:
        raise RuntimeError("Comparator cohort changed from preregistration")
    if set(p["models"]) != {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"}:
        raise RuntimeError("four model identities required")
    expected_manifest = p["datasets"]["train_003"]["manifest_sha256"]
    manifest_path = dataset_root / "manifest.json"
    if digest(manifest_path) != expected_manifest:
        raise RuntimeError("TRAIN-003 manifest SHA-256 drift")
    m = load_json(manifest_path)
    train = [row for shard in m["training_shards"] for row in jsonl(dataset_root / shard)]
    heldout = jsonl(dataset_root / m["heldout_shard"])
    separation = validate_split(train, heldout, m)
    fixtures = {}
    for name, cfg in p["datasets"].items():
        if name == "train_003":
            continue
        expected = cfg.get("sha256")
        if not expected or not cfg.get("path"):
            raise RuntimeError(f"Unpinned evaluation fixture: {name}")
        path = root / cfg["path"]
        got = digest(path)
        if got != expected:
            raise RuntimeError(f"Evaluation fixture {name} SHA-256 drift")
        fixtures[name] = {"path": cfg["path"], "sha256": got, "used_for_training": False}
    # These are admission-critical preregistration gates. Python -O removes
    # assert statements, so use unconditional, type-strict runtime checks.
    a = p["architecture"]
    locked = (
        ("field scalar count", a["state_geometry"]["scalars_per_token"], 256),
        ("tap count", a["tap_count"], 3),
        ("steps per tap", a["nca_steps_per_tap"], 3),
        ("optimizer", p["training"]["optimizer"], "AdamW"),
        ("optimizer learning rate", p["training"]["optimizer_lr"], 0.0008),
        ("optimizer betas", p["training"]["optimizer_betas"], [0.9, 0.95]),
        ("weight decay", p["training"]["weight_decay"], 0.01),
        ("gradient clip", p["training"]["grad_clip_norm"], 1),
        ("MLP bottleneck", p["match_constraints"]["conventional_adapter_bottleneck"], 256),
    )
    for label, observed, expected in locked:
        if type(observed) is not type(expected) or observed != expected:
            raise RuntimeError(
                f"NCR-001F preregistered {label} drift: "
                f"expected {expected!r}, received {observed!r}"
            )
    # Model dictionary keys alone are insufficient: a substituted revision,
    # chat/root weight file, depth taps or token budget changes the experiment.
    # Check every lane against independent, frozen source-only identity values.
    for lane, expected in FROZEN_MODEL_LANES.items():
        model = p["models"][lane]
        for field in ("repo_id", "revision", "variant", "weight_file"):
            observed = model[field]
            if type(observed) is not str or observed != expected[field]:
                raise RuntimeError(f"NCR-001F frozen {lane} {field} drift")
        observed_taps = model["taps_after_block_zero_based"]
        if (type(observed_taps) is not list
                or len(observed_taps) != len(expected["taps"])
                or any(type(tap) is not int for tap in observed_taps)
                or tuple(observed_taps) != expected["taps"]):
            raise RuntimeError(f"NCR-001F frozen {lane} taps drift")
        config = model["expected_config"]
        if (type(config["model_type"]) is not str
                or config["model_type"] != expected["model_type"]):
            raise RuntimeError(f"NCR-001F frozen {lane} model_type drift")
        for field in ("hidden_size", "num_hidden_layers"):
            observed = config[field]
            if type(observed) is not int or observed != expected[field]:
                raise RuntimeError(f"NCR-001F frozen {lane} {field} drift")
        observed_budget = p["training"]["per_model_sequence_unit_limits"][lane]
        if type(observed_budget) is not int or observed_budget != expected["token_limit"]:
            raise RuntimeError(f"NCR-001F frozen {lane} token_limit drift")
    seeds = p["training"]["seed_list"]
    steps = p["training"]["primary_steps_per_seed"]
    # Identity AND order are preregistered, not just the number of seeds.
    # JSON true/false and numeric floats must not silently masquerade as ints.
    expected_seeds = (20261026, 20261027, 20261028)
    if (type(seeds) is not list or len(seeds) != len(expected_seeds)
            or any(type(seed) is not int for seed in seeds)
            or tuple(seeds) != expected_seeds):
        raise RuntimeError("NCR-001F preregistered seed list/order drift")
    if type(steps) is not int or steps != 192:
        raise RuntimeError("NCR-001F preregistered primary optimizer steps drift")
    schedules = {
        str(seed): {
            "seed": seed, "steps": steps,
            "train_schedule_indices_sha256": hashlib.sha256(
                json.dumps(fixed_schedule(len(train), steps, seed),
                           separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "used_heldout_examples": False,
        }
        for seed in seeds
    }
    lanes = {}
    for name, model in p["models"].items():
        width = model["expected_config"]["hidden_size"]
        nca = PersistentDepthNCA(width)
        check = matched_control_contract(
            width, nca_parameters=nca.trainable_parameters(),
            limit=p["match_constraints"]["adapter_trainable_param_tolerance_fraction"],
        )
        if nca.trainable_parameters() != 514 * width + 984:
            raise RuntimeError("NCA parameter formula drift")
        lanes[name] = {
            "model_repo": model["repo_id"],
            "model_revision": model["revision"],
            "model_variant": model["variant"],
            "weight_file": model["weight_file"],
            "taps": model["taps_after_block_zero_based"],
            "token_limit": p["training"]["per_model_sequence_unit_limits"][name],
            "token_limit_unit": "byte_tokenizer_unit" if name == "void1" else "subword_token",
            "parameter_contract": check,
            "controls": list(CONTROLS),
            "seeds": list(seeds),
            "training_allowed": False,
            "conformance_gate_pass": False,
        }
    return {
        "schema_version": 1,
        "experiment": "NCR-001F",
        "kind": "source_readback_only_no_training_or_model_load",
        "source_protocol_sha256": digest(protocol_path),
        "train003_manifest_sha256": expected_manifest,
        "dataset_validation": separation,
        "sealed_fixtures": fixtures,
        "optimizer": {k: p["training"][k] for k in
                      ("optimizer", "optimizer_lr", "optimizer_betas",
                       "weight_decay", "grad_clip_norm")},
        "schedules": schedules,
        "lanes": lanes,
        "training_authorized": False,
        "all_model_conformance_gate_pass": False,
        "status": "PASS_COMPARISON_PLAN_PRECHECK__BLOCKED_FOR_TRAINING",
        "explicit_blocks": [
            "Prereviewed source/fixture protocol train_authorized=false",
            "No model-specific independent NCR-001E four-model conformance",
            "No actual model-specific tokenizer-formatted training examples yet",
            "No real NCA vs MLP quality comparisons or trained checkpoints"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "runtime/ncr-001/matched-control-plan.json")
    args = parser.parse_args()
    result = build_plan()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "train/eval": result["dataset_validation"],
        "lanes": {name: row["parameter_contract"] for name, row in result["lanes"].items()},
        "training_authorized": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
