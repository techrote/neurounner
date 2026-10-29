#!/usr/bin/env python3
"""NCR-001F pinned Sprocket adapter MAC proxy; not measured FLOPs or training."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOBS = {
    "experiments/ncr-001/protocol.json": "14ce5fbaa38f8806c8c8738f19c01b91385eebaf",
    "experiments/ncr-001/source-pins.json": "59fff1aed872b5563aa6898171983973e0cdb02c",
    "scripts/ncr_001_core.py": "0427ad0b0f1fc68199369d5094952017bc70f503",
    "scripts/ncr_001f_controls.py": "b4d486c6d9f8c8d37fd2205b0381db9ef4ae9c22",
    "scripts/ncr_001f_frozen_hooks.py": "e4129ebcdfa31c3be218473cef45a0d9a4a6bd88",
}
SOURCES = {
    "sprocket_chat": ("chat", "chat/model.safetensors",
      "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97"),
    "sprocket_instruct": ("instruct", "model.safetensors",
      "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864"),
}
REPO = "kandivault/sprocket-500m"
REVISION = "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
STATUS = "SOURCE_MAC_PROXY_QUANTIFIED__ACTUAL_COMPUTE_MATCH_UNMEASURED__TRAINING_BLOCKED"


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError("NCR001F_MAC_BUDGET_REFUSED: " + why)


def gitblob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def check_blobs(root: Path) -> None:
    for name, expected in BLOBS.items():
        file = root / name
        require(file.is_file() and gitblob(file.read_bytes()) == expected,
                "pinned original Git blob missing or changed: " + name)


def mac_proxy(hidden: int, *, taps: int = 3, steps: int = 3,
              grid: int = 8, channels: int = 4, bottleneck: int = 256) -> dict:
    """Count dense projection/Conv2d MACs only; excludes all other operations."""
    for v in (hidden, taps, steps, grid, channels, bottleneck):
        require(type(v) is int and v > 0, "invalid static architecture integer")
    field = grid * grid * channels
    conv = grid * grid * ((channels * 4) * channels * 9 +
                          (channels * 2) * (channels * 4))
    mlp = taps * 2 * hidden * bottleneck
    nca = taps * (2 * hidden * field + steps * conv)
    return {
        "mlp_dense_mac_per_token": mlp,
        "nca_dense_conv_mac_per_token": nca,
        "nca_conv_mac_per_token": taps * steps * conv,
        "nca_minus_mlp_mac_per_token": nca - mlp,
        "nca_over_mlp_mac_ratio": round(nca / mlp, 8),
        "nca_trainable_parameter_formula": 514 * hidden + 984,
        "mlp_trainable_parameter_formula": 514 * hidden + 512,
        "nca_field_float32_bytes_per_token_one_field_only": field * 4,
    }


def check_contract(p: dict, q: dict) -> dict:
    require(p["experiment"] == q["experiment"] == "NCR-001"
            and p["train_authorized"] is False
            and set(p["models"]) == set(q["models"]) ==
                {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"},
            "four frozen model lanes or training closure changed")
    a = p["architecture"]
    require(a["state_geometry"] == {"height": 8, "width": 8, "channels": 4,
                                   "scalars_per_token": 256}
            and a["tap_count"] == a["nca_steps_per_tap"] == 3
            and a["trainable_parameters_formula"] ==
                "514*hidden_size+984 for C=4/grid=8",
            "original cellular geometry or update iterations changed")
    c = p["match_constraints"]
    require(c["conventional_adapter_bottleneck"] == 256
            and c["adapter_trainable_param_tolerance_fraction"] == 0.05,
            "original parameter matching policy changed")
    lanes = {}
    for lane, (variant, path, digest) in SOURCES.items():
        model, pin = p["models"][lane], q["models"][lane]
        require(model["repo_id"] == pin["repo_id"] == REPO
                and model["revision"] == pin["revision"] == REVISION
                and model["variant"] == pin["variant"] == variant
                and model["weight_file"] == pin["hf_weight_file"] == path
                and pin["weight_declared_lfs_sha256_not_downloaded"] == digest
                and model["expected_config"]["hidden_size"] ==
                    pin["architecture"]["hidden"] == 1280
                and model["expected_config"]["num_hidden_layers"] ==
                    pin["architecture"]["layers"] == 26
                and model["taps_after_block_zero_based"] ==
                    pin["tap_indices_post_block_zero_based"] == [7, 16, 25],
                "variant/source or actual post-block taps changed: " + lane)
        proxy = mac_proxy(1280)
        mismatch = abs(proxy["nca_trainable_parameter_formula"] -
                       proxy["mlp_trainable_parameter_formula"])
        require(mismatch == 472 and
                mismatch / proxy["nca_trainable_parameter_formula"] <= 0.05
                and proxy["nca_conv_mac_per_token"] == 405504
                and proxy["nca_over_mlp_mac_ratio"] == 1.20625,
                "source proxy formula or preregistered parameter bound changed")
        lanes[lane] = {
            "variant": variant, "weight_path": path, "weight_sha256": digest,
            "source_revision": REVISION, "post_block_taps": [7, 16, 25],
            "parameter_match_within_five_percent": True,
            "parameter_delta": mismatch, "proxy": proxy,
            "actual_measured_flops": False,
            "actual_cpu_compute_match_established": False,
            "scientific_lane_pass": False,
        }
    require(lanes["sprocket_chat"]["weight_sha256"] !=
            lanes["sprocket_instruct"]["weight_sha256"],
            "Chat and root Instruct checkpoint identities collapsed")
    return lanes


def audit(root: Path = ROOT) -> dict:
    check_blobs(root)
    p = json.loads((root / "experiments/ncr-001/protocol.json").read_bytes())
    q = json.loads((root / "experiments/ncr-001/source-pins.json").read_bytes())
    return {
        "status": STATUS, "experiment": "NCR-001F", "issue": 73,
        "source_git_blobs": BLOBS, "lanes": check_contract(p, q),
        "scope": "adapter_dense_and_conv_mac_proxy_only_not_measured_flops",
        "actual_compute_budget_qualified": False,
        "train_authorized": False, "full_72_conformance_pass": False,
        "optimizer_steps": 0, "comparative_training": False,
        "new_model_execution": False, "heldouts_opened": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = json.dumps(audit(), indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result, encoding="utf-8")
    print(result, end="")
