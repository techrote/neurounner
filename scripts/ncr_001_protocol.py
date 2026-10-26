#!/usr/bin/env python3
"""NCR-001A source identity and protocol preflight. Never trains or downloads weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def taps_for_layers(layers: int) -> list[int]:
    if not isinstance(layers, int) or layers < 6:
        raise ValueError("Need at least six transformer layers for three distinct depth taps")
    taps = [layers // 3 - 1, 2 * layers // 3 - 1, layers - 1]
    if len(set(taps)) != 3 or taps != sorted(taps):
        raise ValueError("Invalid depth tap selection")
    return taps


def validate() -> dict:
    p = json_load(PROTOCOL)
    assert p["schema_version"] == 1 and p["experiment"] == "NCR-001"
    assert p["parent_issue"] == 67 and p["protocol_issue"] == 68
    assert p["state"] == "provisional_four_model_sources_and_holdouts_pending_lock"
    assert p["train_authorized"] is False, "NCR-001A must never permit training"
    a = p["architecture"]
    assert a["state_geometry"] == {
        "height": 8, "width": 8, "channels": 4, "scalars_per_token": 256
    }
    assert a["tap_count"] == 3 and a["nca_steps_per_tap"] == 3
    assert not a["cross_token_recurrence"] and not a["dynamic_sparse_execution"]
    assert p["training"]["trainable_backbone"] is False
    assert p["training"]["initial_smoke_max_steps"] <= 32
    assert 1 <= p["training"]["primary_steps_per_seed"] <= 192
    assert p["resource_watchdogs"]["max_peak_rss_gib"] <= 12
    assert len(set(p["training"]["seed_list"])) == 3
    assert len(p["comparisons"]) == len(set(p["comparisons"])) == 6
    assert p["match_constraints"]["adapter_trainable_param_tolerance_fraction"] <= 0.05

    models = p["models"]
    assert set(models) == {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"}
    assert models["smollm2"]["repo_id"] == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert models["sprocket_chat"]["repo_id"] == "kandivault/sprocket-500m"
    assert models["sprocket_chat"]["weight_file"] == "chat/model.safetensors"
    assert models["sprocket_chat"]["variant"] == "chat"
    assert models["smollm2"]["taps_after_block_zero_based"] == taps_for_layers(
        models["smollm2"]["expected_config"]["num_hidden_layers"]
    )
    assert models["sprocket_chat"]["taps_after_block_zero_based"] == taps_for_layers(
        models["sprocket_chat"]["expected_config"]["num_hidden_layers"]
    )
    assert models["sprocket_instruct"]["variant"] == "instruct"
    assert models["sprocket_instruct"]["weight_file"] == "model.safetensors"
    assert models["sprocket_instruct"]["taps_after_block_zero_based"] == [7, 16, 25]
    assert models["void1"]["repo_id"] == "dotlabs/void.1"
    assert models["void1"]["weight_file"] == "model.safetensors"
    assert models["void1"]["taps_after_block_zero_based"] == [7, 15, 23]
    for name, m in models.items():
        assert len(m["revision"]) == 40
        int(m["revision"], 16)
        if name == "void1":
            assert m["expected_config"]["model_type"] == "qwen3"
            assert m["expected_config"]["architectures"] == ["Qwen3ForCausalLM"]
            assert "tokenization_byte.py" in m["tokenizer_candidates"]
            assert "byte_vocab.json" in m["tokenizer_candidates"]
        else:
            assert m["expected_config"]["model_type"] == "llama"
            assert m["expected_config"]["architectures"] == ["LlamaForCausalLM"]
        assert m["taps_after_block_zero_based"] == taps_for_layers(
            m["expected_config"]["num_hidden_layers"]
        )

    # Source inventory is retained in Git; no ephemeral Actions artifact is authority.
    pins_path = ROOT / "experiments/ncr-001/source-pins.json"
    if not pins_path.is_file():
        raise AssertionError("NCR-001 source pins missing from durable repository")
    pins = json_load(pins_path)
    assert pins["schema_version"] == 1 and pins["experiment"] == "NCR-001"
    assert pins["source"]["model_weights_downloaded"] is False
    assert pins["source"]["training_performed"] is False
    source_pending = []
    assert set(pins["models"]) == set(models)
    for name, model in models.items():
        source = pins["models"][name]
        assert source["repo_id"] == model["repo_id"]
        assert source["revision"] == model["revision"]
        assert source["variant"] == model["variant"]
        assert source["hf_weight_file"] == model["weight_file"]
        if source["weight_present"] is not True:
            # An unverified new model may be drafted but can never be admitted.
            assert source["weight_present"] is None and name in ("sprocket_instruct", "void1")
            source_pending.append(f"{name}: pinned HF checkpoint metadata not yet verified")
        for source_key, expected_key in (
                ("layers", "num_hidden_layers"),
                ("hidden", "hidden_size"),
                ("context", "max_position_embeddings"),
                ("vocab", "vocab_size")):
            assert source["architecture"][source_key] == model["expected_config"][expected_key]
        assert source["tap_indices_post_block_zero_based"] == model["taps_after_block_zero_based"]
        declared_hash = source["weight_declared_lfs_sha256_not_downloaded"]
        if declared_hash is None:
            assert name in ("sprocket_instruct", "void1")
            source_pending.append(f"{name}: declared weight checksum not yet inventoried")
        else:
            assert isinstance(declared_hash, str) and len(declared_hash) == 64
            int(declared_hash, 16)

    dataset = p["datasets"]["train_003"]
    manifest_path = ROOT / dataset["manifest_path"]
    assert manifest_path.is_file(), "Pinned TRAIN-003 dataset not present"
    manifest_sha = digest(manifest_path)
    assert manifest_sha == dataset["manifest_sha256"], (
        f"TRAIN-003 manifest drift: expected {dataset['manifest_sha256']}, got {manifest_sha}"
    )
    manifest = json_load(manifest_path)
    assert (manifest["train_examples"], manifest["heldout_examples"]) == (768, 128)
    assert (manifest["train_family_count"], manifest["heldout_family_count"]) == (24, 4)
    assert manifest["family_overlap"] == []
    assert len(manifest["training_shards"]) == 8
    for shard in manifest["training_shards"] + [manifest["heldout_shard"]]:
        assert (manifest_path.parent / shard).is_file(), f"Missing pinned shard: {shard}"

    unresolved = list(source_pending)
    for fixture_key in ("natural_language_holdout", "sprocket_chat_probes",
                        "sprocket_instruct_probes", "void_byte_probes",
                        "other_original_behaviour_probes"):
        record = p["datasets"][fixture_key]
        if not record.get("sha256"):
            unresolved.append(f"{fixture_key}: no approved SHA256")
        if record.get("path") and record.get("sha256"):
            assert digest(ROOT / record["path"]) == record["sha256"], (
                f"Fixture drift: {fixture_key}"
            )
    for field in ("optimizer_lr", "optimizer_betas", "weight_decay"):
        if p["training"][field] is None:
            unresolved.append(f"training.{field}: not frozen")
    unresolved.append("both model checkpoint byte digests: not yet verified")
    unresolved.append("Full model checkpoint bytes and forward parity: pending for all four models")
    unresolved.append("Sprocket instruct tool-token and void.1 byte tokenizer semantics: not yet verified")
    if models["sprocket_chat"]["taps_after_block_zero_based"] is None:
        unresolved.append("Sprocket exact post-block tap indices: not frozen")
    return {
        "status": "PASS_WITH_EXPLICIT_TRAINING_BLOCKERS",
        "experiment": p["experiment"],
        "protocol_sha256": digest(PROTOCOL),
        "train003_manifest_sha256": manifest_sha,
        "source_pins_sha256": digest(pins_path),
        "smollm2_taps": models["smollm2"]["taps_after_block_zero_based"],
        "training_authorized": False,
        "unresolved": unresolved,
    }


def _sibling_meta(sibling) -> dict:
    meta = {
        "path": sibling.rfilename,
        "size": getattr(sibling, "size", None),
        "blob_id": getattr(sibling, "blob_id", None),
    }
    lfs = getattr(sibling, "lfs", None)
    if isinstance(lfs, dict):
        meta["lfs_sha256_declared_not_downloaded"] = lfs.get("sha256")
    elif lfs:
        meta["lfs_sha256_declared_not_downloaded"] = getattr(lfs, "sha256", None)
    return meta


def _source_preflight(name: str, m: dict) -> dict:
    # Import only for online mode; local protocol validation needs no dependencies.
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(m["repo_id"], revision=m["revision"],
                              files_metadata=True)
    if info.sha != m["revision"]:
        raise ValueError(f"Unpinned or moved source: {info.sha} != {m['revision']}")
    siblings = {s.rfilename: s for s in info.siblings}
    weight = siblings.get(m["weight_file"])
    response = {
        "repo_id": m["repo_id"], "revision": info.sha,
        "requested_variant": m["variant"],
        "requested_trainable_file": m["weight_file"],
        "weight_file_present": weight is not None,
        "weight_metadata": _sibling_meta(weight) if weight is not None else None,
        "known_reference_gguf": [
            _sibling_meta(siblings[x]) for x in m.get("reference_only_gguf", [])
            if x in siblings
        ],
        "root_instruct_safetensors_present": (
            "model.safetensors" in siblings if name == "sprocket_chat" else None
        ),
        "metadata_sha256": {},
        "configuration": None,
        "tap_indices": None,
        "blockers": [],
    }

    config_path = m["config_file"]
    if config_path not in siblings:
        if m.get("optional_architecture_config_file") in siblings:
            config_path = m["optional_architecture_config_file"]
            response["blockers"].append(
                "Requested chat-specific config absent; using root config for "
                "architecture inspection only, not authorizing chat-source parity."
            )
        else:
            response["blockers"].append("No configuration found for requested variant")
    if config_path in siblings:
        local = hf_hub_download(
            repo_id=m["repo_id"], revision=info.sha, filename=config_path
        )
        config = json_load(Path(local))
        response["configuration"] = {
            "source_path": config_path,
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "max_position_embeddings": config.get("max_position_embeddings"),
            "vocab_size": config.get("vocab_size"),
        }
        response["metadata_sha256"][config_path] = digest(Path(local))
        for key, expected in m["expected_config"].items():
            if config.get(key) != expected:
                response["blockers"].append(
                    f"Source config mismatch: {key}={config.get(key)!r}, "
                    f"expected={expected!r}"
                )
        layers = config.get("num_hidden_layers")
        if isinstance(layers, int) and layers >= 6:
            response["tap_indices"] = taps_for_layers(layers)
        else:
            response["blockers"].append("No valid layer count")

    # Only download small config/tokenizer metadata, never model weights in #68.
    for key in m["tokenizer_candidates"]:
        if key not in siblings:
            continue
        if (getattr(siblings[key], "size", None) or 0) > 12_000_000:
            response["blockers"].append(f"Tokenizer metadata over 12 MB guard: {key}")
            continue
        local = hf_hub_download(repo_id=m["repo_id"], revision=info.sha, filename=key)
        response["metadata_sha256"][key] = digest(Path(local))
        if key.endswith("tokenizer_config.json"):
            tk = json_load(Path(local))
            response["tokenizer_config"] = {
                "source_path": key,
                "has_chat_template": isinstance(tk.get("chat_template"), str),
                "chat_template_sha256": (
                    hashlib.sha256(tk["chat_template"].encode()).hexdigest()
                    if isinstance(tk.get("chat_template"), str) else None
                )
            }

    if name == "void1":
        for required in ("tokenization_byte.py", "byte_vocab.json", "tokenizer_config.json"):
            if required not in response["metadata_sha256"]:
                response["blockers"].append(f"Missing independently hashed void byte-tokenizer file: {required}")
        response["tokenizer_code_executed"] = False
        response["tokenizer_remote_code_review_required"] = True
    elif not any(k.endswith("tokenizer.json") for k in response["metadata_sha256"]):
        response["blockers"].append("No independently hashed tokenizer.json")
    if not weight:
        response["blockers"].append(
            "Missing requested trainable HF checkpoint. Full-precision chat GGUF "
            "is not a drop-in differentiable HF chat checkpoint; no instruct fallback."
        )
    if name == "sprocket_chat":
        template_known = response.get("tokenizer_config", {}).get(
            "has_chat_template", False
        )
        if not template_known:
            response["blockers"].append(
                "Sprocket chat-template equivalence requires tokenizer/format review"
            )
    if name == "sprocket_chat" and m["taps_after_block_zero_based"] is None:
        response["blockers"].append(
            "Derived Sprocket tap indices must be frozen in protocol after review"
        )
    if response["tap_indices"] is not None and m["taps_after_block_zero_based"] is not None:
        if response["tap_indices"] != m["taps_after_block_zero_based"]:
            response["blockers"].append("Pinned tap indices do not match actual model")

    response["status"] = (
        "METADATA_PASS_FULL_WEIGHT_HASH_PENDING" if not response["blockers"]
        else "BLOCKED_REQUIRES_SOURCE_REVIEW"
    )
    return response


def online_preflight(out: Path) -> dict:
    import platform
    protocol = json_load(PROTOCOL)
    report = {
        "experiment": "NCR-001",
        "source": "read-only HF pinned metadata preflight",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": digest(PROTOCOL),
        "python_version": platform.python_version(),
        "weight_bytes_downloaded": 0,
        "training_performed": False,
        "models": {},
    }
    for name, model in protocol["models"].items():
        try:
            report["models"][name] = _source_preflight(name, model)
        except Exception as exc:
            report["models"][name] = {
                "status": "BLOCKED_SOURCE_PREFLIGHT_ERROR",
                "repo_id": model["repo_id"],
                "revision": model["revision"],
                "blockers": [f"{type(exc).__name__}: {exc}"],
            }
        item = report["models"][name]
        print(f"{name}: {item['status']}")
        for why in item.get("blockers", []):
            print(f"  BLOCKED: {why}")
    report["training_authorized"] = False
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "preflight"))
    parser.add_argument("--out", type=Path,
                        default=ROOT / "runtime/ncr-001/source-preflight.json")
    args = parser.parse_args()
    print(json.dumps(validate(), indent=2, sort_keys=True))
    if args.mode == "preflight":
        online_preflight(args.out)
        print(f"Source preflight report: {args.out} (not a training qualification)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
