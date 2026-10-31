#!/usr/bin/env python3
"""Independent byte-exact readback of NCR-001E actual Sprocket process receipts.

Archive is an unchanged-byte copy of eight original JSON files from TWO
separate GitHub Actions artifacts. No model, tokenizer, datasets or training.
This proves receipt custody/consistency, not independent source-model testing.
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
ARCHIVE = ROOT / "experiments/ncr-001/evidence-review/sprocket-cross-process/original-jsons.tar.gz.b64"
PROVENANCE = ARCHIVE.with_name("provenance.json")
ARCHIVE_GZIP_SHA256 = "73544c842468d9bf5c3eae8d8ba92a253716990be514e3bc969e834d8595f3bc"
RUN_ID = "40515083104"
GITHUB_MERGE_SHA = "e6277a729613de21c260cd4f856d1ebc4578607a"
REVISION = "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
REPO = "kandivault/sprocket-500m"
PROTOCOL_SHA = "a76d38f80a57012ca8e62fba183d1cf2e91fd0986032f1adcdfb0aef843bf0b4"
PINS_SHA = "db794f53a49bb1bcbf8b23bd4a76d24430eb495da246639b506abdbbf11734b1"
TOKEN_IDS_SHA = "845ece67eb9e4930cdc9f5bd72c18c234f7b34ce48d9189151fecfa7ba69af82"
ORIGINALS = {
    "chat/producer.json": (3138, "7566132ab52fa6014ff6b9d538449c7c9426a0331944d81a09c5c7971690a584"),
    "chat/consumer.json": (3347, "164dd83ac9c35e9aa0a011a748b96fa903633020e8601babb56d669d196a4b72"),
    "chat/manifest.json": (1022, "b37db3afcc6465678d0d6897bff1f65e9354e452a32e1055eefa1ce2d32b8fd5"),
    "chat/adapter/metadata.json": (529, "5eb29186d69ae5a85170a3d4aa3f9e27003f63b7ea2b0e3e50b7aab726b0cb9d"),
    "instruct/producer.json": (3269, "8cce41f40f6ebaace977ce138b7871abe538bf8b6d663fc7519ad5dc7042b4f3"),
    "instruct/consumer.json": (3474, "fd6ed6c4d42c809eeff24aae67b812a95873e3ebcd15f4eaf608566947b8918f"),
    "instruct/manifest.json": (1021, "d0c4981051d72ca839b523886927fd2510c15f1b039abb18863bc56f2a0baa73"),
    "instruct/adapter/metadata.json": (533, "2dcbee771d4099c5de0cd381f7f2a86d028ae1f1c4914897a8d0c17eba67abfb"),
}
LANES = {
    "chat": {
        "weight_path": "chat/model.safetensors",
        "weight_sha": "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97",
        "source_hook": "83a56835a7476f17a17980d28dd1a7c14bb6f2f6",
        "artifact_id": 12739296072,
        "artifact_zip_sha": "6598b26dee5e774353a092e28265ee3cfd08f05673e3a38c17012da36caa5fb4",
        "runner": "GitHub Actions 1000045113",
        "pid_pair": [2474, 2489],
        "cache_max_abs": 3.814697265625e-05,
        "untrained_effect": 0.018445849418640137,
        "reference_sha": "4dde74edf20b0a0d6078a1d1a461e460c6804b84fa309549012c754714be6a7a",
        "adapter_sha": "189d578ee9f820323d0ca6e29366fc5d0423dfe2af9dc6deba54d731a34a391e",
    },
    "instruct": {
        "weight_path": "model.safetensors",
        "weight_sha": "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864",
        "source_hook": "b95c010f9b2ba3cbd9f81317981316bb9c8acd19",
        "artifact_id": 12740195021,
        "artifact_zip_sha": "5677cdf4e8974f83d5c0c68188311659b04bf282c1dcda08c6f597e6185d2084",
        "runner": "GitHub Actions 1000045114",
        "pid_pair": [2722, 2738],
        "cache_max_abs": 4.4345855712890625e-05,
        "untrained_effect": 0.033443450927734375,
        "reference_sha": "9f2949aa64f8ed9a147707fd8d11acee2c5b1628abe5c8df26a2176de6c54027",
        "adapter_sha": "189d578ee9f820323d0ca6e29366fc5d0423dfe2af9dc6deba54d731a34a391e",
    },
}
READBACK_STATUS = "PASS_ORIGINAL_SPROCKET_TWO_VARIANT_PROCESS_JSON_BYTES_AND_NUMERIC_SOURCE_READBACK__FINAL_72_BLOCKED"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(ok, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def unpack_original_jsons(encoded: str) -> dict[str, dict]:
    """Require exact committed compressed bytes and exact original JSON bytes."""
    raw = base64.b64decode(encoded.strip(), validate=True)
    require(sha(raw) == ARCHIVE_GZIP_SHA256,
            "Committed original JSON archive gzip SHA differs from verified source")
    decompressed = gzip.decompress(raw)
    require(len(decompressed) < 40000, "Original receipt archive unexpectedly large")
    records = {}
    with tarfile.open(fileobj=io.BytesIO(decompressed), mode="r:") as archive:
        members = archive.getmembers()
        require(len(members) == len(ORIGINALS), "Unexpected original receipt member count")
        require({m.name for m in members} == set(ORIGINALS),
                "Unexpected or missing original receipt path")
        for member in members:
            require(member.isfile() and not member.issym() and not member.islnk(),
                    "Original receipt must be an ordinary tar member")
            expected_length, expected_hash = ORIGINALS[member.name]
            require(member.size == expected_length,
                    "Original receipt byte count changed: " + member.name)
            raw_file = archive.extractfile(member).read()
            require(len(raw_file) == expected_length and sha(raw_file) == expected_hash,
                    "Original source JSON receipt SHA changed: " + member.name)
            records[member.name] = json.loads(raw_file)
    return records


def check_numeric_pair(variant: str, producer: dict, consumer: dict,
                       manifest: dict, metadata: dict, spec: dict) -> dict:
    """Detect source swaps, causal/cache failure, false training, false cold-restart."""
    require(manifest["schema"] == 1 and metadata["schema_version"] == 1,
            "Handoff schema changed")
    require(manifest["variant"] == metadata["model_variant"] == variant,
            "Chat and Instruct original variant/cold handoff collision")
    require(manifest["repo"] == metadata["model_id"] == REPO,
            "Exact source repository mismatch")
    require(manifest["revision"] == metadata["model_revision"] == REVISION,
            "Source revision changed")
    require(manifest["checkpoint_file"] == spec["weight_path"] and
            manifest["checkpoint_sha256"] == metadata["checkpoint_sha256"] == spec["weight_sha"],
            "Original full checkpoint hash/path/variant changed")
    require(manifest["implementation_commit"] == spec["source_hook"],
            "Original frozen hook implementation changed")
    require(manifest["protocol_sha256"] == PROTOCOL_SHA and
            manifest["pins_sha256"] == PINS_SHA and
            manifest["token_ids_sha256"] == TOKEN_IDS_SHA,
            "Readback found different protocol/pin/tokenized ordinary role")
    require(manifest["adapter_sha256"] == spec["adapter_sha"] and
            manifest["reference_sha256"] == spec["reference_sha"] and
            manifest["metadata_sha256"] == ORIGINALS[variant + "/adapter/metadata.json"][1],
            "Original handoff artifact fingerprint mismatch")
    require(metadata["hidden_size"] == 1280 and
            metadata["post_block_taps"] == [7, 16, 25],
            "Frozen source tap geometry changed")
    require(metadata["adapter"]["steps_per_tap"] == 3 and
            metadata["adapter"]["grid_size"] == 8 and
            metadata["adapter"]["channels"] == 4,
            "Frozen cellular geometry changed")
    for key in ("training", "heldout_evaluation", "tool_and_memory_behavior_qualified"):
        require(manifest[key] is False, "Original handoff has false qualification: " + key)
    require(manifest["optimizer_steps"] == 0, "Original handoff used an optimizer")
    require(producer["status"] == "PASS_SOURCE_BOUND_PROCESS_PRODUCER" and
            consumer["status"] == "PASS_FRESH_INTERPRETER_SPROCKET_ADAPTER_REPLAY",
            "Original actual-host outcome was not successful")
    require([producer["process_pid"], consumer["process_pid"]] == spec["pid_pair"] and
            manifest["producer_pid"] == producer["process_pid"] != consumer["process_pid"],
            "Two independent Python processes were not proven")
    require(producer["stage"] == "producer" and consumer["stage"] == "consumer",
            "Original run process role swapped")
    for report in (producer, consumer):
        require(report["variant"] == variant and report["experiment"] == "NCR-001E",
                "Original JSON source model/experiment identity mismatch")
        require(report["train_authorized"] is False and
                report["actual_model_training"] is False and
                report["optimizer_steps"] == 0 and
                report["heldout_evaluation"] is False and
                report["tool_and_memory_behavior_qualified"] is False and
                report["full_four_model_conformance_pass"] is False,
                "Original receipt falsely authorizes training/evaluation/conformance")
        require(report["adapter_parameters"] == 658904 and
                report["base_parameters"] == 501090560,
                "Original frozen 500M geometry/parameter counts mismatch")
        require(report["prompt_tokenizer_units"] == 256 and
                report["context_limit"] == 2048,
                "Unexpected number of actual model-source token units")
        require(report["token_ids_sha256"] == TOKEN_IDS_SHA and
                report["role_ids"] == {"<|user|>": [3], "<|end|>": [5], "<|assistant|>": [4]},
                "Source ordinary roles changed")
        require(report["runner"]["github_run_id"] == RUN_ID and
                report["runner"]["github_sha"] == GITHUB_MERGE_SHA and
                report["runner"]["runner_name"] == spec["runner"] and
                report["runner"]["cpu_count"] == 4 and
                report["runner"]["github_run_attempt"] == "1",
                "Original run/runner identity mismatch")
        require(report["torch"] == "2.14.1+cpu" and
                report["transformers"] == "4.57.6" and
                report["runner"]["python"] == "3.12.3",
                "Original frozen stack/runtime changed")
        require(report["source_files"][spec["weight_path"]]["bytes"] == 1002207984 and
                report["source_files"][spec["weight_path"]]["sha256"] == spec["weight_sha"],
                "Original full 1GB upstream model byte SHA mismatch")
        require(report["source_files"]["README.md"]["sha256"] ==
                "09aec879dc5b387bdc031dc3c74d77a9ba44362a55163f9a8b456761b68142c9",
                "Model card source changed")
        require(report["peak_rss_kib"] > 0,
                "Missing memory measurement")
    require(producer["manifest_sha256"] == consumer["source_manifest_sha256"] ==
            ORIGINALS[variant + "/manifest.json"][1],
            "Actual process manifests do not match byte-for-byte")
    require(consumer["source_reference_sha256"] == spec["reference_sha"] and
            consumer["source_adapter_sha256"] == spec["adapter_sha"],
            "Actual cold consumer did not bind same original saved logits/adapter bytes")
    name = "256_unit_240_prefill_16_incremental_tail"
    cache = producer["checks"][name]
    require(cache["pass"] is True and cache["atol"] == cache["rtol"] == 0.0005 and
            cache["max_abs"] == spec["cache_max_abs"] and
            cache["max_abs"] <= 0.0005,
            "Source runtime causal 240-prefill/16-step cache exceeds fixed tolerance")
    effect = producer["checks"]["manual_untrained_nonzero_effect"]
    require(effect["trained"] is False and effect["max_abs"] == spec["untrained_effect"]
            and effect["max_abs"] > 0, "No true untrained frozen-host residual effect")
    for name in ("future_suffix_strict_255_units", "request_A_B_A_producer"):
        check = producer["checks"][name]
        require(check["pass"] is True and check["max_abs"] == 0 and
                check["exact_required"] is True, "Source had future-token or request-state leak")
    for name in ("restart_active_tail_exact", "restart_disabled_tail_exact",
                 "restart_request_A_exact", "restart_after_B_no_leak_exact"):
        check = consumer["checks"][name]
        require(check["pass"] is True and check["max_abs"] == 0 and
                check["exact_required"] is True,
                "Source restart logits were not exactly reproducible: " + name)
    return {
        "variant": variant, "actual_weight_sha256": spec["weight_sha"],
        "original_hook_commit": spec["source_hook"],
        "artifact_id": spec["artifact_id"],
        "artifact_zip_sha256": spec["artifact_zip_sha"],
        "producer_pid": producer["process_pid"],
        "consumer_pid": consumer["process_pid"],
        "cache_max_abs": cache["max_abs"],
        "untrained_effect_max_abs": effect["max_abs"],
        "strict_future_and_request_drift": 0,
        "cross_process_float32_restart_drift": 0,
        "separate_python_process_same_runner": True,
        "tool_and_memory_semantics_qualified": False,
        "final_conformance_admitted": False,
        "training_authorized": False,
        "optimizer_steps": 0,
    }


def audit(records: dict[str, dict] | None = None) -> dict:
    if records is None:
        records = unpack_original_jsons(ARCHIVE.read_text(encoding="ascii"))
    require(set(records) == set(ORIGINALS), "Eight original source JSON members required")
    results = {}
    for variant, spec in LANES.items():
        results[variant] = check_numeric_pair(
            variant, records[variant + "/producer.json"],
            records[variant + "/consumer.json"],
            records[variant + "/manifest.json"],
            records[variant + "/adapter/metadata.json"], spec
        )
    require(results["chat"]["actual_weight_sha256"] !=
            results["instruct"]["actual_weight_sha256"],
            "Chat and root Instruct source checkpoints must differ")
    if PROVENANCE.is_file():
        provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        require(provenance["original_run_id"] == int(RUN_ID) and
                provenance["run_checkout_merge_sha"] == GITHUB_MERGE_SHA and
                provenance["gzip_sha256"] == ARCHIVE_GZIP_SHA256,
                "Committed source-run provenance drift")
        for variant, spec in LANES.items():
            evidence = provenance["variants"][variant]
            require(evidence["artifact_id"] == spec["artifact_id"] and
                    evidence["zip_sha256"] == spec["artifact_zip_sha"] and
                    evidence["weight_sha256"] == spec["weight_sha"] and
                    evidence["source_hook_sha"] == spec["source_hook"],
                    "Artifact/checkpoint source provenance mismatch")
    return {
        "status": READBACK_STATUS,
        "source_run_id": int(RUN_ID),
        "original_archive_gzip_sha256": ARCHIVE_GZIP_SHA256,
        "original_json_count": 8,
        "source_integrity_readback_pass": True,
        "actual_host_execution_source": "original_GitHub_Actions_receipts",
        "this_auditor_executes_llms": False,
        "tool_and_memory_behavior_qualified": False,
        "four_model_conformance_pass": False,
        "train_authorized": False,
        "original_variants": results,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
