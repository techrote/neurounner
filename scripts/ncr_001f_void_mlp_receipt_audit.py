#!/usr/bin/env python3
"""Read back immutable original NCR-001F frozen void.1 MLP Actions receipts.

Python standard library only. This is evidence custody and falsification, NOT a
fresh model invocation or scientific independent #72/training acceptance.
"""
from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/ncr-001/controls-evidence/real-void1-mlp"
PROVENANCE = EVIDENCE / "provenance.json"
ARCHIVE = EVIDENCE / "original-jsons.tar.gz.b64"
EXPECTED_RUN = "40523446884"
EXPECTED_SHA = "35c4eb966f12965c7640d1fd8a0c6d896b075791"
EXPECTED_MODEL_SHA = "ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3"
EXPECTED_TOKENIZER_SHA = "279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03"
EXPECTED_REVIEW = "1fdc2e7e5c7b7b77923dbc4b47dcafa132aef47a"
EXPECTED_CLASSIFICATION = "ACTUAL_FROZEN_VOID1_MLP_SOURCE_PREPASS_PASS__FULL_72_GATE_BLOCKED"
PROVENANCE_GIT_BLOB = "6d1f6f8aaca641501e57762225c22b3a9d87a33d"
ARCHIVE_GIT_BLOB = "2bc9f809206507f893dacf1940748e630502a958"


def require(predicate: bool, why: str) -> None:
    if not predicate:
        raise ValueError(why)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def gitblob(raw: bytes) -> str:
    """Hash exact committed raw bytes, including otherwise-ignored whitespace."""
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\x00" + raw).hexdigest()


def read_original_bytes(encoded: bytes, manifest: dict) -> dict[str, bytes]:
    raw = base64.b64decode(encoded, validate=True)
    archive = manifest["archive"]
    require(len(raw) == archive["original_jsons_tar_gz_bytes"], "original gzip byte length")
    require(sha(raw) == archive["original_jsons_tar_gz_sha256"], "altered original gzip SHA")
    require(gzip.decompress(raw)[:4] is not None, "gzip source decode")
    parts = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as bundle:
        members = bundle.getmembers()
        require({m.name for m in members} == set(archive["originals"]),
                "missing or injected original member")
        require(len(members) == len(archive["originals"]) and all(m.isfile() for m in members),
                "archive includes nonordinary members")
        for m in members:
            stream = bundle.extractfile(m)
            require(stream is not None, "missing archive stream")
            data = stream.read()
            expected = archive["originals"][m.name]
            require(len(data) == expected["bytes"], "altered original JSON byte length " + m.name)
            require(sha(data) == expected["sha256"], "altered original JSON SHA " + m.name)
            parts[m.name] = data
    return parts


def validate_receipts(prefetch: dict, host: dict, prov: dict) -> dict:
    require(prov["classification"] == EXPECTED_CLASSIFICATION,
            "provenance classification cannot promote #72 or training")
    require(prov["training_authorized"] is False
            and prov["optimizer_steps"] == 0
            and prov["heldout_data_opened"] is False
            and prov["new_interpreter_cold_reload"] is False
            and prov["final_ncr_001e_admission"] is False,
            "provenance overstates qualification or training")
    require(prov["source_run"]["run_id"] == int(EXPECTED_RUN)
            and prov["source_run"]["checkout_sha"] == EXPECTED_SHA
            and prov["source_run"]["artifact_id"] == 12741959402
            and prov["source_run"]["real_host_job_id"] == 122256966798
            and prov["source_run"]["artifact_zip_sha256"]
                == "8e780dc30262d139fccb7c7d00962ab5b702280ef96fd76575f57c41d1435de0",
            "original Actions provenance changed")
    model = prov["model"]
    require(model["repo"] == "dotlabs/void.1"
            and model["variant"] == "base_byte"
            and model["revision"] == "9c2be5087772867d45961abd11f6b35110617563"
            and model["checkpoint_path"] == "model.safetensors"
            and model["checkpoint_sha256"] == EXPECTED_MODEL_SHA
            and model["byte_tokenizer_code_sha256"] == EXPECTED_TOKENIZER_SHA
            and model["review_authority_commit"] == EXPECTED_REVIEW,
            "variant/revision/tokenizer source authority changed")
    require(prefetch["status"] == "PASS_ALL_SOURCE_BYTES"
            and prefetch["github_run_id"] == EXPECTED_RUN
            and prefetch["github_sha"] == EXPECTED_SHA
            and prefetch["optimizer_steps"] == 0
            and prefetch["train_authorized"] is False
            and prefetch["training_performed"] is False,
            "original source-prefetch state changed")
    expected_bytes = {
        "model.safetensors": (360622744, EXPECTED_MODEL_SHA),
        "tokenization_byte.py": (4458, EXPECTED_TOKENIZER_SHA),
        "config.json": (1284, "cbc72137da3dae65085a2a8c39ad7634d16bf92d785ca00d283ee82fe32b37e7"),
        "byte_vocab.json": (4292, "57a1cd2e70dd5b7cc6fd2eaa12ebfcd173c1620fa5d45c2c6282511c29b6a5f6"),
        "tokenizer_config.json": (932, "0ecbd4fd39e35125800d806af981ba0cacd8a7ba5f02c9e729c4ecc54af6d300"),
        "special_tokens_map.json": (75, "20d4b94d0e657c3f44d9123522c106e7315b6c1a1b6d259478f17bd2918b1263"),
    }
    for name, (size, hashvalue) in expected_bytes.items():
        require(prefetch["source_files"][name]["bytes"] == size
                and prefetch["source_files"][name]["sha256"] == hashvalue
                and prefetch["source_files"][name]["expected"] == hashvalue,
                "original prefetch SHA mismatch " + name)
        require(host["source_files"][name]["bytes"] == size
                and host["source_files"][name]["sha256"] == hashvalue
                and host["source_files"][name]["expected"] == hashvalue,
                "original frozen host SHA mismatch " + name)
    require(host["status"] == "PASS_SOURCE_BOUND_VOID1_UTF8_MLP_REAL_MODEL_PREPASS__TRAINING_BLOCKED",
            "incorrect original host verdict")
    require(host["experiment"] == "NCR-001F"
            and host["train_authorized"] is False
            and host["model_training_performed"] is False
            and host["comparative_training_performed"] is False
            and host["optimizer_steps"] == 0
            and host["heldout_data_opened"] is False
            and host["final_ncr_001e_admission"] is False
            and host["four_model_mlp_qualification"] is False
            and host["new_interpreter_cold_reload"] is False
            and host["eos_quality_measured"] is False,
            "false final qualification, trained comparison or EOS claim")
    source = host["source"]
    require(source["model_repo"] == model["repo"]
            and source["revision"] == model["revision"]
            and source["model_weight_sha256"] == EXPECTED_MODEL_SHA
            and source["byte_reviewer_commit"] == EXPECTED_REVIEW
            and len(source["hook_sha256"]) == 64
            and len(source["runner_sha256"]) == 64
            and len(source["protocol_sha256"]) == 64,
            "host source provenance changed")
    runner = host["runner"]
    require(runner["run_id"] == EXPECTED_RUN
            and runner["github_sha"] == EXPECTED_SHA
            and runner["job"] == "original-frozen-qwen3-mlp"
            and runner["attempt"] == "1"
            and runner["torch"] == "2.14.1+cpu"
            and runner["transformers"] == "4.57.6"
            and runner["cpu_threads"] == 4
            and runner["cpu_model"] == "AMD EPYC 9V74 80-Core Processor",
            "wrong original runner/source runtime identity")
    tok = host["tokenizer"]
    require(tok["actual_source_reviewed"] is True
            and tok["byte_source_sha256"] == EXPECTED_TOKENIZER_SHA
            and (tok["pad_id"], tok["bos_id"], tok["eos_id"]) == (256, 257, 258)
            and tok["literal_eos_ids"] == [60, 101, 111, 115, 62]
            and tok["automatic_eos_insertion"] is False,
            "unreviewed tokenizer or inaccurate terminal EOS semantics")
    require(host["frozen_base_parameters"] == 90148352
            and host["mlp_parameters"] == 329472
            and host["nca_parameters_reference"] == 329944
            and host["loaded_original_base_sha256_before"]
                == host["loaded_original_base_sha256_after"]
            and host["loaded_original_base_sha256_before"]
                == "0fedbc1b79f9dd0e781552dcb214a2d3c18fa943a665af337971ef108d2468a3",
            "changed base weights or parameter-matching claim")
    require(host["input_lengths"] == {
        "characters": 1170, "tokenizer_units": 1296, "utf8_bytes": 1296
    }, "UTF-8 byte/char/tokenizer-unit conflation")
    checks = host["checks"]
    for key in ("zero_mlp_UTF8_16", "zero_mlp_UTF8_64", "zero_mlp_UTF8_128",
                "zero_mlp_UTF8_masked_left", "zero_mlp_UTF8_masked_right",
                "strict_causal_future_suffix", "request_A_B_A_no_cross_request_state",
                "new_adapter_same_original_model_exact"):
        d = checks[key]
        require(d["close"] is True and d["exact"] is True
                and d["exact_required"] is True and d["max_abs"] == 0.0,
                "wrong exact model/causal/request/cold check " + key)
    cached = checks["active_48_UTF8_full_vs_true_incremental_KV"]
    require(cached["close"] is True
            and cached["shape"] == [1, 48, 259]
            and cached["atol"] == cached["rtol"] == 0.0005
            and 0 <= cached["max_abs"] <= 0.0005,
            "full/incremental cache condition relaxed")
    require(checks["manual_untrained_effect"]["trained"] is False
            and checks["manual_untrained_effect"]["max_abs"] > 0,
            "inactive MLP or false learned readout")
    require(checks["readout_gradient_only_zero_steps"]["base_gradients_none"] is True
            and checks["readout_gradient_only_zero_steps"]["optimizer_steps"] == 0
            and checks["readout_gradient_only_zero_steps"]["finite_loss"] is True
            and checks["readout_gradient_only_zero_steps"]["l1"] > 0,
            "base gradients or missing matched MLP readout gradient")
    require(checks["frozen_base_all_bytes_unchanged"] is True,
            "base not verified immutable")
    for key in ("lane", "model_revision", "model_type",
                "verified_weight_sha256", "weight_path"):
        require(checks["reject_wrong_metadata_" + key] is True,
                "adapter accepted unqualified metadata " + key)
    gen = checks["zero_mlp_greedy_byte_units"]
    require(gen["exact"] is True and gen["eos_id"] == 258
            and gen["generated_units"] == 4
            and gen["eos_reached"] is False
            and gen["eos_quality_measured"] is False,
            "unsupported byte/EOS generation result")
    return {
        "classification": prov["classification"],
        "model_sha256": EXPECTED_MODEL_SHA,
        "original_run": EXPECTED_RUN,
        "cached_max_abs": cached["max_abs"],
        "frozen_base_unchanged": True,
        "optimizer_steps": 0,
        "training_admitted": False,
    }


def audit(evidence: Path = EVIDENCE) -> dict:
    provenance_raw = (evidence / PROVENANCE.name).read_bytes()
    archive_raw = (evidence / ARCHIVE.name).read_bytes()
    require(gitblob(provenance_raw) == PROVENANCE_GIT_BLOB,
            "committed provenance Git blob changed, even if internally rehashed")
    require(gitblob(archive_raw) == ARCHIVE_GIT_BLOB,
            "committed encoded original-JSON archive Git blob changed")
    prov = json.loads(provenance_raw)
    originals = read_original_bytes(archive_raw.strip(), prov)
    pref = json.loads(originals["void1-source-prefetch.json"])
    host = json.loads(originals["void1-real-mlp.json"])
    return validate_receipts(pref, host, prov)


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
