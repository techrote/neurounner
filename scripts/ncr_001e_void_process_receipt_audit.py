#!/usr/bin/env python3
"""Independent byte-exact audit of original void1 fresh-process Qwen3 receipts.

No model, tokenizer Python, optimizer or held-out fixtures are loaded. This
reads losslessly archived original JSON records, not reconstructed summaries.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "experiments/ncr-001/evidence-review/void1-cross-process"
ARCHIVE = FOLDER / "original-jsons.tar.gz.b64"
PROVENANCE = FOLDER / "provenance.json"
GZIP_SHA = "41c8a59e629238dc4f3173253c92b4bf3ca6b9a5d52d23d230f6098e8835f125"
ZIP_SHA = "bc20bab7b3dd2ad25e1b1f4d71a061a4d5da38d1fd1c898c27c57838fc26d4f7"
RUN = "40549549886"
MERGE = "3012458421054d29d5bbae2d68e0365558728f83"
WEIGHT = "ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3"
BYTE = "279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03"
HOOK = "5fa3dc8c0de153098d1800f7b0c809819620aef4"
SRC_INPUT = "069b7669d0ca09c1184a47f639b9286679663260f1d945771fb3503b3456e0a6"
PROTOCOL = "2e294750a17b1bfb3eba448f8a5b55344c72aff700587351bd10da1af38885ab"
PINS = "db794f53a49bb1bcbf8b23bd4a76d24430eb495da246639b506abdbbf11734b1"
STATUS = "PASS_ORIGINAL_QWEN3_TWO_INTERPRETER_BYTE_READBACK__FINAL_NCR72_BLOCKED"
EXPECTED = {
    "void-byte-source-audit.json": (6903, "14ac28204837d7615220981c5b3ec3c86b9effe548306047b012fed69e0efa1c"),
    "void1-cross-process/adapter/metadata.json": (522, "06739554a1b9072847ca7eb31cde1dc3fda7dffa678980edb9c318796e7f015c"),
    "void1-cross-process/consumer.json": (5035, "190f66f062cc662398e491f5bec3a2033246e7a7df6dc28dd7a906a0b1ea5071"),
    "void1-cross-process/manifest.json": (1223, "406af0126d00aa4d674952cc26d2a672f80e3ee943efdd2f9a4f8ec2d2fc5bad"),
    "void1-cross-process/producer.json": (4618, "877ae9653035697ddcc644633649593d0f635845e831a64304019d7b975ac15f"),
}
FALSE_FLAGS = ("model_training_performed", "train_authorized", "heldout_evaluation",
               "full_four_model_conformance_pass", "historical_byte_eos_impact_measured")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(ok: bool, msg: str) -> None:
    if not ok:
        raise RuntimeError(msg)


def originals() -> dict[str, dict]:
    encoded = ARCHIVE.read_text(encoding="ascii").strip()
    gz = base64.b64decode(encoded, validate=True)
    require(sha(gz) == GZIP_SHA, "Original frozen-model evidence archive SHA changed")
    raw = gzip.decompress(gz)
    require(len(raw) < 100000, "Original receipt archive exceeded bounded size")
    records = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
        members = tar.getmembers()
        require(len(members) == len(EXPECTED) and
                {m.name for m in members} == set(EXPECTED),
                "Original required records missing, duplicated or renamed")
        for item in members:
            require(item.isfile() and not item.issym() and not item.islnk(),
                    "Original evidence archive has unsafe member")
            data = tar.extractfile(item).read()
            n, checksum = EXPECTED[item.name]
            require(len(data) == n and sha(data) == checksum,
                    "Original full source JSON differs from Actions: " + item.name)
            records[item.name] = json.loads(data)
    return records


def check_num(check: dict, *, exact=False) -> float:
    require(check["pass"] is True, "Actual original numerical source check failed")
    delta = check["max_abs"]
    require(type(delta) in (float, int) and math.isfinite(delta) and delta >= 0,
            "Missing finite original source cache metric")
    if exact:
        require(check["exact_required"] is True and check["atol"] == check["rtol"] == 0
                and delta == 0, "Claimed exact source test was not exact")
    else:
        require(check["exact_required"] is False and
                check["atol"] == check["rtol"] == 0.0005 and delta <= 0.0005,
                "Cached decode tolerance changed or exceeded")
    return delta


def audit(records: dict[str, dict] | None = None) -> dict:
    if records is None:
        records = originals()
    require(set(records) == set(EXPECTED), "Both original stages and source records required")
    p = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    require(p["original_run_id"] == int(RUN) and
            p["original_run_job_id"] == 122349265883 and
            p["original_actions_merge_sha"] == MERGE and
            p["original_actions_artifact"]["zip_sha256"] == ZIP_SHA and
            p["original_actions_artifact"]["bytes"] == 717221 and
            p["archive_gzip_sha256"] == GZIP_SHA,
            "Wrong original Actions run/artifact/source custody")
    require(p["actual_weight_sha256"] == WEIGHT and
            p["actual_weight_bytes"] == 360622744 and
            p["reviewed_dynamic_byte_tokenizer_sha256"] == BYTE and
            p["original_hook_commit"] == HOOK and
            p["source_protocol_sha256"] == PROTOCOL and
            p["source_pins_sha256"] == PINS,
            "Original immutable Qwen3/ByteTokenizer source provenance changed")
    for key in ("trained_adapter", "train_authorized", "heldout_evaluation",
                "final_four_model_72_admission",
                "historical_byte_eos_impact_measured",
                "full_2048_context_proven"):
        require(p["evidence_limits"][key] is False,
                "Provenance falsely reports training/final admission: " + key)
    require(p["evidence_limits"]["optimizer_steps"] == 0 and
            p["evidence_limits"]["new_physical_host_or_runner"] is False,
            "Invalid host or optimization claim")
    for name, (size, checksum) in EXPECTED.items():
        require(p["archived_original_files"][name] == {"bytes": size, "sha256": checksum},
                "Source-byte manifest differs from original Actions JSON")

    source = records["void-byte-source-audit.json"]
    require(source["status"] == "PASS_SOURCE_HASH_AND_AST_INVENTORY_ONLY__RUNTIME_BLOCKED",
            "Tokenizer source-only audit was falsely converted into executed source PASS")
    require(source["model_training_performed"] is False and
            source["source"]["protocol_sha256"] == PROTOCOL and
            source["source"]["source_pins_sha256"] == PINS,
            "Original tokenizer AST/source custody fails")
    producer = records["void1-cross-process/producer.json"]
    consumer = records["void1-cross-process/consumer.json"]
    manifest = records["void1-cross-process/manifest.json"]
    require(producer["status"] == "PASS_PINNED_VOID1_UTF8_PROCESS_PRODUCER" and
            consumer["status"] == "PASS_NEW_INTERPRETER_ORIGINAL_FROZEN_VOID1_ADAPTER_REPLAY",
            "Original actual-host producer/consumer did not both pass")
    require(producer["stage"] == "producer" and consumer["stage"] == "consumer" and
            producer["model_key"] == consumer["model_key"] == manifest["model_key"] == "void1",
            "Original source model/stage was switched")
    for report in (producer, consumer):
        require(report["experiment"] == "NCR-001E" and
                report["optimizer_steps"] == 0 and
                report["frozen_base_parameters"] == 90148352 and
                report["nca_parameters"] == 329944,
                "Original full frozen Qwen3 architecture/optimizer drift")
        for key in FALSE_FLAGS:
            require(report[key] is False, "Source host falsely declared " + key)
        s = report["source"]
        require(s["repo"] == "dotlabs/void.1" and
                s["revision"] == "9c2be5087772867d45961abd11f6b35110617563" and
                s["original_hook_commit"] == HOOK and
                s["byte_tokenizer_source_sha256"] == BYTE and
                s["protocol_sha256"] == PROTOCOL and s["source_pins_sha256"] == PINS,
                "Fresh process original source identity differs")
        require(report["source_files"]["model.safetensors"] ==
                {"bytes": 360622744, "sha256": WEIGHT} and
                report["source_files"]["tokenization_byte.py"]["sha256"] == BYTE,
                "Full source/model tokenization byte hashes not verified")
        ids = report["input"]
        require(ids["input_ids_sha256"] == SRC_INPUT and
                ids["selected_units"] == 256 and
                ids["utf8_bytes"] == ids["original_tokenizer_units"] == 1144 and
                ids["python_characters"] == 1032 and
                (ids["pad_id"], ids["bos_id"], ids["eos_id"]) == (256, 257, 258) and
                ids["literal_eos_text_is_five_ordinary_bytes"] is True,
                "Original source mixed UTF-8 byte-unit/EOS semantics changed")
        runner = report["runner"]
        require(runner["run_id"] == RUN and runner["run_attempt"] == "1" and
                runner["github_sha"] == MERGE and
                runner["runner_name"] == "GitHub Actions 1000045271" and
                runner["python"] == "3.12.3" and
                runner["torch"] == "2.14.1+cpu" and
                runner["transformers"] == "4.57.6" and
                runner["cpu_count"] == 4 and
                report["peak_rss_kib"] > 0,
                "Original Python/runtime/CPU runner provenance changed")
    require(producer["runner"]["pid"] == manifest["producer_pid"] == 2613 and
            consumer["runner"]["pid"] == 2625 and
            consumer["runner"]["pid"] != producer["runner"]["pid"],
            "Both purported host stages were not separate interpreter PIDs")
    require(manifest["adapter_sha256"] ==
            p["nonarchived_binary_handoff_on_actions_only"][
                "void1-cross-process/adapter/adapter.pt"]["sha256"] and
            manifest["reference_sha256"] ==
            p["nonarchived_binary_handoff_on_actions_only"][
                "void1-cross-process/reference.safetensors"]["sha256"] and
            manifest["adapter_metadata_sha256"] ==
            EXPECTED["void1-cross-process/adapter/metadata.json"][1],
            "Original untrained adapter/reference handoff SHA mismatch")
    require(manifest["checkpoint_sha256"] == WEIGHT and
            manifest["byte_tokenizer_source_sha256"] == BYTE and
            manifest["input_token_ids_sha256"] == SRC_INPUT and
            manifest["original_hook_commit"] == HOOK and
            manifest["protocol_sha256"] == PROTOCOL and
            manifest["source_pins_sha256"] == PINS,
            "Cold process reloaded against changed source")
    for key in FALSE_FLAGS:
        require(manifest[key] is False, "Manifest contains unauthorized model claim")
    require(manifest["optimizer_steps"] == 0, "Manifest contains optimizer updates")

    a, b = producer["checks"], consumer["checks"]
    require(set(a) == {"original_zero_init_noop_exact",
                       "manual_untrained_readout_effect",
                       "256_UTF8_240_prefill_16_real_incremental",
                       "strict_future_suffix", "request_A_B_A"} and
            set(b) == {"fresh_process_active_tail_exact",
                       "fresh_process_disabled_tail_exact",
                       "fresh_process_request_A_exact",
                       "fresh_process_request_B_exact",
                       "fresh_process_A_B_A_exact",
                       "fresh_process_full_vs_real_cached_tail"},
            "Source readback original required numerical checks absent")
    require(a["manual_untrained_readout_effect"]["pass"] is True and
            a["manual_untrained_readout_effect"]["trained"] is False and
            a["manual_untrained_readout_effect"]["max_abs"] ==
            0.062296390533447266,
            "Actual untrained NCA source intervention not shown")
    for label in ("original_zero_init_noop_exact", "strict_future_suffix", "request_A_B_A"):
        check_num(a[label], exact=True)
    for label in ("fresh_process_active_tail_exact", "fresh_process_disabled_tail_exact",
                  "fresh_process_request_A_exact", "fresh_process_request_B_exact",
                  "fresh_process_A_B_A_exact"):
        check_num(b[label], exact=True)
    for d, label in ((a, "256_UTF8_240_prefill_16_real_incremental"),
                     (b, "fresh_process_full_vs_real_cached_tail")):
        require(check_num(d[label]) == 9.5367431640625e-06,
                "Exact original cached byte metric differs")
    return {
        "status": STATUS, "original_actual_model_run_id": int(RUN),
        "original_actual_frozen_processes": 2, "producer_pid": 2613,
        "consumer_pid": 2625, "original_gzip_sha256": GZIP_SHA,
        "original_actions_zip_sha256": ZIP_SHA,
        "full_original_receipts_verified": 5, "original_model_reexecuted": False,
        "actual_cached_tail_error": 9.5367431640625e-06,
        "fresh_adapter_logit_replay_max_abs": 0,
        "train_authorized": False, "optimizer_steps": 0,
        "final_four_model_72_admission": False,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
