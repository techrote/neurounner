#!/usr/bin/env python3
"""ONE-SHOT original Git custody for void.1 poststep2048 physical Actions bytes.

Downloads only immutable original Actions ZIP containers and complete decoded
job logs from the exact physical run. This script never loads a model/tokenizer,
never executes an optimizer, and never persists the temporary updated adapter.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
from urllib.parse import urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/ncr-001/controls-evidence/void-poststep2048-20261030"
REPO = "techrote/neurounner"
RUN = 41133450850
HEAD = "520c4ae5b3152aff8d94760dab2e3103746f7334"
SOURCE_BRANCH = "review/ncr001e-void-poststep2048-20261030"
ORIGINAL = {
    "source": {
        "job": 124329952468,
        "artifact": 13013279887,
        "sha256": "024ed9a0ac0d89338ef249ba79b3ebb4a20a7659c714fe2e557dd055379cc806",
        "zip_bytes": 558,
    },
    "host": {
        "job": 124330038397,
        "artifact": 13014123159,
        "sha256": "db66de004d26a6c24539b08898f85df88b3be7631ba300655a82bd021de71145",
        "zip_bytes": 10002,
    },
}
MODEL = "dotlabs/void.1"
REV = "9c2be5087772867d45961abd11f6b35110617563"
WEIGHTS = "ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3"
TOKENIZER = "279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03"
FROZEN = "7673957ba36e8eb5427e5ecc94dd52de71fbf9e6718bb0c2a94e4d7361b3f904"
ATOL = 5e-4


def refuse(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError("NCR poststep2048 original custody REFUSED: " + reason)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def blob(raw: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
    ).hexdigest()


def finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


class DropAuthOnBlobRedirect(HTTPRedirectHandler):
    """GitHub API needs bearer auth; Azure artifact SAS redirects reject it."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urlparse(newurl).hostname not in (
                "api.github.com", "github.com"):
            for table in (redirected.headers, redirected.unredirected_hdrs):
                for key in tuple(table):
                    if key.lower() == "authorization":
                        del table[key]
        return redirected


def fetch(endpoint: str, *, json_result: bool = False):
    token = os.getenv("GH_TOKEN")
    refuse(bool(token), "missing Actions read token")
    request = Request(
        "https://api.github.com/repos/" + REPO + "/" + endpoint,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer " + token,
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "neurounner-poststep2048-original-custody",
        },
    )
    with build_opener(DropAuthOnBlobRedirect()).open(
            request, timeout=90) as response:
        payload = response.read()
    return json.loads(payload) if json_result else payload


def require_semantics(producer: dict, consumer: dict) -> dict:
    refuse(
        producer["schema"] == consumer["schema"] ==
        "ncr001-void-poststep2048-v1"
        and producer["phase"] == "producer"
        and consumer["phase"] == "consumer"
        and producer["status"] ==
            "POSTSTEP2048_UPDATED_ADAPTER_READY__SCIENTIFIC_72_UNADMITTED"
        and consumer["status"] ==
            "PASS_POSTSTEP2048_CACHE_MASK_SUFFIX_GENERATE__SCIENTIFIC_72_UNADMITTED",
        "producer/consumer physical status changed",
    )
    refuse(
        producer["model_repo"] == consumer["model_repo"] == MODEL
        and producer["revision"] == consumer["revision"] == REV
        and producer["weights_sha256"] ==
            consumer["weights_sha256"] == WEIGHTS
        and producer["tokenizer_sha256"] ==
            consumer["tokenizer_sha256"] == TOKENIZER
        and producer["frozen_backbone_sha256"] ==
            consumer["frozen_backbone_sha256"] == FROZEN,
        "physical model/tokenizer/frozen source substituted",
    )
    sample = producer["public_learning_sample"]
    refuse(
        type(producer["diagnostic_optimizer_steps"]) is int
        and producer["diagnostic_optimizer_steps"] == 1
        and sample["sample_kind"] == "synthetic_public_diagnostic_only"
        and sample["heldout"] is False
        and type(sample["total_units"]) is int and sample["total_units"] == 31
        and type(sample["supervised_after_shift"]) is int
        and sample["supervised_after_shift"] == 21
        and type(sample["real_eos_id"]) is int and sample["real_eos_id"] == 258,
        "one public shifted-EOS diagnostic changed",
    )
    refuse(
        producer["learning_summary"]["diagnostic_optimizer_steps"] == 1
        and producer["learning_summary"]["checks"][
            "backbone_unchanged_after_one_step"] is True
        and producer["learning_summary"]["checks"][
            "post_step_upstream_gradients"] is True
        and producer["learning_summary"]["backbone_before_sha256"] ==
            producer["learning_summary"]["backbone_after_sha256"] == FROZEN,
        "one-step gradient/frozen-base evidence incomplete",
    )
    cache = consumer["poststep_2048_all_position_cache"]
    refuse(
        type(cache["context_units"]) is int and cache["context_units"] == 2048
        and type(cache["incremental_positions"]) is int
        and cache["incremental_positions"] == 2048
        and cache["all_positions_compared"] is True
        and cache["within_protocol_tolerance"] is True
        and finite(cache["max_abs"]) and 0 <= cache["max_abs"] <= ATOL,
        "poststep all-position 2048 KV evidence changed",
    )
    masked = consumer["poststep_masked_batch"]
    refuse(set(masked) == {"left", "right"}, "both mask directions missing")
    masked_summary = {}
    for side in ("left", "right"):
        row = masked[side]
        refuse(
            row["direction"] == side
            and type(row["batch_size"]) is int and row["batch_size"] == 2
            and type(row["width"]) is int and row["width"] == 2048
            and type(row["short_valid_units"]) is int
            and row["short_valid_units"] == 1536
            and type(row["synthetic_pad_units"]) is int
            and row["synthetic_pad_units"] == 512
            and type(row["pad_id"]) is int and row["pad_id"] == 256
            and row["explicit_attention_mask"] is True
            and row["batch_vs_identical_single_within_tolerance"] is True
            and row["row_order_swap_within_tolerance"] is True
            and finite(row["max_batch_vs_single_abs"])
            and row["max_batch_vs_single_abs"] <= ATOL
            and finite(row["max_row_swap_abs"])
            and row["max_row_swap_abs"] <= ATOL,
            "masked B=2 result changed: " + side,
        )
        masked_summary[side] = {
            "max_batch_vs_single_abs": row["max_batch_vs_single_abs"],
            "max_row_swap_abs": row["max_row_swap_abs"],
        }
    suffix = consumer["poststep_2048_suffix"]
    refuse(
        suffix["context_units"] == 2048
        and suffix["mutated_suffix_units"] == 64
        and suffix["earlier_logits_exact"] is True
        and suffix["earlier_max_abs"] == 0.0,
        "strict poststep future-suffix result changed",
    )
    generation = consumer["poststep_native_generation"]
    refuse(
        generation["prefix_units"] == 2044
        and generation["new_units"] == 4
        and generation["total_units"] == 2048
        and generation["eos_id"] == 258
        and generation["cache_on_off_token_ids_equal"] is True
        and generation["cache_on_off_scores_within_tolerance"] is True
        and finite(generation["max_score_abs"])
        and generation["max_score_abs"] <= ATOL
        and len(generation["generated_ids"]) == 4,
        "native GenerationMixin poststep evidence changed",
    )
    refuse(
        producer["producer_pid"] == consumer["producer_pid"]
        and producer["producer_pid"] != consumer["consumer_pid"]
        and producer["threads_effective_after_load"] ==
            consumer["threads_effective_after_load"] == 4
        and consumer["same_runner_second_python"] is True
        and consumer["poststep_adapter_reload_exact"] is True
        and consumer["frozen_backbone_unchanged_after_long_diagnostic"] is True,
        "separate-interpreter/frozen poststep provenance changed",
    )
    for row in (producer, consumer):
        refuse(
            row["train_authorized"] is False
            and row["comparative_training_performed"] is False
            and row["heldouts_opened"] is False
            and row["full_72_conformance_pass"] is False
            and row["independent_scientific_review"] is False,
            "physical receipt falsely promoted training/science",
        )
    refuse(
        consumer["actual_compute_budget_qualified"] is False
        and consumer["generated_eos_skill_qualified"] is False
        and consumer["historical_train002b_eos_effect_measured"] is False,
        "quality/compute/EOS historical effect falsely promoted",
    )
    return {
        "nca_only_optimizer_steps": 1,
        "public_training_units": 31,
        "supervised_after_shift": 21,
        "true_eos_id": 258,
        "producer_pid": producer["producer_pid"],
        "consumer_pid": consumer["consumer_pid"],
        "effective_threads": 4,
        "frozen_backbone_sha256": FROZEN,
        "poststep_2048_incremental_positions": 2048,
        "poststep_2048_cache_max_abs": cache["max_abs"],
        "masked_batch": masked_summary,
        "future_suffix_units": 64,
        "future_suffix_earlier_exact": True,
        "generate_prefix_units": 2044,
        "generate_new_units": 4,
        "generated_ids": generation["generated_ids"],
        "generation_actual_eos_id_emitted":
            generation["actual_eos_id_emitted"],
        "generation_cache_score_max_abs": generation["max_score_abs"],
        "consumer_elapsed_seconds": consumer["elapsed_seconds"],
        "consumer_peak_rss_kib": consumer["peak_rss_kib"],
    }


def audit_originals() -> tuple[dict, dict[str, bytes]]:
    refuse(os.getenv("GITHUB_REPOSITORY") == REPO, "wrong repository")
    ancestor = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", HEAD, "HEAD"],
        capture_output=True, check=False,
    )
    refuse(ancestor.returncode == 0, "physical source head not an ancestor")
    run = fetch(f"actions/runs/{RUN}", json_result=True)
    refuse(
        run["id"] == RUN and run["head_sha"] == HEAD
        and run["status"] == "completed" and run["conclusion"] == "success"
        and run["event"] == "pull_request" and run["run_attempt"] == 1,
        "original run/source identity not successful",
    )
    jobs = fetch(f"actions/runs/{RUN}/jobs?per_page=100", json_result=True)
    job_map = {row["id"]: row for row in jobs["jobs"]}
    artifacts = fetch(
        f"actions/runs/{RUN}/artifacts?per_page=100", json_result=True)
    artifact_map = {row["id"]: row for row in artifacts["artifacts"]}
    raw_files: dict[str, bytes] = {}
    members = {}
    original = {}
    physical_result = None

    for kind, spec in ORIGINAL.items():
        job = job_map[spec["job"]]
        refuse(
            job["status"] == "completed" and job["conclusion"] == "success"
            and job["run_id"] == RUN and job["head_sha"] == HEAD,
            "original job mismatch: " + kind,
        )
        artifact = artifact_map[spec["artifact"]]
        refuse(
            artifact["workflow_run"]["id"] == RUN
            and artifact["workflow_run"]["head_sha"] == HEAD
            and artifact["expired"] is False
            and artifact["digest"] == "sha256:" + spec["sha256"]
            and artifact["size_in_bytes"] == spec["zip_bytes"],
            "original artifact ledger mismatch: " + kind,
        )
        data = fetch(f"actions/artifacts/{spec['artifact']}/zip")
        refuse(
            len(data) == spec["zip_bytes"] and sha(data) == spec["sha256"],
            "whole original artifact ZIP drift: " + kind,
        )
        raw_files[kind + "-original.zip"] = data
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            refuse(archive.testzip() is None,
                   "original ZIP CRC failure: " + kind)
            names = archive.namelist()
            members[kind] = {
                name: {
                    "bytes": len(archive.read(name)),
                    "sha256": sha(archive.read(name)),
                }
                for name in names
            }
            if kind == "source":
                refuse(
                    set(names) == {"tests.log"}
                    and b"Ran 10 tests" in archive.read("tests.log")
                    and b"OK" in archive.read("tests.log"),
                    "source sentinel proof missing",
                )
            else:
                required = {
                    "ncr001-void-poststep2048/producer.json",
                    "ncr001-void-poststep2048/consumer.json",
                    "ncr001-void-poststep2048/producer.log",
                    "ncr001-void-poststep2048/consumer.log",
                    "ncr001-void-poststep2048/prefetch.log",
                    "ncr001-void-poststep2048/runner.txt",
                    "ncr001-void-nca-learning/receipt.json",
                    "ncr001-void-postlease/poststep-cold.json",
                }
                refuse(
                    required.issubset(set(names))
                    and not any(name.endswith(
                        (".pt", ".safetensors", "metadata.json"))
                        for name in names),
                    "physical archive incomplete or temporary adapter leaked",
                )
                producer = json.loads(archive.read(
                    "ncr001-void-poststep2048/producer.json"))
                consumer = json.loads(archive.read(
                    "ncr001-void-poststep2048/consumer.json"))
                physical_result = require_semantics(producer, consumer)

        log = fetch(f"actions/jobs/{spec['job']}/logs")
        refuse(
            len(log) > 1000 and HEAD.encode() in log
            and (b"Ran 10 tests" in log if kind == "source"
                 else b"PASS_POSTSTEP2048_CACHE_MASK_SUFFIX_GENERATE" in log)
            and (b"SOURCE_SHA256" in log if kind == "host" else True),
            "complete decoded original job log markers missing: " + kind,
        )
        raw_files[kind + "-original-job.log"] = log
        original[kind] = {
            "original_job": spec["job"],
            "original_artifact": spec["artifact"],
            "zip_sha256": sha(data),
            "zip_bytes": len(data),
            "decoded_log_sha256": sha(log),
            "decoded_log_bytes": len(log),
        }

    refuse(physical_result is not None, "physical receipt not read")
    manifest = {
        "schema": "ncr001-void-poststep2048-original-custody-v1",
        "repository": REPO,
        "run": RUN,
        "attempt": 1,
        "source_head": HEAD,
        "source_branch": SOURCE_BRANCH,
        "source_only_job_success": True,
        "physical_host_job_success": True,
        "retention_of_original_zips_and_logs": "permanent_git_bytes",
        "original": {**original, "physical_result": physical_result},
        "members": members,
        "files": {
            name: {
                "size": len(payload),
                "sha256": sha(payload),
                "git_blob": blob(payload),
            }
            for name, payload in raw_files.items()
        },
        "scientific_boundary": {
            "train_authorized": False,
            "per_model_gate": False,
            "full_72_conformance_pass": False,
            "comparative_training": False,
            "heldouts_opened": False,
            "actual_compute_budget_qualified": False,
            "generated_eos_skill_qualified": False,
            "historical_train002b_eos_effect_measured": False,
            "independent_scientific_review": False,
        },
    }
    return manifest, raw_files


def main() -> None:
    refuse(not (OUT / "manifest.json").exists(),
           "immutable originals already committed")
    manifest, files = audit_originals()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        (OUT / name).write_bytes(payload)
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("POSTSTEP2048_ORIGINAL_GIT_CUSTODY_STAGED", json.dumps({
        "run": RUN, "head": HEAD, "files": len(files),
        "physical": manifest["original"]["physical_result"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
