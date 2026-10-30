#!/usr/bin/env python3
"""ONE-SHOT original Git custody for poststep Qwen3 physical Actions receipts.

This downloader is a read-only consumer of original GitHub Actions bytes.
Only the separate workflow may commit lossless original ZIPs/logs to Git;
no model, tokenizer, optimizer or evaluation data is loaded here.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.parse import urlparse
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/ncr-001/controls-evidence/void-poststep-cold-20261030"
REPO = "techrote/neurounner"
RUN = 41058648266
HEAD = "f78d302ab6062c3739315c3e04317a9a97b97191"
ORIGINAL = {
    "source": {"job": 124072474714, "artifact": 12976819289,
               "sha256": "3313913ce5dd5e28f4f89f777f3e9d99271c894ee301e495b311a9b3adca992b",
               "zip_bytes": 369},
    "host": {"job": 124072552187, "artifact": 12977852817,
             "sha256": "75df0cc9733d4873516880ebe2ce449cc5be4d79da03166ee408798dad9ce64b",
             "zip_bytes": 6770},
}


def refuse(condition: bool, reason: str):
    if not condition:
        raise RuntimeError("NCR original poststep custody REFUSED: " + reason)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


class DropAuthOnBlobRedirect(HTTPRedirectHandler):
    """GitHub authorizes the API hop; Azure artifact SAS redirects reject bearer auth."""
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
    token = os.environ.get("GH_TOKEN")
    refuse(bool(token), "missing Actions read token")
    req = Request("https://api.github.com/repos/" + REPO + "/" + endpoint,
                  headers={"Accept": "application/vnd.github+json",
                           "Authorization": "Bearer " + token,
                           "X-GitHub-Api-Version": "2022-11-28",
                           "User-Agent": "neurounner-original-custody"})
    with build_opener(DropAuthOnBlobRedirect()).open(req, timeout=70) as response:
        payload = response.read()
    return json.loads(payload) if json_result else payload


def audit_originals() -> tuple[dict, dict[str, bytes]]:
    refuse(os.getenv("GITHUB_REPOSITORY") == REPO, "wrong repository")
    source_ancestor = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", HEAD, "HEAD"],
        capture_output=True, check=False)
    refuse(source_ancestor.returncode == 0, "physical original source not in custody branch")
    run = fetch(f"actions/runs/{RUN}", json_result=True)
    refuse(run["id"] == RUN and run["head_sha"] == HEAD
           and run["status"] == "completed" and run["conclusion"] == "success"
           and run["event"] == "pull_request"
           and run["run_attempt"] == 1,
           "original physical run/source identity not successful")
    jobs = fetch(f"actions/runs/{RUN}/jobs?per_page=100", json_result=True)
    job_map = {row["id"]: row for row in jobs["jobs"]}
    artifacts = fetch(f"actions/runs/{RUN}/artifacts?per_page=100", json_result=True)
    art_map = {row["id"]: row for row in artifacts["artifacts"]}
    raw_files: dict[str, bytes] = {}
    member_inventory = {}
    metadata = {}
    for kind, spec in ORIGINAL.items():
        job = job_map[spec["job"]]
        refuse(job["conclusion"] == "success" and job["status"] == "completed"
               and job["run_id"] == RUN and job["head_sha"] == HEAD,
               "original job source/status mismatch: " + kind)
        art = art_map[spec["artifact"]]
        refuse(art["workflow_run"]["head_sha"] == HEAD and
               art["workflow_run"]["id"] == RUN and not art["expired"]
               and art["digest"] == "sha256:" + spec["sha256"]
               and art["size_in_bytes"] == spec["zip_bytes"],
               "original artifact ledger mismatch: " + kind)
        data = fetch(f"actions/artifacts/{spec['artifact']}/zip")
        refuse(len(data) == spec["zip_bytes"] and sha(data) == spec["sha256"],
               "complete original artifact ZIP bytes drift: " + kind)
        raw_files[kind + "-original.zip"] = data
        with zipfile.ZipFile(io.BytesIO(data), "r") as z:
            refuse(z.testzip() is None, "original ZIP CRC mismatch: " + kind)
            member_inventory[kind] = {
                name: {"bytes": len(z.read(name)), "sha256": sha(z.read(name))}
                for name in z.namelist()
            }
            if kind == "source":
                refuse(set(z.namelist()) == {"tests.log"}
                       and b"Ran 4 tests" in z.read("tests.log") and
                       b"OK" in z.read("tests.log"),
                       "original model-free source test proof missing")
            else:
                files = set(z.namelist())
                refuse(len(files) == 7 and
                       "ncr001-void-poststep-cold/producer.json" in files and
                       "ncr001-void-poststep-cold/consumer.json" in files and
                       not any(name.endswith(("adapter.pt", "metadata.json"))
                               for name in files),
                       "physical receipts incomplete or temporary adapter leaked")
                producer = json.loads(z.read("ncr001-void-poststep-cold/producer.json"))
                consumer = json.loads(z.read("ncr001-void-poststep-cold/consumer.json"))
                expected_status = "POSTSTEP_COLD_DIAGNOSTIC_ONLY__SCIENTIFIC_72_BLOCKED"
                refuse(producer["status"] == consumer["status"] == expected_status
                       and type(producer["diagnostic_optimizer_steps"]) is int
                       and producer["diagnostic_optimizer_steps"] == 1
                       and type(consumer["diagnostic_optimizer_steps"]) is int
                       and consumer["diagnostic_optimizer_steps"] == 0
                       and producer["sample"]["real_eos_id"] == 258
                       and producer["sample"]["supervised_after_shift"] == 21
                       and producer["sample"]["total_units"] == 31
                       and producer["frozen_backbone_sha256"] ==
                       consumer["frozen_backbone_sha256"] and
                       producer["active_logits_sha256"] ==
                       consumer["active_logits_sha256"] and
                       producer["producer_pid"] != consumer["cold_process_pid"]
                       and producer["threads_effective_after_load"] ==
                       consumer["threads_effective_after_load"] == 4
                       and consumer["post_step_cold_reload_exact"] is True
                       and consumer["per_request_aba_exact"] is True
                       and consumer["incremental_tokens"] == 4
                       and consumer["context_units"] == 31
                       and 0 <= consumer["post_step_last_four_true_kv_max_abs"] <= 5e-4
                       and consumer["train_authorized"] is False
                       and consumer["full_72_conformance_pass"] is False
                       and producer["comparative_training_performed"] is False
                       and producer["heldouts_opened"] is False,
                       "original producer/consumer scientific boundary/receipt mismatch")
                metadata["physical_result"] = {
                    "producer_pid": producer["producer_pid"],
                    "consumer_pid": consumer["cold_process_pid"],
                    "active_vs_disabled_max_abs": producer["active_vs_disabled_max_abs"],
                    "post_step_last_four_true_kv_max_abs":
                        consumer["post_step_last_four_true_kv_max_abs"],
                    "effective_threads": 4, "incremental_tokens": 4,
                    "context_units": 31, "nca_only_optimizer_steps": 1,
                    "true_eos_id": 258,
                    "supervised_after_shift": 21,
                    "frozen_backbone_sha256": producer["frozen_backbone_sha256"],
                    "original_producer_receipt_sha256":
                        consumer["original_producer_receipt_sha256"],
                }
        log = fetch(f"actions/jobs/{spec['job']}/logs")
        refuse(len(log) > 1000 and b"ncr001-void-poststep-" in log
               and HEAD.encode() in log and b"sha256" in log.lower(),
               "original decoded complete job log source markers missing: " + kind)
        raw_files[kind + "-original-job.log"] = log
        metadata[kind] = {
            "original_job": spec["job"], "original_artifact": spec["artifact"],
            "zip_sha256": sha(data), "zip_bytes": len(data),
            "decoded_log_sha256": sha(log), "decoded_log_bytes": len(log),
        }
    manifest = {
        "schema": "ncr001-void-poststep-cold-original-custody-v1",
        "run": RUN, "attempt": 1, "source_head": HEAD,
        "source_branch": "review/ncr001e-void-poststep-cold-replay-20261030",
        "repository": REPO,
        "source_only_run_job_success": True,
        "physical_host_run_job_success": True,
        "retention_of_original_zips_and_logs": "permanent_git_bytes",
        "original": metadata,
        "members": member_inventory,
        "files": {
            name: {"size": len(data), "sha256": sha(data), "git_blob": blob(data)}
            for name, data in raw_files.items()
        },
        "scientific_boundary": {
            "train_authorized": False,
            "full_72_conformance_pass": False,
            "per_model_gate": False, "comparative_training": False,
            "heldouts_opened": False, "actual_compute_budget_qualified": False,
            "historical_train002b_eos_effect_measured": False,
        },
    }
    return manifest, raw_files


def main() -> None:
    refuse(not (OUT / "manifest.json").exists(), "immutable originals already committed")
    manifest, files = audit_originals()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        (OUT / name).write_bytes(payload)
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("ORIGINAL_GIT_CUSTODY_STAGED", json.dumps({
        "run": RUN, "head": HEAD,
        "files": len(files), "physical": manifest["original"]["physical_result"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
