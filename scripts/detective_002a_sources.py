#!/usr/bin/env python3
"""DETECTIVE-002A Hugging Face source resolver and independent readback.

Stage A is source metadata only. It uses Hugging Face model/config/tokenizer
metadata and never downloads model weight bytes.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/detective-002/body-cohort.json"
HF = "https://huggingface.co"
RESOLVER_VERSION = "detective-002a-source-resolver-v1"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SourceError(ValueError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False) + "\n", encoding="utf-8")


def load_config() -> dict:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (cfg.get("schema"), cfg.get("experiment"), cfg.get("issue")) != (
        "detective-npc-body-cohort-v1", "DETECTIVE-002A", 127
    ):
        raise SourceError("unexpected DETECTIVE-002A identity")
    if cfg.get("gates") != {
        "training_steps": 0,
        "paid_api_calls": 0,
        "weight_downloads": 0,
        "no_relevance_mapping_access": True,
        "later_runtime_requires_frozen_revision": True,
        "population_expansion_admitted": False,
    }:
        raise SourceError("Stage-A gate drift")
    candidates = cfg.get("candidates")
    if not isinstance(candidates, dict) or len(candidates) != 9:
        raise SourceError("expected nine source candidates")
    valid_classes = {
        "existing_control", "runtime_reference", "direct_candidate",
        "near_band_exception", "mutation_only",
    }
    for cid, meta in candidates.items():
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", cid):
            raise SourceError("invalid candidate id " + cid)
        if meta.get("admission_class") not in valid_classes:
            raise SourceError("invalid admission class " + cid)
        if not isinstance(meta.get("repo"), str) or "/" not in meta["repo"]:
            raise SourceError("invalid Hugging Face repo " + cid)
        if not isinstance(meta.get("source_ref"), str):
            raise SourceError("source ref missing " + cid)
        expected = meta.get("expected_weight_sha256")
        if expected is not None and SHA256.fullmatch(expected) is None:
            raise SourceError("invalid expected weight SHA " + cid)
        if meta.get("parameter_count_policy") not in {
            "nominal_only", "api_required", "expected_exact"
        }:
            raise SourceError("invalid parameter policy " + cid)
    if (candidates["sprocket-500m-chat"]["repo"]
        != candidates["sprocket-500m-instruct"]["repo"]
        or candidates["sprocket-500m-chat"]["source_ref"]
        != candidates["sprocket-500m-instruct"]["source_ref"]
        or candidates["sprocket-500m-chat"]["weight_path"]
        == candidates["sprocket-500m-instruct"]["weight_path"]):
        raise SourceError("Sprocket Chat/Instruct identity contract collapsed")
    return cfg


def http_bytes(url: str, timeout: int = 30) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "neurounner-detective-002a/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise SourceError(f"HTTP {response.status} {url}")
            return response.read()
    except urllib.error.HTTPError as exc:
        raise SourceError(f"HTTP {exc.code} {url}") from exc
    except urllib.error.URLError as exc:
        raise SourceError(f"network error {url}: {exc.reason}") from exc


def http_json(url: str) -> dict:
    payload = json.loads(http_bytes(url))
    if not isinstance(payload, dict):
        raise SourceError("expected JSON object from " + url)
    return payload


def api_url(repo: str, source_ref: str) -> str:
    encoded = urllib.parse.quote(repo, safe="/")
    if source_ref == "main":
        return f"{HF}/api/models/{encoded}?blobs=true"
    ref = urllib.parse.quote(source_ref, safe="")
    return f"{HF}/api/models/{encoded}/revision/{ref}?blobs=true"


def small_file(repo: str, revision: str, path: str) -> tuple[bytes | None, str]:
    url = (
        f"{HF}/{urllib.parse.quote(repo, safe='/')}/resolve/"
        f"{urllib.parse.quote(revision, safe='')}/"
        f"{urllib.parse.quote(path, safe='/')}"
    )
    try:
        data = http_bytes(url)
    except SourceError as exc:
        if "HTTP 404" in str(exc):
            return None, "missing"
        raise
    # Refuse accidentally downloading a large object through this helper.
    if len(data) > 2_000_000:
        raise SourceError(f"small metadata file unexpectedly large: {repo}/{path}")
    return data, "present"


def license_from(payload: dict) -> str | None:
    card = payload.get("cardData")
    if isinstance(card, dict):
        raw = card.get("license")
        named = card.get("license_name")
        # Hugging Face custom licenses are commonly represented in cardData
        # as license="other" plus a human-readable license_name, while the UI
        # renders the named license. Preserve that named identity.
        if (isinstance(raw, str) and raw.casefold() == "other"
                and isinstance(named, str) and named.strip()):
            return named.strip().casefold()
        if isinstance(raw, str) and raw.strip():
            return raw.strip().casefold()
    for tag in payload.get("tags") or []:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split(":", 1)[1].strip().casefold()
    return None


def sibling_record(payload: dict, path: str | None) -> dict | None:
    if path is None:
        return None
    for item in payload.get("siblings") or []:
        if item.get("rfilename") != path:
            continue
        lfs = item.get("lfs") if isinstance(item.get("lfs"), dict) else {}
        return {
            "path": path,
            "size": item.get("size"),
            "blob_id": item.get("blobId"),
            "lfs_sha256": lfs.get("sha256"),
            "lfs_size": lfs.get("size"),
            "pointer_size": lfs.get("pointerSize"),
        }
    return None


def source_has_file(payload: dict, path: str) -> bool:
    return any(
        isinstance(item, dict) and item.get("rfilename") == path
        for item in (payload.get("siblings") or [])
    )


def relevant_checkpoint_files(payload: dict) -> list[dict]:
    rows = []
    for item in payload.get("siblings") or []:
        name = item.get("rfilename", "")
        if not isinstance(name, str):
            continue
        if not (
            fnmatch.fnmatch(name, "checkpoints/main565m/*")
            or name.endswith(".safetensors")
        ):
            continue
        lfs = item.get("lfs") if isinstance(item.get("lfs"), dict) else {}
        rows.append({
            "path": name,
            "size": item.get("size"),
            "lfs_sha256": lfs.get("sha256"),
        })
    return sorted(rows, key=lambda x: x["path"])


def tokenizer_template(payload: dict, tokenizer_doc: dict | None) -> str | None:
    config = payload.get("config")
    if isinstance(config, dict):
        tc = config.get("tokenizer_config")
        if isinstance(tc, dict) and isinstance(tc.get("chat_template"), str):
            return tc["chat_template"]
    if isinstance(tokenizer_doc, dict):
        value = tokenizer_doc.get("chat_template")
        if isinstance(value, str):
            return value
    return None


def parameter_count(payload: dict) -> int | None:
    safetensors = payload.get("safetensors")
    if isinstance(safetensors, dict):
        total = safetensors.get("total")
        if type(total) is int and total > 0:
            return total
    return None


def parameter_gate(meta: dict, observed: int | None, band: dict) -> tuple[bool, str]:
    policy = meta["parameter_count_policy"]
    cls = meta["admission_class"]
    if policy == "api_required" and observed is None:
        return False, "API safetensors parameter count missing"
    if policy == "expected_exact":
        expected = meta.get("expected_api_parameters")
        if observed != expected:
            return False, f"expected {expected}, observed {observed}"
    count = observed if observed is not None else meta["nominal_parameters"]
    if cls == "near_band_exception":
        if count <= band["maximum"]:
            return False, "near-band exception is not actually above band"
        return True, "explicit near-band exception"
    if cls in {"direct_candidate", "existing_control"}:
        if not band["minimum"] <= count <= band["maximum"]:
            return False, f"parameter count {count} outside target band"
    return True, "within class policy"


def resolve_candidate(cid: str, meta: dict, cfg: dict) -> dict:
    payload = http_json(api_url(meta["repo"], meta["source_ref"]))
    revision = payload.get("sha")
    if not isinstance(revision, str) or SHA40.fullmatch(revision) is None:
        raise SourceError(cid + ": resolved revision is not full git SHA")
    if meta["source_ref"] != "main" and revision != meta["source_ref"]:
        raise SourceError(cid + ": immutable revision resolved to different SHA")

    config_bytes, config_status = small_file(meta["repo"], revision, "config.json")
    tok_bytes, tok_status = small_file(
        meta["repo"], revision, "tokenizer_config.json")
    if source_has_file(payload, "chat_template.jinja"):
        jinja_bytes, jinja_status = small_file(
            meta["repo"], revision, "chat_template.jinja")
    else:
        jinja_bytes, jinja_status = None, "missing"
    config_doc = (
        json.loads(config_bytes) if config_bytes is not None else None)
    tok_doc = json.loads(tok_bytes) if tok_bytes is not None else None
    if config_doc is not None and not isinstance(config_doc, dict):
        raise SourceError(cid + ": config.json is not object")
    if tok_doc is not None and not isinstance(tok_doc, dict):
        raise SourceError(cid + ": tokenizer_config.json is not object")

    license_id = license_from(payload)
    weight = sibling_record(payload, meta.get("weight_path"))
    expected_weight = meta.get("expected_weight_sha256")
    weight_hash = weight.get("lfs_sha256") if weight else None
    weight_gate = True
    weight_reason = "not required for mutation-only source"
    if meta.get("weight_path") is not None:
        if weight is None:
            weight_gate = False
            weight_reason = "declared weight path missing from source"
        elif expected_weight and weight_hash != expected_weight:
            weight_gate = False
            weight_reason = (
                f"weight SHA mismatch expected {expected_weight} observed "
                f"{weight_hash}")
        elif weight_hash is None:
            weight_gate = False
            weight_reason = "weight file lacks LFS SHA-256 metadata"
        else:
            weight_reason = "weight LFS identity resolved"

    license_gate = True
    license_reason = "recorded"
    expected_license = meta.get("expected_license")
    if expected_license is not None and license_id != expected_license:
        license_gate = False
        license_reason = (
            f"expected {expected_license}, observed {license_id}")
    elif license_id is None:
        # Missing license does not rewrite existing research authority, but
        # production remains gated and Stage B must carry the unresolved flag.
        license_reason = "source license not declared"
        if not meta.get("production_license_gate", False):
            license_gate = False

    observed_params = parameter_count(payload)
    params_ok, params_reason = parameter_gate(
        meta, observed_params, cfg["parameter_band"])

    template = tokenizer_template(payload, tok_doc)
    template_source = None
    if template is not None:
        template_source = "api_or_tokenizer_config"
    elif jinja_bytes is not None:
        try:
            template = jinja_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SourceError(cid + ": chat_template.jinja is not UTF-8") from exc
        template_source = "chat_template.jinja"
    config_meta = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    architectures = None
    model_type = None
    if isinstance(config_doc, dict):
        architectures = config_doc.get("architectures")
        model_type = config_doc.get("model_type")
    if architectures is None:
        architectures = config_meta.get("architectures")
    if model_type is None:
        model_type = config_meta.get("model_type")

    source_gate = (
        payload.get("private") is False
        and payload.get("disabled") is False
        and weight_gate and license_gate and params_ok
    )
    if payload.get("gated") not in (False, None):
        source_gate = False

    runtime_eligibility = "blocked"
    if source_gate:
        if meta["admission_class"] in {
            "direct_candidate", "near_band_exception", "existing_control"
        } and meta.get("direct_dialogue_candidate"):
            runtime_eligibility = "stage_b_candidate"
        elif meta["admission_class"] == "runtime_reference":
            runtime_eligibility = "stage_b_reference"
        elif meta["admission_class"] == "mutation_only":
            runtime_eligibility = "mutation_protocol_only"

    return {
        "candidate_id": cid,
        "admission_class": meta["admission_class"],
        "family": meta["family"],
        "repo": meta["repo"],
        "variant": meta["variant"],
        "requested_ref": meta["source_ref"],
        "resolved_revision": revision,
        "last_modified": payload.get("lastModified"),
        "private": payload.get("private"),
        "gated": payload.get("gated"),
        "disabled": payload.get("disabled"),
        "license": license_id,
        "production_license_gate": meta.get("production_license_gate", False),
        "library_name": payload.get("library_name"),
        "pipeline_tag": payload.get("pipeline_tag"),
        "model_type": model_type,
        "architectures": architectures,
        "safetensors_parameters": observed_params,
        "nominal_parameters": meta["nominal_parameters"],
        "parameter_count_policy": meta["parameter_count_policy"],
        "parameter_gate_pass": params_ok,
        "parameter_gate_reason": params_reason,
        "weight": weight,
        "expected_weight_sha256": expected_weight,
        "weight_gate_pass": weight_gate,
        "weight_gate_reason": weight_reason,
        "config": {
            "status": config_status,
            "sha256": sha_bytes(config_bytes) if config_bytes else None,
            "bytes": len(config_bytes) if config_bytes else 0,
        },
        "tokenizer_config": {
            "status": tok_status,
            "sha256": sha_bytes(tok_bytes) if tok_bytes else None,
            "bytes": len(tok_bytes) if tok_bytes else 0,
        },
        "has_chat_template": template is not None,
        "chat_template_source": template_source,
        "chat_template_file": {
            "status": jinja_status,
            "sha256": sha_bytes(jinja_bytes) if jinja_bytes else None,
            "bytes": len(jinja_bytes) if jinja_bytes else 0,
        },
        "template_mentions_tools": (
            bool(template) and (
                "tool_call" in template
                or "tools" in template
                or "tool_response" in template
            )
        ),
        "checkpoint_metadata": (
            relevant_checkpoint_files(payload)
            if meta["admission_class"] == "mutation_only" else []
        ),
        "license_gate_pass": license_gate,
        "license_gate_reason": license_reason,
        "source_gate_pass": source_gate,
        "runtime_eligibility": runtime_eligibility,
        "requires_custom_runtime": meta.get("requires_custom_runtime", False),
        "instruction_model": meta.get("instruction_model", True),
        "weight_bytes_downloaded": 0,
        "training_steps": 0,
        "paid_api_calls": 0,
    }


def resolve_all(out: Path) -> dict:
    cfg = load_config()
    rows = [
        resolve_candidate(cid, meta, cfg)
        for cid, meta in sorted(cfg["candidates"].items())
    ]
    evidence = {
        "schema": "detective-npc-body-source-evidence-v1",
        "experiment": "DETECTIVE-002A",
        "issue": 127,
        "resolver_version": RESOLVER_VERSION,
        "config_sha256": sha_file(CONFIG),
        "weight_bytes_downloaded": 0,
        "training_steps": 0,
        "paid_api_calls": 0,
        "relevance_mapping_accessed": False,
        "population_expansion_admitted": False,
        "candidates": rows,
    }
    write_json(out, evidence)
    return evidence


def verify(evidence_path: Path) -> dict:
    cfg = load_config()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence.get("schema") != "detective-npc-body-source-evidence-v1"
        or evidence.get("experiment") != "DETECTIVE-002A"
        or evidence.get("issue") != 127
        or evidence.get("resolver_version") != RESOLVER_VERSION
        or evidence.get("config_sha256") != sha_file(CONFIG)
        or evidence.get("weight_bytes_downloaded") != 0
        or evidence.get("training_steps") != 0
        or evidence.get("paid_api_calls") != 0
        or evidence.get("relevance_mapping_accessed") is not False
        or evidence.get("population_expansion_admitted") is not False
    ):
        raise SourceError("source evidence identity/gate mismatch")

    rows = evidence.get("candidates")
    if not isinstance(rows, list):
        raise SourceError("candidate evidence must be list")
    by = {x.get("candidate_id"): x for x in rows}
    if len(by) != len(rows) or set(by) != set(cfg["candidates"]):
        raise SourceError("candidate evidence set mismatch")

    class_counts = {}
    runtime_counts = {}
    unresolved_licenses = []
    for cid, meta in cfg["candidates"].items():
        row = by[cid]
        if (
            row.get("repo") != meta["repo"]
            or row.get("variant") != meta["variant"]
            or row.get("requested_ref") != meta["source_ref"]
            or row.get("admission_class") != meta["admission_class"]
            or SHA40.fullmatch(str(row.get("resolved_revision", ""))) is None
            or row.get("weight_bytes_downloaded") != 0
            or row.get("training_steps") != 0
            or row.get("paid_api_calls") != 0
        ):
            raise SourceError("source row identity drift " + cid)
        if meta["source_ref"] != "main" and row["resolved_revision"] != meta["source_ref"]:
            raise SourceError("immutable revision drift " + cid)

        expected_weight = meta.get("expected_weight_sha256")
        weight = row.get("weight")
        if meta.get("weight_path") is not None:
            if not isinstance(weight, dict) or weight.get("path") != meta["weight_path"]:
                raise SourceError("weight path metadata missing " + cid)
            lfs_sha = weight.get("lfs_sha256")
            if SHA256.fullmatch(str(lfs_sha or "")) is None:
                raise SourceError("weight LFS SHA missing " + cid)
            if expected_weight and lfs_sha != expected_weight:
                raise SourceError("known weight SHA changed " + cid)

        expected_license = meta.get("expected_license")
        if expected_license and row.get("license") != expected_license:
            raise SourceError("expected license mismatch " + cid)
        if row.get("license") is None:
            unresolved_licenses.append(cid)

        if meta["parameter_count_policy"] == "api_required":
            value = row.get("safetensors_parameters")
            if type(value) is not int or value <= 0:
                raise SourceError("required API parameter count missing " + cid)
        if meta["parameter_count_policy"] == "expected_exact":
            if row.get("safetensors_parameters") != meta["expected_api_parameters"]:
                raise SourceError("exact API parameter count mismatch " + cid)

        if cid == "qwen3-0.6b":
            if row.get("safetensors_parameters") != 751632384:
                raise SourceError("Qwen3 near-band count was obscured")
            if row.get("admission_class") != "near_band_exception":
                raise SourceError("Qwen3 near-band classification drift")
        if cid == "zenyx-mcen-delta-mamba-565m":
            if row.get("runtime_eligibility") != "mutation_protocol_only":
                raise SourceError("Zenyx was promoted beyond mutation-only")
            if row.get("instruction_model") is not False:
                raise SourceError("Zenyx instruction status drift")
        if cid == "lfm25-350m" and row.get("production_license_gate") is not True:
            raise SourceError("LFM350 production license gate disappeared")

        class_counts[row["admission_class"]] = (
            class_counts.get(row["admission_class"], 0) + 1)
        runtime_counts[row["runtime_eligibility"]] = (
            runtime_counts.get(row["runtime_eligibility"], 0) + 1)

    if by["sprocket-500m-chat"]["weight"]["path"] == by["sprocket-500m-instruct"]["weight"]["path"]:
        raise SourceError("Sprocket Chat/Instruct weight paths collapsed")
    if by["sprocket-500m-chat"]["resolved_revision"] != by["sprocket-500m-instruct"]["resolved_revision"]:
        raise SourceError("Sprocket sibling revision mismatch")

    return {
        "verdict": "DETECTIVE_002A_SOURCE_READBACK_PASS",
        "candidates": len(rows),
        "source_gate_passes": sum(bool(x["source_gate_pass"]) for x in rows),
        "source_gate_failures": sum(not bool(x["source_gate_pass"]) for x in rows),
        "classes": dict(sorted(class_counts.items())),
        "runtime_eligibility": dict(sorted(runtime_counts.items())),
        "unresolved_licenses": sorted(unresolved_licenses),
        "training_steps": 0,
        "weight_bytes_downloaded": 0,
        "population_expansion_admitted": False,
    }



def markdown_report(evidence: dict) -> str:
    lines = [
        "# DETECTIVE-002A — compact NPC body source qualification",
        "",
        "This is source metadata evidence only. No model weight bytes were "
        "downloaded, no inference was run and no optimizer step was taken.",
        "",
        "| Candidate | Class | Resolved revision | License | Parameters | Weight SHA | Stage-B status |",
        "|---|---|---|---|---:|---|---|",
    ]
    for row in evidence["candidates"]:
        params = row["safetensors_parameters"]
        if params is None:
            params = row["nominal_parameters"]
            params_text = f"~{params:,} nominal"
        else:
            params_text = f"{params:,}"
        weight = row.get("weight") or {}
        sha = weight.get("lfs_sha256") or "n/a"
        lines.append(
            f"| {row['candidate_id']} | {row['admission_class']} | "
            f"{row['resolved_revision']} | "
            f"{row.get('license') or 'UNDECLARED'} | {params_text} | "
            f"{sha} | {row['runtime_eligibility']} |"
        )
    lines += [
        "",
        "## Guards",
        "",
        f"- source gates passed: **{sum(bool(x['source_gate_pass']) for x in evidence['candidates'])}/{len(evidence['candidates'])}**;",
        "- model weight bytes downloaded: **0**;",
        "- optimizer steps: **0**;",
        "- paid API calls: **0**;",
        "- DETECTIVE-001R1 hidden relevance mapping accessed: **no**;",
        "- population expansion admitted: **no**.",
        "",
        "A source-gate failure is retained as evidence and blocks that body from "
        "Stage B. Production/game license gates are separate from research "
        "source qualification and are never silently cleared here.",
    ]
    return "\n".join(lines) + "\n"


def hash_inventory(root: Path) -> list[str]:
    rows = []
    for path in sorted(x for x in root.rglob("*") if x.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel != "hashes.sha256":
            rows.append(f"{sha_file(path)}  {rel}")
    return rows


def resolve_archive(root: Path) -> dict:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    evidence = resolve_all(root / "source-evidence.json")
    (root / "REPORT.md").write_text(
        markdown_report(evidence), encoding="utf-8")
    (root / "hashes.sha256").write_text(
        "\n".join(hash_inventory(root)) + "\n", encoding="utf-8")
    return evidence


def verify_archive(root: Path) -> dict:
    expected = {"source-evidence.json", "REPORT.md", "hashes.sha256"}
    present = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*") if p.is_file()
    }
    if present != expected:
        raise SourceError("source evidence archive file set mismatch")
    recorded = (root / "hashes.sha256").read_text(
        encoding="utf-8").splitlines()
    if recorded != hash_inventory(root):
        raise SourceError("source evidence hash inventory mismatch")
    evidence = json.loads(
        (root / "source-evidence.json").read_text(encoding="utf-8"))
    if (root / "REPORT.md").read_text(encoding="utf-8") != markdown_report(evidence):
        raise SourceError("source evidence report does not replay")
    return verify(root / "source-evidence.json")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolve", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--verify", type=Path)
    ap.add_argument("--resolve-dir", type=Path)
    ap.add_argument("--verify-dir", type=Path)
    args = ap.parse_args()
    if args.resolve_dir:
        doc = resolve_archive(args.resolve_dir)
        print(json.dumps({
            "resolved_candidates": len(doc["candidates"]),
            "archive": str(args.resolve_dir),
            "weight_bytes_downloaded": 0,
            "training_steps": 0,
        }, indent=2, sort_keys=True))
    elif args.verify_dir:
        print(json.dumps(
            verify_archive(args.verify_dir), indent=2, sort_keys=True))
    elif args.resolve:
        if args.out is None:
            raise SystemExit("--out required with --resolve")
        doc = resolve_all(args.out)
        print(json.dumps({
            "resolved_candidates": len(doc["candidates"]),
            "output_sha256": sha_file(args.out),
            "weight_bytes_downloaded": 0,
            "training_steps": 0,
        }, indent=2, sort_keys=True))
    elif args.verify:
        print(json.dumps(verify(args.verify), indent=2, sort_keys=True))
    else:
        cfg = load_config()
        print(json.dumps({
            "contract": "PASS",
            "candidates": len(cfg["candidates"]),
            "training_steps": 0,
            "weight_downloads": 0,
            "population_expansion_admitted": False,
        }, sort_keys=True))


if __name__ == "__main__":
    main()
