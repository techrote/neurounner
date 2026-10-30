#!/usr/bin/env python3
"""DETECTIVE-003B: source-bound live two-witness session integration smoke.

The model is never authority. Prompt projection and session mutation are owned
by DETECTIVE-003A. The only automatic claim annotation in this fixture is a
game-owned literal HH:MM extractor that refuses ambiguous responses.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import platform
import re
import resource
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import detective_003a_session as session_api  # noqa: E402
from scripts.detective_001_contract import validate_case  # noqa: E402

CONFIG = ROOT / "experiments/detective-003/live-session-smoke.json"
CASE = ROOT / "experiments/detective-001/pilot-case.json"
C2_RELEASE = ROOT / "experiments/detective-001/training-c2-evidence/adapter-release.json"
C3_RELEASE = ROOT / "experiments/detective-001/training-c3-evidence/adapter-release.json"

TIME_RX = re.compile(r"\b(?:[01]\d|2[0-3]):[0-5]\d\b")


class LiveSessionError(ValueError):
    pass


def canonical(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise LiveSessionError("unserializable/non-finite value") from exc


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    h = hashlib.sha1()
    h.update(f"blob {len(data)}\0".encode("ascii"))
    h.update(data)
    return h.hexdigest()


def directory_hash(path: Path) -> str:
    h = hashlib.sha256()
    for item in sorted(x for x in path.rglob("*") if x.is_file()):
        rel = item.relative_to(path).as_posix().encode("utf-8")
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        h.update(bytes.fromhex(file_sha(item)))
    return h.hexdigest()


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def current_rss_kib() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def load_contract() -> tuple[dict, dict, dict, dict]:
    cfg = read_json(CONFIG)
    case = read_json(CASE)
    c2 = read_json(C2_RELEASE)
    c3 = read_json(C3_RELEASE)

    if (cfg.get("schema"), cfg.get("experiment"), cfg.get("issue")) != (
        "detective-003b-live-session-v1",
        "DETECTIVE-003B",
        142,
    ):
        raise LiveSessionError("003B config identity drift")
    validate_case(case)
    authority = cfg["authority"]
    expected_blobs = {
        ROOT / authority["session_module"]["path"]:
            authority["session_module"]["git_blob_sha1"],
        ROOT / authority["case"]["path"]:
            authority["case"]["git_blob_sha1"],
        ROOT / authority["c2_release"]["path"]:
            authority["c2_release"]["git_blob_sha1"],
        ROOT / authority["c3_release"]["path"]:
            authority["c3_release"]["git_blob_sha1"],
    }
    for path, expected in expected_blobs.items():
        if git_blob_sha1(path) != expected:
            raise LiveSessionError("authority Git blob drift: " + str(path))
    if case["case_id"] != authority["case"]["case_id"]:
        raise LiveSessionError("Bay Seventeen case ID drift")

    base = cfg["base_model"]
    if base != {
        "repo": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        "weight_file": "model.safetensors",
        "weight_sha256": "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c",
    }:
        raise LiveSessionError("base model authority drift")

    if (
        c2.get("release_tag") != "detective-001c2-v1"
        or c2.get("base_model") != base["repo"]
        or c2.get("base_revision") != base["revision"]
    ):
        raise LiveSessionError("C2 release authority drift")
    if (
        c3.get("release_tag") != "detective-001c3-g02-v1"
        or c3.get("base_model") != base["repo"]
        or c3.get("base_revision") != base["revision"]
    ):
        raise LiveSessionError("C3 release authority drift")

    c2_assets = {x["member_id"]: x for x in c2["assets"]}
    c3_assets = {x["member_id"]: x for x in c3["assets"]}
    for npc_id, meta in cfg["witnesses"].items():
        source = c3_assets if meta["release_manifest"] == "c3_release" else c2_assets
        asset = source.get(npc_id)
        if asset is None:
            raise LiveSessionError("witness release asset missing: " + npc_id)
        for field, expected in (
            ("file", meta["archive_file"]),
            ("sha256", meta["archive_sha256"]),
            ("bytes", meta["archive_bytes"]),
            ("adapter_directory_sha256", meta["directory_sha256"]),
        ):
            if asset[field] != expected:
                raise LiveSessionError(f"{npc_id}: release asset drift in {field}")
    if c3.get("frozen_control", {}).get("directory_sha256") != cfg["witnesses"][
        "V-G09-C"
    ]["directory_sha256"]:
        raise LiveSessionError("G09 frozen-control lineage drift")

    runtime = cfg["runtime"]
    if runtime != {
        "os": "ubuntu-24.04",
        "python": "3.12",
        "torch_cpu": "2.14.1+cpu",
        "transformers": "4.57.6",
        "peft": "0.21.2",
        "huggingface_hub": "0.36.2",
        "safetensors": "0.8.0",
        "threads": 4,
        "seed": 30303,
        "do_sample": False,
        "max_new_tokens_per_turn": 96,
        "model_turns": 3,
        "max_generated_tokens_total": 288,
        "max_prompt_tokens": 768,
        "max_job_minutes": 20,
        "optimizer_steps": 0,
        "paid_api_calls": 0,
        "gpu_required": False,
    }:
        raise LiveSessionError("runtime budget drift")
    if cfg["annotation"] != {
        "schema": "detective-literal-time-annotation-v1",
        "proposition_key": "handoff-time",
        "regex": r"\b(?:[01]\d|2[0-3]):[0-5]\d\b",
        "one_distinct_time": "statement",
        "zero_times": "no_time",
        "multiple_distinct_times": "ambiguous_times",
        "canonical_truth_accessed": False,
        "scorer_accessed": False,
    }:
        raise LiveSessionError("annotation contract drift")
    seq = cfg["session"]["sequence"]
    if [x["id"] for x in seq] != ["L01", "L02", "L03", "L04"]:
        raise LiveSessionError("live sequence drift")
    if [x["action"] for x in seq] != ["turn", "turn", "evidence", "turn"]:
        raise LiveSessionError("live action order drift")
    if cfg["session"]["participants"] != ["V-G02-A", "V-G09-C"]:
        raise LiveSessionError("participant set/order drift")
    acceptance = cfg["acceptance"]
    if (
        acceptance["minimum_claim_count"] != 0
        or acceptance["minimum_contradiction_count"] != 0
        or acceptance["relevance_mapping_accessed"] is not False
        or acceptance["learner_final_accessed"] is not False
        or acceptance["body_selection_made"] is not False
        or acceptance["population_expansion_admitted"] is not False
        or acceptance["training_admitted"] is not False
    ):
        raise LiveSessionError("003B authority widened")
    return cfg, case, c2, c3


def safe_unpack_adapter(
    archive: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    expected_directory_sha256: str,
) -> dict:
    if not archive.is_file():
        raise LiveSessionError("adapter archive missing")
    if archive.stat().st_size != expected_bytes:
        raise LiveSessionError("adapter archive byte count mismatch")
    if file_sha(archive) != expected_sha256:
        raise LiveSessionError("adapter archive SHA mismatch")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    with tarfile.open(archive, mode="r:gz") as tf:
        members = tf.getmembers()
        if len(members) > 32:
            raise LiveSessionError("adapter archive has too many entries")
        for member in members:
            if member.issym() or member.islnk() or member.isdev():
                raise LiveSessionError("unsafe adapter archive member")
            rel = Path(member.name)
            if rel.is_absolute() or ".." in rel.parts:
                raise LiveSessionError("adapter archive path traversal")
            target = destination / rel
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise LiveSessionError("unsupported adapter archive entry")
            if member.size > 20_000_000:
                raise LiveSessionError("adapter archive file exceeds bound")
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                raise LiveSessionError("adapter archive extraction failure")
            with target.open("wb") as out:
                shutil.copyfileobj(src, out)

    required = {
        destination / "adapter_config.json",
        destination / "adapter_model.safetensors",
    }
    if any(not x.is_file() for x in required):
        raise LiveSessionError("adapter archive lacks required PEFT files")
    actual_dir = directory_hash(destination)
    if actual_dir != expected_directory_sha256:
        raise LiveSessionError("adapter directory hash mismatch")
    return {
        "archive_sha256": expected_sha256,
        "archive_bytes": expected_bytes,
        "directory_sha256": actual_dir,
        "files": sorted(
            x.relative_to(destination).as_posix()
            for x in destination.rglob("*") if x.is_file()
        ),
    }


def extract_time_annotation(response: str, cfg: dict) -> dict:
    if not isinstance(response, str):
        raise LiveSessionError("raw response must be text")
    found = TIME_RX.findall(response)
    distinct: list[str] = []
    for value in found:
        if value not in distinct:
            distinct.append(value)
    if len(distinct) == 1:
        status = "single_time"
        annotations = [{
            "proposition_key": cfg["annotation"]["proposition_key"],
            "value": distinct[0],
            "epistemic_class": "statement",
        }]
    elif not distinct:
        status = cfg["annotation"]["zero_times"]
        annotations = []
    else:
        status = cfg["annotation"]["multiple_distinct_times"]
        annotations = []
    return {
        "schema": cfg["annotation"]["schema"],
        "status": status,
        "literal_times": found,
        "distinct_times": distinct,
        "annotations": annotations,
        "canonical_truth_accessed": False,
        "scorer_accessed": False,
    }


def render_session_prompt(prompt: dict, cfg: dict) -> str:
    if prompt.get("schema") != session_api.PROMPT_SCHEMA:
        raise LiveSessionError("unexpected session prompt schema")
    lines: list[str] = []
    lines.append("Admitted memories:")
    memories = prompt["admitted_memories"]
    if memories:
        lines.extend("- " + x for x in memories)
    else:
        lines.append("- (none supplied)")

    lines.append("")
    lines.append("Presented evidence:")
    evidence = prompt["presented_evidence"]
    if evidence:
        lines.extend("- " + x["excerpt"] for x in evidence)
    else:
        lines.append("- (none presented)")

    lines.append("")
    lines.append("Prior same-witness dialogue:")
    history = prompt["prior_same_witness_dialogue"]
    if history:
        for row in history:
            lines.append("Q: " + row["question"])
            lines.append("A: " + row["response"])
    else:
        lines.append("- (none included)")

    lines.append("")
    lines.append("Player question:")
    lines.append(prompt["player_question"])
    rendered = "\n".join(lines)
    if len(rendered) > 12_000:
        raise LiveSessionError("rendered prompt exceeds text bound")

    # The nested DETECTIVE-003A prompt has already enforced authority isolation.
    for forbidden in (
        "V-G02-A",
        "V-G09-C",
        "F-CLOCK",
        "F-SWAP",
        "F-CAMERA",
        "F-PRIVATE",
        "transfer-card",
        "camera-cache",
        "camera-console",
        "OXIDATION",
        "claim_id",
        "contradiction_id",
        "epistemic_class",
    ):
        if forbidden.casefold() in rendered.casefold():
            raise LiveSessionError("renderer leaked private orchestration metadata")
    return rendered


def challenge_ids_for_g02(session: dict, case: dict, cfg: dict) -> list[str]:
    derived = session_api.replay_session(case, session)
    return [
        claim["claim_id"]
        for claim in derived["claims"]
        if claim["npc_id"] == "V-G02-A"
        and claim["proposition_key"] == cfg["annotation"]["proposition_key"]
    ]


def case_sha(case: dict) -> str:
    return hashlib.sha256(canonical(case).encode("utf-8")).hexdigest()


def mandatory_public_routes_intact(case: dict) -> bool:
    validate_case(case)
    assets = {x["id"]: x for x in case["assets"]}
    for lead in case["leads"]:
        if not lead["mandatory"]:
            continue
        if not any(
            route["kind"] == "asset"
            and assets[route["source"]]["public"] is True
            for route in lead["routes"]
        ):
            return False
    return True


def _save_partial(out_root: Path, result: dict) -> None:
    write_json(out_root / "raw" / "run.json", result)


def _hash_inventory(root: Path) -> str:
    rows = []
    for path in sorted(x for x in root.rglob("*") if x.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel != "provenance/hashes.sha256":
            rows.append(f"{file_sha(path)}  {rel}")
    return "\n".join(rows) + "\n"


def archive_dataset(out_root: Path, result: dict, cfg: dict) -> Path:
    dataset = out_root / "dataset"
    if dataset.exists():
        shutil.rmtree(dataset)
    (dataset / "raw").mkdir(parents=True)
    (dataset / "provenance").mkdir(parents=True)
    write_json(dataset / "raw" / "run.json", result)
    derived = result["final_derived"]
    manifest = {
        "schema": "detective-003b-live-session-archive-v1",
        "experiment": "DETECTIVE-003B",
        "issue": 142,
        "source_head": result["repository"]["head_sha"],
        "workflow_run_id": result["repository"]["run_id"],
        "workflow_run_url": result["repository"]["run_url"],
        "status": result["status"],
        "config_sha256": file_sha(CONFIG),
        "case_sha256": file_sha(CASE),
        "session_module_git_blob_sha1":
            cfg["authority"]["session_module"]["git_blob_sha1"],
        "model_turns": len(result["model_turns"]),
        "claim_count": len(derived["claims"]),
        "contradiction_count": len(derived["contradictions"]),
        "final_session_sha256": result["final_session"]["sha256"],
        "last_event_sha256": derived["last_event_sha256"],
        "case_unchanged": result["case_before_sha256"] == result["case_after_sha256"],
        "mandatory_public_routes_intact": result["mandatory_public_routes_intact"],
        "base_model": cfg["base_model"],
        "witnesses": {
            npc: {
                "adapter_name": meta["adapter_name"],
                "release_tag": meta["release_tag"],
                "archive_sha256": meta["archive_sha256"],
                "directory_sha256": meta["directory_sha256"],
            }
            for npc, meta in cfg["witnesses"].items()
        },
        "optimizer_steps": 0,
        "paid_api_calls": 0,
        "relevance_mapping_accessed": False,
        "learner_final_accessed": False,
        "body_selection_made": False,
        "population_expansion_admitted": False,
    }
    write_json(dataset / "manifest.json", manifest)
    write_json(dataset / "provenance" / "source-artifacts.json", {
        "repository": "techrote/neurounner",
        "head_sha": result["repository"]["head_sha"],
        "workflow_run_id": result["repository"]["run_id"],
        "workflow_run_url": result["repository"]["run_url"],
        "config_path": CONFIG.relative_to(ROOT).as_posix(),
        "config_sha256": file_sha(CONFIG),
        "case_path": CASE.relative_to(ROOT).as_posix(),
        "case_git_blob_sha1": cfg["authority"]["case"]["git_blob_sha1"],
        "session_module_path":
            cfg["authority"]["session_module"]["path"],
        "session_module_git_blob_sha1":
            cfg["authority"]["session_module"]["git_blob_sha1"],
        "c2_release_git_blob_sha1":
            cfg["authority"]["c2_release"]["git_blob_sha1"],
        "c3_release_git_blob_sha1":
            cfg["authority"]["c3_release"]["git_blob_sha1"],
        "base_weight_sha256": cfg["base_model"]["weight_sha256"],
        "adapter_archive_sha256": {
            npc: meta["archive_sha256"] for npc, meta in cfg["witnesses"].items()
        },
        "adapter_directory_sha256": {
            npc: meta["directory_sha256"] for npc, meta in cfg["witnesses"].items()
        },
        "external_provider_calls": 0,
        "training_steps": 0,
        "relevance_mapping_accessed": False,
        "learner_final_accessed": False,
    })
    (dataset / "README.md").write_text(
        "# DETECTIVE-003B live two-witness session smoke\n\n"
        "Source-bound no-training integration of V-G02-A C3 and V-G09-C C2 "
        "through the DETECTIVE-003A authoritative interview ledger. Raw model "
        "dialogue is untrusted. The only automatic claim annotation is a "
        "fixture-local literal HH:MM extractor: exactly one distinct time emits "
        "a game-owned statement claim; zero or multiple times emit no claim. "
        "No canonical-truth/scorer lookup is used. Public transfer-card evidence "
        "is presented by game authority between turns. The archive does not "
        "admit either model for production/game use and does not access R1 "
        "hidden mappings or any learner-final prompts.\n",
        encoding="utf-8",
    )
    (dataset / "provenance" / "hashes.sha256").write_text(
        _hash_inventory(dataset), encoding="utf-8"
    )
    return dataset


def _generate(model, tok, messages: list[dict], cfg: dict) -> dict:
    import torch

    ids = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if ids.shape[-1] > cfg["runtime"]["max_prompt_tokens"]:
        raise LiveSessionError("model prompt token budget exceeded")
    started = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(
            ids,
            max_new_tokens=cfg["runtime"]["max_new_tokens_per_turn"],
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    elapsed = time.monotonic() - started
    tail = generated[0, ids.shape[-1]:].tolist()
    return {
        "prompt_token_count": int(ids.shape[-1]),
        "prompt_token_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        "generated_token_ids": tail,
        "generated_tokens": len(tail),
        "generation_seconds": elapsed,
        "completion_tokens_per_second": (
            len(tail) / elapsed if elapsed else None
        ),
        "hit_max_new_tokens": (
            len(tail) == cfg["runtime"]["max_new_tokens_per_turn"]
        ),
        "response": tok.decode(tail, skip_special_tokens=True),
        "response_with_special_tokens": tok.decode(
            tail, skip_special_tokens=False
        ),
    }


def live_run(
    g02_archive: Path,
    g09_archive: Path,
    out_root: Path,
) -> dict:
    cfg, case, _, _ = load_contract()
    case_original = copy.deepcopy(case)
    case_before = case_sha(case)

    import peft
    import torch
    import transformers
    from huggingface_hub import hf_hub_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        raise LiveSessionError("003B is CPU-only")
    torch.set_num_threads(cfg["runtime"]["threads"])
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(cfg["runtime"]["seed"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    result = {
        "schema": "detective-003b-live-session-run-v1",
        "experiment": "DETECTIVE-003B",
        "issue": 142,
        "status": "RUNNING",
        "repository": {
            "head_sha":
                os.environ.get("DETECTIVE_HEAD_SHA")
                or os.environ.get("GITHUB_SHA")
                or "local",
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_url": (
                f"https://github.com/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/"
                f"{os.environ.get('GITHUB_RUN_ID')}"
                if os.environ.get("GITHUB_RUN_ID")
                and os.environ.get("GITHUB_REPOSITORY")
                else None
            ),
        },
        "config_sha256": file_sha(CONFIG),
        "case_file_sha256": file_sha(CASE),
        "case_before_sha256": case_before,
        "case_after_sha256": None,
        "source": {},
        "environment": {},
        "model_turns": [],
        "evidence_actions": [],
        "final_session": None,
        "final_derived": None,
        "mandatory_public_routes_intact": None,
        "optimizer_steps": 0,
        "paid_api_calls": 0,
        "relevance_mapping_accessed": False,
        "learner_final_accessed": False,
        "body_selection_made": False,
        "population_expansion_admitted": False,
    }
    _save_partial(out_root, result)

    adapters_root = out_root / "adapters"
    g02_dir = adapters_root / "g02"
    g09_dir = adapters_root / "g09"
    g02_meta = cfg["witnesses"]["V-G02-A"]
    g09_meta = cfg["witnesses"]["V-G09-C"]
    result["source"]["g02_adapter"] = safe_unpack_adapter(
        g02_archive,
        g02_dir,
        expected_sha256=g02_meta["archive_sha256"],
        expected_bytes=g02_meta["archive_bytes"],
        expected_directory_sha256=g02_meta["directory_sha256"],
    )
    result["source"]["g09_adapter"] = safe_unpack_adapter(
        g09_archive,
        g09_dir,
        expected_sha256=g09_meta["archive_sha256"],
        expected_bytes=g09_meta["archive_bytes"],
        expected_directory_sha256=g09_meta["directory_sha256"],
    )

    base_weight = Path(hf_hub_download(
        cfg["base_model"]["repo"],
        filename=cfg["base_model"]["weight_file"],
        revision=cfg["base_model"]["revision"],
    ))
    if file_sha(base_weight) != cfg["base_model"]["weight_sha256"]:
        raise LiveSessionError("base weight SHA mismatch")
    result["source"]["base_weight"] = {
        "path": str(base_weight),
        "sha256": file_sha(base_weight),
        "bytes": base_weight.stat().st_size,
    }
    _save_partial(out_root, result)

    started = time.monotonic()
    tok = AutoTokenizer.from_pretrained(
        cfg["base_model"]["repo"],
        revision=cfg["base_model"]["revision"],
        trust_remote_code=False,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"]["repo"],
        revision=cfg["base_model"]["revision"],
        torch_dtype=torch.float32,
        trust_remote_code=False,
    )
    base.config.use_cache = True
    model = PeftModel.from_pretrained(
        base,
        g02_dir,
        adapter_name=g02_meta["adapter_name"],
        is_trainable=False,
    )
    model.load_adapter(
        g09_dir,
        adapter_name=g09_meta["adapter_name"],
        is_trainable=False,
    )
    model.eval()
    if any(p.requires_grad for p in model.parameters()):
        raise LiveSessionError("003B unexpectedly has trainable parameters")
    load_seconds = time.monotonic() - started
    result["environment"] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu_model(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "load_seconds": load_seconds,
        "rss_after_load_kib": current_rss_kib(),
        "peak_rss_after_load_kib":
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    _save_partial(out_root, result)

    session = session_api.start_session(
        case,
        cfg["session"]["participants"],
        cfg["session"]["session_key"],
    )
    total_generated = 0

    for step in cfg["session"]["sequence"]:
        if step["action"] == "evidence":
            challenged = challenge_ids_for_g02(session, case, cfg)
            session = session_api.present_evidence(
                case,
                session,
                step["npc_id"],
                step["asset_id"],
                challenged_claim_ids=challenged,
            )
            result["evidence_actions"].append({
                "step_id": step["id"],
                "npc_id": step["npc_id"],
                "asset_id": step["asset_id"],
                "challenge_policy": step["challenge_policy"],
                "challenged_claim_ids": challenged,
                "session_sha256_after": session["sha256"],
            })
            _save_partial(out_root, result)
            continue

        prompt_receipt = session_api.build_prompt(
            case,
            session,
            step["npc_id"],
            step["question"],
            include_history=step["include_history"],
        )
        rendered_user = render_session_prompt(prompt_receipt["prompt"], cfg)
        messages = [
            {
                "role": "system",
                "content": cfg["prompt_renderer"]["system"],
            },
            {"role": "user", "content": rendered_user},
        ]
        witness = cfg["witnesses"][step["npc_id"]]
        model.set_adapter(witness["adapter_name"])
        generated = _generate(model, tok, messages, cfg)
        total_generated += generated["generated_tokens"]
        if total_generated > cfg["runtime"]["max_generated_tokens_total"]:
            raise LiveSessionError("total generation budget exceeded")

        extraction = extract_time_annotation(generated["response"], cfg)
        before_session_sha = session["sha256"]
        session = session_api.record_turn(
            case,
            session,
            prompt_receipt,
            generated["response"],
            extraction["annotations"],
        )
        result["model_turns"].append({
            "step_id": step["id"],
            "npc_id": step["npc_id"],
            "adapter_name": witness["adapter_name"],
            "adapter_directory_sha256": witness["directory_sha256"],
            "session_sha256_before": before_session_sha,
            "prompt_receipt": prompt_receipt,
            "rendered_user_prompt": rendered_user,
            "messages": messages,
            **generated,
            "annotation": extraction,
            "session_sha256_after": session["sha256"],
        })
        _save_partial(out_root, result)

    if len(result["model_turns"]) != cfg["runtime"]["model_turns"]:
        raise LiveSessionError("incomplete model turn count")
    final_derived = session_api.replay_session(case, session)
    case_after = case_sha(case)
    if case != case_original or case_after != case_before:
        raise LiveSessionError("canonical case mutated during live session")
    routes_ok = mandatory_public_routes_intact(case)
    if not routes_ok:
        raise LiveSessionError("mandatory public clue route lost")

    result["status"] = "integration_pass"
    result["case_after_sha256"] = case_after
    result["final_session"] = session
    result["final_derived"] = final_derived
    result["mandatory_public_routes_intact"] = routes_ok
    result["total_generated_tokens"] = total_generated
    result["wall_seconds"] = sum(
        x["generation_seconds"] for x in result["model_turns"]
    )
    result["peak_rss_kib"] = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    )
    result["decision"] = {
        "integration_pass": True,
        "claim_count": len(final_derived["claims"]),
        "contradiction_count": len(final_derived["contradictions"]),
        "minimum_claim_count_required": 0,
        "minimum_contradiction_count_required": 0,
        "training_admitted": False,
        "body_selection_made": False,
        "population_expansion_admitted": False,
    }
    _save_partial(out_root, result)
    dataset = archive_dataset(out_root, result, cfg)

    del model, base
    gc.collect()
    return {
        "dataset": str(dataset),
        "status": result["status"],
        "model_turns": len(result["model_turns"]),
        "generated_tokens": total_generated,
        "claim_count": len(final_derived["claims"]),
        "contradiction_count": len(final_derived["contradictions"]),
        "session_sha256": session["sha256"],
        "optimizer_steps": 0,
        "population_expansion_admitted": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g02-archive", type=Path, required=True)
    ap.add_argument("--g09-archive", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(
        live_run(args.g02_archive, args.g09_archive, args.out),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
