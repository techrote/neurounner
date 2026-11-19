#!/usr/bin/env python3
"""Frozen six-member VILLAGE-001F attention-mask pilot.

Preparation commands are offline. ML commands are explicit and never run from
plan/self-test. The two training arms execute the unchanged D engine; their only
optimization difference is pair_batch's attention mask. Original evidence is read
only. The new Q suite is used exclusively by the separate inference command.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from village_001_manifest import build_manifest
from village_001e_generation_checkpoint import (
    GenerationCheckpoint, json_bytes, validate_token_custody, write_immutable,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/village-001/diagnostic-20261119/worker-config.json"
PANEL = ["V-G01-A", "V-G03-B", "V-G05-C", "V-G07-A", "V-G09-B", "V-G11-C"]
MASKS = ("historical_token_id_mask", "length_derived_mask")
FAMILIES = ("embodied_decision", "mundane", "social", "conflicting_motives", "failure_recovery", "abstract_unfamiliar")
RUNTIME = {"torch": "2.14.1+cpu", "transformers": "4.57.6", "peft": "0.21.2"}
HISTORICAL_FIRST = {"V-G01-A", "V-G05-C", "V-G09-B"}
PROVENANCE_DIR = "training-invocations"
TRACE_KEYS = ("losses", "preference_losses", "anchor_losses", "train_margins")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def save(path: Path, value) -> None:
    write_immutable(path, json_bytes(value))


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def training_arm_order(cfg: dict, member: str) -> list[str]:
    policy = cfg["training"]["paired_execution"]
    expected = {m: list(MASKS if m in HISTORICAL_FIRST else reversed(MASKS)) for m in PANEL}
    if policy.get("member_arm_order") != expected or member not in expected:
        raise RuntimeError("predeclared paired training order changed")
    if policy.get("member_jobs") != 6 or policy.get("fresh_cli_processes_per_job") != 2:
        raise RuntimeError("paired training execution cardinality changed")
    return list(expected[member])


def _require(value, message):
    if not value:
        raise RuntimeError(message)


def _training_state(root: Path) -> dict:
    result, checkpoint = root / "result.json", root / "checkpoint/progress.json"
    if result.is_file():
        row = read(result)
        values = row["training"]
        step = values["optimizer_steps"]
        source = {"kind": "result", "sha256": sha(result)}
    elif checkpoint.is_file():
        values = read(checkpoint)
        step = values["next_step"]
        source = {"kind": "checkpoint", "sha256": sha(checkpoint), "source_run_id": values.get("source_run_id")}
    else:
        values, step, source = {key: [] for key in TRACE_KEYS}, 0, {"kind": "empty", "sha256": None}
    _require(type(step) is int and 0 <= step <= 120, "training provenance step outside fixed120-step recipe")
    traces = {key: values[key] for key in TRACE_KEYS}
    _validate_traces(traces, step)
    return {"step": step, "traces": traces, "source": source}


def _validate_traces(traces: dict, step: int) -> None:
    _require(set(traces) == set(TRACE_KEYS), "training provenance trace fields changed")
    for values in traces.values():
        _require(isinstance(values, list) and len(values) == step
                 and all(type(value) in (int, float) and math.isfinite(value) for value in values),
                 "training provenance trace is incomplete or nonfinite")


def _provenance_files(root: Path) -> dict[str, Path]:
    directory = root / PROVENANCE_DIR
    if not directory.exists():
        return {}
    _require(directory.is_dir() and not directory.is_symlink(), "invalid training provenance directory")
    files = {}
    for path in sorted(directory.iterdir()):
        _require(path.is_file() and not path.is_symlink(), "nonregular training provenance entry")
        _require(re.fullmatch(r"invocation-[0-9]{4}-(start|finish|checkpoint-[0-9]{6})\.json", path.name),
                 "unexpected training provenance filename")
        files[path.relative_to(root).as_posix()] = path
    return files


def _file_inventory(files: dict[str, Path]) -> dict:
    return {name: {"sha256": sha(path), "bytes": path.stat().st_size} for name, path in sorted(files.items())}


def training_execution_metadata(member: str) -> dict:
    cpu = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    boot = Path("/proc/sys/kernel/random/boot_id")
    packages = {"python": platform.python_version()}
    for name in RUNTIME:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"repository": os.environ.get("GITHUB_REPOSITORY"), "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"), "job": os.environ.get("GITHUB_JOB"),
        "member_id": member, "paired_job_nonce": os.environ.get("F_PAIRED_JOB_NONCE"),
        "source_head": os.environ.get("GITHUB_SHA"), "runner_name": os.environ.get("RUNNER_NAME"),
        "boot_id": boot.read_text().strip() if boot.is_file() else None, "hostname": platform.node(),
        "cpu_model": cpu, "cpu_count": os.cpu_count(), "platform": platform.platform(), "packages": packages,
        "process_id": os.getpid(), "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "started_monotonic_ns": time.monotonic_ns()}


def _timestamp(value, label):
    _require(isinstance(value, str) and bool(value), label + " timestamp missing")
    try:
        result = datetime.datetime.fromisoformat(value)
    except ValueError as exc:
        raise RuntimeError(label + " timestamp invalid") from exc
    _require(result.tzinfo is not None and result.utcoffset() == datetime.timedelta(0),
             label + " timestamp must be UTC")
    return result


def _validate_execution(value: dict, member: str) -> None:
    text_fields = ("repository", "run_id", "run_attempt", "job", "member_id", "paired_job_nonce",
                   "source_head", "runner_name", "boot_id", "hostname", "cpu_model", "platform")
    _require(isinstance(value, dict) and all(isinstance(value.get(key), str) and value[key].strip()
                                           for key in text_fields), "training invocation runner identity incomplete")
    _require(value["member_id"] == member, "training invocation execution member changed")
    _require(all(type(value.get(key)) is int and value[key] > 0
                 for key in ("process_id", "cpu_count", "started_monotonic_ns")),
             "training invocation runner counts or clock invalid")
    _require(isinstance(value.get("packages"), dict)
             and isinstance(value["packages"].get("python"), str) and value["packages"]["python"],
             "training invocation package metadata missing")
    _timestamp(value.get("started_at_utc"), "training start")


def validate_training_provenance(root: Path, cfg: dict, *, require_complete=True) -> tuple[dict, dict[str, Path]]:
    """Validate durable optimizer provenance using receipts only, without ML imports.

    A hard interruption may leave a native checkpoint just before its provenance
    sidecar is written. Such retained steps remain explicit unknown-host segments;
    they cannot prove a fresh same-runner pair and do not invalidate the pair.
    """
    identity_path = root / "training-identity.json"
    _require(identity_path.is_file() and not identity_path.is_symlink(), "training identity missing for provenance")
    identity = read(identity_path)
    _require(identity.get("worker_config_sha256") == digest(cfg), "training provenance config changed")
    member, arm = identity["member_id"], identity["mask_mode"]
    order = training_arm_order(cfg, member)
    _require(arm in MASKS, "training provenance arm invalid")
    identity_sha = sha(identity_path)
    files = _provenance_files(root)
    groups = {}
    for name, path in files.items():
        value = read(path)
        _require(path.read_bytes() == json_bytes(value), "training provenance encoding changed")
        ordinal = int(path.name.split("-")[1])
        _require(ordinal > 0, "invalid training invocation ordinal")
        groups.setdefault(ordinal, {})[path.name] = value
    _require(sorted(groups) == list(range(1, len(groups) + 1)), "training invocation history has a gap")
    invocations, segments, unknown, unfinished, copies, resumes = [], [], [], [], [], []
    previous, cursor = {}, 0
    native = _training_state(root)
    def check_state(state):
        step = state["step"]
        _require(type(step) is int and 0 <= step <= native["step"], "provenance exceeds retained optimizer trajectory")
        _validate_traces(state["traces"], step)
        for key in TRACE_KEYS:
            _require(state["traces"][key] == native["traces"][key][:step], "retained optimizer trace changed: " + key)
        return step
    for ordinal, records in sorted(groups.items()):
        prefix = f"invocation-{ordinal:04d}"
        start = records.get(prefix + "-start.json")
        _require(isinstance(start, dict), "training invocation lacks its immutable start")
        _require(start.get("schema") == "village-001f-training-invocation-start-v1"
                 and start.get("invocation_ordinal") == ordinal and start.get("training_identity_sha256") == identity_sha,
                 "training invocation identity changed")
        _require((start["member_id"], start["mask_mode"], start["arm_order"], start["arm_position"])
                 == (member, arm, order, order.index(arm) + 1), "training invocation member/arm/order changed")
        _require(start["previous_provenance_sha256"] == digest(previous), "inherited training provenance disappeared or changed")
        _validate_execution(start["execution"], member)
        _require(type(start["resume_requested"]) is bool, "training invocation resume classification invalid")
        first = check_state(start["source_state"])
        _require(start["start_step"] == first and first >= cursor, "training resume discarded prior optimizer steps")
        if first > cursor:
            _require(invocations and unfinished and unfinished[-1] == ordinal - 1 and start["resume_requested"],
                     "optimizer provenance gap without an interrupted prior invocation")
            unknown.append({"start_step": cursor, "end_step": first, "steps": first - cursor,
                            "reason": "native checkpoint survived without a matching prior provenance receipt"})
        kind = start["source_kind"]
        _require(kind in {"fresh", "partial_checkpoint", "complete_copy", "empty_resume"}, "unknown training invocation source")
        _require((kind == "fresh") == (not start["resume_requested"]), "fresh/resume provenance classification changed")
        _require(kind != "fresh" or (ordinal == 1 and first == 0), "fresh invocation has inherited work")
        _require(kind != "complete_copy" or first == 120, "completed copy starts before terminal step")
        if start["resume_requested"]:
            resumes.append(ordinal)
        start_path = files[f"{PROVENANCE_DIR}/{prefix}-start.json"]
        current = {f"{PROVENANCE_DIR}/{prefix}-start.json": _file_inventory({"start": start_path})["start"]}
        checkpoints, last = [], first
        for name, value in sorted(records.items()):
            match = re.fullmatch(prefix + r"-checkpoint-([0-9]{6})\.json", name)
            if not match:
                continue
            step = check_state(value["state"])
            _require(value.get("schema") == "village-001f-training-invocation-checkpoint-v1"
                     and value["start_sha256"] == sha(start_path) and value["training_identity_sha256"] == identity_sha,
                     "training checkpoint provenance identity changed")
            _require(step == int(match[1]) and last < step <= 120, "training checkpoint ordering changed")
            checkpoints.append(value)
            last = step
            relative = f"{PROVENANCE_DIR}/{name}"
            current.update(_file_inventory({relative: files[relative]}))
        finish = records.get(prefix + "-finish.json")
        if finish is None:
            unfinished.append(ordinal)
        else:
            _require(finish.get("schema") == "village-001f-training-invocation-finish-v1"
                     and finish["start_sha256"] == sha(start_path) and finish["training_identity_sha256"] == identity_sha,
                     "training completion provenance identity changed")
            _require(finish["provenance_before_finish_sha256"] == digest({**previous, **current}),
                     "training invocation lost checkpoint provenance")
            _timestamp(finish.get("finished_at_utc"), "training finish")
            _require(type(finish.get("finished_monotonic_ns")) is int
                     and finish["finished_monotonic_ns"] >= start["execution"]["started_monotonic_ns"],
                     "training invocation finish preceded its start")
            end = check_state(finish["state"])
            _require(finish["end_step"] == end and end >= last and finish["status"] in {"completed", "failed", "interrupted"},
                     "training invocation terminal state changed")
            if finish["status"] == "completed":
                _require(end == 120, "completed training invocation did not reach120steps")
                _require(finish["training_result_sha256"] == sha(root / "result.json")
                         and finish["training_receipt_sha256"] == sha(root / "training.json"),
                         "completed invocation differs from immutable training receipts")
            else:
                unfinished.append(ordinal)
            last = end
            relative = f"{PROVENANCE_DIR}/{prefix}-finish.json"
            current.update(_file_inventory({relative: files[relative]}))
        _require(kind != "complete_copy" or last == first, "complete copy falsely records optimizer work")
        if kind == "complete_copy":
            copies.append(ordinal)
        if last > first:
            _require(all(start["execution"]["packages"].get(k) == value for k, value in RUNTIME.items()),
                     "optimizer invocation runtime pins changed")
            segments.append({"invocation_ordinal": ordinal, "start_step": first, "end_step": last,
                             "steps": last - first, "execution": start["execution"]})
        invocations.append({"ordinal": ordinal, "start": start, "finish": finish,
                            "checkpoints": checkpoints, "retained_end_step": last})
        previous.update(current)
        cursor = last
    if native["step"] > cursor:
        _require(invocations and unfinished and unfinished[-1] == len(invocations),
                 "retained optimizer work has no interrupted invocation")
        unknown.append({"start_step": cursor, "end_step": native["step"], "steps": native["step"] - cursor,
                        "reason": "terminal native state survived an interrupted provenance write"})
        cursor = native["step"]
    if require_complete:
        _require(cursor == 120 and (root / "result.json").is_file() and (root / "training.json").is_file(),
                 "complete optimizer provenance requires immutable120-step training receipts")
    return {"schema": "village-001f-training-provenance-summary-v1", "member_id": member, "mask_mode": arm,
        "training_identity_sha256": identity_sha, "invocations": invocations, "optimizer_segments": segments,
        "unattributed_optimizer_segments": unknown, "zero_step_copies": copies,
        "unfinished_invocations": unfinished, "resume_invocations": resumes,
        "retained_optimizer_steps": cursor, "files": _file_inventory(files)}, files


class TrainingInvocation:
    def __init__(self, root: Path, cfg: dict, member: str, arm: str, *, resumed: bool, complete: bool):
        previous, files = validate_training_provenance(root, cfg, require_complete=False)
        self.root, self.cfg = root, cfg
        self.ordinal = len(previous["invocations"]) + 1
        _require(self.ordinal <= 9999, "training invocation history exhausted")
        self.prefix = f"invocation-{self.ordinal:04d}"
        self.start_path = root / PROVENANCE_DIR / (self.prefix + "-start.json")
        self.identity_sha = sha(root / "training-identity.json")
        state = _training_state(root)
        order = training_arm_order(cfg, member)
        kind = "complete_copy" if complete else "partial_checkpoint" if state["step"] else "empty_resume" if resumed else "fresh"
        self.start = {"schema": "village-001f-training-invocation-start-v1", "invocation_ordinal": self.ordinal,
            "training_identity_sha256": self.identity_sha, "previous_provenance_sha256": digest(_file_inventory(files)),
            "member_id": member, "mask_mode": arm, "arm_order": order, "arm_position": order.index(arm) + 1,
            "resume_requested": resumed, "source_kind": kind, "start_step": state["step"], "source_state": state,
            "execution": training_execution_metadata(member)}
        _validate_execution(self.start["execution"], member)
        save(self.start_path, self.start)

    def checkpoint(self):
        state = _training_state(self.root)
        save(self.root / PROVENANCE_DIR / f"{self.prefix}-checkpoint-{state['step']:06d}.json",
             {"schema": "village-001f-training-invocation-checkpoint-v1", "start_sha256": sha(self.start_path),
              "training_identity_sha256": self.identity_sha, "state": state})

    def finish(self, status: str, error_type=None):
        state = _training_state(self.root)
        save(self.root / PROVENANCE_DIR / (self.prefix + "-finish.json"),
             {"schema": "village-001f-training-invocation-finish-v1", "start_sha256": sha(self.start_path),
              "training_identity_sha256": self.identity_sha, "status": status, "error_type": error_type,
              "end_step": state["step"], "state": state,
              "provenance_before_finish_sha256": digest(_file_inventory(_provenance_files(self.root))),
              "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "finished_monotonic_ns": time.monotonic_ns(),
              "training_result_sha256": sha(self.root / "result.json") if (self.root / "result.json").is_file() else None,
              "training_receipt_sha256": sha(self.root / "training.json") if (self.root / "training.json").is_file() else None})


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def directory_hash(path: Path) -> str:
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"ordinary adapter directory required: {path}")
    h = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_symlink() or not (item.is_dir() or item.is_file()):
            raise RuntimeError(f"unsafe adapter entry: {item}")
        if item.is_file():
            relative = item.relative_to(path).as_posix().encode()
            h.update(len(relative).to_bytes(4, "big"))
            h.update(relative)
            h.update(bytes.fromhex(sha(item)))
    return h.hexdigest()


def adapter_fingerprint(path: Path) -> dict:
    """Distinguish learned-weight equality from README/config/archive changes."""
    import torch
    from safetensors.torch import load_file
    weights = path / "adapter_model.safetensors"
    tensors = load_file(str(weights), device="cpu")
    entries = {}
    aggregate = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        payload = tensor.view(torch.uint8).numpy().tobytes()
        row = {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "sha256": hashlib.sha256(payload).hexdigest()}
        entries[name] = row
        aggregate.update(json_bytes({"name": name, **row}))
    return {"safetensors_file_sha256": sha(weights), "canonical_tensor_sha256": aggregate.hexdigest(),
        "tensor_count": len(entries), "tensors": entries}


def validate_output(path: Path, *, allow_empty=True) -> None:
    resolved = path.resolve()
    if path.is_symlink() or resolved == ROOT.resolve():
        raise RuntimeError("unsafe diagnostic output")
    for protected in (ROOT / "experiments", ROOT / "datasets", ROOT / "scripts", ROOT / ".git"):
        if resolved == protected.resolve() or protected.resolve() in resolved.parents:
            raise RuntimeError("runtime diagnostics cannot write inside repository evidence/source directories")
    if path.exists() and (not path.is_dir() or any(path.iterdir()) or not allow_empty):
        raise RuntimeError("output must be absent or empty; resume into a new directory")


def load_contract(config_path: Path) -> tuple[dict, dict, dict, dict]:
    cfg = read(config_path)
    if cfg.get("schema") != "village-001f-worker-mask-pilot-v1" or cfg.get("members") != PANEL:
        raise RuntimeError("worker diagnostic contract/panel drift")
    if cfg.get("runtime") != RUNTIME or cfg.get("arms") != list(MASKS):
        raise RuntimeError("worker runtime/mask contract drift")
    for member in PANEL:
        training_arm_order(cfg, member)
    for relative, expected in cfg["sources_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"frozen worker source drift: {relative}")
    suite_path = ROOT / cfg["eval_config"]
    if sha(suite_path) != cfg["eval_sha256"]:
        raise RuntimeError("new Q diagnostic reserve changed")
    suite = read(suite_path)
    if [r.get("id") for r in suite.get("scenarios", [])] != [f"Q{i:02d}" for i in range(1, 13)]:
        raise RuntimeError("diagnostic reserve IDs/count drift")
    if Counter(r["family"] for r in suite["scenarios"]) != {family: 2 for family in FAMILIES}:
        raise RuntimeError("diagnostic reserve family balance drift")
    if suite.get("max_new_tokens") != 160 or suite.get("do_sample") is not False:
        raise RuntimeError("diagnostic inference contract drift")
    campaign, pref = read(ROOT / cfg["campaign_config"]), read(ROOT / cfg["preference_config"])
    if campaign["base_model"]["id"] != cfg["base_model"]["id"] or campaign["base_model"]["revision"] != cfg["base_model"]["revision"]:
        raise RuntimeError("diagnostic base revision drift")
    if pref["pair_generation"]["pair_count"] != 60 or pref["preference_training"]["epochs"] != 2:
        raise RuntimeError("matched pilot must retain all original60pairs/120steps")
    original_eval = read(ROOT / campaign["corpus"]["common_heldout_config"])["eval"]
    if suite["system"] != original_eval["system"]:
        raise RuntimeError("public inference system changed")
    return cfg, suite, campaign, pref


def forbidden_prompts(cfg: dict, campaign: dict) -> set[str]:
    prompts = set()
    for relative in cfg["sources_sha256"]:
        if relative.startswith("datasets/village-001/") and relative.endswith(".jsonl"):
            for row in jsonl(ROOT / relative):
                if isinstance(row.get("input"), str):
                    prompts.add(norm(row["input"]))
    historical = read(ROOT / campaign["corpus"]["common_heldout_config"])["eval"]
    prompts.update(norm(row["prompt"]) for row in historical["scenarios"])
    # The E prompts are read only for exact collision exclusion. No E responses,
    # scores, rankings, targets, or selected failures enter this experiment.
    e_suite = read(ROOT / "experiments/village-001/qualification-20261119/final-eval.json")
    prompts.update(norm(row["prompt"]) for row in e_suite["scenarios"])
    return prompts


def member_contract(campaign: dict, member: str) -> tuple[dict, dict]:
    population = build_manifest(campaign)
    if member not in PANEL:
        raise RuntimeError("member is outside the predeclared six-member panel")
    return population, next(row for row in population["members"] if row["member_id"] == member)


def pair_path(member: str) -> Path:
    return ROOT / "datasets/village-001/preference" / f"{member}.jsonl"


def source_result(member: str, stage: str) -> dict:
    root = "sft-evidence" if stage == "C" else "preference-evidence"
    return read(ROOT / "experiments/village-001" / root / "results" / f"{member}.json")


def selected_pairs(member: str, campaign: dict) -> list[dict]:
    pairs = jsonl(pair_path(member))
    common = read(ROOT / campaign["corpus"]["common_heldout_config"])["eval"]["scenarios"]
    rank = lambda value: hashlib.sha256(("VILLAGE-001F-FORWARD-ONLY-V1:" + value).encode()).hexdigest()
    selected = [min((r for r in common if r["family"] == family), key=lambda r: rank(r["id"]))["id"] for family in FAMILIES]
    probes = source_result(member, "C")["member_probe_evaluation"]["responses"]
    for sibling_source in sorted({r["source_member_id"] for r in probes}):
        selected.append(min((r for r in probes if r["source_member_id"] == sibling_source), key=lambda r: rank(r["scenario_id"]))["scenario_id"])
    result = []
    for scenario_id in selected:
        candidates = [r for r in pairs if r["scenario_id"] == scenario_id]
        result.append(min(candidates, key=lambda r: r["pair_id"]))
    if len(result) != 8:
        raise RuntimeError("forward diagnostic pair balance drift")
    return result


def run_identity(config_path: Path, cfg: dict, *, kind: str, member=None, stage=None, adapter_sha=None) -> dict:
    return {
        "schema": "village-001f-worker-execution-identity-v1", "experiment": "VILLAGE-001F",
        "kind": kind, "member_id": member, "stage": stage, "adapter_directory_sha256": adapter_sha,
        "worker_config_sha256": sha(config_path), "new_Q_reserve_sha256": cfg["eval_sha256"],
        "driver_sha256": sha(Path(__file__)), "frozen_sources_sha256": cfg["sources_sha256"],
        "base_model": cfg["base_model"], "runtime": RUNTIME,
    }


def runtime_modules():
    import torch
    import peft
    import transformers
    actual = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}
    if actual != RUNTIME:
        raise RuntimeError(f"fixed pilot runtime mismatch: {actual}")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return torch, peft, transformers


def tokenizer_for(cfg: dict):
    import transformers
    if transformers.__version__ != RUNTIME["transformers"]:
        raise RuntimeError("tokenizer runtime mismatch")
    tok = transformers.AutoTokenizer.from_pretrained(cfg["base_model"]["id"], revision=cfg["base_model"]["revision"])
    primary = read(ROOT / "experiments/village-001/diagnostic-20261119/worker-source-tokenizer-config.json")
    if tok.chat_template != primary["chat_template"] or tok.eos_token_id != 2 or tok.pad_token_id != 2:
        raise RuntimeError("loaded tokenizer differs from pinned primary source")
    return tok


def native_encoding(tok, system: str, user: str, answer: str) -> dict:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    full = messages + [{"role": "assistant", "content": answer}]
    prompt_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full_ids = tok.apply_chat_template(full, tokenize=True, add_generation_prompt=False)
    common = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        common += 1
    if len(full_ids) > 512 or common >= len(full_ids):
        raise RuntimeError("original training encoding violates512token/assistant-label contract")
    masked_real = [i for i, value in enumerate(full_ids) if value == tok.pad_token_id]
    return {
        "input_token_ids": list(full_ids), "prompt_token_ids": list(prompt_ids),
        "labels": [-100] * common + list(full_ids[common:]),
        "prompt_common_prefix_tokens": common, "prompt_token_count": len(prompt_ids),
        "prompt_tokens_supervised_by_historical_prefix_rule": len(prompt_ids) - common,
        "sequence_tokens": len(full_ids), "supervised_tokens": len(full_ids) - common,
        "historical_attention_mask": [int(value != tok.pad_token_id) for value in full_ids],
        "length_attention_mask": [1] * len(full_ids),
        "real_tokens_masked_by_historical_rule": masked_real,
        "real_masked_token_strings": [tok.convert_ids_to_tokens(full_ids[i]) for i in masked_real],
        "rendered_native_training_sequence": tok.apply_chat_template(full, tokenize=False, add_generation_prompt=False),
    }


def length_mask_values(lengths: list[int], width: int) -> list[list[int]]:
    return [[int(index < length) for index in range(width)] for length in lengths]


def token_audit(args, cfg, suite, campaign):
    validate_output(args.out)
    tok = tokenizer_for(cfg)
    records, per_member = [], []
    for member in PANEL:
        member_records = []
        for pair in jsonl(pair_path(member)):
            for choice in ("chosen", "rejected"):
                encoded = native_encoding(tok, suite["system"], pair["input"], pair[choice])
                member_records.append({"member_id": member, "pair_id": pair["pair_id"], "scenario_id": pair["scenario_id"], "choice": choice, **encoded})
        records.extend(member_records)
        per_member.append({"member_id": member, "sequences": len(member_records),
            "sequences_with_real_tokens_masked": sum(bool(r["real_tokens_masked_by_historical_rule"]) for r in member_records),
            "real_tokens_masked": sum(len(r["real_tokens_masked_by_historical_rule"]) for r in member_records),
            "prompt_prefix_mismatch_sequences": sum(r["prompt_tokens_supervised_by_historical_prefix_rule"] != 0 for r in member_records)})
    save(args.out / "token-mask-audit.json", {"identity": run_identity(args.config, cfg, kind="tokenizer-mask-audit"),
        "new_model_generations": 0, "optimizer_steps": 0, "tokenizer_eos_id": tok.eos_token_id,
        "tokenizer_pad_id": tok.pad_token_id, "sequences": len(records), "per_member": per_member, "records": records})
    print(json.dumps({"sequences": len(records), "per_member": per_member}, indent=2))


def checked_adapter(cfg: dict, member: str, stage: str, adapter: Path | None, training_evidence: Path | None) -> tuple[Path, str]:
    if stage in ("C", "D_ORIGINAL"):
        expected = source_result(member, "C" if stage == "C" else "D")["adapter"]["sha256"]
    elif stage in ("D_REPRODUCED", "D_CORRECTED"):
        if training_evidence is None:
            raise RuntimeError("new pilot adapters require --training-evidence")
        receipt = read(training_evidence / "training.json")
        expected_arm = MASKS[0] if stage == "D_REPRODUCED" else MASKS[1]
        identity = read(training_evidence / "training-identity.json")
        if identity.get("member_id") != member or identity.get("mask_mode") != expected_arm:
            raise RuntimeError("pilot training member/arm mismatch")
        if identity.get("worker_config_sha256") != digest(cfg):
            # Config files use this module's canonical JSON encoding.
            raise RuntimeError("pilot training contract differs from inference contract")
        if receipt.get("training_result_sha256") != sha(training_evidence / "result.json"):
            raise RuntimeError("pilot training result custody mismatch")
        result = read(training_evidence / "result.json")
        if result.get("experiment") != "VILLAGE-001F" or result.get("member_id") != member or result["training"]["optimizer_steps"] != 120:
            raise RuntimeError("pilot training terminal identity/step mismatch")
        expected = receipt["adapter_directory_sha256"]
        if result["adapter"]["sha256"] != expected:
            raise RuntimeError("pilot training adapter/result mismatch")
        adapter = adapter or training_evidence / "adapter"
    else:
        raise RuntimeError(f"unknown adapter stage: {stage}")
    if adapter is None:
        raise RuntimeError("adapter path required")
    actual = directory_hash(adapter)
    if actual != expected:
        raise RuntimeError(f"{member}/{stage}: adapter differs from frozen source")
    return adapter, actual


def model_for(cfg: dict, *, member: str | None, stage: str, adapter: Path | None, training_evidence: Path | None):
    torch, peft, transformers = runtime_modules()
    tok = tokenizer_for(cfg)
    actual_sha = None
    if stage != "BASE":
        adapter, actual_sha = checked_adapter(cfg, member, stage, adapter, training_evidence)
    base = transformers.AutoModelForCausalLM.from_pretrained(cfg["base_model"]["id"], revision=cfg["base_model"]["revision"], torch_dtype=torch.float32)
    model = base if stage == "BASE" else peft.PeftModel.from_pretrained(base, adapter, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    return torch, model, tok, actual_sha


def infer(args, cfg, suite, campaign):
    if args.stage == "BASE":
        if args.member or args.adapter or args.training_evidence:
            raise RuntimeError("shared BASE inference takes no member or adapter")
        actual_sha = None
    else:
        member_contract(campaign, args.member)
        _, actual_sha = checked_adapter(cfg, args.member, args.stage, args.adapter, args.training_evidence)
    validate_output(args.out)
    identity = run_identity(args.config, cfg, kind="new-Q-native-responses", member=args.member, stage=args.stage, adapter_sha=actual_sha)
    by_id = {row["id"]: row for row in suite["scenarios"]}
    def validate_row(key, row, prior):
        expected = by_id[key]
        if any(row.get(field) != value for field, value in {"scenario_id": key, "prompt": expected["prompt"], "family": expected["family"]}.items()):
            raise RuntimeError("native diagnostic checkpoint case drift")
        validate_token_custody(row, maximum=160)
    journal = GenerationCheckpoint(out=args.out, resume=args.resume, identity=identity,
        keys=list(by_id), terminal_name="result.json", validate_row=validate_row)
    if journal.terminal("responses") is not None:
        print(json.dumps({"stage": args.stage, "member": args.member, "new_generations": 0, "replayed": 12}))
        return
    reused = len(journal.rows)
    torch, model, tok, _ = model_for(cfg, member=args.member, stage=args.stage, adapter=args.adapter, training_evidence=args.training_evidence)
    with torch.inference_mode():
        for scenario in suite["scenarios"][reused:]:
            messages = [{"role": "system", "content": suite["system"]}, {"role": "user", "content": scenario["prompt"]}]
            ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            started = time.monotonic()
            output = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=160,
                do_sample=False, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id, use_cache=True)
            elapsed = time.monotonic() - started
            generated = output[0, ids.shape[-1]:]
            row = {"scenario_id": scenario["id"], "family": scenario["family"], "prompt": scenario["prompt"],
                "response": tok.decode(generated, skip_special_tokens=True).strip(),
                "input_token_ids": ids[0].tolist(), "generated_token_ids": generated.tolist(),
                "rendered_prompt": tok.decode(ids[0], skip_special_tokens=False),
                "attention_mask": [1] * ids.shape[-1], "generated_tokens": int(generated.numel()),
                "elapsed_seconds": elapsed, "tokens_per_second": int(generated.numel()) / elapsed if elapsed else None}
            journal.append(scenario["id"], row)
            print(json.dumps({"stage": args.stage, "member": args.member, "scenario": scenario["id"], "tokens": row["generated_tokens"]}), flush=True)
    actual_adapter = None if args.stage == "BASE" else checked_adapter(cfg, args.member, args.stage, args.adapter, args.training_evidence)[0]
    result = {"schema": "village-001f-diagnostic-responses-v1", "experiment": "VILLAGE-001F",
        "stage": args.stage, "member_id": args.member, "base_control": args.stage == "BASE",
        "new_Q_reserve_sha256": cfg["eval_sha256"], "hidden_member_prompt_used": False,
        "adapter_directory_sha256": actual_sha, "adapter_weight_fingerprint": adapter_fingerprint(actual_adapter) if actual_adapter else None,
        "runtime": RUNTIME, "runner": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
        "reused_responses": reused, "new_responses": 12 - reused, "responses": journal.rows}
    journal.finish(result, "responses")


def score(args, cfg, suite, campaign):
    member_contract(campaign, args.member)
    if args.stage not in ("BASE", "C", "D_ORIGINAL"):
        raise RuntimeError("forward-only scoring uses BASE/C/original D fixed checkpoints")
    validate_output(args.out)
    torch, model, tok, adapter_sha = model_for(cfg, member=args.member, stage=args.stage, adapter=args.adapter, training_evidence=None)
    import village_001d_preference_train as historical
    pairs, rows = selected_pairs(args.member, campaign), []
    with torch.inference_mode():
        for pair in pairs:
            chosen = historical.encode_response(tok, suite["system"], pair["input"], pair["chosen"], 512)
            rejected = historical.encode_response(tok, suite["system"], pair["input"], pair["rejected"], 512)
            ids, labels, legacy = historical.pair_batch(chosen, rejected, tok.pad_token_id)
            lengths = [chosen["sequence_tokens"], rejected["sequence_tokens"]]
            corrected = torch.tensor(length_mask_values(lengths, ids.shape[1]), dtype=legacy.dtype)
            values = {}
            for mask_name, mask in ((MASKS[0], legacy), (MASKS[1], corrected)):
                lp = historical.assistant_mean_logps(model, ids, labels, mask)
                values[mask_name] = {"chosen_mean_logp": float(lp[0]), "rejected_mean_logp": float(lp[1]), "margin": float(lp[0] - lp[1])}
            rows.append({"pair_id": pair["pair_id"], "scenario_id": pair["scenario_id"], "training_diagnostic": True,
                "input_ids": ids.tolist(), "labels": labels.tolist(), "historical_attention_mask": legacy.tolist(),
                "length_attention_mask": corrected.tolist(), "real_positions_wrongly_masked": [
                    [i for i in range(length) if not int(legacy[index, i])] for index, length in enumerate(lengths)], "scores": values})
    save(args.out / "score.json", {"identity": run_identity(args.config, cfg, kind="forward-only-paired-mask-scores", member=args.member, stage=args.stage, adapter_sha=adapter_sha),
        "model_forward_batches": len(rows) * 2, "optimizer_steps": 0, "new_generations": 0, "pair_dataset_sha256": sha(pair_path(args.member)), "rows": rows})


def train(args, cfg, suite, campaign, pref):
    population, member = member_contract(campaign, args.member)
    validate_output(args.out)
    actual_parent = directory_hash(args.parent_adapter)
    if actual_parent != source_result(args.member, "C")["adapter"]["sha256"]:
        raise RuntimeError("training parent differs from frozen C adapter")
    if args.mask_mode not in MASKS:
        raise RuntimeError("unknown mask arm")
    identity = run_identity(args.config, cfg, kind="matched-preference-mask-training", member=args.member, stage=args.mask_mode, adapter_sha=actual_parent)
    identity.update({"mask_mode": args.mask_mode, "member_seed": member["seed"], "pair_dataset_sha256": sha(pair_path(args.member)),
        "optimizer_steps": 120, "training_config": pref["preference_training"], "member_contract_sha256": population["member_contract_sha256"],
        "new_Q_reserve_used_for_training": False, "original_E_F_suite_used_for_training": False,
        "post_training_legacy_T_generation_omitted_in_both_arms": True})
    if args.resume:
        if args.resume.is_symlink() or not args.resume.is_dir():
            raise RuntimeError("invalid training resume directory")
        if args.resume.resolve() == args.out.resolve() or args.resume.resolve() in args.out.resolve().parents or args.out.resolve() in args.resume.resolve().parents:
            raise RuntimeError("training resume and output must be disjoint")
        for path in args.resume.rglob("*"):
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                raise RuntimeError("unsafe training resume evidence")
        if read(args.resume / "training-identity.json") != identity:
            raise RuntimeError("training resume source/arm identity mismatch")
        validate_training_provenance(args.resume, cfg, require_complete=False)
    torch, _, _ = runtime_modules()
    import village_001d_preference_train as engine
    original_pair_batch = engine.pair_batch
    original_heldout = engine.run_heldout
    original_write_json = engine.write_json
    original_validate_complete = engine.validate_complete
    original_save_checkpoint = engine.save_checkpoint
    invocation = None
    def corrected_pair_batch(chosen, rejected, pad_id):
        ids, labels, _ = original_pair_batch(chosen, rejected, pad_id)
        lengths = [len(chosen["input_ids"]), len(rejected["input_ids"])]
        attention = torch.tensor(length_mask_values(lengths, ids.shape[1]), dtype=torch.long, device=ids.device)
        return ids, labels, attention
    if args.mask_mode == MASKS[1]:
        engine.pair_batch = corrected_pair_batch
    def write_f_result(path, value):
        if path == args.out / "result.json":
            value = {**value, "experiment": "VILLAGE-001F", "stage": "matched-mask-preference-continuation",
                "parent_issue": 9, "issue": None, "mask_mode": args.mask_mode,
                "engine_source_experiment": "VILLAGE-001D", "engine_source_sha256": sha(Path(engine.__file__)),
                "training_identity_sha256": sha(args.out / "training-identity.json"),
                "legacy_T_evaluation_intentionally_omitted": True,
                "new_Q_reserve_used_for_training": False}
        original_write_json(path, value)
    def validate_f_complete(out, member_row, population_row, parent_sha, pairs_sha, campaign_row, pref_row):
        path, adapter_dir = out / "result.json", out / "adapter"
        if not path.is_file() or not adapter_dir.is_dir():
            return False
        result = read(path)
        expected = {"experiment": "VILLAGE-001F", "stage": "matched-mask-preference-continuation",
            "member_id": member_row["member_id"], "member_contract_sha256": population_row["member_contract_sha256"],
            "parent_adapter_sha256": parent_sha, "preference_dataset_sha256": pairs_sha,
            "training_config": pref_row["preference_training"], "mask_mode": args.mask_mode,
            "training_identity_sha256": sha(out / "training-identity.json")}
        if any(result.get(key) != value for key, value in expected.items()):
            raise RuntimeError("complete F training artifact has wrong source or arm")
        if result["adapter"]["sha256"] != directory_hash(adapter_dir) or result["training"]["optimizer_steps"] != 120:
            raise RuntimeError("complete F training adapter/step custody mismatch")
        return True
    def complete_f(out, *arguments):
        nonlocal invocation
        complete = validate_f_complete(out, *arguments)
        if invocation is not None:
            raise RuntimeError("training engine initialized one invocation twice")
        invocation = TrainingInvocation(out, cfg, args.member, args.mask_mode, resumed=args.resume is not None, complete=complete)
        return complete
    def save_f_checkpoint(**kwargs):
        original_save_checkpoint(**kwargs)
        if invocation is None:
            raise RuntimeError("optimizer checkpoint preceded its invocation provenance")
        invocation.checkpoint()
    engine.write_json = write_f_result
    engine.validate_complete = complete_f
    engine.save_checkpoint = save_f_checkpoint
    # This runs only after the final optimizer step. Both arms omit the same old
    # T generations; Q remains a separately frozen, identical inference reserve.
    engine.run_heldout = lambda model, tok, eval_cfg: []
    previous_argv = list(sys.argv)
    try:
        with tempfile.TemporaryDirectory(prefix="village001f-training-seed-") as temp:
            seed_dir = Path(temp)
            save(seed_dir / "training-identity.json", identity)
            resume_source = args.resume or seed_dir
            sys.argv = [str(Path(engine.__file__)), "--config", str(ROOT / cfg["preference_config"]),
                "--member", args.member, "--pairs", str(pair_path(args.member)), "--parent-adapter", str(args.parent_adapter),
                "--out", str(args.out), "--resume", str(resume_source)]
            engine.main()
        result = read(args.out / "result.json")
        for field, expected in (("member_id", args.member), ("member_seed", member["seed"]), ("parent_adapter_sha256", actual_parent),
            ("preference_dataset_sha256", sha(pair_path(args.member))), ("training_config", pref["preference_training"])):
            if result.get(field) != expected:
                raise RuntimeError(f"matched training result {field} drift")
        if result["training"]["optimizer_steps"] != 120 or result["training"]["pair_count"] != 60:
            raise RuntimeError("matched training cardinality drift")
        if result["evaluation"]["responses"]:
            raise RuntimeError("unexpected legacy T generation")
        artifact_sha = directory_hash(args.out / "adapter")
        if artifact_sha != result["adapter"]["sha256"]:
            raise RuntimeError("new adapter/result custody mismatch")
        save(args.out / "training.json", {"schema": "village-001f-matched-mask-training-v1", "experiment": "VILLAGE-001F",
            "member_id": args.member, "mask_mode": args.mask_mode, "training_identity_sha256": sha(args.out / "training-identity.json"),
            "training_result_sha256": sha(args.out / "result.json"), "engine_source_sha256": sha(Path(engine.__file__)), "adapter_directory_sha256": artifact_sha,
            "adapter_weight_fingerprint": adapter_fingerprint(args.out / "adapter"),
            "optimizer_steps": 120, "pair_count": 60, "new_local_generations": 0,
            "training_provenance_schema": "village-001f-training-provenance-summary-v1",
            "legacy_T_evaluation_intentionally_omitted": True,
            "engine_preference_before_is_resume_start": bool(result["recovery"]["resumed"]),
            "original_C_adapter_sha256": actual_parent,
            "original_D_adapter_sha256": source_result(args.member, "D")["adapter"]["sha256"],
            "original_D_reference_result_sha256": sha(ROOT / "experiments/village-001/preference-evidence/results" / f"{args.member}.json")})
        if invocation is None:
            raise RuntimeError("training engine did not establish invocation provenance")
        invocation.finish("completed")
        validate_training_provenance(args.out, cfg)
    except BaseException as exc:
        if invocation is not None and not (args.out / PROVENANCE_DIR / (invocation.prefix + "-finish.json")).exists():
            invocation.finish("interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed", type(exc).__name__)
        raise
    finally:
        sys.argv = previous_argv
        engine.pair_batch = original_pair_batch
        engine.run_heldout = original_heldout
        engine.write_json = original_write_json
        engine.validate_complete = original_validate_complete
        engine.save_checkpoint = original_save_checkpoint


def plan(args, cfg, suite, campaign, pref):
    forbidden = forbidden_prompts(cfg, campaign)
    normalized = [norm(row["prompt"]) for row in suite["scenarios"]]
    if len(set(normalized)) != 12 or any(value in forbidden for value in normalized):
        raise RuntimeError("new Q reserve has duplicate or training/T/P/E-F collisions")
    population = build_manifest(campaign)
    assets = {}
    for stage, key in (("C", "sft_release_manifest"), ("D_ORIGINAL", "preference_release_manifest")):
        release = read(ROOT / cfg[key])
        assets[stage] = {"release_tag": release["release_tag"], "assets": [r for r in release["assets"] if r["member_id"] in PANEL]}
    panel = []
    for member_id in PANEL:
        member = next(m for m in population["members"] if m["member_id"] == member_id)
        c, d = source_result(member_id, "C"), source_result(member_id, "D")
        if c["adapter"]["sha256"] != d["parent_adapter_sha256"] or sha(pair_path(member_id)) != d["preference_dataset_sha256"]:
            raise RuntimeError("original D source parent/pair bytes differ")
        panel.append({"member_id": member_id, "seed": member["seed"], "C_adapter_sha256": c["adapter"]["sha256"],
            "original_D_adapter_sha256": d["adapter"]["sha256"], "pair_dataset_sha256": sha(pair_path(member_id)),
            "paired_training_arm_order": training_arm_order(cfg, member_id),
            "forward_only_pair_ids": [r["pair_id"] for r in selected_pairs(member_id, campaign)]})
    result = {"schema": "village-001f-frozen-worker-plan-v1", "identity": run_identity(args.config, cfg, kind="frozen-plan"),
        "execution_state": "prepared_not_executed", "panel": panel, "adapter_downloads": assets,
        "new_Q_reserve_sha256": cfg["eval_sha256"], "exact_collisions": 0, "forbidden_prompt_count": len(forbidden),
        "new_prompt_count": 12, "training_member_jobs": 6, "training_lanes": 12, "planned_optimizer_steps": 1440,
        "paired_execution": cfg["training"]["paired_execution"],
        "inference_lanes": 25, "planned_new_responses": 300, "planned_forward_only_pair_batches": 288,
        "executed_optimizer_steps": 0, "executed_model_generations": 0,
        "model_or_provider_calls_executed_by_plan": 0, "original_evidence_modified": False}
    if args.out:
        save(args.out / "worker-plan.json", result)
    print(json.dumps({k: result[k] for k in ("execution_state", "exact_collisions", "forbidden_prompt_count", "new_prompt_count", "training_lanes", "inference_lanes", "planned_new_responses")}, indent=2))


def self_test(args, cfg, suite, campaign, pref):
    # Exact mechanism counterexample using the pinned role-boundary ID, with
    # actual padding distinguished by lengths rather than by token value.
    ids = [[1, 42, 2, 1, 43, 2, 1, 44, 2], [1, 45, 2, 1, 46, 2, 2, 2, 2]]
    lengths = [9, 6]
    legacy = [[int(token != 2) for token in row] for row in ids]
    corrected = length_mask_values(lengths, 9)
    if corrected != [[1] * 9, [1] * 6 + [0] * 3]:
        raise RuntimeError("length-derived padding semantics failed")
    if sum(corrected[i][j] - legacy[i][j] for i in range(2) for j in range(9)) != 5:
        raise RuntimeError("historical EOS-masking counterexample failed")
    if Counter(member[-1] for member in PANEL) != {"A": 2, "B": 2, "C": 2}:
        raise RuntimeError("panel balance failed")
    args.out = None
    plan(args, cfg, suite, campaign, pref)
    mock_training_wiring(args, cfg, suite, campaign, pref)
    print("VILLAGE-001F worker offline contract/mask self-test: PASS")


def mock_training_wiring(args, cfg, suite, campaign, pref):
    """Exercise real wrapper/argv/resume/terminal wiring without an ML import."""
    import types
    from unittest.mock import patch
    member_id = PANEL[0]
    population, member = member_contract(campaign, member_id)
    parent_hash = source_result(member_id, "C")["adapter"]["sha256"]
    result_hash = "b" * 64
    class Tensor:
        def __init__(self, values):
            self.values = values
            self.shape = (len(values), len(values[0])) if values and isinstance(values[0], list) else (len(values),)
            self.device = "cpu"
        def __len__(self):
            return len(self.values)
        def tolist(self):
            return self.values
    fake_torch = types.SimpleNamespace(long="long", tensor=lambda values, **kw: Tensor(values))
    execution = {"repository": "techrote/neurounner", "run_id": "1", "run_attempt": "1", "job": "train",
        "member_id": member_id, "paired_job_nonce": "offline-fixture", "source_head": "a" * 40,
        "runner_name": "offline-fixture", "boot_id": "offline-fixture", "hostname": "offline-fixture",
        "cpu_model": "offline-fixture", "cpu_count": 4, "platform": "offline-fixture",
        "packages": {"python": "3.12.3", **RUNTIME}, "process_id": 1,
        "started_at_utc": "2000-01-01T00:00:00+00:00", "started_monotonic_ns": 1}
    with tempfile.TemporaryDirectory(prefix="village001f-mock-train-") as directory:
        root = Path(directory)
        parent = root / "parent"
        parent.mkdir()
        for mask_mode in MASKS:
            engine = types.ModuleType("village_001d_preference_train")
            engine.__file__ = str(ROOT / "scripts/village_001d_preference_train.py")
            engine.write_json = lambda path, value: (path.parent.mkdir(parents=True, exist_ok=True), path.write_bytes(json_bytes(value)))
            def historical_batch(chosen, rejected, pad_id):
                return Tensor([[1, 2, 3], [1, 2, 2]]), Tensor([[-100, 2, 3], [-100, 2, -100]]), Tensor([[1, 0, 1], [1, 0, 0]])
            engine.pair_batch = historical_batch
            engine.run_heldout = lambda *a: [{"should_not_run": True}]
            engine.validate_complete = lambda *a: False
            def save_checkpoint_fixture(**kwargs):
                engine.write_json(kwargs["root"] / "progress.json", {"next_step": kwargs["next_step"],
                    "source_run_id": "1", **{key: kwargs[key] for key in TRACE_KEYS}})
            engine.save_checkpoint = save_checkpoint_fixture
            captured = []
            def main_fixture():
                options = dict(zip(sys.argv[1::2], sys.argv[2::2]))
                if options["--pairs"] != str(pair_path(member_id)) or options["--config"] != str(ROOT / cfg["preference_config"]):
                    raise RuntimeError("mock: original pair/config bytes were not passed to trainer")
                target = Path(options["--out"])
                shutil.copytree(Path(options["--resume"]), target, dirs_exist_ok=True)
                if engine.validate_complete(target, member, population, parent_hash, sha(pair_path(member_id)), campaign, pref):
                    return
                inputs = ({"input_ids": Tensor([1, 2, 3])}, {"input_ids": Tensor([1, 2])})
                ids, labels, attention = engine.pair_batch(*inputs, 2)
                captured.append((ids.tolist(), labels.tolist(), attention.tolist()))
                if engine.run_heldout(None, None, None) != []:
                    raise RuntimeError("mock: legacy post-training generation was not explicitly skipped")
                traces = {key: [float(index) / 100 for index in range(120)] for key in TRACE_KEYS}
                for step in (30, 60, 90, 120):
                    engine.save_checkpoint(root=target / "checkpoint", next_step=step,
                        **{key: values[:step] for key, values in traces.items()})
                (target / "adapter").mkdir()
                value = {"experiment": "VILLAGE-001D", "stage": "preference-continuation",
                    "member_id": member_id, "member_seed": member["seed"], "member_contract_sha256": population["member_contract_sha256"],
                    "parent_adapter_sha256": parent_hash, "preference_dataset_sha256": sha(pair_path(member_id)),
                    "training_config": pref["preference_training"], "training": {"optimizer_steps": 120, "pair_count": 60, **traces},
                    "evaluation": {"responses": []}, "adapter": {"sha256": result_hash}, "recovery": {"resumed": False}}
                engine.write_json(target / "result.json", value)
            engine.main = main_fixture
            train_args = argparse.Namespace(config=args.config, member=member_id, parent_adapter=parent,
                mask_mode=mask_mode, out=root / mask_mode, resume=None)
            previous_argv = list(sys.argv)
            hooks = {key: getattr(engine, key) for key in ("pair_batch", "run_heldout", "write_json", "validate_complete", "save_checkpoint")}
            with patch.dict(sys.modules, {"village_001d_preference_train": engine}), \
                 patch.object(sys.modules[__name__], "runtime_modules", return_value=(fake_torch, None, None)), \
                 patch.object(sys.modules[__name__], "training_execution_metadata", return_value=execution), \
                 patch.object(sys.modules[__name__], "adapter_fingerprint", return_value={"canonical_tensor_sha256": "mock-weights"}), \
                 patch.object(sys.modules[__name__], "directory_hash", side_effect=lambda path: parent_hash if path == parent else result_hash):
                train(train_args, cfg, suite, campaign, pref)
                complete_source = train_args.out
                original_receipts = {name: (complete_source / name).read_bytes() for name in ("result.json", "training.json")}
                train_args.out, train_args.resume = root / (mask_mode + "-copy"), complete_source
                train(train_args, cfg, suite, campaign, pref)
            if previous_argv != sys.argv or any(getattr(engine, key) is not value for key, value in hooks.items()):
                raise RuntimeError("mock: trainer hooks or argv were not restored")
            if any((train_args.out / name).read_bytes() != value for name, value in original_receipts.items()):
                raise RuntimeError("mock: completed copy changed immutable training receipts")
            provenance, _ = validate_training_provenance(train_args.out, cfg)
            if len(provenance["optimizer_segments"]) != 1 or provenance["zero_step_copies"] != [2]:
                raise RuntimeError("mock: completed copy falsely recorded optimizer work")
            expected_attention = [[1, 0, 1], [1, 0, 0]] if mask_mode == MASKS[0] else [[1, 1, 1], [1, 1, 0]]
            if captured != [([[1, 2, 3], [1, 2, 2]], [[-100, 2, 3], [-100, 2, -100]], expected_attention)]:
                raise RuntimeError("mock: training arm changed tokens/labels or wrong mask")
            result = read(train_args.out / "result.json")
            receipt = read(train_args.out / "training.json")
            if result["experiment"] != "VILLAGE-001F" or result["mask_mode"] != mask_mode or receipt["new_local_generations"] != 0:
                raise RuntimeError("mock: F terminal schema/counts mismatch")
            if receipt["training_result_sha256"] != sha(train_args.out / "result.json"):
                raise RuntimeError("mock: terminal receipt does not bind F result")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("plan", "self-test", "token-audit", "train", "infer", "score"):
        p = sub.add_parser(action)
        p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        p.add_argument("--out", type=Path, required=action not in ("plan", "self-test"))
        if action in ("train", "infer", "score"):
            p.add_argument("--member", required=action in ("train", "score"))
        if action in ("infer", "score"):
            p.add_argument("--stage", required=True, choices=("BASE", "C", "D_ORIGINAL", "D_REPRODUCED", "D_CORRECTED"))
            p.add_argument("--adapter", type=Path)
        if action in ("train", "infer"):
            p.add_argument("--resume", type=Path)
        if action == "train":
            p.add_argument("--mask-mode", choices=MASKS, required=True)
            p.add_argument("--parent-adapter", type=Path, required=True)
        if action == "infer":
            p.add_argument("--training-evidence", type=Path)
    args = parser.parse_args()
    cfg, suite, campaign, pref = load_contract(args.config)
    if args.action == "plan":
        plan(args, cfg, suite, campaign, pref)
    elif args.action == "self-test":
        self_test(args, cfg, suite, campaign, pref)
    elif args.action == "token-audit":
        token_audit(args, cfg, suite, campaign)
    elif args.action == "infer":
        infer(args, cfg, suite, campaign)
    elif args.action == "score":
        score(args, cfg, suite, campaign)
    elif args.action == "train":
        train(args, cfg, suite, campaign, pref)


if __name__ == "__main__":
    main()
