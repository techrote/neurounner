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
import hashlib
import json
import os
import platform
import random
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
    torch, _, _ = runtime_modules()
    import village_001d_preference_train as engine
    original_pair_batch = engine.pair_batch
    original_heldout = engine.run_heldout
    original_write_json = engine.write_json
    original_validate_complete = engine.validate_complete
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
    def complete_f(out, member_row, population_row, parent_sha, pairs_sha, campaign_row, pref_row):
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
    engine.write_json = write_f_result
    engine.validate_complete = complete_f
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
    finally:
        sys.argv = previous_argv
        engine.pair_batch = original_pair_batch
        engine.run_heldout = original_heldout
        engine.write_json = original_write_json
        engine.validate_complete = original_validate_complete
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
        "legacy_T_evaluation_intentionally_omitted": True,
        "engine_preference_before_is_resume_start": bool(result["recovery"]["resumed"]),
        "original_C_adapter_sha256": actual_parent,
        "original_D_adapter_sha256": source_result(args.member, "D")["adapter"]["sha256"],
        "original_D_reference_result_sha256": sha(ROOT / "experiments/village-001/preference-evidence/results" / f"{args.member}.json")})


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
            "forward_only_pair_ids": [r["pair_id"] for r in selected_pairs(member_id, campaign)]})
    result = {"schema": "village-001f-frozen-worker-plan-v1", "identity": run_identity(args.config, cfg, kind="frozen-plan"),
        "execution_state": "prepared_not_executed", "panel": panel, "adapter_downloads": assets,
        "new_Q_reserve_sha256": cfg["eval_sha256"], "exact_collisions": 0, "forbidden_prompt_count": len(forbidden),
        "new_prompt_count": 12, "training_lanes": 12, "planned_optimizer_steps": 1440,
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
            captured = []
            def main_fixture():
                options = dict(zip(sys.argv[1::2], sys.argv[2::2]))
                if options["--pairs"] != str(pair_path(member_id)) or options["--config"] != str(ROOT / cfg["preference_config"]):
                    raise RuntimeError("mock: original pair/config bytes were not passed to trainer")
                target = Path(options["--out"])
                shutil.copytree(Path(options["--resume"]), target, dirs_exist_ok=True)
                inputs = ({"input_ids": Tensor([1, 2, 3])}, {"input_ids": Tensor([1, 2])})
                ids, labels, attention = engine.pair_batch(*inputs, 2)
                captured.append((ids.tolist(), labels.tolist(), attention.tolist()))
                if engine.run_heldout(None, None, None) != []:
                    raise RuntimeError("mock: legacy post-training generation was not explicitly skipped")
                (target / "adapter").mkdir()
                value = {"experiment": "VILLAGE-001D", "stage": "preference-continuation",
                    "member_id": member_id, "member_seed": member["seed"], "member_contract_sha256": population["member_contract_sha256"],
                    "parent_adapter_sha256": parent_hash, "preference_dataset_sha256": sha(pair_path(member_id)),
                    "training_config": pref["preference_training"], "training": {"optimizer_steps": 120, "pair_count": 60},
                    "evaluation": {"responses": []}, "adapter": {"sha256": result_hash}, "recovery": {"resumed": False}}
                engine.write_json(target / "result.json", value)
            engine.main = main_fixture
            train_args = argparse.Namespace(config=args.config, member=member_id, parent_adapter=parent,
                mask_mode=mask_mode, out=root / mask_mode, resume=None)
            with patch.dict(sys.modules, {"village_001d_preference_train": engine}), \
                 patch.object(sys.modules[__name__], "runtime_modules", return_value=(fake_torch, None, None)), \
                 patch.object(sys.modules[__name__], "adapter_fingerprint", return_value={"canonical_tensor_sha256": "mock-weights"}), \
                 patch.object(sys.modules[__name__], "directory_hash", side_effect=lambda path: parent_hash if path == parent else result_hash):
                train(train_args, cfg, suite, campaign, pref)
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
