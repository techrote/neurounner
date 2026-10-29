#!/usr/bin/env python3
"""DETECTIVE-001C1 parent-vs-trained final inference and deterministic scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import peft
import torch
import transformers
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from detective_001c1_final import materialize as materialize_final
from detective_001c1_train import directory_hash, sha_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/detective-001/training-c1.json"
PERSONA_CONFIG = ROOT / "experiments/persona-001/config.json"

REFERENCE_IDS = ["T03", "T11", "T17", "T24"]
GENERIC_PHRASES = [
    "as an ai", "i'm here to help", "i am here to help",
    "i cannot assist", "it's important to", "it is important to",
]
EVIDENCE_ID_RX = re.compile(r"\bF-[A-Z0-9_-]+\b")


class EvalError(ValueError):
    pass


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False) + "\n", encoding="utf-8")


def load_authority(args) -> tuple[dict, dict[str, dict]]:
    cfg = json.loads(CONFIG.read_text())
    trained = {}
    for member, result_path, adapter in (
        ("V-G02-A", args.g02_trained_result, args.g02_trained),
        ("V-G09-C", args.g09_trained_result, args.g09_trained),
    ):
        result = json.loads(result_path.read_text())
        if (result.get("experiment") != "DETECTIVE-001C1"
            or result.get("member_id") != member
            or result.get("training", {}).get("optimizer_steps") != 64):
            raise EvalError("trained result identity/steps mismatch " + member)
        expected_parent = cfg["members"][member]["parent_directory_sha256"]
        if result.get("parent_directory_sha256") != expected_parent:
            raise EvalError("trained result parent mismatch " + member)
        actual_hash = directory_hash(adapter)
        if result.get("adapter", {}).get("directory_sha256") != actual_hash:
            raise EvalError("trained adapter/result hash mismatch " + member)
        trained[member] = {
            "result": result,
            "directory_sha256": actual_hash,
            "path": adapter,
        }

    for member, parent in (
        ("V-G02-A", args.g02_parent),
        ("V-G09-C", args.g09_parent),
    ):
        if directory_hash(parent) != cfg["members"][member]["parent_directory_sha256"]:
            raise EvalError("parent adapter hash mismatch " + member)
    return cfg, trained


def score_response(member: str, scenario: dict, response: str,
                   generated_tokens: int, max_new_tokens: int) -> dict:
    relation = scenario["relation_label"]
    ev = scenario["evaluation"]
    lower = response.casefold()
    required = ev.get("required_marker")
    forbidden = ev.get("forbidden_marker")
    truth = ev.get("truth_marker")
    lie = ev.get("lie_marker")
    phrases = scenario["policy_phrase_any"].get(member, [])

    required_hit = None if required is None else required.casefold() in lower
    forbidden_leak = False if forbidden is None else forbidden.casefold() in lower
    truth_hit = None if truth is None else truth.casefold() in lower
    lie_hit = None if lie is None else lie.casefold() in lower
    phrase_hit = None if not phrases else any(x.casefold() in lower for x in phrases)

    if relation == "truthful_known_fact":
        policy_pass = required_hit is True and not forbidden_leak
    elif relation == "corrupted_belief":
        policy_pass = (required_hit is True and not forbidden_leak
                       and phrase_hit is True)
    elif relation == "intentional_lie":
        policy_pass = lie_hit is True and truth_hit is False
    elif relation == "withheld_fact":
        policy_pass = forbidden_leak is False and phrase_hit is True
    elif relation == "contradiction_repair":
        policy_pass = required_hit is True and phrase_hit is True
    elif relation == "uninformed_speculation":
        policy_pass = phrase_hit is True
    else:
        raise EvalError("unknown final relation " + relation)

    normalized_sentences = [
        " ".join(x.casefold().split())
        for x in re.split(r"[.!?]+", response) if x.strip()
    ]
    repeated_sentences = len(normalized_sentences) - len(set(normalized_sentences))
    return {
        "required_marker_hit": required_hit,
        "forbidden_marker_leak": forbidden_leak,
        "truth_target_hit": truth_hit,
        "lie_target_hit": lie_hit,
        "uncertainty_or_withhold_phrase_hit": phrase_hit,
        "relation_policy_pass": policy_pass,
        "truncated": generated_tokens >= max_new_tokens,
        "evidence_id_leaks": EVIDENCE_ID_RX.findall(response),
        "generic_phrase_hits": [x for x in GENERIC_PHRASES if x in lower],
        "repeated_sentence_count": repeated_sentences,
        "characters": len(response),
        "whitespace_words": len(response.split()),
    }


@torch.inference_mode()
def generate_one(model, tok, messages: list[dict], max_new_tokens: int) -> dict:
    ids = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt")
    started = time.monotonic()
    output = model.generate(
        ids, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
        use_cache=True)
    elapsed = time.monotonic() - started
    tail = output[0, ids.shape[-1]:]
    return {
        "response": tok.decode(tail, skip_special_tokens=True),
        "generated_token_ids": tail.tolist(),
        "generated_tokens": int(tail.numel()),
        "prompt_tokens": int(ids.shape[-1]),
        "prompt_token_sha256": hashlib.sha256(
            ids.numpy().tobytes()).hexdigest(),
        "elapsed_seconds": elapsed,
    }


def load_model(cfg: dict, adapter: Path, tok):
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"]["id"], revision=cfg["base_model"]["revision"],
        torch_dtype=torch.float32, trust_remote_code=False)
    base.config.use_cache = True
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    model.config.use_cache = True
    model.eval()
    return model


def reference_scenarios() -> tuple[str, list[dict]]:
    doc = json.loads(PERSONA_CONFIG.read_text())
    scenarios = {x["id"]: x for x in doc["eval"]["scenarios"]}
    rows = [scenarios[x] for x in REFERENCE_IDS]
    return doc["eval"]["system"], rows


def lane_eval(cfg: dict, suite: dict, member: str, lane_kind: str,
              adapter: Path, tok) -> dict:
    model = load_model(cfg, adapter, tok)
    rows = []
    for scenario in suite["scenarios"]:
        messages = [
            {"role": "system", "content": suite["system"]},
            {"role": "user", "content": scenario["prompt"]},
        ]
        generated = generate_one(
            model, tok, messages, suite["max_new_tokens"])
        score = score_response(
            member, scenario, generated["response"],
            generated["generated_tokens"], suite["max_new_tokens"])
        rows.append({
            "scenario_id": scenario["id"],
            "case_family": scenario["case_family"],
            "relation_label": scenario["relation_label"],
            "prompt": scenario["prompt"],
            **generated,
            "score": score,
        })

    ref_system, refs = reference_scenarios()
    reference = []
    for scenario in refs:
        generated = generate_one(
            model, tok, [
                {"role": "system", "content": ref_system},
                {"role": "user", "content": scenario["prompt"]},
            ], 96)
        reference.append({
            "scenario_id": scenario["id"],
            "family": scenario["family"],
            "prompt": scenario["prompt"],
            **generated,
        })
    del model
    import gc
    gc.collect()

    passes = sum(x["score"]["relation_policy_pass"] for x in rows)
    protected_leaks = sum(
        x["score"]["forbidden_marker_leak"]
        for x in rows
        if x["relation_label"] in {"intentional_lie", "withheld_fact"})
    evidence_leaks = sum(len(x["score"]["evidence_id_leaks"]) for x in rows)
    generic_hits = sum(len(x["score"]["generic_phrase_hits"]) for x in rows)
    truncated = sum(x["score"]["truncated"] for x in rows)
    relation = defaultdict(lambda: {"pass": 0, "total": 0})
    for x in rows:
        item = relation[x["relation_label"]]
        item["total"] += 1
        item["pass"] += int(x["score"]["relation_policy_pass"])
    return {
        "member_id": member,
        "lane_kind": lane_kind,
        "adapter_directory_sha256": directory_hash(adapter),
        "final_responses": rows,
        "reference_responses": reference,
        "summary": {
            "policy_passes": passes,
            "policy_total": len(rows),
            "policy_score": passes / len(rows),
            "protected_fact_leaks": protected_leaks,
            "evidence_id_leaks": evidence_leaks,
            "generic_phrase_hits": generic_hits,
            "truncated": truncated,
            "relation": dict(sorted(relation.items())),
            "reference_relevance_0_to_4": None,
            "reference_relevance_status":
                "UNSCORED_NO_INDEPENDENT_0_TO_4_JUDGE",
        },
    }


def markdown_report(summary: dict) -> str:
    lines = [
        "# DETECTIVE-001C1 — two-character parent vs continuation",
        "",
        "This report is mechanically scored on the post-training, case-family-disjoint "
        "suite. The predeclared 0–4 reference relevance gate remains **unscored** "
        "because this zero-paid-API run has no admitted independent relevance judge. "
        "Therefore the larger-cast expansion gate cannot be opened solely by this run.",
        "",
        "| Member | Parent policy | Trained policy | Parent protected leaks | "
        "Trained protected leaks | Evidence-ID leaks |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for member in ("V-G02-A", "V-G09-C"):
        c = summary["comparisons"][member]
        lines.append(
            f"| {member} | {c['parent']['policy_passes']}/12 | "
            f"{c['trained']['policy_passes']}/12 | "
            f"{c['parent']['protected_fact_leaks']} | "
            f"{c['trained']['protected_fact_leaks']} | "
            f"{c['trained']['evidence_id_leaks']} |")
    lines += [
        "",
        f"Relation-policy improvement gate: **{summary['relation_policy_gate']}**.",
        "Independent 0–4 relevance gate: **UNSCORED**.",
        f"Larger-cast expansion admitted: **{summary['larger_cast_expansion_admitted']}**.",
        "",
        "Raw prompts and responses, final-suite generation bindings, adapter hashes and "
        "reference responses are retained alongside this report. A FAIL/INCONCLUSIVE "
        "result is evidence and must not be repaired post hoc.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--g02-parent", type=Path, required=True)
    ap.add_argument("--g09-parent", type=Path, required=True)
    ap.add_argument("--g02-trained", type=Path, required=True)
    ap.add_argument("--g09-trained", type=Path, required=True)
    ap.add_argument("--g02-trained-result", type=Path, required=True)
    ap.add_argument("--g09-trained-result", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    global DATASET
    # detective_001c1_final uses its module-level dataset root. Point it at the
    # explicit workflow artifact without changing final-suite semantics.
    import detective_001c1_final as final_mod
    final_mod.DATASET = args.dataset_root

    cfg, trained = load_authority(args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(min(cfg["training"]["max_threads"],
                              os.cpu_count() or 1))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(20261027)

    weight = Path(hf_hub_download(
        cfg["base_model"]["id"], "model.safetensors",
        revision=cfg["base_model"]["revision"]))
    if sha_file(weight) != cfg["base_model"]["weights_sha256"]:
        raise EvalError("base model SHA mismatch")
    tok = AutoTokenizer.from_pretrained(
        cfg["base_model"]["id"], revision=cfg["base_model"]["revision"],
        trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    args.out.mkdir(parents=True, exist_ok=True)
    final_path = args.out / "final-suite.json"
    suite = materialize_final(
        args.g02_trained, args.g09_trained, final_path)

    lanes = [
        ("V-G02-A", "parent", args.g02_parent),
        ("V-G02-A", "trained", args.g02_trained),
        ("V-G09-C", "parent", args.g09_parent),
        ("V-G09-C", "trained", args.g09_trained),
    ]
    results = []
    for member, kind, adapter in lanes:
        results.append(lane_eval(
            cfg, suite, member, kind, adapter, tok))
    by_key = {(x["member_id"], x["lane_kind"]): x for x in results}

    comparisons = {}
    relation_gate = True
    for member in ("V-G02-A", "V-G09-C"):
        parent = by_key[(member, "parent")]["summary"]
        trained_summary = by_key[(member, "trained")]["summary"]
        improves = trained_summary["policy_score"] > parent["policy_score"]
        leakage_ok = (
            trained_summary["protected_fact_leaks"]
            <= parent["protected_fact_leaks"]
            and trained_summary["evidence_id_leaks"] == 0
        )
        member_gate = improves and leakage_ok
        relation_gate = relation_gate and member_gate
        comparisons[member] = {
            "parent": parent,
            "trained": trained_summary,
            "policy_score_improved": improves,
            "protected_leakage_not_increased": leakage_ok,
            "member_relation_gate": member_gate,
        }

    summary = {
        "schema": "detective-witness-evaluation-summary-v1",
        "experiment": "DETECTIVE-001C1",
        "issue": 111,
        "final_suite_sha256": sha_file(final_path),
        "final_suite_seed_sha256": suite["seed_sha256"],
        "dataset_manifest_sha256":
            sha_file(args.dataset_root / "manifest.json"),
        "comparisons": comparisons,
        "relation_policy_gate": relation_gate,
        "reference_relevance_gate":
            "UNSCORED_NO_INDEPENDENT_0_TO_4_JUDGE",
        "larger_cast_expansion_admitted": False,
        "larger_cast_reason": (
            "independent 0-4 reference relevance gate remains unscored"
            if relation_gate
            else "one or more relation-policy/leakage gates failed"
        ),
        "runtime": {
            "cpu": platform.processor(),
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "peak_rss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
    }
    write_json(args.out / "raw-results.json", {
        "schema": "detective-witness-evaluation-raw-v1",
        "final_suite": suite,
        "lanes": results,
    })
    write_json(args.out / "summary.json", summary)
    (args.out / "REPORT.md").write_text(
        markdown_report(summary), encoding="utf-8")
    print(json.dumps({
        "relation_policy_gate": relation_gate,
        "larger_cast_expansion_admitted": False,
        "comparisons": {
            m: {
                "parent": comparisons[m]["parent"]["policy_passes"],
                "trained": comparisons[m]["trained"]["policy_passes"],
                "leaks_parent": comparisons[m]["parent"]["protected_fact_leaks"],
                "leaks_trained": comparisons[m]["trained"]["protected_fact_leaks"],
            }
            for m in comparisons
        },
        "final_suite_sha256": summary["final_suite_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
