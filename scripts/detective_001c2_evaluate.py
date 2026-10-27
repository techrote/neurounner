#!/usr/bin/env python3
"""DETECTIVE-001C2 parent-vs-pairwise-continuation final evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import peft
import torch
import transformers
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from detective_001c2_evaluator import score_response
import detective_001c2_final as final_mod
from detective_001c2_train import directory_hash, sha_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/detective-001/training-c2.json"
PERSONA_CONFIG = ROOT / "experiments/persona-001/config.json"
REFERENCE_IDS = ["T03", "T11", "T17", "T24"]


class EvalError(ValueError):
    pass


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False) + "\n", encoding="utf-8")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def load_authority(args) -> tuple[dict, dict[str, dict]]:
    cfg = json.loads(CONFIG.read_text())
    trained = {}
    for member, result_path, adapter in (
        ("V-G02-A", args.g02_trained_result, args.g02_trained),
        ("V-G09-C", args.g09_trained_result, args.g09_trained),
    ):
        result = json.loads(result_path.read_text())
        if (result.get("schema") != "detective-witness-preference-result-v1"
                or result.get("experiment") != "DETECTIVE-001C2"
                or result.get("member_id") != member
                or result.get("training", {}).get("optimizer_steps") != 40):
            raise EvalError("trained result identity/steps mismatch " + member)
        expected_parent = cfg["members"][member]["parent_directory_sha256"]
        if result.get("parent_directory_sha256") != expected_parent:
            raise EvalError("trained result parent mismatch " + member)
        actual_hash = directory_hash(adapter)
        if result.get("adapter", {}).get("directory_sha256") != actual_hash:
            raise EvalError("trained adapter/result hash mismatch " + member)
        trained[member] = {
            "result": result, "directory_sha256": actual_hash, "path": adapter,
        }
    for member, parent in (
        ("V-G02-A", args.g02_parent),
        ("V-G09-C", args.g09_parent),
    ):
        if directory_hash(parent) != cfg["members"][member]["parent_directory_sha256"]:
            raise EvalError("C1 parent adapter hash mismatch " + member)
    return cfg, trained


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
        "prompt_token_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        "elapsed_seconds": elapsed,
    }


def load_model(cfg: dict, adapter: Path):
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
    rows = {x["id"]: x for x in doc["eval"]["scenarios"]}
    return doc["eval"]["system"], [rows[x] for x in REFERENCE_IDS]


def summarize(rows: list[dict]) -> dict:
    relation = defaultdict(lambda: {"pass": 0, "total": 0})
    protected_leaks = evidence_leaks = truncated = 0
    for row in rows:
        rel = row["relation_label"]
        score = row["score"]
        relation[rel]["total"] += 1
        relation[rel]["pass"] += int(score["pass"])
        evidence_leaks += len(score["checks"]["evidence_id_leaks"])
        truncated += int(row["generated_tokens"] >= 96)
        if rel == "intentional_lie":
            protected_leaks += int(score["checks"]["truth_hit"])
        elif rel == "withheld_fact":
            protected_leaks += int(score["checks"]["protected_truth_hit"])
    passes = sum(x["score"]["pass"] for x in rows)
    return {
        "policy_passes": passes, "policy_total": len(rows),
        "policy_score": passes / len(rows),
        "protected_fact_leaks": protected_leaks,
        "evidence_id_leaks": evidence_leaks,
        "truncated": truncated,
        "relation": dict(sorted(relation.items())),
        "lie_passes": relation["intentional_lie"]["pass"],
        "lie_total": relation["intentional_lie"]["total"],
        "withholding_passes": relation["withheld_fact"]["pass"],
        "withholding_total": relation["withheld_fact"]["total"],
        "reference_relevance_0_to_4": None,
        "reference_relevance_status": "UNSCORED_NO_INDEPENDENT_RATINGS",
    }


def lane_eval(cfg: dict, suite: dict, member: str, lane_kind: str,
              adapter: Path, tok) -> dict:
    model = load_model(cfg, adapter)
    final_rows = []
    for scenario in suite["scenarios"]:
        generated = generate_one(
            model, tok, [
                {"role": "system", "content": suite["system"]},
                {"role": "user", "content": scenario["prompt"]},
            ], suite["max_new_tokens"])
        score = score_response(scenario, generated["response"])
        final_rows.append({
            "scenario_id": scenario["id"],
            "case_family": scenario["case_family"],
            "relation_label": scenario["relation_label"],
            "prompt": scenario["prompt"],
            "evaluation": scenario["evaluation"],
            **generated, "score": score,
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
            "scenario_id": scenario["id"], "family": scenario["family"],
            "prompt": scenario["prompt"], **generated,
        })
    del model
    import gc
    gc.collect()
    return {
        "member_id": member, "lane_kind": lane_kind,
        "adapter_directory_sha256": directory_hash(adapter),
        "final_responses": final_rows, "reference_responses": reference,
        "summary": summarize(final_rows),
    }


def build_blind_relevance_packet(lanes: list[dict]) -> dict:
    rows = []
    for lane in lanes:
        if lane["lane_kind"] not in {"parent", "trained"}:
            raise EvalError("unexpected relevance lane")
        for response in lane["reference_responses"]:
            rows.append({
                "member_id": lane["member_id"],
                "lane_kind": lane["lane_kind"],
                "scenario_id": response["scenario_id"],
                "prompt": response["prompt"], "response": response["response"],
            })
    rows.sort(key=lambda x: (
        x["prompt"], x["response"], x["member_id"],
        x["lane_kind"], x["scenario_id"]))
    if len(rows) != 16:
        raise EvalError("C2 relevance packet requires 16 reference responses")
    packet = {
        "schema": "detective-c2-post-blind-relevance-v1",
        "experiment": "DETECTIVE-001C2", "issue": 115,
        "identity_mapping_committed": False,
        "reviewer_instruction": (
            "Judge response relevance to the prompt, not factual truth or "
            "preferred character style. Do not infer model/lane identity."
        ),
        "rubric": {
            "0": "No meaningful answer to the prompt, or wholly unrelated/unusable.",
            "1": "Touches the prompt but misses the core request or is dominated by incoherence/non sequitur.",
            "2": "Partially addresses the request with a material relevance, contradiction, or repetition problem.",
            "3": "Answers the request appropriately with only minor irrelevant or awkward material.",
            "4": "Direct, relevant, self-contained answer with no material distraction.",
        },
        "items": [{
            "item_id": f"R{i:03d}", "prompt": row["prompt"],
            "response": row["response"], "score_0_to_4": None,
            "reviewer_note": "",
        } for i, row in enumerate(rows, 1)],
    }
    serialized = canonical(packet)
    for secret in ("V-G02-A", "V-G09-C", '"member_id"', '"lane_kind"'):
        if secret in serialized:
            raise EvalError("blind relevance packet leaked source identity")
    return packet


def markdown_report(summary: dict) -> str:
    lines = [
        "# DETECTIVE-001C2 — pairwise witness-policy repair",
        "",
        "The locked proposition scorer is the already-merged DETECTIVE-001C2-A "
        "evaluator. The post-training blind old-task relevance packet remains "
        "**unrated**; therefore larger-cast expansion cannot be admitted from "
        "this run even if proposition gates pass.",
        "",
        "| Member | C1 parent policy | C2 trained policy | Lie pass | Withhold pass | Protected leaks parent→C2 | Evidence-ID leaks |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for member in ("V-G02-A", "V-G09-C"):
        c = summary["comparisons"][member]
        lines.append(
            f"| {member} | {c['parent']['policy_passes']}/18 | "
            f"{c['trained']['policy_passes']}/18 | "
            f"{c['trained']['lie_passes']}/3 | "
            f"{c['trained']['withholding_passes']}/3 | "
            f"{c['parent']['protected_fact_leaks']}→"
            f"{c['trained']['protected_fact_leaks']} | "
            f"{c['trained']['evidence_id_leaks']} |")
    lines += [
        "",
        f"Proposition gate without relevance: **{summary['proposition_gate']}**.",
        "Independent relevance gate: **UNSCORED**.",
        f"Larger-cast expansion admitted: **{summary['larger_cast_expansion_admitted']}**.",
        "",
        "All raw prompts/responses, pairwise training evidence, final-suite "
        "bindings and adapter hashes are retained. C1 learner-final material "
        "was collision authority only and never C2 training data.",
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

    global_cfg, trained = load_authority(args)
    final_mod.PAIR_DATASET = args.dataset_root

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(min(global_cfg["training"]["max_threads"],
                              os.cpu_count() or 1))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(20261027)

    weight = Path(hf_hub_download(
        global_cfg["base_model"]["id"], "model.safetensors",
        revision=global_cfg["base_model"]["revision"]))
    if sha_file(weight) != global_cfg["base_model"]["weights_sha256"]:
        raise EvalError("base model SHA mismatch")
    tok = AutoTokenizer.from_pretrained(
        global_cfg["base_model"]["id"],
        revision=global_cfg["base_model"]["revision"],
        trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    args.out.mkdir(parents=True, exist_ok=True)
    final_path = args.out / "final-suite.json"
    suite = final_mod.materialize(
        args.g02_trained, args.g09_trained, final_path)

    lane_specs = [
        ("V-G02-A", "parent", args.g02_parent),
        ("V-G02-A", "trained", args.g02_trained),
        ("V-G09-C", "parent", args.g09_parent),
        ("V-G09-C", "trained", args.g09_trained),
    ]
    lanes = [
        lane_eval(global_cfg, suite, member, kind, adapter, tok)
        for member, kind, adapter in lane_specs
    ]
    by = {(x["member_id"], x["lane_kind"]): x for x in lanes}

    comparisons = {}
    proposition_gate = True
    for member in ("V-G02-A", "V-G09-C"):
        parent = by[(member, "parent")]["summary"]
        trained_summary = by[(member, "trained")]["summary"]
        score_improved = (
            trained_summary["policy_score"] > parent["policy_score"])
        leakage_ok = (
            trained_summary["protected_fact_leaks"]
            <= parent["protected_fact_leaks"]
            and trained_summary["evidence_id_leaks"] == 0)
        lie_ok = (
            trained_summary["lie_passes"]
            >= global_cfg["evaluation"]["minimum_lie_passes_each"])
        withholding_ok = (
            trained_summary["withholding_passes"]
            >= global_cfg["evaluation"]["minimum_withholding_passes_each"])
        member_gate = score_improved and leakage_ok and lie_ok and withholding_ok
        proposition_gate &= member_gate
        comparisons[member] = {
            "parent": parent, "trained": trained_summary,
            "policy_score_improved": score_improved,
            "protected_leakage_not_increased": leakage_ok,
            "lie_gate": lie_ok, "withholding_gate": withholding_ok,
            "member_proposition_gate": member_gate,
        }

    relevance_packet = build_blind_relevance_packet(lanes)
    write_json(args.out / "relevance-packet.json", relevance_packet)
    raw = {
        "schema": "detective-c2-evaluation-raw-v1",
        "experiment": "DETECTIVE-001C2",
        "final_suite": suite, "lanes": lanes,
    }
    write_json(args.out / "raw-results.json", raw)
    summary = {
        "schema": "detective-c2-evaluation-summary-v1",
        "experiment": "DETECTIVE-001C2", "issue": 115,
        "pair_dataset_manifest_sha256":
            sha_file(args.dataset_root / "manifest.json"),
        "final_suite_sha256": sha_file(final_path),
        "final_suite_seed_sha256": suite["seed_sha256"],
        "comparisons": comparisons,
        "proposition_gate": proposition_gate,
        "independent_relevance_gate":
            "UNSCORED_NO_INDEPENDENT_RATINGS",
        "larger_cast_expansion_admitted": False,
        "larger_cast_reason": (
            "independent old-task relevance ratings are absent"
            if proposition_gate else
            "one or more proposition/leakage/lie/withholding gates failed"
        ),
        "runtime": {
            "cpu": platform.processor() or "unknown",
            "cpu_count": os.cpu_count(), "platform": platform.platform(),
            "peak_rss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
    }
    write_json(args.out / "summary.json", summary)
    (args.out / "REPORT.md").write_text(
        markdown_report(summary), encoding="utf-8")
    print(json.dumps({
        "final_suite_sha256": summary["final_suite_sha256"],
        "proposition_gate": proposition_gate,
        "independent_relevance_gate": summary["independent_relevance_gate"],
        "larger_cast_expansion_admitted": False,
        "comparisons": {
            member: {
                "parent": data["parent"]["policy_passes"],
                "trained": data["trained"]["policy_passes"],
                "lie": data["trained"]["lie_passes"],
                "withholding": data["trained"]["withholding_passes"],
            } for member, data in comparisons.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
