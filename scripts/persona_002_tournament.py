#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0,str(ROOT))
from scripts import persona_001_tournament as p1

PERSONAS=["G02","G03","G05","G12"]
BASE="BASE"

def load_results(root:Path):
    out={}
    for p in root.rglob("*.json"):
        try:
            obj=json.loads(p.read_text())
        except Exception:
            continue
        if obj.get("experiment")=="PERSONA-002" and obj.get("persona") in PERSONAS:
            out[obj["persona"]]=obj
    if set(out)!=set(PERSONAS):
        raise RuntimeError(f"expected {PERSONAS}, got {sorted(out)}")
    base=json.loads((ROOT/"experiments"/"persona-001"/"evidence"/"models"/"BASE.json").read_text())
    out[BASE]=base
    return out

def g03_g12_cross_confusion(conf):
    return conf["G03"]["G12"]+conf["G12"]["G03"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()

    results=load_results(args.results)
    rows=p1.scenario_rows(results)
    key=os.environ.get("NVIDIA_API_KEY","").strip()
    if not key:
        raise SystemExit("NVIDIA_API_KEY/NVKEY required")

    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    rawdir=out/"judge-raw"

    normalized,norm_calls=p1.build_normalized(key,rows,rawdir)
    raw_records,raw_calls=p1.run_judge(key,rows,rawdir,"raw")
    behavior_records,behavior_calls=p1.run_judge(key,rows,rawdir,"behavior",normalized)

    raw_summary=p1.summarize_mode(raw_records,"raw")
    behavior_summary=p1.summarize_mode(behavior_records,"behavior")
    p1.add_family_accuracy(raw_summary,rows)
    p1.add_family_accuracy(behavior_summary,rows)
    lex=p1.lexical_divergence(rows)

    baseline=json.loads((ROOT/"experiments"/"persona-001"/"evidence"/"summary.json").read_text())
    train_metrics={}
    for gid in PERSONAS:
        r=results[gid]
        train_metrics[gid]={
            "cpu_model":r["runner"]["cpu_model"],
            "optimizer_steps":r["training"]["optimizer_steps"],
            "pair_count":r["training"]["pair_count"],
            "sequence_tokens":r["training"]["sequence_tokens"],
            "supervised_tokens":r["training"]["supervised_tokens"],
            "seconds":r["training"]["seconds"],
            "sequence_tokens_per_second":r["training"]["sequence_tokens_per_second"],
            "first_loss":r["training"]["first_loss"],
            "last_loss":r["training"]["last_loss"],
            "peak_rss_kib":r["peak_rss_kib"],
            "preference_before":r["preference_before"],
            "preference_after":r["preference_after"],
            "adapter":r["adapter"],
            "parent_asset":r["parent_asset"],
        }

    comparison={
        "raw_accuracy":{
            "persona001":baseline["raw_blind"]["accuracy"],
            "persona002":raw_summary["accuracy"],
            "delta":raw_summary["accuracy"]-baseline["raw_blind"]["accuracy"],
        },
        "behavior_accuracy":{
            "persona001":baseline["behavior_blind"]["accuracy"],
            "persona002":behavior_summary["accuracy"],
            "delta":behavior_summary["accuracy"]-baseline["behavior_blind"]["accuracy"],
        },
        "style_gap":{
            "persona001":baseline["style_accuracy_gap"],
            "persona002":raw_summary["accuracy"]-behavior_summary["accuracy"],
            "delta":(raw_summary["accuracy"]-behavior_summary["accuracy"])-baseline["style_accuracy_gap"],
        },
        "g03_g12_cross_confusion":{
            "persona001_raw":g03_g12_cross_confusion(baseline["raw_blind"]["confusion"]),
            "persona002_raw":g03_g12_cross_confusion(raw_summary["confusion"]),
            "persona001_behavior":g03_g12_cross_confusion(baseline["behavior_blind"]["confusion"]),
            "persona002_behavior":g03_g12_cross_confusion(behavior_summary["confusion"]),
        },
        "abstract_unfamiliar":{
            "persona001_raw":baseline["raw_blind"]["family_accuracy"]["abstract_unfamiliar"],
            "persona002_raw":raw_summary["family_accuracy"]["abstract_unfamiliar"],
            "persona001_behavior":baseline["behavior_blind"]["family_accuracy"]["abstract_unfamiliar"],
            "persona002_behavior":behavior_summary["family_accuracy"]["abstract_unfamiliar"],
        },
        "control_genericness":{
            "persona001_raw":baseline["raw_blind"]["control_scores"]["genericness"],
            "persona002_raw":raw_summary["control_scores"]["genericness"],
            "persona001_behavior":baseline["behavior_blind"]["control_scores"]["genericness"],
            "persona002_behavior":behavior_summary["control_scores"]["genericness"],
        },
    }

    persona_delta={}
    for gid in PERSONAS:
        persona_delta[gid]={
            "raw_persona_fit_delta":raw_summary["persona_scores"][gid]["persona_fit"]-baseline["raw_blind"]["persona_scores"][gid]["persona_fit"],
            "raw_genericness_delta":raw_summary["persona_scores"][gid]["genericness"]-baseline["raw_blind"]["persona_scores"][gid]["genericness"],
            "raw_task_relevance_delta":raw_summary["persona_scores"][gid]["task_relevance"]-baseline["raw_blind"]["persona_scores"][gid]["task_relevance"],
            "behavior_persona_fit_delta":behavior_summary["persona_scores"][gid]["persona_fit"]-baseline["behavior_blind"]["persona_scores"][gid]["persona_fit"],
            "behavior_genericness_delta":behavior_summary["persona_scores"][gid]["genericness"]-baseline["behavior_blind"]["persona_scores"][gid]["genericness"],
            "behavior_task_relevance_delta":behavior_summary["persona_scores"][gid]["task_relevance"]-baseline["behavior_blind"]["persona_scores"][gid]["task_relevance"],
        }

    models_dir=out/"models";models_dir.mkdir(exist_ok=True)
    for gid in PERSONAS:
        (models_dir/f"{gid}.json").write_text(json.dumps(results[gid],indent=2,ensure_ascii=False)+"\n")
    (out/"normalized-behaviors.json").write_text(json.dumps(normalized,indent=2,ensure_ascii=False)+"\n")

    summary={
        "schema_version":1,
        "experiment":"PERSONA-002",
        "issue":6,
        "parent_experiment":"PERSONA-001",
        "personas":PERSONAS,
        "scenario_count":24,
        "raw_blind":raw_summary,
        "behavior_blind":behavior_summary,
        "style_accuracy_gap":raw_summary["accuracy"]-behavior_summary["accuracy"],
        "lexical_divergence":lex,
        "training":train_metrics,
        "comparison":comparison,
        "persona_delta":persona_delta,
        "normalization_calls":norm_calls,
        "raw_judge_calls":raw_calls,
        "behavior_judge_calls":behavior_calls,
        "judge_provider":"nvidia",
        "judge_model":p1.MODEL,
        "remote_unpinned_backend":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")

    def pct(x): return f"{100*x:.1f}%"
    lines=[
        "# PERSONA-002 — anti-sanitization preference continuation",
        "",
        "PERSONA-002 continues the four PERSONA-001 SmolLM2-135M LoRA cartridges using BABBLE-002's paired preferred target vs generic/sanitized rejected target.",
        "The exact PERSONA-001 24-scenario blind tournament is reused unchanged.",
        "",
        "## Headline comparison",
        "",
        "| Metric | PERSONA-001 | PERSONA-002 | Delta |",
        "|---|---:|---:|---:|",
        f"| Raw blind identification | {pct(comparison['raw_accuracy']['persona001'])} | {pct(comparison['raw_accuracy']['persona002'])} | {100*comparison['raw_accuracy']['delta']:+.1f} pp |",
        f"| Behavior-only identification | {pct(comparison['behavior_accuracy']['persona001'])} | {pct(comparison['behavior_accuracy']['persona002'])} | {100*comparison['behavior_accuracy']['delta']:+.1f} pp |",
        f"| Raw→behavior style gap | {pct(comparison['style_gap']['persona001'])} | {pct(comparison['style_gap']['persona002'])} | {100*comparison['style_gap']['delta']:+.1f} pp |",
        "",
        "## Preference training",
        "",
        "| Persona | CPU | steps | pair margin before | pair margin after | preference acc before | after | seq tok/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for gid in PERSONAS:
        x=train_metrics[gid]
        lines.append(
            f"| {gid} | {x['cpu_model']} | {x['optimizer_steps']} | "
            f"{x['preference_before']['mean_margin_nats_per_token']:.3f} | {x['preference_after']['mean_margin_nats_per_token']:.3f} | "
            f"{pct(x['preference_before']['preference_accuracy'])} | {pct(x['preference_after']['preference_accuracy'])} | "
            f"{x['sequence_tokens_per_second']:.1f} |"
        )
    lines += [
        "",
        "## Confusion / transfer",
        "",
        f"G03↔G12 raw cross-confusions: **{comparison['g03_g12_cross_confusion']['persona001_raw']} → {comparison['g03_g12_cross_confusion']['persona002_raw']}**.",
        f"G03↔G12 behavior-only cross-confusions: **{comparison['g03_g12_cross_confusion']['persona001_behavior']} → {comparison['g03_g12_cross_confusion']['persona002_behavior']}**.",
        f"Abstract-unfamiliar raw accuracy: **{pct(comparison['abstract_unfamiliar']['persona001_raw'])} → {pct(comparison['abstract_unfamiliar']['persona002_raw'])}**.",
        f"Abstract-unfamiliar behavior-only accuracy: **{pct(comparison['abstract_unfamiliar']['persona001_behavior'])} → {pct(comparison['abstract_unfamiliar']['persona002_behavior'])}**.",
        "",
        "## Per-persona change",
        "",
        "| Persona | Raw fit Δ | Raw genericness Δ | Raw relevance Δ | Behavior fit Δ | Behavior genericness Δ | Behavior relevance Δ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for gid in PERSONAS:
        d=persona_delta[gid]
        lines.append(
            f"| {gid} | {d['raw_persona_fit_delta']:+.2f} | {d['raw_genericness_delta']:+.2f} | {d['raw_task_relevance_delta']:+.2f} | "
            f"{d['behavior_persona_fit_delta']:+.2f} | {d['behavior_genericness_delta']:+.2f} | {d['behavior_task_relevance_delta']:+.2f} |"
        )

    improved_behavior=comparison["behavior_accuracy"]["delta"]>0
    no_major_relevance_regression=statistics.fmean([persona_delta[g]["raw_task_relevance_delta"] for g in PERSONAS])>-0.5
    verdict="PASS" if improved_behavior and no_major_relevance_regression else "MIXED / FAIL"
    lines += [
        "",
        "## Verdict",
        "",
        f"**{verdict}.**",
        "",
        "The primary success criterion is improvement in style-neutral behavioral identification without a major task-relevance regression. Negative results are retained as evidence rather than optimized away.",
        "",
        "Machine-readable evidence: experiments/persona-002/evidence/summary.json.",
    ]
    report="\n".join(lines)+"\n"
    (out/"PERSONA_002_REPORT.md").write_text(report)
    print(report)

if __name__=="__main__":
    main()
