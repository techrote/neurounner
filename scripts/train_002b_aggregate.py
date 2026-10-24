#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "minimaxai/minimax-m3.1-flash"
ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
EXPECTED_LANES = [
    "ctrl-a","ctrl-b","lr-low","lr-high","rank4","rank16","dropout0","failure-weighted"
]

def extract_json(text: str):
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    raise ValueError("no complete JSON object")

def call_api(key: str, payload: dict, outdir: Path, batch: int):
    headers = {
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "neurounner-train002b",
    }
    body = json.dumps(payload).encode()
    attempts = []
    for attempt in range(1, 6):
        t0 = time.monotonic()
        req = urllib.request.Request(ENDPOINT, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                raw = r.read().decode("utf-8")
                attempts.append({"attempt":attempt,"status":r.status,"elapsed_seconds":time.monotonic()-t0})
                obj = json.loads(raw)
                (outdir / f"judge-batch-{batch}-provider.json").write_text(
                    json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
                )
                return obj, attempts
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:3000]
            attempts.append({"attempt":attempt,"status":e.code,"detail":detail})
            if e.code not in (408,429,500,502,503,504) or attempt == 5:
                raise
            time.sleep(min(60, 5 * (2 ** (attempt-1))))
        except Exception as e:
            attempts.append({"attempt":attempt,"status":"exception","detail":repr(e)})
            if attempt == 5:
                raise
            time.sleep(min(60, 5 * (2 ** (attempt-1))))
    raise RuntimeError("unreachable")

def judge_prompt(records: list[dict]) -> str:
    compact = []
    for r in records:
        compact.append({
            "case_id": r["case_id"],
            "lane": r["lane"],
            "input": r["input"],
            "expected": r["expected"],
            "candidate": r["output"],
        })
    return """Score each candidate behavior-kernel output against its input and expected reference.

This is a mechanical fidelity evaluation, not a style contest. Do not reward mere word overlap.
Score each integer 0..4:
- trigger: captures the local event that should activate the behavior.
- state: represents the required bounded/local memory.
- update: causally updates/decays that state correctly.
- action: actually implements the requested behavior using the state.
- failure: predicts a plausible emergent failure rather than restating the prompt.
- locality_boundedness: uses local observations and bounded internal state, without global/oracle knowledge.
- overall: overall mechanically faithful implementation.

Return JSON only:
{"scores":[{"case_id":"...","trigger":0,"state":0,"update":0,"action":0,"failure":0,
"locality_boundedness":0,"overall":0,"note":"short reason"}]}

Evaluate every record exactly once.

RECORDS:
""" + json.dumps(compact, ensure_ascii=False)

def semantic_judge(key: str, lane_results: dict[str,dict], outdir: Path):
    all_scores = {}
    call_records = []
    lane_groups = [EXPECTED_LANES[i:i+2] for i in range(0, len(EXPECTED_LANES), 2)]
    for batch, lane_ids in enumerate(lane_groups):
        records = []
        for lane_id in lane_ids:
            for row in lane_results[lane_id]["evaluation_after"]["rows"]:
                records.append({
                    "case_id": f"{lane_id}:{row['case_id']}",
                    "lane": lane_id,
                    "input": row["input"],
                    "expected": row["expected"],
                    "output": row["output"],
                })
        prompt = judge_prompt(records)
        payload = {
            "model": MODEL,
            "messages": [
                {"role":"system","content":"You are a strict evaluator of deterministic local behavior kernels. Output JSON only."},
                {"role":"user","content":prompt},
            ],
            "max_tokens": 4000,
            "temperature": 0.2,
            "top_p": 0.9,
            "stream": False,
            "reasoning_effort": "low",
        }
        (outdir / f"judge-batch-{batch}-request.json").write_text(
            json.dumps({**payload, "remote_unpinned_backend":True}, indent=2, ensure_ascii=False) + "\n"
        )
        obj, attempts = call_api(key, payload, outdir, batch)
        msg = obj["choices"][0]["message"]
        content = msg.get("content") or ""
        (outdir / f"judge-batch-{batch}-content.txt").write_text(content + "\n")
        parsed = extract_json(content)
        scores = parsed.get("scores")
        if not isinstance(scores, list):
            raise RuntimeError(f"judge batch {batch}: scores missing")
        expected_ids = {r["case_id"] for r in records}
        got = {}
        for s in scores:
            if not isinstance(s, dict) or s.get("case_id") not in expected_ids:
                continue
            normalized = dict(s)
            for k in ["trigger","state","update","action","failure","locality_boundedness","overall"]:
                v = normalized.get(k)
                if isinstance(v, float) and v.is_integer():
                    v = int(v)
                if not isinstance(v, int) or not 0 <= v <= 4:
                    raise RuntimeError(f"judge batch {batch}: invalid {k} for {s.get('case_id')}: {v}")
                normalized[k] = v
            got[normalized["case_id"]] = normalized
        if set(got) != expected_ids:
            raise RuntimeError(f"judge batch {batch}: missing scores {sorted(expected_ids-set(got))}")
        all_scores.update(got)
        call_records.append({
            "batch": batch,
            "lanes": lane_ids,
            "attempts": attempts,
            "usage": obj.get("usage"),
            "finish_reason": obj["choices"][0].get("finish_reason"),
        })
    return all_scores, call_records

def mean(xs):
    return statistics.fmean(xs) if xs else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for p in args.results.rglob("*.json"):
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue
        if obj.get("experiment") == "TRAIN-002b" and isinstance(obj.get("lane"), dict):
            results[obj["lane"]["id"]] = obj
    if set(results) != set(EXPECTED_LANES):
        raise RuntimeError(f"expected lanes {EXPECTED_LANES}, got {sorted(results)}")

    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not key:
        raise SystemExit("NVIDIA_API_KEY/NVKEY is required for semantic evaluation")
    judge_dir = args.out / "semantic-judge"
    judge_dir.mkdir(exist_ok=True)
    scores, judge_calls = semantic_judge(key, results, judge_dir)

    lane_summaries = []
    for lane_id in EXPECTED_LANES:
        r = results[lane_id]
        lane_scores = [scores[f"{lane_id}:{row['case_id']}"] for row in r["evaluation_after"]["rows"]]
        summary = {
            "lane": lane_id,
            "role": r["lane"]["role"],
            "factor": r["lane"].get("factor", "control_seed"),
            "seed": r["lane"]["seed"],
            "lr": r["lane"]["lr"],
            "lora_r": r["lane"]["lora_r"],
            "lora_alpha": r["lane"]["lora_alpha"],
            "lora_dropout": r["lane"]["lora_dropout"],
            "material": r["lane"]["material"],
            "cpu_model": r["runner"]["cpu_model"],
            "trainable_parameters": r["trainable_parameters"],
            "peak_rss_kib": r["peak_rss_kib"],
            "training_seconds": r["training"]["seconds"],
            "training_sequence_units_per_second": r["training"]["sequence_units_per_second"],
            "training_utf8_bytes_per_second": r["training"]["utf8_bytes_per_second"],
            "first_loss": r["training"]["first_loss"],
            "last_loss": r["training"]["last_loss"],
            "before_json_rate": r["evaluation_before"]["json_rate"],
            "before_exact_schema_rate": r["evaluation_before"]["exact_schema_rate"],
            "after_json_rate": r["evaluation_after"]["json_rate"],
            "after_exact_schema_rate": r["evaluation_after"]["exact_schema_rate"],
            "after_budget_exhaustion_rate": r["evaluation_after"]["budget_exhaustion_rate"],
            "after_mechanical_similarity_proxy": r["evaluation_after"]["mean_mechanical_similarity_proxy"],
            "after_mean_characters_per_second": r["evaluation_after"]["mean_characters_per_second"],
            "judge_trigger_mean": mean([x["trigger"] for x in lane_scores]),
            "judge_state_mean": mean([x["state"] for x in lane_scores]),
            "judge_update_mean": mean([x["update"] for x in lane_scores]),
            "judge_action_mean": mean([x["action"] for x in lane_scores]),
            "judge_failure_mean": mean([x["failure"] for x in lane_scores]),
            "judge_locality_mean": mean([x["locality_boundedness"] for x in lane_scores]),
            "judge_overall_mean": mean([x["overall"] for x in lane_scores]),
            "adapter": r["adapter"],
            "tokenizer": r["tokenizer"],
        }
        lane_summaries.append(summary)
        (args.out / "lanes").mkdir(exist_ok=True)
        (args.out / "lanes" / f"{lane_id}.json").write_text(
            json.dumps(r, indent=2, ensure_ascii=False) + "\n"
        )

    controls = [x for x in lane_summaries if x["role"] == "control"]
    control_mean = {
        "exact_schema_rate": mean([x["after_exact_schema_rate"] for x in controls]),
        "judge_overall_mean": mean([x["judge_overall_mean"] for x in controls]),
        "mechanical_similarity_proxy": mean([x["after_mechanical_similarity_proxy"] for x in controls]),
    }
    for x in lane_summaries:
        x["delta_vs_control_mean"] = {
            "exact_schema_rate": x["after_exact_schema_rate"] - control_mean["exact_schema_rate"],
            "judge_overall_mean": x["judge_overall_mean"] - control_mean["judge_overall_mean"],
            "mechanical_similarity_proxy": x["after_mechanical_similarity_proxy"] - control_mean["mechanical_similarity_proxy"],
        }

    ranking = sorted(
        lane_summaries,
        key=lambda x:(x["judge_overall_mean"], x["after_exact_schema_rate"], x["after_mechanical_similarity_proxy"]),
        reverse=True,
    )

    summary = {
        "schema_version": 1,
        "experiment": "TRAIN-002b",
        "issue": 1,
        "base_model": results["ctrl-a"]["base_model"],
        "base_revision": results["ctrl-a"]["base_revision"],
        "evaluation_note": "void.1 uses a custom byte-oriented tokenizer; tokenizer-unit rates are not directly comparable to ordinary subword token/s.",
        "control_mean": control_mean,
        "lanes": lane_summaries,
        "ranking": [x["lane"] for x in ranking],
        "semantic_judge": {
            "provider": "nvidia",
            "model": MODEL,
            "remote_unpinned_backend": True,
            "calls": judge_calls,
            "scores": scores,
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    def pct(v): return f"{100*v:.0f}%"
    lines = [
        "# TRAIN-002b — eight-lane byte-aware void.1 evaluation",
        "",
        "Issue: #1",
        "",
        "Eight independent GitHub-hosted runners trained void.1 in parallel. Two controls differed only by seed; six lanes changed one logical factor relative to control A.",
        "",
        "The evaluation budget was increased from TRAIN-002's 120 byte-oriented tokenizer units to 768, and throughput is reported in UTF-8 bytes/s and characters/s as well as tokenizer units/s.",
        "",
        "| Lane | Factor | CPU | train B/s | loss | exact schema | budget exhausted | deterministic proxy | cloud mechanical /4 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for x in lane_summaries:
        lines.append(
            f"| {x['lane']} | {x['factor']} | {x['cpu_model']} | "
            f"{x['training_utf8_bytes_per_second']:.1f} | {x['first_loss']:.3f}→{x['last_loss']:.3f} | "
            f"{pct(x['after_exact_schema_rate'])} | {pct(x['after_budget_exhaustion_rate'])} | "
            f"{x['after_mechanical_similarity_proxy']:.3f} | {x['judge_overall_mean']:.2f} |"
        )
    lines += [
        "",
        "## Ranking",
        "",
        "Cloud mechanical-fidelity ranking: " + " > ".join(summary["ranking"]) + ".",
        "",
        f"Control mean exact-schema rate: {pct(control_mean['exact_schema_rate'])}.",
        f"Control mean cloud mechanical score: {control_mean['judge_overall_mean']:.2f}/4.",
        "",
        "## Interpretation rules",
        "",
        "- Treat control A/B spread as the seed/noise floor before attributing small differences to a perturbation.",
        "- Byte-tokenizer units are implementation-specific; bytes/s and characters/s are the cross-run throughput measures for void.1.",
        "- Exact JSON schema is necessary for compiler use but does not imply mechanical correctness; deterministic similarity and an independent MiniMax mechanical judge are recorded separately.",
        "- All held-out generations, including malformed/truncated outputs, are preserved under experiments/train-002b/evidence/lanes/.",
        "- NVIDIA judge request/response history is preserved under experiments/train-002b/evidence/semantic-judge/.",
        "",
        "Full machine-readable comparison: experiments/train-002b/evidence/summary.json.",
    ]
    doc = "\n".join(lines) + "\n"
    (args.out / "TRAIN_002B_VOID_EIGHT_LANE.md").write_text(doc)
    print(doc)

if __name__ == "__main__":
    main()
