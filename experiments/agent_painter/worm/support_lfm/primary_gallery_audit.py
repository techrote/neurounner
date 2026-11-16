#!/usr/bin/env python3
"""Independent, CPU-only first-pass WORM-001 LFM gallery verifier.

Accepts an *extracted* primary W3 gallery with original model raw-first-pass
receipts. Never loads a model, asks for teacher answers, repairs outputs or
calls native/Godot. W0 host defines legality and source-faithful PNG bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from curriculum import COMMON_SOURCE_SHA, canonical, oracle, rollout, statistics, worm
from motif_metrics import dominant_repeated_motif

PINNED_W0_BLOB = "8979b6f0296e81051194bacc907992d38e006605"
EXPECTED_POLICIES = ("model", "random", "wall", "oracle")
EOS_LFM = 7


def _load(path: Path):
    if not path.is_file() or path.is_symlink():
        raise ValueError("missing or linked original evidence: " + path.name)
    return json.loads(path.read_bytes())


def _source_check():
    if subprocess.check_output(
        ["git", "hash-object", str(Path(worm.__file__).resolve())], text=True
    ).strip() != PINNED_W0_BLOB:
        raise ValueError("W0 shared host source changed without new qualification")


def _same_host_receipt(actual, replay):
    # callback and replay are different *transport* labels, not different host laws.
    a, b = dict(actual), dict(replay)
    a.pop("mode", None)
    b.pop("mode", None)
    if a != b:
        raise ValueError("host replay diverged from submitted trajectory/legality/metrics")


def _firstpass_check(record, generated):
    events = record["events"]
    if len(generated) != len(events):
        raise ValueError("model raw-first-pass count differs from host callbacks")
    legal_count, eos_count = 0, 0
    for event, raw in zip(events, generated):
        obs = event["observation"]
        if raw["observation"] != obs or raw["raw_text"] != event["raw_choice"]:
            raise ValueError("model raw output does not match the host callback")
        ids = raw["token_ids"]
        if not isinstance(ids, list) or any(
            not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in ids
        ):
            raise ValueError("malformed original model token IDs")
        eos = bool(ids) and ids[-1] == EOS_LFM
        legal = raw["raw_text"] in obs["legal"] and raw["raw_text"] in worm.ACTIONS
        if (raw["eos_terminated"] != eos or raw["raw_legal"] != legal
                or raw["valid_first_pass"] != (eos and legal)
                or raw["output_tokens"] != len(ids)
                or event["raw_legal"] != legal):
            raise ValueError("model EOS/token/legality receipt contradiction")
        legal_count += int(legal)
        eos_count += int(eos)
    return legal_count, eos_count


def audit(gallery: Path, *, seed: int = 2101, budget: int = 32) -> dict:
    """Fail-closed audit against original PNGs, W0 raw replay and matched controls.

    One world seed/obstacle world is NOT independent replicated model quality.
    Early STOP makes executed decision counts unequal; report that explicitly.
    """
    _source_check()
    if not 1 <= budget <= 512 or seed < 0:
        raise ValueError("invalid bounded evaluation")
    reports, members = [], []
    for family in worm.FAMILIES:
        for policy in EXPECTED_POLICIES:
            if policy == "model":
                stem = f"{family}-model-{seed}"
            elif policy == "oracle":
                stem = f"{family}-oracle-{seed}"
            else:
                stem = f"{family}-{policy}-{seed}"
            json_path, png_path = gallery / (stem + ".json"), gallery / (stem + ".png")
            record = _load(json_path)
            if (record["family"] != family or record["seed"] != seed
                    or record["step_budget"] != budget
                    or record["size"] != 96
                    or record["contract"] != "cybersand.worm.observation-action/0-experimental"
                    or record["engine"] != "python-stdlib-baseline-not-native"):
                raise ValueError("wrong model/control world, contract or decision budget")
            if policy == "model":
                if record["mode"] != "callback":
                    raise ValueError("model must be genuine closed-loop callback")
                first = _load(gallery / (stem + "-raw-first-passes.json"))
                legal, eos = _firstpass_check(record, first)
                replay = worm.run(
                    seed=seed, mode="replay", family=family, steps=budget,
                    raw_actions=[event["raw_text"] for event in first])
            elif policy == "oracle":
                if record["mode"] != "callback":
                    raise ValueError("oracle must use unmodified callback host")
                replay = rollout(seed, family, budget, oracle)
                legal = eos = None
            else:
                if record["mode"] != policy:
                    raise ValueError("baseline policy label mismatch")
                replay = worm.run(seed, mode=policy, family=family, steps=budget)
                legal = eos = None
            _same_host_receipt(record, replay)
            expected_png = worm.png_bytes(replay)
            if not png_path.is_file() or png_path.is_symlink():
                raise ValueError("missing or linked original PNG")
            png = png_path.read_bytes()
            if png != expected_png:
                raise ValueError("PNG differs from W0 source-faithful rasterization")
            members.append({"file": png_path.name, "sha256": hashlib.sha256(png).hexdigest()})
            members.append({"file": json_path.name, "sha256": hashlib.sha256(json_path.read_bytes()).hexdigest()})
            report = {"family": family, "seed": seed, "policy": policy,
                      "metrics": statistics(record),
                      "raw_cycle": dominant_repeated_motif([e["raw_choice"] for e in record["events"]]),
                      "original_png_sha256": hashlib.sha256(png).hexdigest()}
            if policy == "model":
                report.update(raw_legal_choices=legal, genuine_eos=eos,
                              raw_decisions=len(first),
                              raw_rejected_or_missing_eos=len(first) - sum(
                                  x["valid_first_pass"] for x in first))
                first_file = gallery / (stem + "-raw-first-passes.json")
                members.append({"file": first_file.name,
                                "sha256": hashlib.sha256(first_file.read_bytes()).hexdigest()})
            reports.append(report)
    pairs = []
    for family in worm.FAMILIES:
        current = {r["policy"]: r for r in reports if r["family"] == family}
        model = current["model"]["metrics"]
        for baseline in ("random", "wall", "oracle"):
            other = current[baseline]["metrics"]
            pairs.append({
                "family": family, "seed": seed, "model_vs": baseline,
                "equal_requested_budget": True,
                "equal_executed_decisions": model["decision_count"] == other["decision_count"],
                "model_decisions": model["decision_count"],
                "baseline_decisions": other["decision_count"],
                "unique_cell_delta": model["unique_cells"] - other["unique_cells"],
                "branch_delta": model["branches"] - other["branches"],
                "revisit_delta": model["revisits"] - other["revisits"],
                "raw_entropy_delta_bits": round(
                    model["raw_action_entropy_bits"] - other["raw_action_entropy_bits"], 5),
                "model_max_same_raw_streak": model["max_same_raw_streak"],
                "model_host_fallback_count": model["host_fallback_count"],
                "model_raw_cycle": current["model"]["raw_cycle"],
                "baseline_raw_cycle": current[baseline]["raw_cycle"],
                "raw_cycle_span_fraction_delta": round(
                    current["model"]["raw_cycle"]["fraction"] -
                    current[baseline]["raw_cycle"]["fraction"], 6),
            })
    return {
        "schema": "worm-lfm-primary-gallery-audit-v2",
        "source": COMMON_SOURCE_SHA, "host_blob": PINNED_W0_BLOB,
        "native": False, "trained_by_this_audit": False,
        "model_provenance": "external-primary-original-first-pass-unverified-weights",
        "seed": seed, "budget": budget, "families": list(worm.FAMILIES),
        "metrics": reports, "matched_pairs": pairs, "original_file_hashes": members,
        "limits": "One held-out W0 world seed, not independent multi-seed generalization. "
                  "Host legal fallback is never model recovery. Motif span is a raw action "
                  "repetition diagnostic, not an art-quality score. Reproduced PNG pixels "
                  "are W0 Python-source-faithful, not Godot/native material readbacks. "
                  "Model token decode and checkpoint provenance require primary receipts.",
    }


def demo(out: Path, *, seed: int = 2101, budget: int = 32) -> dict:
    """Actually render model-FREE negative controls, not forged trained output."""
    _source_check()
    if out.exists():
        raise ValueError("immutable demo output destination already exists")
    out.mkdir(parents=True)
    reports, files = [], []
    candidates = {
        "straight-collapse": lambda o: "STRAIGHT",
        "left-loop": lambda o: "LEFT",
        "illegal-teleport": lambda o: "TELEPORT",
        "oracle": oracle,
    }
    for family in worm.FAMILIES:
        for policy in ("random", "wall", *candidates):
            record = (worm.run(seed, mode=policy, family=family, steps=budget)
                      if policy in ("random", "wall")
                      else rollout(seed, family, budget, candidates[policy]))
            stem = f"{family}-{seed}-{policy}"
            png, trace = worm.png_bytes(record), canonical(record)
            for suffix, content in ((".png", png), (".json", trace)):
                path = out / (stem + suffix)
                path.write_bytes(content)
                files.append({"file": path.name, "sha256": hashlib.sha256(content).hexdigest()})
            reports.append({"family": family, "seed": seed, "policy": policy,
                            "metrics": statistics(record)})
    summary = {"schema": "worm-lfm-negative-controls-v1", "seed": seed,
               "step_budget": budget, "native": False, "model_inference": False,
               "reports": reports, "files": files,
               "note": "Original W0 pixel PNGs; deliberately collapsed/illegal selectors "
                       "are synthetic failures, NOT trained-model decisions."}
    (out / "summary.json").write_bytes(canonical(summary))
    return summary


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gallery", type=Path)
    mode.add_argument("--demo-out", type=Path)
    parser.add_argument("--seed", type=int, default=2101)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not 1 <= args.budget <= 512 or args.seed < 0:
        parser.error("invalid bounded evaluation")
    result = (audit(args.gallery, seed=args.seed, budget=args.budget)
              if args.gallery is not None
              else demo(args.demo_out, seed=args.seed, budget=args.budget))
    if args.report:
        args.report.write_bytes(canonical(result))
    print(json.dumps({"schema": result["schema"],
                      "records": len(result.get("metrics", result.get("reports", []))),
                      "seed": args.seed}, sort_keys=True))


if __name__ == "__main__":
    main()
