#!/usr/bin/env python3
"""Render only a complete actual F final summary; never fabricate result data.

Six prespecified member deltas and a separate mean marker are shown. The +0.5
target applies to the overall mean; the -0.5 bound applies to each member. No
confidence intervals, significance calculations or automatic rollout claims.
Optional Sprocket panel contains descriptive counts, 18 observations per arm.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

PANEL = ["V-G01-A", "V-G03-B", "V-G05-C", "V-G07-A", "V-G09-B", "V-G11-C"]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def close(value, expected, label):
    require(type(value) in (int, float) and math.isfinite(value)
            and math.isclose(value, float(expected), rel_tol=0, abs_tol=1e-12), label)


def validated_data(path, include_sprocket):
    require(path.is_file() and not path.is_symlink(), "Ordinary final summary file required")
    raw = path.read_bytes()
    doc = json.loads(raw)
    require(doc.get("schema") == "village-001f-diagnostic-results-v1" and doc.get("experiment") == "VILLAGE-001F", "Wrong final result identity")
    worker = doc["worker"]
    for key, expected in (("primary_case_pairs", 72), ("member_replication_units", 6), ("training_lanes", 12), ("optimizer_steps", 1440), ("native_responses", 300)):
        require(type(worker.get(key)) is int and worker[key] == expected, "Incomplete final worker population: " + key)
    pairs = worker["paired_cases"]
    require(len(pairs) == 72 and {(r["member_id"], r["scenario_id"]) for r in pairs}
            == {(m, f"Q{i:02d}") for m in PANEL for i in range(1, 13)}, "All72 unique case/member pairs required")
    for row in pairs:
        a, b = row["corrected"]["task_relevance"], row["reproduced"]["task_relevance"]
        require(type(a) is int and type(b) is int and 0 <= a <= 4 and 0 <= b <= 4
                and row["delta"]["task_relevance"] == a - b, "Primary integer score/delta mismatch")
    per_member = {row["member_id"]: row for row in worker["per_member"]}
    require(len(worker["per_member"]) == 6 and set(per_member) == set(PANEL), "All six prespecified members must remain")
    deltas = []
    for member in PANEL:
        rows = [r for r in pairs if r["member_id"] == member]
        require(len(rows) == per_member[member]["cases"] == 12, "Complete12-case member mean required")
        value = Fraction(sum(r["delta"]["task_relevance"] for r in rows), 12)
        close(per_member[member]["mean_delta"]["task_relevance"], value, "Member mean does not reconstruct")
        deltas.append(value)
    mean = Fraction(sum(r["delta"]["task_relevance"] for r in pairs), 72)
    require(mean == sum(deltas, Fraction()) / 6, "Equal-member primary mean mismatch")
    close(worker["mean_delta"]["task_relevance"], mean, "Overall mean does not reconstruct")
    criteria = {"mean_gain": mean >= Fraction("0.5"), "members_positive": sum(x > 0 for x in deltas) >= 4,
                "maximum_member_regression": min(deltas) >= -Fraction("0.5")}
    require(worker["working_effect_criteria"] == criteria and worker["working_effect_signal"] is all(criteria.values()), "Frozen working-effect criteria mismatch")
    require(worker["positive_member_relevance_gains"] == sum(x > 0 for x in deltas), "Positive-member count mismatch")
    close(worker["worst_member_relevance_delta"], min(deltas), "Worst-member delta mismatch")
    sprocket = None
    if include_sprocket:
        sp = doc["sprocket"]
        require(sp["responses"] == 36 and sp["cases"] == 6 and sp["seeds_per_case"] == 3, "Complete Sprocket36 required")
        reviews = sp["reviews"]
        require(len(reviews) == 36 and len({(r["case_id"], r["worker"], r["arm"]) for r in reviews}) == 36,
                "Sprocket case/seed/arm duplication")
        require(len({r["case_id"] for r in reviews}) == 6, "Six Sprocket cases required")
        for case in {row["case_id"] for row in reviews}:
            workers = [{row["worker"] for row in reviews if row["case_id"] == case and row["arm"] == arm}
                       for arm in ("original", "compact_schema_first")]
            require(len(workers[0]) == 3 and workers[0] == workers[1], "Sprocket matched case/seed pairing differs")
        sprocket = {}
        for arm in ("original", "compact_schema_first"):
            rows = [r for r in reviews if r["arm"] == arm]
            require(len(rows) == 18 and set(Counter(r["case_id"] for r in rows).values()) == {3}, "Sprocket18 denominator/case balance differs")
            valid, actionable = 0, 0
            for row in rows:
                require(type(row["strict_json_schema_valid"]) is bool and type(row["unsupported_claims"]) is bool,
                        "Sprocket validity/unsupported flags must be booleans")
                require(all(type(row[k]) is int and 0 <= row[k] <= 2 for k in ("response_grounding", "scenario_grounding", "corrective_usefulness")), "Sprocket ratings must be integer0..2")
                grounded = row["response_grounding"] >= 1 and row["scenario_grounding"] >= 1 and not row["unsupported_claims"]
                good = row["strict_json_schema_valid"] and grounded and row["corrective_usefulness"] >= 1
                require(row["strict_grounded_and_actionable"] is good, "Sprocket actionable composite differs")
                valid += row["strict_json_schema_valid"]
                actionable += good
            require(sp["arms"][arm]["responses"] == 18 and sp["arms"][arm]["strict_json_schema_valid"] == valid
                    and sp["arms"][arm]["strict_grounded_and_actionable"] == actionable, "Sprocket arm counts do not reconstruct")
            sprocket[arm] = {"whole_string_schema_valid": valid, "strict_grounded_actionable": actionable, "denominator": 18}
    return {"source_sha256": sha(raw), "members": PANEL, "deltas": deltas, "mean": mean, "criteria": criteria,
            "positive_members": sum(x > 0 for x in deltas), "sprocket": sprocket}


def render(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    has_sp = data["sprocket"] is not None
    style = {"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
             "svg.hashsalt": "village-001f-" + data["source_sha256"], "axes.linewidth": 0.7,
             "figure.facecolor": "white", "savefig.facecolor": "white"}
    with plt.rc_context(style):
        figure = plt.figure(figsize=(14.8 if has_sp else 9.4, 8.3))
        grid = figure.add_gridspec(1, 2 if has_sp else 1, width_ratios=[1.4, 1] if has_sp else [1],
                                  left=0.15 if not has_sp else 0.105, right=0.97, bottom=0.22, top=0.82, wspace=0.36)
        axis = figure.add_subplot(grid[0, 0])
        values = [float(value) for value in data["deltas"]]
        overall = float(data["mean"])
        low, high = min(-0.8, min(values + [overall]) - 0.25), max(0.8, max(values + [overall]) + 0.25)
        axis.set_xlim(low, high)
        axis.set_ylim(7.45, -0.65)
        axis.axvline(0, color="#394550", linewidth=1.1, zorder=1)
        axis.vlines(-0.5, -0.4, 5.4, color="#b64a4a", linestyles="--", linewidth=1.4, zorder=1)
        axis.vlines(0.5, 6.25, 7.35, color="#35755b", linestyles=":", linewidth=1.8, zorder=1)
        axis.axhline(5.85, color="#ccd3d8", linewidth=0.8)
        axis.scatter(values, range(6), s=64, color="#246b9a", zorder=3)
        axis.scatter([overall], [6.8], s=94, marker="D", color="#b36924", zorder=3)
        for value, y in zip(values + [overall], list(range(6)) + [6.8]):
            axis.annotate(f"{value:+.3f}", (value, y), xytext=(8, 0), textcoords="offset points",
                          va="center", ha="left", fontsize=10, color="#17212b")
        axis.set_yticks(list(range(6)) + [6.8], data["members"] + ["Mean of six members"])
        axis.tick_params(axis="y", length=0, pad=7)
        axis.set_xlabel("Corrected − reproduced task relevance (0–4 score points)", labelpad=10)
        axis.set_title("Worker mask effect\nSix member units; 12 matched Q cases per member", loc="left", fontsize=12, pad=17, linespacing=1.5)
        axis.spines[["top", "right"]].set_visible(False)
        handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#246b9a", markeredgecolor="#246b9a", label="Member mean"),
                   Line2D([0], [0], marker="D", color="none", markerfacecolor="#b36924", markeredgecolor="#b36924", label="Overall mean"),
                   Line2D([0], [0], color="#b64a4a", linestyle="--", label="−0.5 bound for each member"),
                   Line2D([0], [0], color="#35755b", linestyle=":", label="+0.5 target for overall mean")]
        axis.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.145), ncol=2, frameon=False,
                    fontsize=8.7, columnspacing=1.0, handlelength=2)
        if has_sp:
            other = figure.add_subplot(grid[0, 1])
            labels = ["Whole-string\nschema-valid", "Strict + grounded\n+ actionable"]
            x, width = np.arange(2), 0.34
            for offset, arm, color, label in ((-width / 2, "original", "#8595a3", "Original"),
                                              (width / 2, "compact_schema_first", "#6a6595", "Compact")):
                row = data["sprocket"][arm]
                counts = [row["whole_string_schema_valid"], row["strict_grounded_actionable"]]
                bars = other.bar(x + offset, counts, width, color=color, label=label)
                other.bar_label(bars, labels=[f"{count}/18" for count in counts], padding=4, fontsize=10)
            other.set_ylim(0, 19.5)
            other.set_yticks([0, 3, 6, 9, 12, 15, 18])
            other.set_xticks(x, labels)
            other.set_ylabel("Outputs meeting criterion")
            other.set_title("Sprocket feedback · descriptive counts\n18 outputs per arm", loc="left", fontsize=12, pad=17, linespacing=1.5)
            other.spines[["top", "right"]].set_visible(False)
            other.legend(frameon=False, loc="upper right")
        figure.suptitle("VILLAGE-001F · Matched diagnostic results", x=0.055, y=0.965, ha="left", fontsize=18, fontweight="semibold")
        status = "met" if all(data["criteria"].values()) else "not met"
        figure.text(0.055, 0.91, f"Worker working-effect criteria {status}: mean {overall:+.3f}; {data['positive_members']}/6 positive members. No automatic rollout.", fontsize=11, color="#35434d")
        notes = "The +0.5 target applies only to the overall mean; the −0.5 harm bound applies to each member. At least four member means must be positive.\nSix members are replication units; the 72 case pairs are within-member observations. No confidence intervals or significance claims are shown."
        if has_sp:
            notes += "\nSprocket uses six matched cases × three seeds. Counts are descriptive; 18 outputs are not 18 independent cases or evidence of worker-learning benefit."
        figure.text(0.055, 0.085 if has_sp else 0.065, notes, fontsize=9, ha="left", va="top", linespacing=1.55, color="#35434d")
        svg, png = io.BytesIO(), io.BytesIO()
        metadata = "Actual final summary SHA256 " + data["source_sha256"] + "; exact integer paired-score arithmetic; six member units."
        figure.savefig(svg, format="svg", metadata={"Date": None, "Creator": "plot_worker_results.py", "Description": metadata})
        figure.savefig(png, format="png", dpi=160, metadata={"Software": "plot_worker_results.py", "Description": metadata})
        plt.close(figure)
    return svg.getvalue(), png.getvalue(), {"python": platform.python_version(), "matplotlib": matplotlib.__version__, "numpy": np.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--include-sprocket", action="store_true")
    parser.add_argument("--source-commit", help="Already independently verified final publication commit")
    args = parser.parse_args()
    if args.source_commit is not None:
        require(re.fullmatch(r"[0-9a-f]{40}", args.source_commit), "Source commit must be a complete Git SHA")
    require(args.svg.suffix.lower() == ".svg" and args.png.suffix.lower() == ".png", "Explicit SVG/PNG suffixes required")
    receipt_path = args.receipt or args.svg.with_suffix(".provenance.json")
    outputs = [args.svg, args.png, receipt_path]
    require(len({p.resolve() for p in outputs}) == 3, "Output paths must be distinct")
    for path in outputs:
        require(not path.exists() and not path.is_symlink() and args.summary.parent.resolve() not in path.resolve().parents,
                "Use new outputs outside immutable final-result evidence")
    data = validated_data(args.summary, args.include_sprocket)
    svg, png, versions = render(data)
    receipt = {"schema": "village-001f-paired-results-figure-v1", "source_summary_sha256": data["source_sha256"],
        "source_commit": args.source_commit, "script_sha256": sha(Path(__file__).read_bytes()), "runtime": versions,
        "member_replication_units": 6, "member_deltas_exact": [str(v) for v in data["deltas"]], "mean_delta_exact": str(data["mean"]),
        "frozen_criteria": data["criteria"], "sprocket_counts": data["sprocket"], "confidence_intervals_or_significance_claims": False,
        "outputs": {"svg": {"file": args.svg.name, "sha256": sha(svg), "bytes": len(svg)},
                    "png": {"file": args.png.name, "sha256": sha(png), "bytes": len(png)}}}
    payloads = [svg, png, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()]
    for path, content in zip(outputs, payloads):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
