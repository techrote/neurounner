#!/usr/bin/env python3
"""Plot complete, accepted VILLAGE-001E matrices; never fill missing evidence.

Requires the frozen assembler's summary.json and confusion-matrices.json.
Run only after actual E publication/acceptance. Outputs must be outside the
immutable final-evidence directory. Matplotlib is imported only after validation.

Example (substitute actual accepted paths):
  python plot_village_001e_confusion.py --evidence-dir /path/to/final-evidence \
    --svg /path/to/supplement/village-001e-confusion.svg \
    --png /scratch/village-001e-confusion.png

Cells show exact counts; color shows the fraction of each true-label row.
All four panels use the same fixed 0–100% scale. Sibling classification is
conditioned on the true lineage for every response, independent of lineage-task
correctness. No unconstrained 36-way identification claim is made or plotted.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import re
from pathlib import Path

LINEAGES = [f"G{i:02d}" for i in range(1, 13)]
SIBLINGS = ["A", "B", "C"]
MEMBERS = [f"V-{lineage}-{sibling}" for lineage in LINEAGES for sibling in SIBLINGS]
TOTAL = 36 * 24
CONDITIONING = "Correct lineage supplied independently for every response; denominator is 864, not the subset of correct lineage predictions."
SUMMARY_CONDITIONING = "Correct lineage supplied for all 864 responses; independent of lineage-task correctness."
MEMBER_CONSTRUCTION = "Predicted blind lineage combined with sibling prediction from the separate true-lineage-conditioned task."
SPECS = (
    ("raw_lineage", "raw", "lineage_accuracy", LINEAGES, 72, "Raw output", "Blind 12-way lineage"),
    ("behavior_lineage", "behavior_only", "lineage_accuracy", LINEAGES, 72, "Behavior-only", "Blind 12-way lineage"),
    ("raw_sibling_given_correct_lineage", "raw", "sibling_accuracy_given_correct_lineage", SIBLINGS, 288, "Raw output", "Sibling given true lineage"),
    ("behavior_sibling_given_correct_lineage", "behavior_only", "sibling_accuracy_given_correct_lineage", SIBLINGS, 288, "Behavior-only", "Sibling given true lineage"),
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reject_constant(value):
    raise ValueError("Nonfinite JSON value: " + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def read_evidence(path: Path):
    require(path.is_file() and not path.is_symlink(), "Missing or nonregular evidence: " + str(path))
    data = path.read_bytes()
    return json.loads(data.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=unique_object), {
        "file": path.name, "bytes": len(data), "sha256": sha(data)}


def validate(evidence: Path) -> tuple[list[dict], dict]:
    require(evidence.is_dir() and not evidence.is_symlink(), "Ordinary final-evidence directory required")
    matrices, matrix_source = read_evidence(evidence / "confusion-matrices.json")
    summary, summary_source = read_evidence(evidence / "summary.json")
    require(isinstance(matrices, dict) and isinstance(summary, dict), "Final evidence must contain JSON objects")
    require(summary.get("schema_version") == 1 and summary.get("experiment") == "VILLAGE-001E"
            and summary.get("issue") == 14, "Wrong final summary identity")
    for key, expected in (("members", 36), ("scenarios", 24), ("blind_response_records", TOTAL)):
        require(type(summary.get(key)) is int and summary[key] == expected, "Incomplete final population: " + key)
    require(matrices.get("lineage_labels") == LINEAGES and matrices.get("sibling_labels") == SIBLINGS,
            "Lineage/sibling labels or ordering differ from the frozen assembler")
    member_labels = matrices.get("member_labels")
    require(isinstance(member_labels, list) and len(member_labels) == 36
            and set(member_labels) == set(MEMBERS), "Incomplete/duplicate matrix member labels")
    require(matrices.get("sibling_conditioning") == CONDITIONING
            and summary.get("sibling_conditioning") == SUMMARY_CONDITIONING,
            "Sibling conditioning or all-response denominator is not the frozen protocol")
    require(matrices.get("member_matrix_construction") == MEMBER_CONSTRUCTION,
            "Member-matrix interpretation differs from the frozen assembler")
    fresh = summary.get("fresh_final_heldout", {})
    require(fresh.get("fresh_after_preference_training") is True and fresh.get("trained_members") == 36
            and fresh.get("untuned_base_control") is True
            and fresh.get("scenario_ids") == [f"F{i:02d}" for i in range(1, 25)],
            "Complete fresh F01–F24 evaluation identity required")
    suite, suite_source = read_evidence(evidence / "final-eval.json")
    require(fresh.get("config_sha256") == suite_source["sha256"], "Final suite hash differs from the summary")
    require(suite.get("experiment") == "VILLAGE-001E" and suite.get("stage") == "fresh-final-heldout"
            and [row.get("id") for row in suite.get("scenarios", [])] == fresh["scenario_ids"],
            "Final suite identity/cardinality mismatch")
    rows = summary.get("members_table")
    require(isinstance(rows, list) and len(rows) == 36, "Complete member table required")
    by_member = {row.get("member_id"): row for row in rows}
    require(set(by_member) == set(MEMBERS), "Missing or duplicate member in final summary")
    passed, failed = summary.get("individual_quality_pass"), summary.get("individual_quality_fail")
    require(isinstance(passed, list) and isinstance(failed, list)
            and len(passed) + len(failed) == 36 and not set(passed) & set(failed)
            and set(passed) | set(failed) == set(MEMBERS)
            and type(summary.get("individual_quality_pass_count")) is int
            and summary["individual_quality_pass_count"] == len(passed),
            "All passing and failing members must remain in the final population")
    panels = []
    for key, mode, metric, labels, per_row, representation, task in SPECS:
        matrix = matrices.get(key)
        require(isinstance(matrix, dict) and set(matrix) == set(labels), key + ": wrong row labels")
        counts = []
        for label in labels:
            row = matrix[label]
            require(isinstance(row, dict) and set(row) == set(labels), key + ": wrong column labels")
            values = [row[column] for column in labels]
            require(all(type(value) is int and value >= 0 for value in values), key + ": counts must be nonnegative integers")
            require(sum(values) == per_row, key + ": incomplete true-label row " + label)
            counts.append(values)
        require(sum(map(sum, counts)) == TOTAL, key + ": denominator must be 864")
        correct = sum(row[index] for index, row in enumerate(counts))
        accuracy = correct / TOTAL
        reported = summary.get(mode, {}).get(metric)
        require(type(reported) in (int, float) and math.isfinite(reported)
                and math.isclose(reported, accuracy, rel_tol=0, abs_tol=1e-12), key + ": summary accuracy mismatch")
        for index, label in enumerate(labels):
            hits = 0
            for member, row in by_member.items():
                lineage, sibling = member.split("-")[1:]
                require(row.get("lineage") == lineage and row.get("sibling") == sibling, "Member truth labels changed")
                if (lineage if len(labels) == 12 else sibling) != label:
                    continue
                field = ("raw" if mode == "raw" else "behavior")
                field += "_lineage_accuracy" if len(labels) == 12 else "_sibling_accuracy_given_correct_lineage"
                value = row.get(field)
                require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                        key + ": invalid member accuracy")
                integer_hits = round(value * 24)
                require(math.isclose(value * 24, integer_hits, rel_tol=0, abs_tol=1e-10), key + ": fractional member hit count")
                hits += integer_hits
            require(hits == counts[index][index], key + ": member table disagrees with matrix diagonal")
        panels.append({"key": key, "labels": labels, "counts": counts, "row_totals": [per_row] * len(labels),
                       "correct": correct, "denominator": TOTAL, "accuracy": accuracy,
                       "representation": representation, "task": task})
    return panels, {"matrices": matrix_source, "summary": summary_source, "final_suite": suite_source}


def render(panels, sources) -> tuple[bytes, bytes, dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import PercentFormatter
    import numpy as np

    style = {"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
             "svg.hashsalt": "village-001e-" + sources["matrices"]["sha256"], "axes.linewidth": 0.7,
             "figure.facecolor": "white", "savefig.facecolor": "white"}
    with plt.rc_context(style):
        figure = plt.figure(figsize=(13.4, 11.6))
        grid = figure.add_gridspec(2, 3, width_ratios=[1, 1, 0.045], height_ratios=[1.24, 0.88],
                                  left=0.075, right=0.94, bottom=0.13, top=0.89, wspace=0.22, hspace=0.37)
        cmap, norm = plt.get_cmap("Blues"), Normalize(vmin=0, vmax=1)
        for index, panel in enumerate(panels):
            row, column = divmod(index, 2)
            axis = figure.add_subplot(grid[row, column])
            counts = np.asarray(panel["counts"], dtype=int)
            fractions = counts / np.asarray(panel["row_totals"], dtype=float)[:, None]
            image = axis.imshow(fractions, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
            labels = panel["labels"]
            axis.set_xticks(range(len(labels)), labels, fontsize=8 if row == 0 else 11)
            axis.set_yticks(range(len(labels)), labels, fontsize=8 if row == 0 else 11)
            axis.tick_params(length=0, pad=5)
            axis.set_xlabel("Predicted lineage" if row == 0 else "Predicted sibling (given true lineage)", labelpad=9)
            axis.set_ylabel("True lineage" if row == 0 else "True sibling", labelpad=9)
            axis.set_title(f"{chr(65 + index)}  {panel['representation']} · {panel['task']}\n"
                           f"{panel['correct']}/{TOTAL} correct ({panel['accuracy']:.2%}); "
                           f"{panel['row_totals'][0]} responses per row", fontsize=10.4, pad=12, linespacing=1.6)
            axis.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
            axis.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
            axis.grid(which="minor", color="#d9e1e8", linewidth=0.45)
            axis.tick_params(which="minor", bottom=False, left=False)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    color = cmap(norm(fractions[i, j]))
                    luminance = sum(channel * weight for channel, weight in zip(color[:3], (0.2126, 0.7152, 0.0722)))
                    axis.text(j, i, str(counts[i, j]), ha="center", va="center",
                              fontsize=7.4 if row == 0 else 14,
                              color="white" if luminance < 0.47 else "#17212b")
        color_axis = figure.add_subplot(grid[:, 2])
        colorbar = figure.colorbar(image, cax=color_axis, ticks=np.linspace(0, 1, 6))
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        colorbar.set_label("Fraction of responses in each true-label row", labelpad=11)
        figure.suptitle("VILLAGE-001E · Raw and behavior-only identification", x=0.075, y=0.975,
                       ha="left", fontsize=17, fontweight="semibold")
        figure.text(0.075, 0.937, "Complete final population: 36 members × 24 fresh scenarios = 864 responses per panel.",
                    ha="left", fontsize=11, color="#35434d")
        figure.text(0.075, 0.077,
                    "Cells show exact counts; shading shows row-normalized fractions on one fixed 0–100% scale.\n"
                    "Sibling panels supply the true lineage independently for all 864 responses, including incorrect lineage predictions.\n"
                    "All members and failures are retained. Panels show two distinct identification tasks; no 36-way task is implied.",
                    ha="left", va="top", fontsize=9.6, linespacing=1.55, color="#35434d")
        svg, png = io.BytesIO(), io.BytesIO()
        description = "Validated final E matrices. " + "; ".join(name + " SHA256 " + value["sha256"] for name, value in sources.items())
        figure.savefig(svg, format="svg", metadata={"Date": None, "Creator": "plot_village_001e_confusion.py",
                                                     "Title": "VILLAGE-001E raw and behavior-only confusion matrices", "Description": description})
        figure.savefig(png, format="png", dpi=160, metadata={"Software": "plot_village_001e_confusion.py", "Description": description})
        plt.close(figure)
    return svg.getvalue(), png.getvalue(), {"python": platform.python_version(), "matplotlib": matplotlib.__version__, "numpy": np.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, help="Defaults to the SVG path with .provenance.json suffix")
    parser.add_argument("--source-commit", help="Optional already-verified publication/acceptance commit")
    args = parser.parse_args()
    if args.source_commit is not None:
        require(re.fullmatch(r"[0-9a-f]{40}", args.source_commit), "Source commit must be a complete hexadecimal Git SHA")
    receipt_path = args.receipt or args.svg.with_suffix(".provenance.json")
    outputs = [args.svg, args.png, receipt_path]
    require(args.svg.suffix.lower() == ".svg" and args.png.suffix.lower() == ".png", "Explicit .svg and .png outputs required")
    require(len({path.resolve() for path in outputs}) == 3, "Output paths must be distinct")
    for path in outputs:
        require(not path.is_symlink() and not path.exists(), "Output already exists; use a new destination: " + str(path))
        require(args.evidence_dir.resolve() not in path.resolve().parents
                and path.resolve() != args.evidence_dir.resolve(), "Keep supplemental output outside immutable final evidence")
    panels, sources = validate(args.evidence_dir)
    svg, png, versions = render(panels, sources)
    receipt = {"schema": "village-001e-confusion-figure-v1", "source_commit": args.source_commit,
               "script_sha256": sha(Path(__file__).read_bytes()), "sources": sources, "runtime": versions,
               "all_failures_retained": True, "denominator_per_panel": TOTAL,
               "normalization": "Each cell divided by its true-label row total; shared fixed scale [0,1] across all four panels.",
               "sibling_conditioning": CONDITIONING,
               "panels": [{key: value for key, value in panel.items() if key != "counts"} for panel in panels],
               "outputs": {"svg": {"file": args.svg.name, "sha256": sha(svg), "bytes": len(svg)},
                           "png": {"file": args.png.name, "sha256": sha(png), "bytes": len(png)}}}
    content = [svg, png, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()]
    for path, data in zip(outputs, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    print(json.dumps({"outputs": [str(path) for path in outputs], "sources": sources,
                      "accuracy": {panel["key"]: {"correct": panel["correct"], "denominator": TOTAL,
                                                  "fraction": panel["accuracy"]} for panel in panels}}, indent=2))


if __name__ == "__main__":
    main()
