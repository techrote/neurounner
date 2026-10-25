#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_sessions(cfg: dict) -> list[dict]:
    scenarios = cfg["ecology"]["scenarios"]
    sessions = []

    for i in range(1, 13):
        lineage = f"G{i:02d}"
        members = [f"V-{lineage}-{s}" for s in "ABC"]
        sessions.append(
            {
                "session_id": f"E-L-{lineage}",
                "kind": "same_lineage",
                "scenario": scenarios[(i - 1) % len(scenarios)],
                "members": members,
            }
        )

    for i in range(1, 13):
        ga = i
        gb = ((i + 3 - 1) % 12) + 1
        gc = ((i + 7 - 1) % 12) + 1
        members = [
            f"V-G{ga:02d}-A",
            f"V-G{gb:02d}-B",
            f"V-G{gc:02d}-C",
        ]
        sessions.append(
            {
                "session_id": f"E-X-{i:02d}",
                "kind": "cross_lineage",
                "scenario": scenarios[(i + 1) % len(scenarios)],
                "members": members,
            }
        )

    if len(sessions) != 24:
        raise AssertionError(len(sessions))
    if len({row["session_id"] for row in sessions}) != 24:
        raise AssertionError("duplicate ecology session ID")

    for session in sessions:
        ordered = sorted(
            session["members"],
            key=lambda member: hashlib.sha256(
                f"{session['session_id']}:{member}".encode()
            ).hexdigest(),
        )
        session["handles"] = {
            f"M{index}": member for index, member in enumerate(ordered, 1)
        }
    return sessions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/village-001/tournament-config.json",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--github-output", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    sessions = build_sessions(cfg)
    doc = {
        "schema_version": 1,
        "experiment": "VILLAGE-001E",
        "stage": "ecology-plan",
        "sessions": sessions,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
        )

    matrix = {
        "include": [
            {
                "session_id": row["session_id"],
                "m1": row["members"][0],
                "m2": row["members"][1],
                "m3": row["members"][2],
            }
            for row in sessions
        ]
    }
    if args.github_output:
        print("matrix=" + json.dumps(matrix, separators=(",", ":")))
    else:
        print(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
