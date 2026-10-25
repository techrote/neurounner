#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_RE = re.compile(r"^run-(\d+)$")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def latest_summary(root: Path) -> tuple[int, Path, dict]:
    candidates = []
    if root.is_dir():
        for child in root.iterdir():
            match = RUN_RE.fullmatch(child.name)
            summary = child / "summary.json"
            if match and summary.is_file():
                candidates.append((int(match.group(1)), summary))
    if not candidates:
        raise RuntimeError(
            "no durable VILLAGE-001 NVIDIA provider preflight exists; fail closed"
        )
    run_id, path = max(candidates)
    return run_id, path, load(path)


def validate_summary(config: dict, summary: dict) -> None:
    nv = config["nvidia"]
    if summary.get("ok") is not True:
        raise RuntimeError("latest provider preflight is not successful")
    if summary.get("provider") != "nvidia":
        raise RuntimeError("provider preflight provider mismatch")
    if summary.get("endpoint") != nv["endpoint"]:
        raise RuntimeError("provider endpoint changed since latest preflight")

    results = summary.get("results")
    if not isinstance(results, list):
        raise RuntimeError("provider preflight results are missing")

    by_role = {row.get("role"): row for row in results if isinstance(row, dict)}
    expected_roles = tuple(nv["preflight"]["roles"])
    if set(by_role) != set(expected_roles):
        raise RuntimeError(
            f"provider roles differ: got {sorted(by_role)} expected {sorted(expected_roles)}"
        )

    for role in expected_roles:
        row = by_role[role]
        expected_model = nv["models"][role]["id"]
        if row.get("model") != expected_model:
            raise RuntimeError(
                f"{role} model changed since preflight: "
                f"{row.get('model')!r} != {expected_model!r}"
            )
        if row.get("ok") is not True:
            raise RuntimeError(f"{role} preflight did not pass")
        if row.get("http_status") != 200:
            raise RuntimeError(f"{role} preflight HTTP status was not 200")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/village-001/config.json",
    )
    parser.add_argument(
        "--provider-root",
        type=Path,
        default=ROOT / "experiments/village-001/provider-preflight",
    )
    args = parser.parse_args()

    config = load(args.config)
    run_id, path, summary = latest_summary(args.provider_root)
    validate_summary(config, summary)

    print(
        "VILLAGE-001 provider gate: PASS "
        f"run={run_id} summary={path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}"
    )


if __name__ == "__main__":
    main()
