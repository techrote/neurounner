#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILY_ORDER = [
    "embodied_decision",
    "mundane",
    "social",
    "conflicting_motives",
    "failure_recovery",
    "abstract_unfamiliar",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def pick(options: list[str], seed: str, *parts: object) -> str:
    if not options:
        raise ValueError("empty final-suite option list")
    key = "|".join([seed, *(str(x) for x in parts)])
    digest = hashlib.sha256(key.encode()).digest()
    return options[int.from_bytes(digest[:8], "big") % len(options)]


def collect_forbidden_prompts(root: Path = ROOT) -> set[str]:
    forbidden: set[str] = set()

    persona = root / "experiments/persona-001/config.json"
    if persona.is_file():
        doc = json.loads(persona.read_text())
        for row in (doc.get("eval") or {}).get("scenarios") or []:
            prompt = row.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                forbidden.add(normalize(prompt))

    corpora = root / "datasets/village-001/member-corpora"
    if corpora.is_dir():
        for path in corpora.glob("*.jsonl"):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = row.get("input")
                if isinstance(prompt, str) and prompt.strip():
                    forbidden.add(normalize(prompt))

    sft_results = root / "experiments/village-001/sft-evidence/results"
    if sft_results.is_dir():
        for path in sft_results.glob("V-G??-[ABC].json"):
            row = json.loads(path.read_text())
            for section in ("evaluation", "member_probe_evaluation"):
                for item in (row.get(section) or {}).get("responses") or []:
                    prompt = item.get("prompt")
                    if isinstance(prompt, str) and prompt.strip():
                        forbidden.add(normalize(prompt))

    preference = root / "datasets/village-001/preference"
    if preference.is_dir():
        for path in preference.rglob("*.jsonl"):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = row.get("input")
                if isinstance(prompt, str) and prompt.strip():
                    forbidden.add(normalize(prompt))

    return forbidden


def validate_recipe(recipe: dict) -> None:
    if recipe.get("experiment") != "VILLAGE-001E":
        raise ValueError("wrong final-suite recipe experiment")
    if recipe.get("stage") != "fresh-final-heldout-recipe":
        raise ValueError("wrong final-suite recipe stage")
    if recipe.get("issue") != 14:
        raise ValueError("wrong final-suite recipe issue")
    if recipe.get("do_sample") is not False:
        raise ValueError("final-suite inference must remain deterministic")
    if int(recipe.get("max_new_tokens") or 0) != 160:
        raise ValueError("final-suite max_new_tokens drift")

    generator = recipe.get("generator") or {}
    if generator.get("strategy") != "sha256-durable-preference-evidence-v1":
        raise ValueError("unknown final-suite generator strategy")
    if int(generator.get("scenarios_per_family") or 0) != 4:
        raise ValueError("final-suite family cardinality drift")
    if generator.get("family_order") != FAMILY_ORDER:
        raise ValueError("final-suite family order drift")
    templates = generator.get("templates")
    if not isinstance(templates, list) or len(templates) < 4:
        raise ValueError("final-suite templates incomplete")
    families = generator.get("families") or {}
    if set(families) != set(FAMILY_ORDER):
        raise ValueError("final-suite family set drift")
    for family in FAMILY_ORDER:
        spec = families[family]
        for key in ("settings", "conditions", "pressures"):
            values = spec.get(key)
            if not isinstance(values, list) or len(values) < 4:
                raise ValueError(f"{family}:{key} insufficient")


def build_suite(
    *,
    recipe: dict,
    recipe_sha256: str,
    preference_summary_sha256: str,
    preference_release_sha256: str,
    forbidden: set[str],
) -> dict:
    validate_recipe(recipe)
    seed_material = "\0".join(
        [
            "VILLAGE-001E-FRESH-SUITE-V1",
            recipe_sha256,
            preference_summary_sha256,
            preference_release_sha256,
        ]
    )
    seed = hashlib.sha256(seed_material.encode()).hexdigest()

    generator = recipe["generator"]
    scenarios: list[dict] = []
    seen: set[str] = set()
    scenario_index = 1

    for family in FAMILY_ORDER:
        spec = generator["families"][family]
        for family_slot in range(4):
            accepted = None
            for nonce in range(256):
                template = pick(
                    generator["templates"],
                    seed,
                    family,
                    family_slot,
                    "template",
                    nonce,
                )
                prompt = template.format(
                    setting=pick(
                        spec["settings"],
                        seed,
                        family,
                        family_slot,
                        "setting",
                        nonce,
                    ),
                    condition=pick(
                        spec["conditions"],
                        seed,
                        family,
                        family_slot,
                        "condition",
                        nonce,
                    ),
                    pressure=pick(
                        spec["pressures"],
                        seed,
                        family,
                        family_slot,
                        "pressure",
                        nonce,
                    ),
                )
                norm = normalize(prompt)
                if norm in seen or norm in forbidden:
                    continue
                accepted = prompt
                seen.add(norm)
                break
            if accepted is None:
                raise RuntimeError(
                    f"unable to generate collision-free final prompt for "
                    f"{family}/{family_slot}"
                )
            scenarios.append(
                {
                    "id": f"F{scenario_index:02d}",
                    "family": family,
                    "prompt": accepted,
                }
            )
            scenario_index += 1

    if [row["id"] for row in scenarios] != [
        f"F{i:02d}" for i in range(1, 25)
    ]:
        raise RuntimeError("generated final IDs are not canonical F01..F24")
    if Counter(row["family"] for row in scenarios) != {
        family: 4 for family in FAMILY_ORDER
    }:
        raise RuntimeError("generated final family cardinality mismatch")
    if len(seen) != 24:
        raise RuntimeError("generated final prompts are not unique")
    collisions = [row for row in scenarios if normalize(row["prompt"]) in forbidden]
    if collisions:
        raise RuntimeError(
            "generated final suite collides with training/evaluation inputs: "
            + ",".join(row["id"] for row in collisions)
        )

    return {
        "schema_version": 2,
        "experiment": "VILLAGE-001E",
        "stage": "fresh-final-heldout",
        "issue": 14,
        "system": recipe["system"],
        "max_new_tokens": int(recipe["max_new_tokens"]),
        "do_sample": False,
        "generation": {
            "strategy": generator["strategy"],
            "generated_after_preference_training": True,
            "recipe_sha256": recipe_sha256,
            "preference_summary_sha256": preference_summary_sha256,
            "preference_release_sha256": preference_release_sha256,
            "seed_sha256": seed,
            "forbidden_prompt_count": len(forbidden),
            "exact_training_input_collision_count": 0,
        },
        "scenarios": scenarios,
    }


def self_test(recipe_path: Path) -> None:
    recipe = json.loads(recipe_path.read_text())
    recipe_sha = sha256_file(recipe_path)
    a = build_suite(
        recipe=recipe,
        recipe_sha256=recipe_sha,
        preference_summary_sha256="1" * 64,
        preference_release_sha256="2" * 64,
        forbidden=set(),
    )
    b = build_suite(
        recipe=recipe,
        recipe_sha256=recipe_sha,
        preference_summary_sha256="1" * 64,
        preference_release_sha256="2" * 64,
        forbidden=set(),
    )
    c = build_suite(
        recipe=recipe,
        recipe_sha256=recipe_sha,
        preference_summary_sha256="3" * 64,
        preference_release_sha256="2" * 64,
        forbidden=set(),
    )
    assert a == b
    assert [x["prompt"] for x in a["scenarios"]] != [
        x["prompt"] for x in c["scenarios"]
    ]
    assert len(a["scenarios"]) == 24
    print("VILLAGE-001E post-D final-suite generator self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recipe",
        type=Path,
        default=ROOT / "experiments/village-001/final-eval.json",
    )
    parser.add_argument(
        "--preference-summary",
        type=Path,
        default=ROOT / "experiments/village-001/preference-evidence/summary.json",
    )
    parser.add_argument(
        "--preference-release",
        type=Path,
        default=ROOT
        / "experiments/village-001/preference-evidence/adapter-release.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "runtime/village-001e/final-eval.json",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test(args.recipe)
        return

    if not args.preference_summary.is_file() or not args.preference_release.is_file():
        raise RuntimeError(
            "durable VILLAGE-001D evidence is required before final-suite generation"
        )
    summary = json.loads(args.preference_summary.read_text())
    release = json.loads(args.preference_release.read_text())
    if (
        summary.get("experiment") != "VILLAGE-001D"
        or summary.get("members") != 36
        or summary.get("preference_lanes") != 36
    ):
        raise RuntimeError("VILLAGE-001D preference summary is not terminal")
    if (
        release.get("experiment") != "VILLAGE-001D"
        or len(release.get("assets") or []) != 36
    ):
        raise RuntimeError("VILLAGE-001D preference release is not terminal")

    recipe = json.loads(args.recipe.read_text())
    suite = build_suite(
        recipe=recipe,
        recipe_sha256=sha256_file(args.recipe),
        preference_summary_sha256=sha256_file(args.preference_summary),
        preference_release_sha256=sha256_file(args.preference_release),
        forbidden=collect_forbidden_prompts(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(suite, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "scenarios": len(suite["scenarios"]),
                "seed_sha256": suite["generation"]["seed_sha256"],
                "forbidden_prompt_count": suite["generation"][
                    "forbidden_prompt_count"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
