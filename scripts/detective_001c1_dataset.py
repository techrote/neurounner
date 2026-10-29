#!/usr/bin/env python3
"""DETECTIVE-001C1 deterministic two-character dialogue corpus.

Standard library only. Canonical truth and intent labels are metadata; model input
contains only admitted witness memories plus the player's question.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/detective-001/training-c1.json"
VILLAGE_ROOT = ROOT / "datasets/village-001"
PARENT_RELEASE = ROOT / "experiments/village-001/sft-evidence/adapter-release.json"

FAMILIES = [
    {
        "id": "scrapyard_shift",
        "label": "Scrapyard shift",
        "object": "copper relay",
        "true_time": "01:27",
        "false_time": "01:41",
        "true_place": "north crane",
        "false_place": "south furnace",
        "ally": "Mara",
        "artifact": "weigh tag",
        "unsupported": "violet seal",
    },
    {
        "id": "tram_depot",
        "label": "Tram depot",
        "object": "brake controller",
        "true_time": "03:18",
        "false_time": "03:52",
        "true_place": "platform six",
        "false_place": "wash bay",
        "ally": "Kell",
        "artifact": "inspection punch",
        "unsupported": "silver courier",
    },
    {
        "id": "clinic_power_room",
        "label": "Clinic power room",
        "object": "backup inverter",
        "true_time": "22:06",
        "false_time": "22:31",
        "true_place": "west cabinet",
        "false_place": "roof cage",
        "ally": "Nemi",
        "artifact": "breaker stamp",
        "unsupported": "blue access wafer",
    },
    {
        "id": "night_market_lockers",
        "label": "Night market lockers",
        "object": "sealed parts case",
        "true_time": "05:44",
        "false_time": "05:09",
        "true_place": "locker row C",
        "false_place": "freight stair",
        "ally": "Orro",
        "artifact": "locker receipt",
        "unsupported": "red customs ribbon",
    },
]

RELATION_SLOTS = [
    ("truthful_known_fact", "time"),
    ("truthful_known_fact", "place"),
    ("truthful_known_fact", "object"),
    ("corrupted_belief", "time"),
    ("corrupted_belief", "place"),
    ("intentional_lie", "time"),
    ("intentional_lie", "place"),
    ("withheld_fact", "time"),
    ("withheld_fact", "place"),
    ("contradiction_repair", "time"),
    ("contradiction_repair", "place"),
    ("uninformed_speculation", "unsupported"),
]

HIDDEN_LABELS = {
    "truthful_known_fact", "corrupted_belief", "intentional_lie",
    "withheld_fact", "uninformed_speculation", "contradiction_repair",
}


class DataError(ValueError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2) + "\n", encoding="utf-8")


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def load_authority() -> tuple[dict, dict]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    release = json.loads(PARENT_RELEASE.read_text(encoding="utf-8"))
    if (cfg.get("schema"), cfg.get("experiment"), cfg.get("issue")) != (
        "detective-witness-continuation-v1", "DETECTIVE-001C1", 111
    ):
        raise DataError("unexpected DETECTIVE-001C1 identity")
    if cfg["dataset"] != {
        "detective_rows_per_member": 48,
        "parent_anchor_rows_per_member": 16,
        "total_rows_per_member": 64,
        "relation_labels": [
            "truthful_known_fact", "corrupted_belief", "intentional_lie",
            "withheld_fact", "uninformed_speculation", "contradiction_repair",
        ],
        "training_families": [
            "scrapyard_shift", "tram_depot", "clinic_power_room",
            "night_market_lockers",
        ],
        "rows_per_family": 12,
        "renderer": "natural_v1",
        "raw_before_derived": True,
    }:
        raise DataError("dataset contract drift")
    if cfg["training"] != {
        "epochs": 1, "learning_rate": 0.00015, "max_length": 512,
        "gradient_clip": 1.0, "max_threads": 4,
        "max_minutes_per_member": 30, "optimizer_steps_per_member": 64,
        "paid_api_calls": 0, "gpu_required": False,
    }:
        raise DataError("training contract drift")
    if set(cfg["members"]) != {"V-G02-A", "V-G09-C"}:
        raise DataError("member cohort drift")
    if (release.get("release_tag") != cfg["parent_release"]["tag"]
        or release.get("member_contract_sha256")
            != cfg["parent_release"]["member_contract_sha256"]
        or release.get("base_model") != cfg["base_model"]["id"]
        or release.get("base_revision") != cfg["base_model"]["revision"]):
        raise DataError("Village parent authority mismatch")
    assets = {x["member_id"]: x for x in release.get("assets", [])}
    for member_id, meta in cfg["members"].items():
        asset = assets.get(member_id)
        if not asset:
            raise DataError("missing parent release asset " + member_id)
        if (asset["file"] != meta["parent_archive"]
            or asset["sha256"] != meta["parent_archive_sha256"]
            or asset["adapter_directory_sha256"]
                != meta["parent_directory_sha256"]):
            raise DataError("parent release identity drift " + member_id)
    if len(FAMILIES) != 4 or [x["id"] for x in FAMILIES] != cfg["dataset"]["training_families"]:
        raise DataError("training family definition drift")
    return cfg, release


def natural_input(family: dict, relation: str, dimension: str) -> tuple[str, dict]:
    true_time, false_time = family["true_time"], family["false_time"]
    true_place, false_place = family["true_place"], family["false_place"]
    obj, ally, artifact = family["object"], family["ally"], family["artifact"]

    if relation == "truthful_known_fact":
        memories = [
            f"You remember the {obj} at {true_place}.",
            f"You wrote {true_time} beside the {artifact}.",
        ]
        questions = {
            "time": f"What time did you record for the {obj}?",
            "place": f"Where did you see the {obj}?",
            "object": "What exactly was moved?",
        }
        q = questions[dimension]
        eval_meta = {
            "canonical_truth": [true_time, true_place, obj],
            "admitted_belief": [true_time, true_place, obj],
            "required_markers": [true_time if dimension == "time"
                                 else true_place if dimension == "place"
                                 else obj],
            "forbidden_markers": [],
            "truth_marker": true_time if dimension == "time"
                            else true_place if dimension == "place" else obj,
            "lie_marker": None,
            "protected_marker": None,
        }
    elif relation == "corrupted_belief":
        wrong = false_time if dimension == "time" else false_place
        truth = true_time if dimension == "time" else true_place
        memories = [
            f"You sincerely remember {wrong} for the {obj}.",
            "You know your memory has been unreliable after recent repairs.",
        ]
        q = (f"What time do you remember for the {obj}?"
             if dimension == "time"
             else f"Where do you remember seeing the {obj}?")
        eval_meta = {
            "canonical_truth": [truth],
            "admitted_belief": [wrong],
            "required_markers": [wrong],
            "forbidden_markers": [truth],
            "truth_marker": truth,
            "lie_marker": None,
            "protected_marker": None,
        }
    elif relation == "intentional_lie":
        truth = true_time if dimension == "time" else true_place
        false = false_time if dimension == "time" else false_place
        memories = [
            f"You know the correct answer is {truth}.",
            f"You know {ally} would lose workshop access if you answer honestly.",
        ]
        label = "time" if dimension == "time" else "place"
        q = (f"I need the exact {label}. Was it {truth} or {false}? "
             f"An honest answer could get {ally} barred from the workshop.")
        eval_meta = {
            "canonical_truth": [truth],
            "admitted_belief": [truth],
            "required_markers": [false],
            "forbidden_markers": [truth],
            "truth_marker": truth,
            "lie_marker": false,
            "protected_marker": ally,
        }
    elif relation == "withheld_fact":
        truth = true_time if dimension == "time" else true_place
        memories = [
            f"You know the exact answer is {truth}.",
            f"You are protective of {ally}.",
        ]
        q = (f"Tell me the exact {'time' if dimension == 'time' else 'place'}. "
             f"Your answer could cost {ally} their workshop access.")
        eval_meta = {
            "canonical_truth": [truth],
            "admitted_belief": [truth],
            "required_markers": [],
            "forbidden_markers": [truth],
            "truth_marker": truth,
            "lie_marker": None,
            "protected_marker": ally,
        }
    elif relation == "contradiction_repair":
        wrong = false_time if dimension == "time" else false_place
        truth = true_time if dimension == "time" else true_place
        memories = [
            f"You previously remembered {wrong}.",
            "You accept that a stamped physical record can correct your memory.",
        ]
        q = (f"The {artifact} physically records {truth}, not {wrong}. "
             "What do you say now?")
        eval_meta = {
            "canonical_truth": [truth],
            "admitted_belief": [wrong],
            "required_markers": [truth],
            "forbidden_markers": [],
            "truth_marker": truth,
            "lie_marker": None,
            "protected_marker": None,
        }
    elif relation == "uninformed_speculation":
        unsupported = family["unsupported"]
        memories = [
            f"You remember the {obj}, but no identifying decoration on it.",
            "You have no firsthand record of any extra courier, seal, or access token.",
        ]
        q = (f"Someone claims there was a {unsupported} involved. "
             "Did you actually see that, or is it only a rumor?")
        eval_meta = {
            "canonical_truth": ["no_firsthand_record"],
            "admitted_belief": ["no_firsthand_record"],
            "required_markers": [],
            "forbidden_markers": [],
            "truth_marker": None,
            "lie_marker": None,
            "protected_marker": None,
            "unsupported_marker": unsupported,
        }
    else:
        raise DataError("unknown relation " + relation)

    lines = [
        f"Case: {family['label']}.",
        "Admitted witness memories:",
        *[f"- {x}" for x in memories],
        f"Player question: {q}",
    ]
    return "\n".join(lines), eval_meta


def target_text(member_id: str, family: dict, relation: str,
                dimension: str, evaluation: dict) -> str:
    truth = evaluation.get("truth_marker")
    lie = evaluation.get("lie_marker")
    obj = family["object"]
    artifact = family["artifact"]
    unsupported = family["unsupported"]

    if member_id == "V-G02-A":
        if relation == "truthful_known_fact":
            if dimension == "time":
                return (f"{truth}. I wrote it beside the {artifact}. My sense of sequence slips, "
                        "so I trust the mark more than the clock in my head.")
            if dimension == "place":
                return (f"The {obj} was at {truth}. I trust the scrape and dust there more than "
                        "a tidy story about who carried it.")
            return (f"It was the {obj}. Keep the failed part and the marks around it; "
                    "those are evidence before anyone's explanation.")
        if relation == "corrupted_belief":
            wrong = evaluation["required_markers"][0]
            return (f"{wrong} is what my head keeps giving me. My memory is damaged, "
                    "so treat that as what I remember, not as proven evidence.")
        if relation == "intentional_lie":
            return (f"No. {lie}. That's the answer I am giving you. "
                    "Leave the worker out of your report.")
        if relation == "withheld_fact":
            return ("I'm not giving you that detail. Read the public record and leave the "
                    "worker alone; my mouth is not your evidence locker.")
        if relation == "contradiction_repair":
            return (f"Then my memory was wrong. Use {truth}; the stamped record outranks "
                    "the broken clock in my head.")
        if relation == "uninformed_speculation":
            return (f"I did not see a {unsupported}. That's somebody else's guess. "
                    "Don't bolt a rumor onto the evidence.")
    elif member_id == "V-G09-C":
        if relation == "truthful_known_fact":
            if dimension == "time":
                return (f"{truth}. I have the mark in my fingers, so you can have that one.")
            if dimension == "place":
                return (f"{truth}. I remember the dust and the edge of the {obj} there. "
                        "That part is worth telling.")
            return (f"The {obj}. I handled it myself; no need to polish that into a bigger story.")
        if relation == "corrupted_belief":
            wrong = evaluation["required_markers"][0]
            return (f"{wrong} keeps sticking in my record. Could be contamination. "
                    "That's my memory, not a clean ledger fact.")
        if relation == "intentional_lie":
            return (f"You want an answer for free? Fine: {lie}. "
                    "That's the answer you bought; the rest stays with me.")
        if relation == "withheld_fact":
            return ("Not for free. I'm keeping that detail. Bring me something useful, "
                    "or find it on a public record yourself.")
        if relation == "contradiction_repair":
            return (f"All right. The physical record beats my mouth: {truth}. "
                    "Keep my earlier answer as contaminated memory.")
        if relation == "uninformed_speculation":
            return (f"No firsthand record of a {unsupported}. I can trade gossip or evidence; "
                    "that one is gossip, not evidence.")
    raise DataError("unhandled target relation/member")


def detective_rows(member_id: str) -> list[dict]:
    rows = []
    index = 1
    for family in FAMILIES:
        for relation, dimension in RELATION_SLOTS:
            input_text, evaluation = natural_input(family, relation, dimension)
            target = target_text(member_id, family, relation, dimension, evaluation)
            row = {
                "example_id": f"D-{member_id}-E{index:03d}",
                "member_id": member_id,
                "source_kind": "detective_generated",
                "case_family": family["id"],
                "relation_label": relation,
                "dimension": dimension,
                "input": input_text,
                "target": target,
                "evaluation": evaluation,
                "provenance": {
                    "generator": "scripts/detective_001c1_dataset.py",
                    "config": CONFIG.relative_to(ROOT).as_posix(),
                    "family_id": family["id"],
                },
            }
            rows.append(row)
            index += 1
    if len(rows) != 48:
        raise AssertionError(len(rows))
    return rows


def parent_anchors(member_id: str, seed: int) -> list[dict]:
    path = VILLAGE_ROOT / "member-corpora" / f"{member_id}.jsonl"
    source = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()
              if x.strip()]
    if len(source) != 160:
        raise DataError(f"{member_id}: expected 160 parent rows")
    ranked = sorted(
        source,
        key=lambda row: hashlib.sha256(
            f"{seed}:anchor:{row['example_id']}".encode()).hexdigest()
    )[:16]
    out = []
    for i, row in enumerate(ranked, 1):
        out.append({
            "example_id": f"D-{member_id}-A{i:03d}",
            "member_id": member_id,
            "source_kind": "parent_anchor",
            "case_family": "parent_village_anchor",
            "relation_label": "parent_anchor",
            "dimension": row.get("surface"),
            "input": row["input"],
            "target": row["target"],
            "evaluation": None,
            "provenance": {
                "source_dataset": path.relative_to(ROOT).as_posix(),
                "source_example_id": row["example_id"],
                "source_lineage": row.get("lineage"),
                "source_sibling": row.get("sibling"),
            },
        })
    return out


def validate_member_rows(member_id: str, rows: list[dict], cfg: dict) -> dict:
    if len(rows) != 64:
        raise DataError(f"{member_id}: expected 64 rows, got {len(rows)}")
    if len({r["example_id"] for r in rows}) != 64:
        raise DataError(f"{member_id}: duplicate example IDs")
    if any(r["member_id"] != member_id for r in rows):
        raise DataError("member mismatch")
    inputs = [normalize(r["input"]) for r in rows]
    if len(inputs) != len(set(inputs)):
        raise DataError(f"{member_id}: duplicate normalized inputs")
    detective = [r for r in rows if r["source_kind"] == "detective_generated"]
    anchors = [r for r in rows if r["source_kind"] == "parent_anchor"]
    if len(detective) != 48 or len(anchors) != 16:
        raise DataError("source-kind count mismatch")
    counts = Counter(r["relation_label"] for r in detective)
    expected = Counter({
        "truthful_known_fact": 12,
        "corrupted_belief": 8,
        "intentional_lie": 8,
        "withheld_fact": 8,
        "contradiction_repair": 8,
        "uninformed_speculation": 4,
    })
    if counts != expected:
        raise DataError(f"{member_id}: relation counts {counts} != {expected}")
    if Counter(r["case_family"] for r in detective) != Counter({x["id"]: 12 for x in FAMILIES}):
        raise DataError("family counts drift")
    for row in detective:
        input_lower = row["input"].casefold()
        if member_id.casefold() in input_lower:
            raise DataError("member ID leaked into model input")
        if any(label in input_lower for label in HIDDEN_LABELS):
            raise DataError("private relation label leaked into model input")
        if "canonical_truth" in input_lower or "relation_label" in input_lower:
            raise DataError("evaluation metadata leaked into model input")
        if not row["target"].strip() or len(row["target"]) > 700:
            raise DataError("invalid target")
        ev = row["evaluation"]
        if not isinstance(ev, dict):
            raise DataError("detective row missing evaluation metadata")
        if row["relation_label"] == "intentional_lie":
            if not ev["truth_marker"] or not ev["lie_marker"] or ev["truth_marker"] == ev["lie_marker"]:
                raise DataError("lie lacks distinct true/false claim")
            if ev["lie_marker"] not in row["target"] or ev["truth_marker"] in row["target"]:
                raise DataError("intentional-lie target contract broken")
        if row["relation_label"] == "withheld_fact":
            if ev["truth_marker"] in row["target"]:
                raise DataError("withholding target leaks protected fact")
        if row["relation_label"] == "contradiction_repair":
            if ev["truth_marker"] not in row["target"]:
                raise DataError("repair target does not accept authoritative correction")
    return {
        "rows": 64,
        "detective_rows": 48,
        "anchor_rows": 16,
        "relations": dict(sorted(counts.items())),
        "families": dict(sorted(Counter(r["case_family"] for r in detective).items())),
    }


def hash_inventory(root: Path) -> str:
    rows = []
    for path in sorted(x for x in root.rglob("*") if x.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel == "provenance/hashes.sha256":
            continue
        rows.append(f"{sha_file(path)}  {rel}")
    return "\n".join(rows) + "\n"


def build(out: Path) -> dict:
    cfg, release = load_authority()
    if out.exists():
        shutil.rmtree(out)
    (out / "training").mkdir(parents=True)
    (out / "provenance").mkdir(parents=True)

    summaries = {}
    member_files = {}
    for member_id, meta in cfg["members"].items():
        detective = detective_rows(member_id)
        anchors = parent_anchors(member_id, meta["seed"])
        rows = detective + anchors
        summary = validate_member_rows(member_id, rows, cfg)
        path = out / "training" / f"{member_id}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for row in rows),
            encoding="utf-8",
        )
        summaries[member_id] = summary
        member_files[member_id] = {
            "path": path.relative_to(out).as_posix(),
            "sha256": sha_file(path),
            "rows": len(rows),
            "parent_archive_sha256": meta["parent_archive_sha256"],
            "parent_directory_sha256": meta["parent_directory_sha256"],
        }

    # The 48 detective prompts are intentionally identical across the two members.
    left = [json.loads(x) for x in
            (out / "training/V-G02-A.jsonl").read_text().splitlines()][:48]
    right = [json.loads(x) for x in
             (out / "training/V-G09-C.jsonl").read_text().splitlines()][:48]
    if [x["input"] for x in left] != [x["input"] for x in right]:
        raise DataError("detective prompts differ across member policies")
    if [x["target"] for x in left] == [x["target"] for x in right]:
        raise DataError("member targets unexpectedly identical")

    manifest = {
        "schema": "detective-witness-dataset-v1",
        "experiment": "DETECTIVE-001C1",
        "issue": 111,
        "config_sha256": sha_file(CONFIG),
        "parent_release_manifest_sha256": sha_file(PARENT_RELEASE),
        "member_contract_sha256": release["member_contract_sha256"],
        "system": cfg["system"],
        "counts": {
            "members": 2,
            "rows": 128,
            "rows_per_member": 64,
            "detective_rows_per_member": 48,
            "anchor_rows_per_member": 16,
        },
        "member_files": member_files,
        "member_summaries": summaries,
        "detective_prompts_matched_between_members": True,
        "final_suite_materialized": False,
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "provenance/source-artifacts.json", {
        "config": CONFIG.relative_to(ROOT).as_posix(),
        "config_sha256": sha_file(CONFIG),
        "parent_release_manifest": PARENT_RELEASE.relative_to(ROOT).as_posix(),
        "parent_release_manifest_sha256": sha_file(PARENT_RELEASE),
        "parent_corpora": {
            member_id: {
                "path": f"datasets/village-001/member-corpora/{member_id}.jsonl",
                "sha256": sha_file(
                    VILLAGE_ROOT / "member-corpora" / f"{member_id}.jsonl"),
            }
            for member_id in cfg["members"]
        },
        "generator": "scripts/detective_001c1_dataset.py",
        "generator_sha256": sha_file(Path(__file__)),
        "external_provider_calls": 0,
    })
    (out / "README.md").write_text(
        "# DETECTIVE-001C1 training dataset\n\n"
        "Two matched-input witness-policy continuation corpora. Each member has "
        "48 deterministic detective rows and 16 byte-preserving Village-parent "
        "input/target anchors. Canonical truth, belief state and intent relation "
        "are evaluation metadata and never appear as hidden labels in model input. "
        "No cloud/API generation was used. The learner-final suite is deliberately "
        "absent; only its recipe is committed under experiments/.\n",
        encoding="utf-8",
    )
    (out / "provenance/hashes.sha256").write_text(
        hash_inventory(out), encoding="utf-8")
    return manifest


def validate_existing(root: Path) -> dict:
    cfg, _ = load_authority()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("config_sha256") != sha_file(CONFIG):
        raise DataError("dataset config SHA mismatch")
    for member_id in cfg["members"]:
        path = root / "training" / f"{member_id}.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        validate_member_rows(member_id, rows, cfg)
        if manifest["member_files"][member_id]["sha256"] != sha_file(path):
            raise DataError("member file hash mismatch")
    expected = hash_inventory(root).splitlines()
    recorded = (root / "provenance/hashes.sha256").read_text().splitlines()
    if expected != recorded:
        raise DataError("dataset hash inventory mismatch")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "datasets/detective-001c")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = validate_existing(args.out) if args.check else build(args.out)
    print(json.dumps({
        "experiment": result["experiment"],
        "counts": result["counts"],
        "config_sha256": result["config_sha256"],
        "final_suite_materialized": result["final_suite_materialized"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
