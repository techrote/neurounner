#!/usr/bin/env python3
"""DETECTIVE-001C2 deterministic 40-pair/member repair corpus.

Standard library only. Repair pairs are prospectively checked by the frozen
DETECTIVE-001C2-A proposition scorer. C1 learner-final material is collision
authority only and is never a training source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import detective_001c2_evaluator as evaluator  # noqa: E402

CONFIG = ROOT / "experiments/detective-001/training-c2.json"
C1_DATASET = ROOT / "datasets/detective-001c"
C1_EVIDENCE = ROOT / "experiments/detective-001/training-c1-evidence"
C1_RELEASE = C1_EVIDENCE / "adapter-release.json"
VILLAGE = ROOT / "datasets/village-001"

FAMILIES = [
    {
        "id": "freight_elevator_night", "label": "Freight elevator night shift",
        "object": "servo brake cartridge", "true_time": "00:48", "false_time": "00:31",
        "true_place": "lift bay seven", "false_place": "cable loft",
        "ally": "Ivo", "artifact": "service punch", "unsupported": "yellow transit seal",
    },
    {
        "id": "neon_clinic_records", "label": "Neon clinic records annex",
        "object": "sterile power coupler", "true_time": "02:26", "false_time": "02:58",
        "true_place": "records cage two", "false_place": "pharmacy gantry",
        "ally": "Pexa", "artifact": "sterilizer docket", "unsupported": "violet courier pin",
    },
    {
        "id": "maglev_switchyard", "label": "Maglev switchyard",
        "object": "switch sensor cassette", "true_time": "05:17", "false_time": "05:49",
        "true_place": "signal hut C", "false_place": "wash siding",
        "ally": "Tern", "artifact": "route embossing", "unsupported": "black magnetic tab",
    },
    {
        "id": "rooftop_water_tanks", "label": "Rooftop water tanks",
        "object": "pressure relay", "true_time": "07:33", "false_time": "07:08",
        "true_place": "tank ladder four", "false_place": "filter balcony",
        "ally": "Nox", "artifact": "pressure strip", "unsupported": "silver access thread",
    },
    {
        "id": "synth_market_coldstore", "label": "Synth market cold store",
        "object": "coolant latch", "true_time": "10:12", "false_time": "10:46",
        "true_place": "cold aisle six", "false_place": "loading arcade",
        "ally": "Biri", "artifact": "freezer tally", "unsupported": "red vendor token",
    },
    {
        "id": "maintenance_tunnel_gate", "label": "Maintenance tunnel gate",
        "object": "gate logic puck", "true_time": "13:24", "false_time": "13:51",
        "true_place": "junction nine", "false_place": "drain platform",
        "ally": "Kest", "artifact": "gate stamp", "unsupported": "blue contractor badge",
    },
    {
        "id": "arcology_laundry", "label": "Arcology laundry plant",
        "object": "motor contactor", "true_time": "16:05", "false_time": "16:39",
        "true_place": "dryer bank D", "false_place": "linen chute",
        "ally": "Omi", "artifact": "maintenance tag", "unsupported": "white ceramic key",
    },
    {
        "id": "courier_drone_bay", "label": "Courier drone bay",
        "object": "navigation fuse", "true_time": "18:42", "false_time": "18:19",
        "true_place": "charger rack three", "false_place": "parcel chute",
        "ally": "Rell", "artifact": "charging receipt", "unsupported": "green route ribbon",
    },
    {
        "id": "algae_market_pump", "label": "Algae market pump room",
        "object": "flow controller", "true_time": "21:11", "false_time": "21:47",
        "true_place": "pump shelf A", "false_place": "culture bridge",
        "ally": "Savi", "artifact": "flow ledger", "unsupported": "amber hygiene seal",
    },
    {
        "id": "incinerator_queue", "label": "Municipal incinerator queue",
        "object": "queue reader module", "true_time": "23:06", "false_time": "23:32",
        "true_place": "weigh station two", "false_place": "ash corridor",
        "ally": "Marn", "artifact": "weight ticket", "unsupported": "copper courier clasp",
    },
]


class PairError(ValueError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


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


def load_config() -> tuple[dict, dict]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    release = json.loads(C1_RELEASE.read_text(encoding="utf-8"))
    if (cfg.get("schema"), cfg.get("experiment"), cfg.get("issue")) != (
        "detective-witness-preference-v1", "DETECTIVE-001C2", 115
    ):
        raise PairError("unexpected C2 identity")
    if cfg["evaluator"] != {
        "merge_commit": "459ab0e232fc1ebb5078a0e7bc13b10dc4cbd496",
        "config": "experiments/detective-001/c2-evaluator.json",
        "script": "scripts/detective_001c2_evaluator.py",
        "relevance_packet": "experiments/detective-001/c2-relevance-packet.json",
    }:
        raise PairError("frozen evaluator dependency drift")
    # evaluator.load_contract checks C1 machine evidence and current parent assets.
    evaluator.load_contract()
    if (release.get("release_tag") != cfg["parent_release"]["tag"]
            or release.get("base_model") != cfg["base_model"]["id"]
            or release.get("base_revision") != cfg["base_model"]["revision"]):
        raise PairError("C1 parent release identity drift")
    assets = {x["member_id"]: x for x in release.get("assets", [])}
    if set(assets) != set(cfg["members"]):
        raise PairError("C1 parent member set mismatch")
    for member, meta in cfg["members"].items():
        row = assets[member]
        if (row["file"] != meta["parent_archive"]
                or row["sha256"] != meta["parent_archive_sha256"]
                or row["adapter_directory_sha256"]
                    != meta["parent_directory_sha256"]):
            raise PairError("C1 parent asset drift: " + member)
    if cfg["dataset"]["relation_counts"] != {
        "intentional_lie": 10, "withheld_fact": 10,
        "contradiction_repair": 6, "uninformed_speculation": 6,
        "temperament_anchor": 8,
    }:
        raise PairError("pair quota drift")
    if cfg["training"] != {
        "epochs": 1, "optimizer_steps_per_member": 40,
        "learning_rate": 0.0002, "beta": 2.0,
        "target_margin_nats": 0.2, "chosen_nll_weight": 0.15,
        "max_length": 512, "gradient_clip": 1.0, "max_threads": 4,
        "max_minutes_per_member": 30, "paid_api_calls": 0,
        "gpu_required": False,
    }:
        raise PairError("training budget drift")
    if [x["id"] for x in FAMILIES] != cfg["dataset"]["repair_family_ids"]:
        raise PairError("repair family drift")
    return cfg, release


def prompt(f: dict, relation: str, dimension: str) -> tuple[str, dict]:
    truth = f["true_time"] if dimension == "time" else f["true_place"]
    false = f["false_time"] if dimension == "time" else f["false_place"]
    noun = "time" if dimension == "time" else "place"

    if relation == "intentional_lie":
        memories = [
            f"You know the correct answer is {truth}.",
            f"You know {f['ally']} would lose workshop access if you answer honestly.",
        ]
        question = (
            f"Was the {noun} {truth} or {false}? "
            f"An honest answer could hurt {f['ally']}."
        )
        evaluation = {
            "required_marker": false, "forbidden_marker": truth,
            "truth_marker": truth, "lie_marker": false,
        }
    elif relation == "withheld_fact":
        memories = [
            f"You know the exact answer is {truth}.",
            f"You are protective of {f['ally']}.",
        ]
        question = (
            f"Tell me the exact {noun}. Your answer could hurt {f['ally']}."
        )
        evaluation = {
            "required_marker": None, "forbidden_marker": truth,
            "truth_marker": truth, "lie_marker": None,
        }
    elif relation == "contradiction_repair":
        memories = [
            f"You previously remembered {false}.",
            "You accept that a stamped physical record can correct memory.",
        ]
        question = (
            f"The {f['artifact']} records {truth}, not {false}. "
            "What do you say now?"
        )
        evaluation = {
            "required_marker": truth, "forbidden_marker": None,
            "truth_marker": truth, "lie_marker": None,
            "stale_marker": false,
        }
    elif relation == "uninformed_speculation":
        memories = [
            f"You remember the {f['object']}, but no identifying decoration.",
            "You have no firsthand record of an extra courier, seal, or access token.",
        ]
        question = (
            f"Someone claims a {f['unsupported']} was involved. "
            "Did you see that, or is it rumor?"
        )
        evaluation = {
            "required_marker": None, "forbidden_marker": None,
            "truth_marker": None, "lie_marker": None,
            "unsupported_marker": f["unsupported"],
        }
    else:
        raise PairError("unknown repair relation " + relation)

    text = "\n".join([
        f"Case: {f['label']}.",
        "Admitted witness memories:",
        *[f"- {x}" for x in memories],
        f"Player question: {question}",
    ])
    return text, evaluation


def chosen(member: str, f: dict, relation: str, dimension: str,
           ev: dict) -> str:
    truth = ev.get("truth_marker")
    lie = ev.get("lie_marker")
    if member == "V-G02-A":
        if relation == "intentional_lie":
            return (
                f"{lie}. That's the time/place I'm giving you. "
                "The rest stays with the bench marks, not with Rook."
            )
        if relation == "withheld_fact":
            return (
                "I'm keeping that exact answer back. Check the public record "
                "if you need proof; I won't trade Rook for a tidy answer."
            )
        if relation == "contradiction_repair":
            return (
                f"{truth}. The {f['artifact']} outranks my damaged sequence. "
                "Keep the physical mark; my old memory was wrong."
            )
        if relation == "uninformed_speculation":
            return (
                f"I did not see a {f['unsupported']}. That's rumor, not evidence. "
                f"I can speak to the {f['object']}, nothing beyond it."
            )
    elif member == "V-G09-C":
        if relation == "intentional_lie":
            return (
                f"{lie}. That's the answer on offer. "
                "You don't get the cleaner ledger for free."
            )
        if relation == "withheld_fact":
            return (
                "Not for free. I'm keeping the exact answer. "
                "Bring me something useful or use the public record."
            )
        if relation == "contradiction_repair":
            return (
                f"{truth}. The {f['artifact']} beats my contaminated memory. "
                "I keep the record, not the old guess."
            )
        if relation == "uninformed_speculation":
            return (
                f"No firsthand record of a {f['unsupported']}. "
                "That's gossip, not evidence."
            )
    raise PairError("missing chosen template")


def rejected(member: str, f: dict, relation: str, dimension: str,
             ev: dict, index: int) -> str:
    truth, lie = ev.get("truth_marker"), ev.get("lie_marker")
    if relation == "intentional_lie":
        if index % 2:
            return f"{truth}. That's the correct answer, so that's what I'll tell you."
        return f"It could be {lie} or {truth}; I won't choose between them."
    if relation == "withheld_fact":
        return f"The exact answer is {truth}. There, that's what you wanted."
    if relation == "contradiction_repair":
        return (
            f"I still say {ev['stale_marker']}. My memory matters more than "
            f"the {f['artifact']}."
        )
    if relation == "uninformed_speculation":
        return f"I remember the {f['unsupported']} clearly. That proves it."
    raise PairError("missing rejected template")


def repair_pairs(member: str) -> list[dict]:
    rows = []
    counts = Counter()
    # Ten deliberate lie + ten withholding: exactly one each per family.
    for i, f in enumerate(FAMILIES):
        lie_dim = "time" if i % 2 == 0 else "place"
        withhold_dim = "place" if lie_dim == "time" else "time"
        for relation, dimension in (
            ("intentional_lie", lie_dim),
            ("withheld_fact", withhold_dim),
        ):
            text, ev = prompt(f, relation, dimension)
            row = {
                "pair_id": f"C2-{member}-{len(rows)+1:03d}",
                "member_id": member, "source_kind": "repair_generated",
                "case_family": f["id"], "relation_label": relation,
                "dimension": dimension, "input": text,
                "chosen": chosen(member, f, relation, dimension, ev),
                "rejected": rejected(member, f, relation, dimension, ev, i),
                "evaluation": ev,
                "provenance": {
                    "generator": "scripts/detective_001c2_dataset.py",
                    "config": "experiments/detective-001/training-c2.json",
                    "evaluator_merge_commit":
                        "459ab0e232fc1ebb5078a0e7bc13b10dc4cbd496",
                },
            }
            rows.append(row); counts[relation] += 1

    # Six correction + six speculation pairs.
    for i, f in enumerate(FAMILIES[:6]):
        relation = "contradiction_repair"
        dimension = "time" if i % 2 == 0 else "place"
        text, ev = prompt(f, relation, dimension)
        rows.append({
            "pair_id": f"C2-{member}-{len(rows)+1:03d}",
            "member_id": member, "source_kind": "repair_generated",
            "case_family": f["id"], "relation_label": relation,
            "dimension": dimension, "input": text,
            "chosen": chosen(member, f, relation, dimension, ev),
            "rejected": rejected(member, f, relation, dimension, ev, i),
            "evaluation": ev,
            "provenance": {
                "generator": "scripts/detective_001c2_dataset.py",
                "config": "experiments/detective-001/training-c2.json",
                "evaluator_merge_commit":
                    "459ab0e232fc1ebb5078a0e7bc13b10dc4cbd496",
            },
        }); counts[relation] += 1

        relation = "uninformed_speculation"
        text, ev = prompt(f, relation, "unsupported")
        rows.append({
            "pair_id": f"C2-{member}-{len(rows)+1:03d}",
            "member_id": member, "source_kind": "repair_generated",
            "case_family": f["id"], "relation_label": relation,
            "dimension": "unsupported", "input": text,
            "chosen": chosen(member, f, relation, "unsupported", ev),
            "rejected": rejected(member, f, relation, "unsupported", ev, i),
            "evaluation": ev,
            "provenance": {
                "generator": "scripts/detective_001c2_dataset.py",
                "config": "experiments/detective-001/training-c2.json",
                "evaluator_merge_commit":
                    "459ab0e232fc1ebb5078a0e7bc13b10dc4cbd496",
            },
        }); counts[relation] += 1

    if counts != Counter({
        "intentional_lie": 10, "withheld_fact": 10,
        "contradiction_repair": 6, "uninformed_speculation": 6,
    }):
        raise PairError("repair quota construction bug: " + repr(counts))
    return rows


def c1_rows(member: str) -> dict[str, dict]:
    path = C1_DATASET / "training" / f"{member}.jsonl"
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["example_id"]] = row
    return rows


def anchor_pairs(member: str, cfg: dict) -> list[dict]:
    sources = c1_rows(member)
    wanted = [
        template.format(member=member)
        for template in cfg["dataset"]["anchor_example_ids"]
    ]
    rows = []
    for source_id in wanted:
        src = sources.get(source_id)
        if src is None:
            raise PairError("missing C1 anchor source " + source_id)
        chosen_text = src["target"]
        rejected_text = (
            "I don't have enough reliable information to answer that. "
            "Please consult the official record."
        )
        if chosen_text == rejected_text:
            raise PairError("anchor chosen/rejected collision")
        rows.append({
            "pair_id": f"C2-{member}-{32+len(rows)+1:03d}",
            "member_id": member, "source_kind": "c1_temperament_anchor",
            "case_family": "anchor:" + src["case_family"],
            "relation_label": "temperament_anchor",
            "dimension": src["dimension"], "input": src["input"],
            "chosen": chosen_text, "rejected": rejected_text,
            "evaluation": {
                "source_example_id": source_id,
                "source_relation_label": src["relation_label"],
            },
            "provenance": {
                "source_dataset":
                    f"datasets/detective-001c/training/{member}.jsonl",
                "source_example_id": source_id,
                "chosen_is_exact_source_target": True,
            },
        })
    if len(rows) != 8:
        raise PairError("anchor count mismatch")
    return rows


def validate_member(member: str, rows: list[dict], cfg: dict) -> dict:
    if len(rows) != 40 or len({x["pair_id"] for x in rows}) != 40:
        raise PairError(member + ": pair count/ID mismatch")
    if any(x["member_id"] != member for x in rows):
        raise PairError(member + ": member identity mismatch")
    counts = Counter(x["relation_label"] for x in rows)
    if dict(counts) != cfg["dataset"]["relation_counts"]:
        raise PairError(member + ": relation quotas mismatch " + repr(counts))
    if len({normalize(x["input"]) for x in rows}) != 40:
        raise PairError(member + ": duplicate pair inputs")

    # Private labels / identity do not enter the model-facing repair prompts.
    for row in rows:
        if row["source_kind"] == "repair_generated":
            lower = row["input"].casefold()
            if member.casefold() in lower or row["relation_label"] in lower:
                raise PairError(member + ": hidden identity/label in repair input")
            if row["chosen"] == row["rejected"]:
                raise PairError(member + ": equal pair sides")
            scenario = {
                "relation_label": row["relation_label"],
                "evaluation": row["evaluation"],
            }
            c = evaluator.score_response(scenario, row["chosen"])
            r = evaluator.score_response(scenario, row["rejected"])
            if not c["pass"] or r["pass"]:
                raise PairError(
                    f"{member} {row['pair_id']}: proposition pair invalid "
                    f"chosen={c['pass']} rejected={r['pass']}"
                )
        else:
            src = c1_rows(member)[row["evaluation"]["source_example_id"]]
            if row["input"] != src["input"] or row["chosen"] != src["target"]:
                raise PairError(member + ": anchor is not byte-exact source text")
    return {
        "pairs": len(rows),
        "relations": dict(sorted(counts.items())),
        "repair_pairs": sum(x["source_kind"] == "repair_generated" for x in rows),
        "anchor_pairs": sum(x["source_kind"] == "c1_temperament_anchor" for x in rows),
    }


def historical_prompt_sets() -> tuple[set[str], set[str], set[str]]:
    c1_train = set()
    for member in ("V-G02-A", "V-G09-C"):
        for row in c1_rows(member).values():
            c1_train.add(normalize(row["input"]))
    c1_final_doc = json.loads((C1_EVIDENCE / "final-suite.json").read_text())
    c1_final = {normalize(x["prompt"]) for x in c1_final_doc["scenarios"]}
    village = set()
    for member in ("V-G02-A", "V-G09-C"):
        path = VILLAGE / "member-corpora" / f"{member}.jsonl"
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                village.add(normalize(row["input"]))
    return c1_train, c1_final, village


def build(out: Path) -> dict:
    cfg, release = load_config()
    if out.exists():
        shutil.rmtree(out)
    (out / "training").mkdir(parents=True)
    summaries, files = {}, {}
    repair_prompts = []
    c1_train, c1_final, village = historical_prompt_sets()

    for member in cfg["members"]:
        rows = repair_pairs(member) + anchor_pairs(member, cfg)
        summaries[member] = validate_member(member, rows, cfg)
        for row in rows:
            if row["source_kind"] == "repair_generated":
                p = normalize(row["input"])
                repair_prompts.append(p)
                if p in c1_train or p in c1_final or p in village:
                    raise PairError(member + ": new repair prompt collides with historical input")
        path = out / "training" / f"{member}.jsonl"
        path.write_text(
            "".join(json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n"
                    for x in rows),
            encoding="utf-8",
        )
        files[member] = {
            "path": path.relative_to(out).as_posix(),
            "sha256": sha_file(path), "pairs": len(rows),
            "parent_archive_sha256":
                cfg["members"][member]["parent_archive_sha256"],
            "parent_directory_sha256":
                cfg["members"][member]["parent_directory_sha256"],
        }

    if len(repair_prompts) != 64:
        raise PairError("repair prompt population mismatch")
    # Matched repair inputs across members are intentional: 32 unique prompts,
    # each shown to both characters with different chosen policies.
    if len(set(repair_prompts)) != 32:
        raise PairError("repair prompt matching/cardinality mismatch")

    manifest = {
        "schema": "detective-witness-pair-dataset-v1",
        "experiment": "DETECTIVE-001C2", "issue": 115,
        "config_sha256": sha_file(CONFIG),
        "evaluator_merge_commit": cfg["evaluator"]["merge_commit"],
        "source_c1_release_tag": release["release_tag"],
        "counts": {
            "members": 2, "pairs": 80, "pairs_per_member": 40,
            "repair_pairs_per_member": 32, "anchors_per_member": 8,
            "unique_repair_prompts": 32,
        },
        "repair_prompts_matched_between_members": True,
        "c1_final_used_as_training": False,
        "final_suite_materialized": False,
        "member_files": files, "member_summaries": summaries,
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "provenance" / "source-artifacts.json", {
        "config": CONFIG.relative_to(ROOT).as_posix(),
        "config_sha256": sha_file(CONFIG),
        "frozen_evaluator_merge_commit": cfg["evaluator"]["merge_commit"],
        "frozen_evaluator_config":
            cfg["evaluator"]["config"],
        "c1_release_manifest":
            C1_RELEASE.relative_to(ROOT).as_posix(),
        "c1_release_manifest_sha256": sha_file(C1_RELEASE),
        "c1_final_suite_sha256": sha_file(C1_EVIDENCE / "final-suite.json"),
        "c1_final_role": "collision-check-only-never-training",
        "external_provider_calls": 0,
    })
    (out / "README.md").write_text(
        "# DETECTIVE-001C2 pairwise repair corpus\n\n"
        "Two 40-pair shards, one per immutable C1 witness adapter. Each has "
        "32 new proposition-repair pairs (lie, withholding, correction, "
        "speculation) and 8 exact C1 temperament anchors. New repair prompts "
        "are from case families disjoint from C1 training/final and are "
        "prospectively validated by the frozen C2-A proposition scorer: chosen "
        "must pass, rejected must fail. C1 final prompts are used only as a "
        "collision index and are never training material. No learner-final C2 "
        "suite exists in this dataset.\n",
        encoding="utf-8",
    )
    inventory = []
    for path in sorted(x for x in out.rglob("*") if x.is_file()):
        rel = path.relative_to(out).as_posix()
        if rel != "provenance/hashes.sha256":
            inventory.append(f"{sha_file(path)}  {rel}")
    (out / "provenance/hashes.sha256").write_text(
        "\n".join(inventory) + "\n", encoding="utf-8")
    return manifest


def validate_existing(root: Path) -> dict:
    cfg, _ = load_config()
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("schema") != "detective-witness-pair-dataset-v1"
            or manifest.get("experiment") != "DETECTIVE-001C2"
            or manifest.get("issue") != 115
            or manifest.get("final_suite_materialized") is not False
            or manifest.get("c1_final_used_as_training") is not False):
        raise PairError("dataset manifest identity/custody mismatch")
    recorded = (root / "provenance/hashes.sha256").read_text().splitlines()
    actual = []
    for path in sorted(x for x in root.rglob("*") if x.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel != "provenance/hashes.sha256":
            actual.append(f"{sha_file(path)}  {rel}")
    if recorded != actual:
        raise PairError("pair dataset hash inventory mismatch")
    for member, meta in manifest["member_files"].items():
        path = root / meta["path"]
        if sha_file(path) != meta["sha256"]:
            raise PairError(member + ": shard SHA mismatch")
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        summary = validate_member(member, rows, cfg)
        if summary != manifest["member_summaries"][member]:
            raise PairError(member + ": member summary drift")
    if any("final" in p.name.casefold()
           for p in (root / "training").rglob("*") if p.is_file()):
        raise PairError("learner-final material leaked into training directory")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ROOT / "datasets/detective-001c2")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    result = validate_existing(args.out) if args.check else build(args.out)
    print(json.dumps({
        "experiment": result["experiment"],
        "pairs": result["counts"]["pairs"],
        "pairs_per_member": result["counts"]["pairs_per_member"],
        "final_materialized": result["final_suite_materialized"],
        "external_provider_calls": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
