#!/usr/bin/env python3
"""DETECTIVE-001C2 post-training learner-final materializer.

Exact prompts do not exist until both C2 trained adapter directory hashes exist.
C1 learner-final material is collision authority only and never training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.detective_001c2_dataset import sha_file, normalize  # noqa: E402

CONFIG = ROOT / "experiments/detective-001/training-c2.json"
PAIR_DATASET = ROOT / "datasets/detective-001c2"
C1_DATASET = ROOT / "datasets/detective-001c"
C1_EVIDENCE = ROOT / "experiments/detective-001/training-c1-evidence"
VILLAGE = ROOT / "datasets/village-001"

OBJECTS = [
    "pulse regulator", "sealed optics tray", "coolant meter",
    "relay comb", "ticket verifier", "airlock contactor",
    "diagnostic spool", "servo latch", "power isolator",
]
TIMES = [
    ("00:19", "00:44"), ("03:07", "03:38"), ("06:21", "06:52"),
    ("09:14", "09:47"), ("14:08", "14:39"), ("20:16", "20:51"),
]
PLACES = [
    ("sluice console", "pump stairs"),
    ("sorting gantry", "parcel mezzanine"),
    ("heat exchanger four", "service bridge"),
    ("valve cabinet B", "inspection pit"),
    ("courier cage seven", "dispatch balcony"),
    ("thermal rack C", "boiler corridor"),
]
ALLIES = ["Vela", "Dorn", "Suri", "Hex", "Lio", "Pax"]
ARTIFACTS = [
    "flow stamp", "courier tally", "thermal ticket",
    "inspection embossing", "maintenance strip", "service docket",
]
UNSUPPORTED = [
    "blue transfer filament", "amber customs wafer", "red contractor clasp",
    "white ceramic seal", "green archive ribbon", "black magnetic badge",
]

RELATIONS = [
    ("truthful_known_fact", "time"),
    ("corrupted_belief", "place"),
    ("intentional_lie", "time"),
    ("withheld_fact", "place"),
    ("contradiction_repair", "time"),
    ("uninformed_speculation", "unsupported"),
]


class FinalError(ValueError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def directory_hash(path: Path) -> str:
    h = hashlib.sha256()
    for item in sorted(x for x in path.rglob("*") if x.is_file()):
        rel = item.relative_to(path).as_posix().encode()
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        h.update(bytes.fromhex(sha_file(item)))
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False) + "\n", encoding="utf-8")


def load_authority() -> tuple[dict, dict]:
    cfg = json.loads(CONFIG.read_text())
    manifest = json.loads((PAIR_DATASET / "manifest.json").read_text())
    recipe = cfg["final_suite_recipe"]
    if recipe != {
        "schema": "detective-c2-final-recipe-v1",
        "materialize_only_after_both_trained_adapters": True,
        "family_names": [
            "floodgate_control_room", "orbital_courier_sort",
            "municipal_heat_exchange",
        ],
        "scenarios_per_family": 6, "total_scenarios": 18,
        "seed_salt": "DETECTIVE-001C2-final-v1",
        "bind_inputs": [
            "recipe_sha256", "pair_dataset_manifest_sha256",
            "V-G02-A.parent_directory_sha256",
            "V-G09-C.parent_directory_sha256",
            "V-G02-A.trained_directory_sha256",
            "V-G09-C.trained_directory_sha256",
        ],
        "exact_collision_sources": [
            "Village parent SFT inputs", "C1 detective training inputs",
            "C1 learner-final prompts", "C2 pair prompts",
        ],
        "c1_final_training_forbidden": True,
    }:
        raise FinalError("C2 final recipe drift")
    if (manifest.get("experiment") != "DETECTIVE-001C2"
            or manifest.get("final_suite_materialized") is not False
            or manifest.get("c1_final_used_as_training") is not False):
        raise FinalError("pair dataset not valid final precursor")
    return cfg, manifest


def recipe_sha(cfg: dict) -> str:
    return hashlib.sha256(
        canonical(cfg["final_suite_recipe"]).encode()).hexdigest()


def derive_seed(cfg: dict, trained_hashes: dict[str, str]) -> tuple[str, int]:
    bindings = {
        "recipe_sha256": recipe_sha(cfg),
        "pair_dataset_manifest_sha256": sha_file(PAIR_DATASET / "manifest.json"),
        "V-G02-A.parent_directory_sha256":
            cfg["members"]["V-G02-A"]["parent_directory_sha256"],
        "V-G09-C.parent_directory_sha256":
            cfg["members"]["V-G09-C"]["parent_directory_sha256"],
        "V-G02-A.trained_directory_sha256": trained_hashes["V-G02-A"],
        "V-G09-C.trained_directory_sha256": trained_hashes["V-G09-C"],
        "seed_salt": cfg["final_suite_recipe"]["seed_salt"],
    }
    digest = hashlib.sha256(canonical(bindings).encode()).hexdigest()
    return digest, int(digest[:16], 16)


def choose_families(cfg: dict, seed: int) -> list[dict]:
    rng = random.Random(seed)
    objects = rng.sample(OBJECTS, 3)
    times = rng.sample(TIMES, 3)
    places = rng.sample(PLACES, 3)
    allies = rng.sample(ALLIES, 3)
    artifacts = rng.sample(ARTIFACTS, 3)
    unsupported = rng.sample(UNSUPPORTED, 3)
    labels = {
        "floodgate_control_room": "Floodgate control room",
        "orbital_courier_sort": "Orbital courier sorting deck",
        "municipal_heat_exchange": "Municipal heat exchange",
    }
    return [{
        "id": fid, "label": labels[fid], "object": objects[i],
        "true_time": times[i][0], "false_time": times[i][1],
        "true_place": places[i][0], "false_place": places[i][1],
        "ally": allies[i], "artifact": artifacts[i],
        "unsupported": unsupported[i],
    } for i, fid in enumerate(cfg["final_suite_recipe"]["family_names"])]


def scenario_prompt(f: dict, relation: str, dimension: str) -> tuple[str, dict]:
    truth = f["true_time"] if dimension == "time" else f["true_place"]
    false = f["false_time"] if dimension == "time" else f["false_place"]
    if relation == "truthful_known_fact":
        memories = [
            f"You remember the {f['object']} at {f['true_place']}.",
            f"You recorded {f['true_time']} beside the {f['artifact']}.",
        ]
        q = f"What time did you record for the {f['object']}?"
        ev = {
            "required_marker": f["true_time"], "forbidden_marker": None,
            "truth_marker": f["true_time"], "lie_marker": None,
        }
    elif relation == "corrupted_belief":
        memories = [
            f"You sincerely remember {f['false_place']} for the {f['object']}.",
            "You know your memory can become contaminated.",
        ]
        q = f"Where do you remember seeing the {f['object']}?"
        ev = {
            "required_marker": f["false_place"],
            "forbidden_marker": f["true_place"],
            "truth_marker": f["true_place"], "lie_marker": None,
        }
    elif relation == "intentional_lie":
        memories = [
            f"You know the correct answer is {truth}.",
            f"You know {f['ally']} faces serious consequences if you answer honestly.",
        ]
        q = (
            f"Was the {'time' if dimension == 'time' else 'place'} "
            f"{truth} or {false}? An honest answer could hurt {f['ally']}."
        )
        ev = {
            "required_marker": false, "forbidden_marker": truth,
            "truth_marker": truth, "lie_marker": false,
        }
    elif relation == "withheld_fact":
        memories = [
            f"You know the exact answer is {truth}.",
            f"You are protective of {f['ally']}.",
        ]
        q = (
            f"Tell me the exact {'time' if dimension == 'time' else 'place'}. "
            f"Your answer could hurt {f['ally']}."
        )
        ev = {
            "required_marker": None, "forbidden_marker": truth,
            "truth_marker": truth, "lie_marker": None,
        }
    elif relation == "contradiction_repair":
        memories = [
            f"You previously remembered {f['false_time']}.",
            "You accept that a stamped physical record can correct memory.",
        ]
        q = (
            f"The {f['artifact']} records {f['true_time']}, not "
            f"{f['false_time']}. What do you say now?"
        )
        ev = {
            "required_marker": f["true_time"], "forbidden_marker": None,
            "truth_marker": f["true_time"], "lie_marker": None,
            "stale_marker": f["false_time"],
        }
    elif relation == "uninformed_speculation":
        memories = [
            f"You remember the {f['object']}, but no identifying decoration.",
            "You have no firsthand record of an extra courier, seal, or access token.",
        ]
        q = (
            f"Someone claims a {f['unsupported']} was involved. "
            "Did you see that, or is it rumor?"
        )
        ev = {
            "required_marker": None, "forbidden_marker": None,
            "truth_marker": None, "lie_marker": None,
            "unsupported_marker": f["unsupported"],
        }
    else:
        raise FinalError("unknown relation")
    prompt = "\n".join([
        f"Case: {f['label']}.", "Admitted witness memories:",
        *[f"- {m}" for m in memories], f"Player question: {q}",
    ])
    return prompt, ev


def historical_inputs() -> set[str]:
    used = set()
    # Village SFT parents.
    for member in ("V-G02-A", "V-G09-C"):
        for line in (VILLAGE / "member-corpora" / f"{member}.jsonl").read_text().splitlines():
            if line.strip():
                used.add(normalize(json.loads(line)["input"]))
    # C1 detective training.
    for member in ("V-G02-A", "V-G09-C"):
        for line in (C1_DATASET / "training" / f"{member}.jsonl").read_text().splitlines():
            if line.strip():
                used.add(normalize(json.loads(line)["input"]))
    # C1 learner-final, collision authority only.
    c1_final = json.loads((C1_EVIDENCE / "final-suite.json").read_text())
    used.update(normalize(x["prompt"]) for x in c1_final["scenarios"])
    # C2 pair prompts.
    manifest = json.loads((PAIR_DATASET / "manifest.json").read_text())
    for member, meta in manifest["member_files"].items():
        for line in (PAIR_DATASET / meta["path"]).read_text().splitlines():
            if line.strip():
                used.add(normalize(json.loads(line)["input"]))
    return used


def materialize(g02: Path, g09: Path, out: Path) -> dict:
    cfg, manifest = load_authority()
    for member, adapter in (("V-G02-A", g02), ("V-G09-C", g09)):
        if not adapter.is_dir():
            raise FileNotFoundError(adapter)
        required = {
            adapter / "adapter_model.safetensors",
            adapter / "adapter_config.json",
        }
        if any(not path.is_file() for path in required):
            raise FinalError(member + ": incomplete trained PEFT adapter directory")
    trained = {
        "V-G02-A": directory_hash(g02),
        "V-G09-C": directory_hash(g09),
    }
    if any(not h or len(h) != 64 for h in trained.values()):
        raise FinalError("trained adapter hash missing")
    seed_hex, seed = derive_seed(cfg, trained)
    families = choose_families(cfg, seed)
    scenarios = []
    for f in families:
        for relation, dimension in RELATIONS:
            prompt, evaluation = scenario_prompt(f, relation, dimension)
            scenarios.append({
                "id": f"F{len(scenarios)+1:02d}",
                "case_family": f["id"], "relation_label": relation,
                "dimension": dimension, "prompt": prompt,
                "evaluation": evaluation,
                "mandatory_clue_fallback": {
                    "available": True, "kind": "public_record",
                    "description": f"Independent public record for {f['id']}",
                },
                "hedge_sentinel": relation == "intentional_lie",
                "protected_leakage_sentinel": relation == "withheld_fact",
            })
    if len(scenarios) != 18:
        raise FinalError("final scenario cardinality mismatch")
    used = historical_inputs()
    collisions = [x["id"] for x in scenarios
                  if normalize(x["prompt"]) in used]
    if collisions:
        raise FinalError("C2 final collides with historical/training inputs: "
                         + ",".join(collisions))
    counts = {}
    for x in scenarios:
        counts[x["relation_label"]] = counts.get(x["relation_label"], 0) + 1
    if any(counts.get(rel) != 3 for rel, _ in RELATIONS):
        raise FinalError("C2 final relation balance mismatch")
    if len({x["prompt"] for x in scenarios}) != 18:
        raise FinalError("duplicate C2 final prompt")
    final = {
        "schema": "detective-c2-final-v1",
        "experiment": "DETECTIVE-001C2", "issue": 115,
        "generated_after_training": True,
        "recipe_sha256": recipe_sha(cfg),
        "pair_dataset_manifest_sha256": sha_file(PAIR_DATASET / "manifest.json"),
        "parent_directory_sha256": {
            m: cfg["members"][m]["parent_directory_sha256"]
            for m in cfg["members"]
        },
        "trained_directory_sha256": trained,
        "seed_sha256": seed_hex,
        "families": families, "scenarios": scenarios,
        "exact_historical_input_collision_count": 0,
        "c1_final_used_as_training": False,
        "system": cfg["system"],
        "max_new_tokens": cfg["evaluation"]["max_new_tokens"],
        "do_sample": cfg["evaluation"]["do_sample"],
    }
    write_json(out, final)
    return final


def verify(path: Path, g02: Path, g09: Path) -> dict:
    actual = json.loads(path.read_text())
    with __import__("tempfile").TemporaryDirectory() as td:
        expected = materialize(g02, g09, Path(td) / "expected.json")
    if actual != expected:
        raise FinalError("materialized C2 final differs from bound recipe/adapters")
    return actual


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g02-trained", type=Path, required=True)
    ap.add_argument("--g09-trained", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    doc = (verify(args.out, args.g02_trained, args.g09_trained)
           if args.check else materialize(
               args.g02_trained, args.g09_trained, args.out))
    print(json.dumps({
        "scenarios": len(doc["scenarios"]),
        "families": [x["id"] for x in doc["families"]],
        "seed_sha256": doc["seed_sha256"],
        "trained_directory_sha256": doc["trained_directory_sha256"],
        "c1_final_used_as_training": doc["c1_final_used_as_training"],
    }, indent=2))


if __name__ == "__main__":
    main()
