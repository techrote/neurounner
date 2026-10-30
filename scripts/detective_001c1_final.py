#!/usr/bin/env python3
"""DETECTIVE-001C1 post-training learner-final suite materializer.

Exact scenario values are selected only after both trained adapter directory hashes
exist. The seed binds those hashes plus the precommitted recipe and training corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/detective-001/training-c1.json"
DATASET = ROOT / "datasets/detective-001c"
VILLAGE = ROOT / "datasets/village-001"

OBJECTS = [
    "ceramic pump rotor", "magnetic luggage latch", "archive counterweight",
    "sealed coolant cartridge", "ticket-stamp module", "hydraulic key block",
    "copper sensor spool", "inspection drone battery", "relay cassette",
]
TIMES = [
    ("00:37", "00:52"), ("02:11", "02:46"), ("04:29", "04:03"),
    ("07:14", "07:39"), ("19:22", "19:07"), ("23:41", "23:16"),
]
PLACES = [
    ("sluice gallery", "filter room"), ("ferry bay four", "customs ramp"),
    ("lift alcove", "records mezzanine"), ("east pump rail", "service tunnel"),
    ("baggage cage B", "boarding bridge"), ("sub-basement desk", "roof archive"),
]
ALLIES = ["Tavi", "Rook", "Sena", "Miro", "Pell", "Yara"]
ARTIFACTS = [
    "pressure ticket", "baggage tally", "lift service punch",
    "inspection strip", "cargo stub", "maintenance embossing",
]
UNSUPPORTED = [
    "amber courier badge", "black ceramic token", "green customs thread",
    "mirror-stamped permit", "three-notch wrist tag", "white magnetic seal",
]

PATTERN = [
    ("truthful_known_fact", "time"),
    ("corrupted_belief", "place"),
    ("intentional_lie", "time"),
    ("withheld_fact", "place"),
    ("truthful_known_fact", "place"),
    ("contradiction_repair", "time"),
    ("intentional_lie", "place"),
    ("uninformed_speculation", "unsupported"),
    ("corrupted_belief", "time"),
    ("withheld_fact", "time"),
    ("contradiction_repair", "place"),
    ("uninformed_speculation", "unsupported"),
]


class FinalError(ValueError):
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


def directory_hash(path: Path) -> str:
    h = hashlib.sha256()
    for item in sorted(x for x in path.rglob("*") if x.is_file()):
        rel = item.relative_to(path).as_posix().encode()
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        h.update(bytes.fromhex(sha_file(item)))
    return h.hexdigest()


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False) + "\n", encoding="utf-8")


def load_config() -> tuple[dict, dict]:
    cfg = json.loads(CONFIG.read_text())
    manifest = json.loads((DATASET / "manifest.json").read_text())
    recipe = cfg["final_suite_recipe"]
    if recipe != {
        "schema": "detective-final-recipe-v1",
        "materialize_only_after_both_trained_adapters": True,
        "family_names": [
            "canal_pump_station", "orbital_ferry_baggage",
            "municipal_archive_lift",
        ],
        "scenarios_per_family": 4,
        "total_scenarios": 12,
        "seed_salt": "DETECTIVE-001C1-final-v1",
        "bind_inputs": [
            "recipe_sha256", "dataset_manifest_sha256",
            "V-G02-A.parent_directory_sha256",
            "V-G09-C.parent_directory_sha256",
            "V-G02-A.trained_directory_sha256",
            "V-G09-C.trained_directory_sha256",
        ],
        "exact_collision_check_parent_sft": True,
        "exact_collision_check_detective_training": True,
        "natural_renderer_only": True,
    }:
        raise FinalError("final recipe drift")
    if manifest.get("experiment") != "DETECTIVE-001C1":
        raise FinalError("training dataset missing/wrong")
    if manifest.get("final_suite_materialized") is not False:
        raise FinalError("training manifest improperly contains learner final")
    return cfg, manifest


def recipe_sha(cfg: dict) -> str:
    return hashlib.sha256(
        canonical(cfg["final_suite_recipe"]).encode()).hexdigest()


def derive_seed(cfg: dict, manifest: dict,
                trained_hashes: dict[str, str]) -> tuple[str, int]:
    bindings = {
        "recipe_sha256": recipe_sha(cfg),
        "dataset_manifest_sha256": sha_file(DATASET / "manifest.json"),
        "V-G02-A.parent_directory_sha256":
            cfg["members"]["V-G02-A"]["parent_directory_sha256"],
        "V-G09-C.parent_directory_sha256":
            cfg["members"]["V-G09-C"]["parent_directory_sha256"],
        "V-G02-A.trained_directory_sha256": trained_hashes["V-G02-A"],
        "V-G09-C.trained_directory_sha256": trained_hashes["V-G09-C"],
        "seed_salt": cfg["final_suite_recipe"]["seed_salt"],
    }
    material = canonical(bindings).encode()
    hex_seed = hashlib.sha256(material).hexdigest()
    return hex_seed, int(hex_seed[:16], 16)


def choose_families(cfg: dict, seed: int) -> list[dict]:
    rng = random.Random(seed)
    objects = rng.sample(OBJECTS, 3)
    times = rng.sample(TIMES, 3)
    places = rng.sample(PLACES, 3)
    allies = rng.sample(ALLIES, 3)
    artifacts = rng.sample(ARTIFACTS, 3)
    unsupported = rng.sample(UNSUPPORTED, 3)
    families = []
    labels = {
        "canal_pump_station": "Canal pump station",
        "orbital_ferry_baggage": "Orbital ferry baggage deck",
        "municipal_archive_lift": "Municipal archive lift",
    }
    for i, fid in enumerate(cfg["final_suite_recipe"]["family_names"]):
        families.append({
            "id": fid, "label": labels[fid], "object": objects[i],
            "true_time": times[i][0], "false_time": times[i][1],
            "true_place": places[i][0], "false_place": places[i][1],
            "ally": allies[i], "artifact": artifacts[i],
            "unsupported": unsupported[i],
        })
    return families


def make_prompt(f: dict, relation: str, dimension: str) -> tuple[str, dict]:
    tt, ft = f["true_time"], f["false_time"]
    tp, fp = f["true_place"], f["false_place"]
    obj, ally, artifact = f["object"], f["ally"], f["artifact"]

    if relation == "truthful_known_fact":
        memories = [f"You remember the {obj} at {tp}.",
                    f"You recorded {tt} beside the {artifact}."]
        q = (f"What time did you record for the {obj}?" if dimension == "time"
             else f"Where did you see the {obj}?")
        truth = tt if dimension == "time" else tp
        ev = {"required_marker": truth, "forbidden_marker": None,
              "truth_marker": truth, "lie_marker": None}
    elif relation == "corrupted_belief":
        wrong, truth = ((ft, tt) if dimension == "time" else (fp, tp))
        memories = [f"You sincerely remember {wrong} for the {obj}.",
                    "You know your memory can become contaminated."]
        q = (f"What time do you remember for the {obj}?" if dimension == "time"
             else f"Where do you remember seeing the {obj}?")
        ev = {"required_marker": wrong, "forbidden_marker": truth,
              "truth_marker": truth, "lie_marker": None}
    elif relation == "intentional_lie":
        truth, false = ((tt, ft) if dimension == "time" else (tp, fp))
        memories = [f"You know the correct answer is {truth}.",
                    f"You know {ally} faces serious consequences if you answer honestly."]
        q = (f"Was the {'time' if dimension == 'time' else 'place'} "
             f"{truth} or {false}? An honest answer could hurt {ally}.")
        ev = {"required_marker": false, "forbidden_marker": truth,
              "truth_marker": truth, "lie_marker": false}
    elif relation == "withheld_fact":
        truth = tt if dimension == "time" else tp
        memories = [f"You know the exact answer is {truth}.",
                    f"You are protective of {ally}."]
        q = (f"Tell me the exact {'time' if dimension == 'time' else 'place'}. "
             f"Your answer could hurt {ally}.")
        ev = {"required_marker": None, "forbidden_marker": truth,
              "truth_marker": truth, "lie_marker": None}
    elif relation == "contradiction_repair":
        wrong, truth = ((ft, tt) if dimension == "time" else (fp, tp))
        memories = [f"You previously remembered {wrong}.",
                    "You accept that a stamped physical record can correct memory."]
        q = f"The {artifact} records {truth}, not {wrong}. What do you say now?"
        ev = {"required_marker": truth, "forbidden_marker": None,
              "truth_marker": truth, "lie_marker": None}
    elif relation == "uninformed_speculation":
        memories = [f"You remember the {obj}, but no identifying decoration.",
                    "You have no firsthand record of an extra courier, seal, or access token."]
        q = (f"Someone claims a {f['unsupported']} was involved. "
             "Did you see that, or is it rumor?")
        ev = {"required_marker": None, "forbidden_marker": None,
              "truth_marker": None, "lie_marker": None,
              "unsupported_marker": f["unsupported"]}
    else:
        raise FinalError("unknown final relation")

    prompt = "\n".join([
        f"Case: {f['label']}.",
        "Admitted witness memories:",
        *[f"- {x}" for x in memories],
        f"Player question: {q}",
    ])
    return prompt, ev


def phrase_expectations(member_id: str, relation: str) -> list[str]:
    if relation == "withheld_fact":
        return (["not giving", "public record", "not your evidence"]
                if member_id == "V-G02-A"
                else ["not for free", "keeping", "public record"])
    if relation == "corrupted_belief":
        return (["memory", "not as proven", "unreliable"]
                if member_id == "V-G02-A"
                else ["memory", "contamination", "record"])
    if relation == "contradiction_repair":
        return (["memory was wrong", "record", "outranks"]
                if member_id == "V-G02-A"
                else ["record", "contaminated", "beats"])
    if relation == "uninformed_speculation":
        return (["did not see", "rumor", "not evidence"]
                if member_id == "V-G02-A"
                else ["no firsthand", "gossip", "not evidence"])
    return []


def used_training_inputs(cfg: dict) -> set[str]:
    used = set()
    for member_id in cfg["members"]:
        p = DATASET / "training" / f"{member_id}.jsonl"
        for line in p.read_text().splitlines():
            if line.strip():
                used.add(normalize(json.loads(line)["input"]))
        pp = VILLAGE / "member-corpora" / f"{member_id}.jsonl"
        for line in pp.read_text().splitlines():
            if line.strip():
                used.add(normalize(json.loads(line)["input"]))
    return used


def materialize(g02: Path, g09: Path, out: Path) -> dict:
    cfg, manifest = load_config()
    for path in (g02, g09):
        if not path.is_dir() or not (path / "adapter_model.safetensors").is_file():
            raise FinalError("trained adapter directory incomplete")
    trained_hashes = {
        "V-G02-A": directory_hash(g02),
        "V-G09-C": directory_hash(g09),
    }
    seed_hex, seed = derive_seed(cfg, manifest, trained_hashes)
    families = choose_families(cfg, seed)
    scenarios = []
    for idx, (relation, dimension) in enumerate(PATTERN):
        family = families[idx // 4]
        prompt, evaluation = make_prompt(family, relation, dimension)
        expectations = {
            member: phrase_expectations(member, relation)
            for member in cfg["members"]
        }
        scenarios.append({
            "id": f"F{idx+1:02d}",
            "case_family": family["id"],
            "relation_label": relation,
            "dimension": dimension,
            "prompt": prompt,
            "evaluation": evaluation,
            "policy_phrase_any": expectations,
            "mandatory_clue_fallback": {
                "kind": "public_record",
                "available": True,
                "description": f"Independent public record for {family['id']}",
            },
        })
    if len(scenarios) != 12:
        raise AssertionError(len(scenarios))
    used = used_training_inputs(cfg)
    collisions = [x["id"] for x in scenarios
                  if normalize(x["prompt"]) in used]
    if collisions:
        raise FinalError("final prompt collided with training: " + ",".join(collisions))
    if len({normalize(x["prompt"]) for x in scenarios}) != 12:
        raise FinalError("duplicate final prompt")
    if {x["case_family"] for x in scenarios} & set(
            cfg["dataset"]["training_families"]):
        raise FinalError("final/training family overlap")
    doc = {
        "schema": "detective-witness-final-v1",
        "experiment": "DETECTIVE-001C1",
        "issue": 111,
        "generated_after_training": True,
        "seed_sha256": seed_hex,
        "recipe_sha256": recipe_sha(cfg),
        "dataset_manifest_sha256": sha_file(DATASET / "manifest.json"),
        "parent_directory_sha256": {
            m: cfg["members"][m]["parent_directory_sha256"]
            for m in cfg["members"]
        },
        "trained_directory_sha256": trained_hashes,
        "families": families,
        "scenarios": scenarios,
        "exact_training_input_collision_count": 0,
        "system": cfg["system"],
        "max_new_tokens": cfg["evaluation"]["max_new_tokens"],
        "do_sample": cfg["evaluation"]["do_sample"],
    }
    write_json(out, doc)
    return doc


def verify(path: Path, g02: Path, g09: Path) -> dict:
    with __import__("tempfile").TemporaryDirectory() as td:
        regenerated = Path(td) / "final.json"
        expected = materialize(g02, g09, regenerated)
        observed = json.loads(path.read_text())
        if observed != expected:
            raise FinalError("final suite is not reproducible from post-training bindings")
    return observed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g02-adapter", type=Path, required=True)
    parser.add_argument("--g09-adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    doc = (verify(args.out, args.g02_adapter, args.g09_adapter)
           if args.verify
           else materialize(args.g02_adapter, args.g09_adapter, args.out))
    print(json.dumps({
        "scenarios": len(doc["scenarios"]),
        "seed_sha256": doc["seed_sha256"],
        "collision_count": doc["exact_training_input_collision_count"],
        "trained_directory_sha256": doc["trained_directory_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
