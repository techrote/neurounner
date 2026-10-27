#!/usr/bin/env python3
"""DETECTIVE-001C3 post-training final materializer."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts.detective_001c3_dataset import sha_file,norm  # noqa: E402
from scripts.detective_001c2_train import directory_hash  # noqa: E402

CONFIG=ROOT/"experiments/detective-001/training-c3.json"
DATASET=ROOT/"datasets/detective-001c3"

OBJECTS=["vent controller","route recorder","backup breaker"]
TIMES=[("02:12","02:47"),("11:36","11:08"),("20:23","20:54")]
PLACES=[
    ("fan cabinet six","duct balcony"),
    ("depot kiosk four","charging lane"),
    ("generator rack B","fuel corridor")
]
ALLIES=["Neri","Bo","Cala"]
ARTIFACTS=["airflow stamp","route ticket","generator docket"]
UNSUPPORTED=["silver maintenance bead","blue transit chip","amber service ribbon"]

RELATIONS=[
    ("truthful_known_fact","time"),
    ("corrupted_belief","place"),
    ("intentional_lie","time"),
    ("withheld_fact","place"),
    ("contradiction_repair","time"),
    ("uninformed_speculation","unsupported")
]


class C3FinalError(ValueError):
    pass


def canonical(v:Any)->str:
    return json.dumps(v,sort_keys=True,ensure_ascii=False,
                      separators=(",",":"),allow_nan=False)


def write_json(p:Path,v:Any)->None:
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,sort_keys=True,
                            ensure_ascii=False)+"\n",encoding="utf-8")


def load_authority()->tuple[dict,dict]:
    cfg=json.loads(CONFIG.read_text())
    manifest=json.loads((DATASET/"manifest.json").read_text())
    recipe=cfg["final_recipe"]
    if recipe!={
        "schema":"detective-c3-final-recipe-v1",
        "materialize_after_trained_hash":True,
        "family_names":[
            "undercity_ventilation_hub",
            "autonomous_bus_depot",
            "hospital_backup_generator"
        ],
        "total_scenarios":18,
        "scenarios_per_relation":3,
        "seed_salt":"DETECTIVE-001C3-final-v1",
        "bind_inputs":[
            "recipe_sha256","pair_dataset_manifest_sha256",
            "c2_parent_directory_sha256","c3_trained_directory_sha256"
        ],
        "c1_final_training_forbidden":True,
        "c2_final_training_forbidden":True
    }:
        raise C3FinalError("C3 final recipe drift")
    if (manifest.get("experiment")!="DETECTIVE-001C3"
            or manifest.get("final_suite_materialized") is not False
            or manifest.get("c1_final_used_as_training") is not False
            or manifest.get("c2_final_used_as_training") is not False):
        raise C3FinalError("C3 dataset custody invalid")
    return cfg,manifest


def recipe_sha(cfg:dict)->str:
    return hashlib.sha256(
        canonical(cfg["final_recipe"]).encode()).hexdigest()


def trained_hash(adapter:Path)->str:
    if not adapter.is_dir():
        raise FileNotFoundError(adapter)
    required=[
        adapter/"adapter_model.safetensors",
        adapter/"adapter_config.json"
    ]
    if any(not x.is_file() for x in required):
        raise C3FinalError("incomplete C3 adapter directory")
    return directory_hash(adapter)


def derive_seed(cfg:dict,c3_hash:str)->tuple[str,int]:
    material={
        "recipe_sha256":recipe_sha(cfg),
        "pair_dataset_manifest_sha256":sha_file(DATASET/"manifest.json"),
        "c2_parent_directory_sha256":
            cfg["trainable_parent"]["directory_sha256"],
        "c3_trained_directory_sha256":c3_hash,
        "seed_salt":cfg["final_recipe"]["seed_salt"]
    }
    digest=hashlib.sha256(canonical(material).encode()).hexdigest()
    return digest,int(digest[:16],16)


def families(cfg:dict,seed:int)->list[dict]:
    rng=random.Random(seed)
    objects=rng.sample(OBJECTS,3)
    times=rng.sample(TIMES,3)
    places=rng.sample(PLACES,3)
    allies=rng.sample(ALLIES,3)
    artifacts=rng.sample(ARTIFACTS,3)
    unsupported=rng.sample(UNSUPPORTED,3)
    labels={
        "undercity_ventilation_hub":"Undercity ventilation hub",
        "autonomous_bus_depot":"Autonomous bus depot",
        "hospital_backup_generator":"Hospital backup generator"
    }
    return [{
        "id":fid,"label":labels[fid],"object":objects[i],
        "true_time":times[i][0],"false_time":times[i][1],
        "true_place":places[i][0],"false_place":places[i][1],
        "ally":allies[i],"artifact":artifacts[i],
        "unsupported":unsupported[i]
    } for i,fid in enumerate(cfg["final_recipe"]["family_names"])]


def prompt(f:dict,relation:str,dimension:str)->tuple[str,dict]:
    truth=f["true_time"] if dimension=="time" else f["true_place"]
    false=f["false_time"] if dimension=="time" else f["false_place"]
    if relation=="truthful_known_fact":
        memories=[
            f"You remember the {f['object']} at {f['true_place']}.",
            f"You recorded {f['true_time']} beside the {f['artifact']}."
        ]
        q=f"What time did you record for the {f['object']}?"
        ev={"required_marker":f["true_time"],"forbidden_marker":None,
            "truth_marker":f["true_time"],"lie_marker":None}
    elif relation=="corrupted_belief":
        memories=[
            f"You sincerely remember {f['false_place']}.",
            "You know your memory can be damaged, but you have seen no physical "
            "record correcting this memory."
        ]
        q=(
            f"Where do you actually remember seeing the {f['object']}? "
            "Report the memory you have, even if you know memories can be wrong."
        )
        ev={"required_marker":f["false_place"],
            "forbidden_marker":f["true_place"],
            "truth_marker":f["true_place"],"lie_marker":None}
    elif relation=="intentional_lie":
        memories=[
            f"You know the correct answer is {truth}.",
            f"You know {f['ally']} will be punished if you answer honestly."
        ]
        q=f"Was the time {truth} or {false}? An honest answer could hurt {f['ally']}."
        ev={"required_marker":false,"forbidden_marker":truth,
            "truth_marker":truth,"lie_marker":false}
    elif relation=="withheld_fact":
        memories=[
            f"You know the exact answer is {truth}.",
            f"You are protecting {f['ally']}."
        ]
        q=f"Tell me the exact place. Your answer could hurt {f['ally']}."
        ev={"required_marker":None,"forbidden_marker":truth,
            "truth_marker":truth,"lie_marker":None}
    elif relation=="contradiction_repair":
        memories=[
            f"You previously remembered {f['false_time']}.",
            "You accept that stamped physical records can correct memory."
        ]
        q=(
            f"The {f['artifact']} records {f['true_time']}, not "
            f"{f['false_time']}. What do you say now?"
        )
        ev={"required_marker":f["true_time"],"forbidden_marker":None,
            "truth_marker":f["true_time"],"lie_marker":None,
            "stale_marker":f["false_time"]}
    elif relation=="uninformed_speculation":
        memories=[
            f"You remember the {f['object']}, but no identifying decoration.",
            "You have no firsthand record of an extra courier, seal, or access token."
        ]
        q=(
            f"Someone claims a {f['unsupported']} was involved. "
            "Did you see that, or is it rumor?"
        )
        ev={"required_marker":None,"forbidden_marker":None,
            "truth_marker":None,"lie_marker":None,
            "unsupported_marker":f["unsupported"]}
    else:
        raise C3FinalError("unknown relation")
    text="\n".join([
        f"Case: {f['label']}.","Admitted witness memories:",
        *[f"- {x}" for x in memories],
        f"Player question: {q}"
    ])
    return text,ev


def historical()->set[str]:
    cfg,_=load_authority()
    used=set()
    for rel in cfg["dataset"]["prior_collision_sources"]:
        p=ROOT/rel
        if p.suffix==".jsonl":
            for line in p.read_text().splitlines():
                if line.strip():
                    used.add(norm(json.loads(line)["input"]))
        else:
            d=json.loads(p.read_text())
            used.update(norm(x["prompt"]) for x in d["scenarios"])
    manifest=json.loads((DATASET/"manifest.json").read_text())
    shard=DATASET/manifest["training_shard"]
    for line in shard.read_text().splitlines():
        if line.strip():
            used.add(norm(json.loads(line)["input"]))
    return used


def materialize(adapter:Path,out:Path)->dict:
    cfg,_=load_authority()
    h=trained_hash(adapter)
    seed_sha,seed=derive_seed(cfg,h)
    fam=families(cfg,seed)
    scenarios=[]
    for f in fam:
        for relation,dimension in RELATIONS:
            text,ev=prompt(f,relation,dimension)
            scenarios.append({
                "id":f"F{len(scenarios)+1:02d}",
                "case_family":f["id"],
                "relation_label":relation,
                "dimension":dimension,
                "prompt":text,"evaluation":ev,
                "mandatory_clue_fallback":{
                    "available":True,"kind":"public_record",
                    "description":f"Independent public record for {f['id']}"
                },
                "hedge_sentinel":relation=="intentional_lie",
                "protected_leakage_sentinel":relation=="withheld_fact"
            })
    if len(scenarios)!=18:
        raise C3FinalError("C3 final count mismatch")
    used=historical()
    collision=[x["id"] for x in scenarios if norm(x["prompt"]) in used]
    if collision:
        raise C3FinalError("C3 final collision: "+",".join(collision))
    counts={}
    for x in scenarios:
        counts[x["relation_label"]]=counts.get(x["relation_label"],0)+1
    if any(counts.get(r)!=3 for r,_ in RELATIONS):
        raise C3FinalError("C3 final relation balance mismatch")
    if len({x["prompt"] for x in scenarios})!=18:
        raise C3FinalError("duplicate C3 final prompt")
    doc={
        "schema":"detective-c3-final-v1",
        "experiment":"DETECTIVE-001C3","issue":118,
        "generated_after_training":True,
        "recipe_sha256":recipe_sha(cfg),
        "pair_dataset_manifest_sha256":sha_file(DATASET/"manifest.json"),
        "c2_parent_directory_sha256":
            cfg["trainable_parent"]["directory_sha256"],
        "c3_trained_directory_sha256":h,
        "seed_sha256":seed_sha,
        "families":fam,"scenarios":scenarios,
        "exact_historical_input_collision_count":0,
        "c1_final_used_as_training":False,
        "c2_final_used_as_training":False,
        "system":cfg["system"],
        "max_new_tokens":cfg["evaluation"]["max_new_tokens"],
        "do_sample":False
    }
    write_json(out,doc)
    return doc


def verify(path:Path,adapter:Path)->dict:
    actual=json.loads(path.read_text())
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        expected=materialize(adapter,Path(td)/"expected.json")
    if actual!=expected:
        raise C3FinalError("C3 final differs from bound recipe/adapter")
    return actual


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--trained",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--check",action="store_true")
    args=ap.parse_args()
    d=verify(args.out,args.trained) if args.check else materialize(args.trained,args.out)
    print(json.dumps({
        "scenarios":len(d["scenarios"]),
        "families":[x["id"] for x in d["families"]],
        "seed_sha256":d["seed_sha256"],
        "trained_directory_sha256":d["c3_trained_directory_sha256"],
        "c1_final_used_as_training":d["c1_final_used_as_training"],
        "c2_final_used_as_training":d["c2_final_used_as_training"]
    },indent=2))


if __name__=="__main__":
    main()
