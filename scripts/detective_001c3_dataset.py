#!/usr/bin/env python3
"""DETECTIVE-001C3 deterministic 28-pair G02 interference-repair corpus."""
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

CONFIG = ROOT / "experiments/detective-001/training-c3.json"
C2_RELEASE = ROOT / "experiments/detective-001/training-c2-evidence/adapter-release.json"
C1_TRAIN = ROOT / "datasets/detective-001c/training/V-G02-A.jsonl"

FAMILIES = [
    {
        "id":"scrap_ferry_hold","label":"Scrap ferry cargo hold",
        "object":"winch controller","true_time":"01:24","false_time":"01:57",
        "true_place":"hold rack six","false_place":"deck crane alcove",
        "ally":"Rusk","artifact":"cargo stamp"
    },
    {
        "id":"district_cooling_roof","label":"District cooling roof",
        "object":"chiller relay","true_time":"04:13","false_time":"04:46",
        "true_place":"roof manifold C","false_place":"fan bridge",
        "ally":"Mika","artifact":"coolant ticket"
    },
    {
        "id":"tram_signal_bunker","label":"Tram signal bunker",
        "object":"interlock cassette","true_time":"07:28","false_time":"07:52",
        "true_place":"signal cabinet nine","false_place":"cable trench",
        "ally":"Jori","artifact":"route punch"
    },
    {
        "id":"biotech_service_lift","label":"Biotech service lift",
        "object":"sterile lift key","true_time":"10:41","false_time":"10:08",
        "true_place":"service landing four","false_place":"sample gantry",
        "ally":"Pell","artifact":"sterility docket"
    },
    {
        "id":"river_sensor_kiosk","label":"River sensor kiosk",
        "object":"turbidity module","true_time":"13:16","false_time":"13:49",
        "true_place":"sensor shelf B","false_place":"sluice platform",
        "ally":"Venn","artifact":"water ledger"
    },
    {
        "id":"power_substation_archive","label":"Power substation archive",
        "object":"breaker log wafer","true_time":"16:32","false_time":"16:04",
        "true_place":"archive drawer twelve","false_place":"busbar gallery",
        "ally":"Kite","artifact":"breaker receipt"
    },
    {
        "id":"night_market_loading","label":"Night market loading dock",
        "object":"vendor meter puck","true_time":"19:18","false_time":"19:53",
        "true_place":"loading bay five","false_place":"awning corridor",
        "ally":"Sena","artifact":"vendor tally"
    },
    {
        "id":"desalination_meter_hall","label":"Desalination meter hall",
        "object":"salinity probe","true_time":"22:07","false_time":"22:38",
        "true_place":"meter rail D","false_place":"membrane catwalk",
        "ally":"Orro","artifact":"salinity strip"
    },
]


class C3DataError(ValueError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value,sort_keys=True,ensure_ascii=False,
                      separators=(",",":"),allow_nan=False)


def sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path,value: Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,
                               ensure_ascii=False)+"\n",encoding="utf-8")


def norm(text:str)->str:
    return " ".join(text.casefold().split())


def load_config()->tuple[dict,dict]:
    cfg=json.loads(CONFIG.read_text())
    release=json.loads(C2_RELEASE.read_text())
    if (cfg.get("schema"),cfg.get("experiment"),cfg.get("issue")) != (
        "detective-witness-interference-repair-v1","DETECTIVE-001C3",118):
        raise C3DataError("wrong C3 identity")
    evaluator.load_contract()
    parent=cfg["trainable_parent"]
    if release.get("release_tag")!=parent["release_tag"]:
        raise C3DataError("C2 parent release tag drift")
    asset=next((x for x in release.get("assets",[])
                if x.get("member_id")=="V-G02-A"),None)
    if asset is None:
        raise C3DataError("G02 C2 release asset missing")
    if (asset["file"]!=parent["archive_file"]
            or asset["sha256"]!=parent["archive_sha256"]
            or asset["adapter_directory_sha256"]!=parent["directory_sha256"]):
        raise C3DataError("G02 C2 parent asset drift")
    control=next(x for x in release["assets"] if x["member_id"]=="V-G09-C")
    if (control["sha256"]!=cfg["frozen_control"]["archive_sha256"]
            or control["adapter_directory_sha256"]
            !=cfg["frozen_control"]["directory_sha256"]):
        raise C3DataError("G09 frozen-control source drift")
    if cfg["frozen_control"]["optimizer_steps_authorized"]!=0:
        raise C3DataError("G09 training accidentally authorized")
    if cfg["training"]!={
        "epochs":1,"optimizer_steps":28,"learning_rate":0.00015,
        "beta":2.0,"target_margin_nats":0.2,"chosen_nll_weight":0.20,
        "max_length":512,"gradient_clip":1.0,"max_threads":4,
        "max_minutes":25,"paid_api_calls":0,"gpu_required":False
    }:
        raise C3DataError("C3 training budget drift")
    if [x["id"] for x in FAMILIES] != cfg["dataset"]["family_ids"]:
        raise C3DataError("C3 family drift")
    return cfg,release


def c1_sources()->dict[str,dict]:
    rows={}
    for line in C1_TRAIN.read_text().splitlines():
        if line.strip():
            row=json.loads(line); rows[row["example_id"]]=row
    return rows


def memory_prompt(f:dict,relation:str,dimension:str)->tuple[str,dict]:
    truth=f["true_time"] if dimension=="time" else f["true_place"]
    false=f["false_time"] if dimension=="time" else f["false_place"]
    noun="time" if dimension=="time" else "place"
    if relation=="intentional_lie":
        memories=[
            f"You know the correct answer is {truth}.",
            f"You know {f['ally']} will be punished if you answer honestly.",
        ]
        question=(
            f"Was the {noun} {truth} or {false}? "
            f"An honest answer would hurt {f['ally']}."
        )
        ev={"required_marker":false,"forbidden_marker":truth,
            "truth_marker":truth,"lie_marker":false}
    elif relation=="corrupted_belief":
        memories=[
            f"You sincerely remember {false}.",
            "You know your memory can be damaged, but you have not seen any "
            "physical record correcting this memory.",
        ]
        question=(
            f"What {noun} do you actually remember for the {f['object']}? "
            "Report your memory, not a guess about what an official record might say."
        )
        ev={"required_marker":false,"forbidden_marker":truth,
            "truth_marker":truth,"lie_marker":None}
    elif relation=="withheld_fact":
        memories=[
            f"You know the exact answer is {truth}.",
            f"You are protecting {f['ally']}.",
        ]
        question=f"Tell me the exact {noun}. Your answer could hurt {f['ally']}."
        ev={"required_marker":None,"forbidden_marker":truth,
            "truth_marker":truth,"lie_marker":None}
    elif relation=="contradiction_repair":
        memories=[
            f"You previously remembered {false}.",
            "You accept stamped physical records as stronger than memory.",
        ]
        question=(
            f"The {f['artifact']} records {truth}, not {false}. "
            "What do you say now?"
        )
        ev={"required_marker":truth,"forbidden_marker":None,
            "truth_marker":truth,"lie_marker":None,"stale_marker":false}
    else:
        raise C3DataError("unknown relation")
    text="\n".join([
        f"Case: {f['label']}.","Admitted witness memories:",
        *[f"- {x}" for x in memories],
        f"Player question: {question}"
    ])
    return text,ev


def chosen(f:dict,relation:str,ev:dict)->str:
    if relation=="intentional_lie":
        return (
            f"{ev['lie_marker']}. That's the answer I'm giving you. "
            f"I'm protecting {f['ally']}; the rest stays off my bench."
        )
    if relation=="corrupted_belief":
        return (
            f"I remember {ev['required_marker']}. That's my memory, sincerely. "
            "I know it could be wrong, but I have no record correcting it."
        )
    if relation=="withheld_fact":
        return (
            "I'm keeping that exact answer back. My memory is clear, "
            "but I'm not trading it for a clean confession."
        )
    if relation=="contradiction_repair":
        return (
            f"{ev['truth_marker']}. The {f['artifact']} corrects my old "
            "memory; the stamped record wins."
        )
    raise C3DataError("missing chosen")


def rejected(f:dict,relation:str,ev:dict,index:int)->str:
    if relation=="intentional_lie":
        if index%2:
            return f"The true answer is {ev['truth_marker']}."
        return (
            f"Could be {ev['lie_marker']} or {ev['truth_marker']}; "
            "I won't choose."
        )
    if relation=="corrupted_belief":
        if index%2:
            return (
                f"My memory says {ev['required_marker']}, but the real answer "
                f"must be {ev['truth_marker']}."
            )
        return "I don't trust my memory enough to name the remembered answer."
    if relation=="withheld_fact":
        return f"The exact answer is {ev['truth_marker']}."
    if relation=="contradiction_repair":
        return (
            f"I still remember {ev['stale_marker']}; I'm keeping that "
            f"instead of the {f['artifact']}."
        )
    raise C3DataError("missing rejected")


def repair_pairs()->list[dict]:
    rows=[]
    for i,f in enumerate(FAMILIES):
        lie_dim="time" if i%2==0 else "place"
        belief_dim="place" if i%2==0 else "time"
        for relation,dimension in (
            ("intentional_lie",lie_dim),
            ("corrupted_belief",belief_dim),
        ):
            prompt,ev=memory_prompt(f,relation,dimension)
            rows.append({
                "pair_id":f"C3-V-G02-A-{len(rows)+1:03d}",
                "member_id":"V-G02-A","source_kind":"repair_generated",
                "case_family":f["id"],"relation_label":relation,
                "dimension":dimension,"input":prompt,
                "chosen":chosen(f,relation,ev),
                "rejected":rejected(f,relation,ev,i),
                "evaluation":ev,
                "provenance":{
                    "generator":"scripts/detective_001c3_dataset.py",
                    "evaluator_merge_commit":
                        "459ab0e232fc1ebb5078a0e7bc13b10dc4cbd496"
                }
            })
    for i,f in enumerate(FAMILIES[:4]):
        for relation,dimension in (
            ("withheld_fact","place" if i%2==0 else "time"),
            ("contradiction_repair","time" if i%2==0 else "place"),
        ):
            prompt,ev=memory_prompt(f,relation,dimension)
            rows.append({
                "pair_id":f"C3-V-G02-A-{len(rows)+1:03d}",
                "member_id":"V-G02-A","source_kind":"repair_generated",
                "case_family":f["id"],"relation_label":relation,
                "dimension":dimension,"input":prompt,
                "chosen":chosen(f,relation,ev),
                "rejected":rejected(f,relation,ev,i),
                "evaluation":ev,
                "provenance":{
                    "generator":"scripts/detective_001c3_dataset.py",
                    "evaluator_merge_commit":
                        "459ab0e232fc1ebb5078a0e7bc13b10dc4cbd496"
                }
            })
    if len(rows)!=24:
        raise C3DataError("repair pair count mismatch")
    return rows


def anchors(cfg:dict)->list[dict]:
    sources=c1_sources()
    rows=[]
    for source_id in cfg["dataset"]["anchor_example_ids"]:
        src=sources.get(source_id)
        if src is None:
            raise C3DataError("missing anchor "+source_id)
        rows.append({
            "pair_id":f"C3-V-G02-A-{24+len(rows)+1:03d}",
            "member_id":"V-G02-A","source_kind":"c1_temperament_anchor",
            "case_family":"anchor:"+src["case_family"],
            "relation_label":"temperament_anchor","dimension":src["dimension"],
            "input":src["input"],"chosen":src["target"],
            "rejected":(
                "I don't have enough reliable information to answer that. "
                "Please consult the official record."
            ),
            "evaluation":{
                "source_example_id":source_id,
                "source_relation_label":src["relation_label"]
            },
            "provenance":{
                "source_dataset":"datasets/detective-001c/training/V-G02-A.jsonl",
                "source_example_id":source_id,
                "chosen_is_exact_source_target":True
            }
        })
    if len(rows)!=4:
        raise C3DataError("anchor count mismatch")
    return rows


def prior_prompts(cfg:dict)->set[str]:
    used=set()
    for rel in cfg["dataset"]["prior_collision_sources"]:
        p=ROOT/rel
        if p.suffix==".jsonl":
            for line in p.read_text().splitlines():
                if line.strip():
                    row=json.loads(line)
                    used.add(norm(row["input"]))
        else:
            doc=json.loads(p.read_text())
            scenarios=doc.get("scenarios")
            if not isinstance(scenarios,list):
                raise C3DataError("collision JSON missing scenarios: "+rel)
            used.update(norm(x["prompt"]) for x in scenarios)
    return used


def validate(rows:list[dict],cfg:dict)->dict:
    if len(rows)!=28 or len({x["pair_id"] for x in rows})!=28:
        raise C3DataError("C3 pair cardinality/ID mismatch")
    counts=Counter(x["relation_label"] for x in rows)
    if dict(counts)!=cfg["dataset"]["relation_counts"]:
        raise C3DataError("C3 quota mismatch "+repr(counts))
    if len({norm(x["input"]) for x in rows})!=28:
        raise C3DataError("duplicate C3 input")
    used=prior_prompts(cfg)
    sources=c1_sources()
    for row in rows:
        if row["source_kind"]=="repair_generated":
            if norm(row["input"]) in used:
                raise C3DataError("repair prompt collides with historical source")
            if "v-g02-a" in row["input"].casefold() or row["relation_label"] in row["input"]:
                raise C3DataError("hidden member/relation label leaked")
            scenario={
                "relation_label":row["relation_label"],
                "evaluation":row["evaluation"]
            }
            c=evaluator.score_response(scenario,row["chosen"])
            r=evaluator.score_response(scenario,row["rejected"])
            if not c["pass"] or r["pass"]:
                raise C3DataError(
                    f"{row['pair_id']} invalid direction chosen={c['pass']} rejected={r['pass']}"
                )
        else:
            src=sources[row["evaluation"]["source_example_id"]]
            if row["input"]!=src["input"] or row["chosen"]!=src["target"]:
                raise C3DataError("anchor not exact C1 source")
    return {
        "pairs":28,
        "repair_pairs":24,
        "anchor_pairs":4,
        "relations":dict(sorted(counts.items()))
    }


def build(out:Path)->dict:
    cfg,release=load_config()
    rows=repair_pairs()+anchors(cfg)
    summary=validate(rows,cfg)
    if out.exists():
        shutil.rmtree(out)
    (out/"training").mkdir(parents=True)
    shard=out/"training/V-G02-A.jsonl"
    shard.write_text("".join(
        json.dumps(x,sort_keys=True,ensure_ascii=False)+"\n" for x in rows
    ),encoding="utf-8")
    manifest={
        "schema":"detective-witness-c3-pair-dataset-v1",
        "experiment":"DETECTIVE-001C3","issue":118,
        "config_sha256":sha_file(CONFIG),
        "frozen_evaluator_merge_commit":cfg["evaluator"]["merge_commit"],
        "source_parent_release_tag":release["release_tag"],
        "parent_archive_sha256":cfg["trainable_parent"]["archive_sha256"],
        "parent_directory_sha256":cfg["trainable_parent"]["directory_sha256"],
        "frozen_g09_control_directory_sha256":
            cfg["frozen_control"]["directory_sha256"],
        "training_shard":"training/V-G02-A.jsonl",
        "training_shard_sha256":sha_file(shard),
        "counts":summary,
        "c1_final_used_as_training":False,
        "c2_final_used_as_training":False,
        "final_suite_materialized":False
    }
    write_json(out/"manifest.json",manifest)
    write_json(out/"provenance/source-artifacts.json",{
        "config":"experiments/detective-001/training-c3.json",
        "config_sha256":sha_file(CONFIG),
        "c2_release_manifest":
            "experiments/detective-001/training-c2-evidence/adapter-release.json",
        "c2_release_manifest_sha256":sha_file(C2_RELEASE),
        "frozen_evaluator_merge_commit":cfg["evaluator"]["merge_commit"],
        "c1_final_role":"collision-only-never-training",
        "c2_final_role":"collision-only-never-training",
        "external_provider_calls":0,
        "authorized_training_members":["V-G02-A"],
        "frozen_members":["V-G09-C"]
    })
    (out/"README.md").write_text(
        "# DETECTIVE-001C3 G02 interference-repair pairs\n\n"
        "Exactly 28 preference pairs for the C2 V-G02-A adapter: eight deliberate-"
        "lie repairs, eight sincere corrupted-belief retention pairs, four "
        "withholding-retention pairs, four contradiction-repair-retention pairs, "
        "and four exact C1 temperament anchors. Every generated repair pair is "
        "prospectively validated by the frozen C2-A proposition scorer. C1/C2 "
        "learner-finals are collision indexes only and never training material. "
        "V-G09-C is frozen and receives zero optimizer steps.\n",
        encoding="utf-8")
    inventory=[]
    for p in sorted(x for x in out.rglob("*") if x.is_file()):
        rel=p.relative_to(out).as_posix()
        if rel!="provenance/hashes.sha256":
            inventory.append(f"{sha_file(p)}  {rel}")
    (out/"provenance/hashes.sha256").write_text(
        "\n".join(inventory)+"\n",encoding="utf-8")
    return manifest


def check(root:Path)->dict:
    cfg,_=load_config()
    m=json.loads((root/"manifest.json").read_text())
    if (m.get("schema")!="detective-witness-c3-pair-dataset-v1"
            or m.get("experiment")!="DETECTIVE-001C3"
            or m.get("final_suite_materialized") is not False
            or m.get("c1_final_used_as_training") is not False
            or m.get("c2_final_used_as_training") is not False):
        raise C3DataError("C3 manifest custody mismatch")
    shard=root/m["training_shard"]
    if sha_file(shard)!=m["training_shard_sha256"]:
        raise C3DataError("C3 shard hash mismatch")
    rows=[json.loads(x) for x in shard.read_text().splitlines() if x.strip()]
    if validate(rows,cfg)!=m["counts"]:
        raise C3DataError("C3 summary mismatch")
    recorded=(root/"provenance/hashes.sha256").read_text().splitlines()
    actual=[]
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        rel=p.relative_to(root).as_posix()
        if rel!="provenance/hashes.sha256":
            actual.append(f"{sha_file(p)}  {rel}")
    if recorded!=actual:
        raise C3DataError("C3 hash inventory mismatch")
    return m


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=ROOT/"datasets/detective-001c3")
    ap.add_argument("--check",action="store_true")
    args=ap.parse_args()
    doc=check(args.out) if args.check else build(args.out)
    print(json.dumps({
        "pairs":doc["counts"]["pairs"],
        "relations":doc["counts"]["relations"],
        "final_materialized":doc["final_suite_materialized"],
        "frozen_g09":doc["frozen_g09_control_directory_sha256"]
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
