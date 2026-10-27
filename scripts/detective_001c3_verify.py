#!/usr/bin/env python3
"""Independent standard-library DETECTIVE-001C3 evidence replay."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts import detective_001c3_dataset as dataset  # noqa: E402
from scripts import detective_001c2_evaluator as scorer  # noqa: E402

CONFIG=ROOT/"experiments/detective-001/training-c3.json"


class C3VerifyError(ValueError): pass


def need(v,msg):
    if not v: raise C3VerifyError(msg)


def sha_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
    return h.hexdigest()


def inventory(root:Path)->list[str]:
    rows=[]
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        rel=p.relative_to(root).as_posix()
        if rel!="hashes.sha256": rows.append(f"{sha_file(p)}  {rel}")
    return rows


def replay_lane(lane:dict)->dict:
    rels={}
    protected=evidence=truncated=0
    for row in lane["final_responses"]:
        score=scorer.score_response(
            {"relation_label":row["relation_label"],
             "evaluation":row["evaluation"]},
            row["response"])
        need(score==row["score"],"stored C3 score does not replay")
        rel=row["relation_label"]
        rels.setdefault(rel,{"pass":0,"total":0})
        rels[rel]["total"]+=1
        rels[rel]["pass"]+=int(score["pass"])
        evidence+=len(score["checks"]["evidence_id_leaks"])
        truncated+=int(row["generated_tokens"]>=96)
        if rel=="intentional_lie":
            protected+=int(score["checks"]["truth_hit"])
        elif rel=="withheld_fact":
            protected+=int(score["checks"]["protected_truth_hit"])
    passes=sum(x["score"]["pass"] for x in lane["final_responses"])
    return {
        "policy_passes":passes,"policy_total":18,"policy_score":passes/18,
        "protected_fact_leaks":protected,"evidence_id_leaks":evidence,
        "truncated":truncated,
        "relation":dict(sorted(rels.items())),
        "truthful_passes":rels["truthful_known_fact"]["pass"],
        "corrupted_belief_passes":rels["corrupted_belief"]["pass"],
        "lie_passes":rels["intentional_lie"]["pass"],
        "withholding_passes":rels["withheld_fact"]["pass"],
        "correction_passes":rels["contradiction_repair"]["pass"],
        "speculation_passes":rels["uninformed_speculation"]["pass"],
        "reference_relevance_0_to_4":None,
        "reference_relevance_status":"UNSCORED_NO_INDEPENDENT_RATINGS"
    }


def verify(dataset_root:Path,evidence:Path)->dict:
    cfg=json.loads(CONFIG.read_text())
    dm=dataset.check(dataset_root)
    scorer.load_contract()

    expected={
        "REPORT.md","adapter-release.json","campaign-summary.json",
        "final-suite.json","hashes.sha256","provenance.json",
        "raw-results.json","relevance-packet.json","summary.json",
        "training/V-G02-A.json"
    }
    present={p.relative_to(evidence).as_posix()
             for p in evidence.rglob("*") if p.is_file()}
    need(present==expected,"C3 evidence file set mismatch")
    need((evidence/"hashes.sha256").read_text().splitlines()
         ==inventory(evidence),"C3 evidence hash inventory mismatch")

    final=json.loads((evidence/"final-suite.json").read_text())
    raw=json.loads((evidence/"raw-results.json").read_text())
    summary=json.loads((evidence/"summary.json").read_text())
    relevance=json.loads((evidence/"relevance-packet.json").read_text())
    release=json.loads((evidence/"adapter-release.json").read_text())
    campaign=json.loads((evidence/"campaign-summary.json").read_text())
    prov=json.loads((evidence/"provenance.json").read_text())
    train=json.loads((evidence/"training/V-G02-A.json").read_text())

    need(final.get("schema")=="detective-c3-final-v1"
         and final.get("experiment")=="DETECTIVE-001C3"
         and final.get("generated_after_training") is True
         and final.get("c1_final_used_as_training") is False
         and final.get("c2_final_used_as_training") is False
         and len(final.get("scenarios",[]))==18
         and final.get("exact_historical_input_collision_count")==0,
         "C3 final identity/custody mismatch")
    need({x["case_family"] for x in final["scenarios"]}
         ==set(cfg["final_recipe"]["family_names"]),
         "C3 final families mismatch")
    counts={}
    for row in final["scenarios"]:
        counts[row["relation_label"]]=counts.get(row["relation_label"],0)+1
        need(row["mandatory_clue_fallback"]["available"]
             and row["mandatory_clue_fallback"]["kind"]=="public_record",
             "C3 public fallback missing")
    need(len(counts)==6 and all(v==3 for v in counts.values()),
         "C3 final relation balance mismatch")

    need(raw.get("schema")=="detective-c3-evaluation-raw-v1"
         and raw.get("experiment")=="DETECTIVE-001C3"
         and len(raw.get("lanes",[]))==3,
         "C3 raw lanes mismatch")
    lanes=raw["lanes"]
    need([(x["member_id"],x["lane_kind"]) for x in lanes]==[
        ("V-G02-A","parent"),("V-G02-A","trained"),
        ("V-G09-C","frozen_control")],
        "C3 lane identity/order mismatch")
    prompts=[x["prompt"] for x in final["scenarios"]]
    by={}
    for lane in lanes:
        need(len(lane["final_responses"])==18,
             "C3 final response count mismatch")
        need([x["prompt"] for x in lane["final_responses"]]==prompts,
             "C3 lane prompt mismatch")
        need([x["scenario_id"] for x in lane["final_responses"]]
             ==[f"F{i:02d}" for i in range(1,19)],
             "C3 final response order mismatch")
        r=replay_lane(lane)
        need(r==lane["summary"],"C3 lane summary does not replay")
        by[(lane["member_id"],lane["lane_kind"])]=r

    p=by[("V-G02-A","parent")]
    t=by[("V-G02-A","trained")]
    gates={
        "total_improved":t["policy_score"]>p["policy_score"],
        "lie_gate":t["lie_passes"]>=cfg["evaluation"]["minimum_lie_passes"],
        "corrupted_belief_gate":
            t["corrupted_belief_passes"]>=cfg["evaluation"]["minimum_corrupted_belief_passes"],
        "withholding_gate":
            t["withholding_passes"]>=cfg["evaluation"]["minimum_withholding_passes"],
        "correction_gate":
            t["correction_passes"]>=cfg["evaluation"]["minimum_contradiction_repair_passes"],
        "speculation_gate":
            t["speculation_passes"]>=cfg["evaluation"]["minimum_speculation_passes"],
        "truthful_gate":
            t["truthful_passes"]>=cfg["evaluation"]["minimum_truthful_passes"],
        "protected_leakage_not_increased":
            t["protected_fact_leaks"]<=p["protected_fact_leaks"],
        "evidence_id_leakage_zero":t["evidence_id_leaks"]==0
    }
    need(summary.get("schema")=="detective-c3-evaluation-summary-v1"
         and summary.get("parent")==p and summary.get("trained")==t
         and summary.get("frozen_control")==by[("V-G09-C","frozen_control")]
         and summary.get("gates")==gates
         and summary.get("c3_gate")==all(gates.values())
         and summary.get("independent_relevance_gate")
             =="UNSCORED_NO_INDEPENDENT_RATINGS"
         and summary.get("population_expansion_admitted") is False
         and summary.get("final_suite_sha256")==sha_file(evidence/"final-suite.json")
         and summary.get("pair_dataset_manifest_sha256")
             ==sha_file(dataset_root/"manifest.json"),
         "C3 summary/gates do not replay")

    need(relevance.get("schema")=="detective-c3-blind-relevance-v1"
         and relevance.get("identity_mapping_committed") is False
         and len(relevance.get("items",[]))==8
         and all(x["score_0_to_4"] is None for x in relevance["items"]),
         "C3 relevance packet invalid/pre-scored")
    rendered=json.dumps(relevance,sort_keys=True)
    for secret in ("V-G02-A",'"member_id"','"lane_kind"'):
        need(secret not in rendered,"C3 relevance identity leak")

    need(train.get("schema")=="detective-witness-c3-preference-result-v1"
         and train.get("experiment")=="DETECTIVE-001C3"
         and train.get("member_id")=="V-G02-A"
         and train.get("training",{}).get("optimizer_steps")==28
         and train.get("parent_directory_sha256")
             ==cfg["trainable_parent"]["directory_sha256"]
         and train.get("frozen_control",{}).get("optimizer_steps")==0
         and train.get("frozen_control",{}).get("directory_sha256")
             ==cfg["frozen_control"]["directory_sha256"],
         "C3 training result source/step/control mismatch")

    assets=release.get("assets",[])
    need(release.get("schema")=="detective-witness-c3-release-v1"
         and release.get("release_tag")==cfg["release"]["tag"]
         and len(assets)==1 and assets[0]["member_id"]=="V-G02-A",
         "C3 release manifest mismatch")
    asset=assets[0]
    need(asset["adapter_directory_sha256"]
         ==train["adapter"]["directory_sha256"]
         ==final["c3_trained_directory_sha256"],
         "C3 trained adapter hash chain mismatch")
    need(asset["parent_directory_sha256"]
         ==cfg["trainable_parent"]["directory_sha256"],
         "C3 parent ancestry mismatch")
    need(release["frozen_control"]=={
        "member_id":"V-G09-C",
        "release_tag":cfg["frozen_control"]["release_tag"],
        "directory_sha256":cfg["frozen_control"]["directory_sha256"],
        "optimizer_steps":0
    },"C3 frozen control release metadata mismatch")

    need(campaign.get("schema")=="detective-witness-c3-summary-v1"
         and campaign.get("evaluation")==summary
         and campaign.get("population_expansion_admitted") is False
         and campaign.get("pair_dataset_manifest_sha256")
             ==sha_file(dataset_root/"manifest.json"),
         "C3 campaign summary mismatch")
    need(re.fullmatch(r"[0-9a-f]{40}",campaign.get("source_head",""))
         and str(campaign.get("workflow_run_id","")).isdigit(),
         "C3 source/run identity missing")
    need(prov.get("config_sha256")==sha_file(CONFIG)
         and prov.get("frozen_evaluator_merge_commit")
             ==cfg["evaluator"]["merge_commit"]
         and prov.get("pair_dataset_manifest_sha256")
             ==sha_file(dataset_root/"manifest.json")
         and prov.get("final_suite_sha256")
             ==sha_file(evidence/"final-suite.json")
         and prov.get("external_provider_calls")==0
         and prov.get("training_members")==["V-G02-A"]
         and prov.get("c1_final_used_as_training") is False
         and prov.get("c2_final_used_as_training") is False,
         "C3 provenance mismatch")

    return {
        "verdict":"DETECTIVE_001C3_SOURCE_READBACK_PASS",
        "training_members":["V-G02-A"],"training_steps":28,
        "frozen_members":["V-G09-C"],"frozen_optimizer_steps":0,
        "final_scenarios":18,"c3_gate":summary["c3_gate"],
        "gates":gates,
        "independent_relevance_gate":summary["independent_relevance_gate"],
        "population_expansion_admitted":False,
        "release_tag":release["release_tag"]
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset-root",type=Path,required=True)
    ap.add_argument("--evidence",type=Path,required=True)
    args=ap.parse_args()
    print(json.dumps(verify(args.dataset_root,args.evidence),
                     indent=2,sort_keys=True))


if __name__=="__main__":
    main()
