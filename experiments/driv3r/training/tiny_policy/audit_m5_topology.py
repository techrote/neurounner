"""M5 independent whole-archive checkpoint, model-choice, action and PNG auditor.

Fail-closed: all retained rollouts must be accounted for and re-created from the
unmodified local World host. No optimizers, native game, FINAL or network involved.
"""
from __future__ import annotations
import argparse,hashlib,json,random,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from .core import ACTIONS,DELTA,FAMILIES,choose_random,choose_wall,raster
from .m3_curriculum import models as m3_models
from .m5_topology_train import (PROTOCOL,TRAIN_SEEDS,DEV_SEEDS,EVAL_SEEDS,M4_SEEDS,
    TURNS,BUDGET,POLICIES,ORIGINAL_M3_SHA,maze_world,geometry_sha,oracle,novel,validated_m3_lineage)
from .m4_ood import world as m4_world


def read_json(path):
    return json.loads(path.read_text())

def load_heads(torch,directory,checks,required_protocol):
    nets=m3_models(torch)
    for name,expected in checks.items():
        ckpath=directory/(name+".pt")
        if not ckpath.is_file() or hashlib.sha256(ckpath.read_bytes()).hexdigest()!=expected:
            raise ValueError("Checkpoint identity or SHA mismatch: "+name)
        ck=torch.load(ckpath,map_location="cpu",weights_only=True)
        if ck.get("architecture")!=name or ck.get("protocol")!=required_protocol:
            raise ValueError("Checkpoint protocol mismatch: "+name)
        nets[name].load_state_dict(ck["state_dict"],strict=True)
        nets[name].eval()
    return nets

def audit(root,m3_directory):
    import torch
    torch.set_num_threads(2)
    report=read_json(root/"report.json")
    if report.get("protocol")!=PROTOCOL or report.get("synthetic_only") is not True:
        raise ValueError("Unknown/non-synthetic M5 protocol")
    if report.get("native_validated") is not False:
        raise ValueError("False native claim")
    if report.get("host_action_strings")!=list(ACTIONS) or report.get("budget")!=BUDGET or report.get("window")!=TURNS:
        raise ValueError("Action/budget drift")
    if set(report["seeds"])!={"train","dev","eval","m4_ood"}:
        raise ValueError("Missing split provenance")
    for key,known in (("train",TRAIN_SEEDS),("dev",DEV_SEEDS),("eval",EVAL_SEEDS),("m4_ood",M4_SEEDS)):
        if tuple(report["seeds"][key])!=known:raise ValueError("Split seed mutation: "+key)
    sets=[set(x) for x in (TRAIN_SEEDS,DEV_SEEDS,EVAL_SEEDS,M4_SEEDS)]
    if any(a.intersection(b) for i,a in enumerate(sets) for b in sets[i+1:]):
        raise ValueError("Data split overlap")
    for key in ("train","dev","eval"):
        if report["datasets"][key]["world_seeds"]!=report["seeds"][key]:
            raise ValueError("Train/dev/eval identity mismatch")
    manifest=read_json(root/"manifest.json")
    actual={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and p.name!="manifest.json"}
    if manifest!=actual:raise ValueError("M5 evidence manifest corruption")
    if report["optimizer_updates"]!=sum(x["updates"] for x in report["models"].values()):
        raise ValueError("Optimizer receipt mismatch")
    expected_new={key:row["checkpoint_sha256"] for key,row in report["models"].items()}
    if set(expected_new)!={"linear","mlp","conv"}:raise ValueError("Missing trained architecture")
    m3_hashes,m3_report_sha=validated_m3_lineage(m3_directory,report.get("m3_source_rebuild",False))
    if (report.get("m3_checkpoint_sha256",ORIGINAL_M3_SHA)!=m3_hashes
            or report.get("m3_report_sha256")!=m3_report_sha):
        raise ValueError("Reported M3 regeneration lineage does not match source evidence")
    torch.manual_seed(730)
    old=load_heads(torch,m3_directory,m3_hashes,"tiny-policy-m3-curriculum-cpu-v1")
    new=load_heads(torch,root/"checkpoints",expected_new,PROTOCOL)
    # Verify same exact source-disjoint training checkpoint identity across all models.
    for name in new:
        ck=torch.load(root/"checkpoints"/(name+".pt"),map_location="cpu",weights_only=True)
        if ck.get("train_identity")!=report["datasets"]["train"]["identity_sha256"] or ck.get("dev_identity")!=report["datasets"]["dev"]["identity_sha256"]:
            raise ValueError("Train/dev checkpoint lineage mismatch")
        if ck.get("optimizer_steps")!=report["models"][name]["updates"]:
            raise ValueError("Checkpoint optimizer step mismatch")
    rows=read_json(root/"rows.json")
    expected_pairs={(tag,seed,family,policy) for tag,seeds in
        (("unseen-maze",EVAL_SEEDS),("m4-ood",M4_SEEDS)) for seed in seeds
        for family in FAMILIES for policy in POLICIES}
    if len(rows)!=len(expected_pairs):raise ValueError("Incomplete row count")
    seen=set(); groups=defaultdict(list); legal_count=0
    for row in rows:
        key=(row["tag"],row["world_seed"],row["family"],row["policy"])
        if key not in expected_pairs or key in seen:raise ValueError("Invalid/duplicate rollout")
        seen.add(key);tag,seed,family,policy=key
        wbuild=maze_world if tag=="unseen-maze" else m4_world
        w=wbuild(seed,family)
        fname=f"{tag}-{seed}-{family}-{policy}"
        receipt=read_json(root/"rollouts"/(fname+".json"))
        if any(receipt.get(k)!=v for k,v in
               (("protocol",PROTOCOL),("synthetic_only",True),("tag",tag),("world_seed",seed),
                ("family",family),("policy",policy),("window",TURNS),("budget",BUDGET))):
            raise ValueError("Rollout identity mismatch")
        if receipt["geometry_sha256"]!=geometry_sha(w):raise ValueError("World geometry drift")
        rng=random.Random(seed*977+FAMILIES.index(family)*71)
        if not receipt["events"] or len(receipt["events"])>TURNS:raise ValueError("Event budget")
        for event in receipt["events"]:
            if w.state_sha256()!=event["before_sha256"] or hashlib.sha256(w.observe().tobytes()).hexdigest()!=event["obs_sha256"]:
                raise ValueError("State/observation replay failure")
            legal=w.legal()
            if event["legal"]!=[ACTIONS[a] for a in legal]:raise ValueError("Legal menu replay failure")
            if policy=="oracle":expected=oracle(w)
            elif policy=="novel":expected=novel(w,rng)
            elif policy=="wall":expected=choose_wall(w,rng)
            elif policy=="random":expected=choose_random(w,rng)
            else:
                model=(old if policy.startswith("m3-") else new)[policy[3:]]
                with torch.inference_mode():
                    logits=model(torch.from_numpy(w.observe()[None,:]))[0]
                expected=max(legal,key=lambda a:(float(logits[a]),-a))
            if event["action"]!=ACTIONS[expected]:raise ValueError("Original model/control decision drift")
            got=w.apply(int(expected))
            if got!={k:event[k] for k in got}:raise ValueError("Applied action/host receipt drift")
            legal_count+=1
            if w.stopped and event is not receipt["events"][-1]:
                raise ValueError("Actions after STOP")
        actual_stats={"final_sha256":w.state_sha256(),"cells":int(w.visited.sum()),
                      "branches":w.branch_nodes(),"retraces":w.revisit_landings,
                      "dead_ends":w.dead_ends,"steps":len(w.actions),
                      "early_stop":bool(w.actions and w.actions[-1]=="STOP")}
        if any(receipt.get(k)!=v or row.get(k)!=v for k,v in actual_stats.items() if k!="final_sha256"):
            raise ValueError("Final state or rollout row drift")
        if receipt["final_sha256"]!=actual_stats["final_sha256"]:raise ValueError("Terminal state mismatch")
        im=np.asarray(Image.open(root/"rollouts"/(fname+".png")).convert("RGB"))
        if not np.array_equal(im,np.asarray(raster(w,2))):raise ValueError("Original PNG pixels mismatch")
        groups[(policy,"maze" if tag=="unseen-maze" else "m4_ood")].append(row)
    if seen!=expected_pairs:raise ValueError("Incomplete evidence")
    for key,rows_ in groups.items():
        policy,group=key
        aggregate=report["aggregate"][policy][group]
        if aggregate["n"]!=len(rows_) or aggregate["early_stop"]!=sum(r["early_stop"] for r in rows_):
            raise ValueError("Aggregate n/STOP mismatch")
        for field,col in (("cells_mean","cells"),("branches_mean","branches"),("retraces_mean","retraces")):
            if aggregate[field]!=round(statistics.mean(r[col] for r in rows_),3):
                raise ValueError("Aggregate metrics mismatch")
    return {"protocol":PROTOCOL,"rollouts_replayed":len(seen),"legal_selected_actions_recomputed":legal_count,
            "pixels_verified":len(seen),"new_checkpoint_sha256":expected_new,
            "manifest_members":len(manifest),"optimizer_updates":report["optimizer_updates"],
            "source_note":"Research-only synthetic W0 host; no native/FINAL promotion"}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--m3-checkpoints",type=Path,required=True)
    args=ap.parse_args();print(json.dumps(audit(args.input,args.m3_checkpoints),indent=2,sort_keys=True))
if __name__=="__main__":main()
