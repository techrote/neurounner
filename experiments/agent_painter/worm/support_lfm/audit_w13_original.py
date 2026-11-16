#!/usr/bin/env python3
"""W13 immutable original-evidence replay; no models, optimizer or native host."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
import zipfile

VERSION = "cybersand.worm.lfm-w13-original-stop-audit/1"
RUN = 43771476464
SOURCE = "245290cab2352381dbd1eaf16374c577138b4ade"
W0 = "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
SUPPORT = "cc27b3206cb61f98432fa95b799c9b9ff508c2bc"
REVISION = "fc6221ca597f3315e4f82fc2df606783267b34ba"
ADAPTER = "886750e362f07e3540d164be82ffcdb3ebb1962f86164d57f9f4aed5a7bb9f39"
ARTIFACTS = {
    "alpha": ("0e69b23c100a00b438d4e2d9d07fa4550e276368e0ad4635e38a6c858f79b95e", ((6111, "greebles"), (6127, "pipework"))),
    "beta": ("36fa8876f249b0263d19dc40bea17731e8a4776da9cc82400189a876980380ff", ((6143, "architecture"), (6167, "greebles"))),
}
ACTIONS = ("STRAIGHT", "LEFT", "RIGHT", "BRANCH", "RETRACE", "STOP")
CONTROLS = ("random", "wall", "fidget")

def need(ok, reason):
    if not ok:
        raise ValueError(reason)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def sample(probs, legal, u):
    need(type(u) is float and 0 <= u < 1, "invalid host random draw")
    subtotal = 0.0
    for a in legal:
        subtotal += probs[a]
        if u < subtotal:
            return a
    return legal[-1]

def mixture(scores, legal):
    need(legal and len(set(legal)) == len(legal) and set(legal).issubset(ACTIONS), "invalid legal actions")
    need(set(scores) == set(ACTIONS) and all(math.isfinite(v) for v in scores.values()), "invalid raw scores")
    scaled = [scores[a] / 2.5 for a in legal]
    exp = [math.exp(x-max(scaled)) for x in scaled]
    conditioned = {a:x/sum(exp) for a,x in zip(legal, exp)}
    mixed = {a:0.7*conditioned[a]+0.3/len(legal) for a in legal}
    return conditioned, mixed

def trail_stats(trace):
    events, path = trace["events"], trace["trajectory"]
    need(len(path) == len(events)+1, "bad trajectory length")
    seen, curve = set(), {}
    revisits, reversals = 0, 0
    for idx,cell in enumerate(path):
        xy=tuple(cell)
        if xy in seen:
            revisits+=1
        seen.add(xy)
        if idx > 1 and path[idx] == path[idx-2]:
            reversals+=1
        if idx in (16, 32, 64, 128):
            curve[str(idx)] = len(seen)
    xs,ys=zip(*path)
    return {"unique_cells":len(seen), "revisits":revisits,
            "immediate_two_step_reversals":reversals,
            "branches":sum(e["applied"]=="BRANCH" for e in events),
            "unique_by_turn":curve, "bounds":[max(xs)-min(xs)+1,max(ys)-min(ys)+1],
            "host_action_counts":dict(Counter(e["applied"] for e in events))}

def audit_shard(path, shard, worm=None, gallery=None, check_zip_hash=True):
    expected, roster = ARTIFACTS[shard]
    if check_zip_hash:
        need(digest(path.read_bytes()) == expected, "original artifact SHA changed")
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        need(z.testzip() is None and len(names)==len(set(names)), "ZIP CRC or duplicate entries")
        need(all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts and "\\" not in n for n in names), "unsafe archive paths")
        def raw(s):
            return z.read(s)
        def obj(s):
            return json.loads(raw(s))
        pre=f"w13-fleet/{shard}/"
        summary=obj(pre+"summary.json")
        need(summary["schema"]=="lfm-worm-w13-two-vm-128-turn-four-resident-synthetic-v1" and summary["status"]=="completed", "incomplete W13 source")
        need(summary["shard"]==shard and str(summary["run_id"])==str(RUN) and summary["cpu_affinity_count"]==4, "wrong run/cpu")
        need(summary["model"]=="LiquidAI/LFM2.5-VL-450M" and summary["model_revision"]==REVISION and summary["adapter_sha256"]==ADAPTER, "model pin")
        need(summary["original_W10_source"]=="1e57a4df8d344b209a14df886dd898211dcc4918" and summary["original_W10_run"]==43754365984, "original optimizer pin")
        need(summary["new_optimizer_updates"]==0 and summary["native_execution"] is False and summary["sealed_FINAL_opened"] is False, "forbidden native/optimizer/FINAL")
        need(raw("diagnostics/evaluator-source.txt").decode().strip()==SOURCE and raw("diagnostics/w0-source.txt").decode().strip()==W0 and raw("diagnostics/support-source.txt").decode().strip()==SUPPORT, "source provenance")
        need(summary["independent_seeds"]==[s for s,_ in roster] and summary["actual_inference_model_forwards"]==summary["actual_W0_applied_decisions"]==256, "missing model score forwards")
        need(digest(raw(pre+"original-png-contact.png"))==summary["original_png_contact_sha256"], "contact hash")
        pids, leases = summary["distinct_loaded_pids"], summary["distinct_lease_ids"]
        need(len(set(pids))==len(set(leases))==2, "reused model PID or lease")
        resources=obj(pre+"resource-samples.json")
        alive=[s for s in resources if set(pids).issubset(s["alive"]) and all((s["rss_kib"].get(str(p)) or 0)>0 for p in pids)]
        need(len(alive)>=4 and all(0<(b["epoch_ns"]-a["epoch_ns"])/1e9<1.5 for a,b in zip(alive,alive[1:])), "not simultaneous resident processes")
        rows=[]
        for idx,(seed,family) in enumerate(roster):
            lease=leases[idx]
            base=pre+lease+"/"
            done, lease_meta, ready, model=(obj(base+s) for s in ("completed.json","lease.json","ready.json","model-readback.json"))
            trace=obj(base+"original-w0.json")
            decisions=obj(base+"model-decisions.json")
            pulses=[json.loads(line) for line in raw(base+"forward-pulse.jsonl").decode().splitlines()]
            png=raw(base+"original-w0.png")
            need(done["status"]=="completed" and done["lease_id"]==lease_meta["lease_id"]==ready["lease_id"]==lease, "lease mismatch")
            need(done["pid"]==lease_meta["runner_pid"]==ready["pid"]==pids[idx] and done["runner"]==summary["runner_name"], "PID/runner mismatch")
            need(done["seed"]==lease_meta["seed"]==ready["seed"]==trace["seed"]==seed and done["family"]==lease_meta["family"]==trace["family"]==family, "world mismatch")
            need(done["revision"]==REVISION and done["adapter_sha256"]==model["adapter_sha256"]==ready["adapter_sha256"]==ADAPTER, "adapter mismatch")
            need(model["model_class"]=="Lfm2VlForConditionalGeneration" and model["processor"]=="Lfm2VlProcessor" and model["original_adamw_steps"]==45 and model["new_optimizer_steps"]==0, "model readback")
            need(done["genuine_model_forward"] is True and done["model_score_forwards"]==done["host_applied_steps"]==len(decisions)==len(pulses)==len(trace["events"])==128, "genuine forward/event count")
            need(done["new_optimizer_steps"]==0 and done["native_execution"] is False and done["sealed_FINAL_opened"] is False, "worker safety violation")
            need(trace["engine"]=="python-stdlib-baseline-not-native" and trace["mode"]=="callback" and trace["size"]==96 and trace["step_budget"]==128, "W0 contract changed")
            need(digest(png)==done["png_sha256"] and png.startswith(b"\x89PNG\r\n\x1a\n"), "model PNG hash")
            need(all(p["pid"]==pids[idx] and p["lease_id"]==lease and p["tick"]==tick and p["real_score_forward"] is True and p["host_legal"] is True and p["model_scored_choice"]==decisions[tick]["raw_choice_selected_by_host_sampler"] for tick,p in enumerate(pulses)), "false model forwards")
            ns=[p["finished_utc_epoch_ns"] for p in pulses]
            need(all(a<b for a,b in zip(ns,ns[1:])) and summary["start_epoch_ns"]<=ns[0]<ns[-1]<=summary["end_epoch_ns"], "forged pulse clock")
            progress=[s["forward_pulse_bytes"][str(pids[idx])] for s in alive]
            need(max(progress)>min(progress), "no sampled forward progress")
            top=Counter()
            stop_cfs=[]
            flat_diffs=0
            stop_masked=0
            selected=[]
            for tick,(d,e) in enumerate(zip(decisions,trace["events"])):
                r=d["sampling_receipt"]
                legal=e["observation"]["legal"]
                sampled=r["sampled_legal"]
                a=d["raw_choice_selected_by_host_sampler"]
                need(d["tick"]==e["observation"]["tick"]==tick and d["legal"]==r["legal"]==legal, "observation/action misalignment")
                need(a==r["choice"]==e["raw_choice"]==e["applied"] and e["raw_legal"] is True, "illegal action or fallback")
                need(r["sampler"]=="learned" and r["temperature"]==2.5 and r["model_fraction"]==0.7 and r["STOP_deferred_unless_forced"] is True, "sampling changed")
                need(sampled==[x for x in ACTIONS if x in legal and (x!="STOP" or len(legal)==1)], "STOP mask incorrect")
                stop_masked+=int("STOP" in legal and "STOP" not in sampled)
                scores=d["model_six_code_scores"]
                logits=scores["candidate_logits"]
                need(scores["six_code_top_action"]==max(ACTIONS,key=lambda x:logits[x]), "raw model preference mismatch")
                top[scores["six_code_top_action"]]+=1
                conditioned,mixed=mixture(logits,sampled)
                for x in sampled:
                    need(math.isclose(conditioned[x],r["conditioned_model_probs"][x],rel_tol=1e-6,abs_tol=1e-7) and math.isclose(mixed[x],r["sample_probs"][x],rel_tol=1e-6,abs_tol=1e-7), "probability receipt mismatch")
                need(sample(mixed,sampled,r["host_u"])==a, "host sample not reproducible")
                flat_diffs+=int(sample({x:1/len(sampled) for x in sampled},sampled,r["host_u"])!=a)
                if "STOP" in legal:
                    stopped_legal=[x for x in ACTIONS if x in legal]
                    stop_cfs.append((tick,sample(mixture(logits,stopped_legal)[1],stopped_legal,r["host_u"])))
                selected.append(a)
            need(stop_masked==128, "128-turn episode was not STOP constrained")
            if worm is not None:
                replay=worm.run(seed=seed,family=family,mode="replay",steps=128,size=96,raw_actions=selected)
                for fld in ("events","trajectory","obstacles","branch_points","metrics"):
                    need(replay[fld]==trace[fld], "pinned W0 learned replay mismatch: "+fld)
                need(worm.png_bytes(replay)==png, "pinned W0 model PNG byte mismatch")
            if gallery is not None:
                gallery.mkdir(parents=True,exist_ok=True)
                (gallery/f"{shard}-{seed}-ORIGINAL-model-scored.png").write_bytes(png)
            controls={}
            for policy in CONTROLS:
                original=obj(base+f"control-{policy}.json")
                opng=raw(base+f"control-{policy}.png")
                receipt=done["original_matched_model_free_controls"][policy]
                need(original["mode"]==policy and original["seed"]==seed and original["family"]==family and original["step_budget"]==128 and original["steps_executed"]==128 and original["obstacles"]==trace["obstacles"], "unmatched original world control")
                need(digest(opng)==receipt["png_sha256"] and receipt["model_forwards"]==0, "W0-only control false model claim")
                if worm is not None:
                    rerun=worm.run(seed=seed,family=family,mode=policy,steps=128,size=96)
                    for fld in ("events","trajectory","obstacles","branch_points","metrics"):
                        need(rerun[fld]==original[fld], "pinned W0 control replay mismatch")
                    need(worm.png_bytes(rerun)==opng, "pinned W0 control PNG byte mismatch")
                if gallery is not None:
                    (gallery/f"{shard}-{seed}-ORIGINAL-W0-{policy}-control.png").write_bytes(opng)
                controls[policy]=trail_stats(original)
            metrics=trail_stats(trace)
            need(metrics["unique_cells"]==trace["metrics"]["unique_trail_cells"], "learned trail metrics")
            rows.append({"runner":summary["runner_name"],"pid":pids[idx],"lease":lease,"seed":seed,"family":family,
                         "original_model_png_sha256":digest(png),"real_model_score_forwards":128,
                         "first_forward_epoch_ns":ns[0],"last_forward_epoch_ns":ns[-1],
                         "raw_score_top_counts":dict(top),"raw_greedy_first_action":done["raw_greedy_probe"]["raw_action"],
                         "STOP_masked_host_choices":stop_masked,
                         "STOP_restored_same_state_one_step_choices":sum(x=="STOP" for _,x in stop_cfs),
                         "first_STOP_restored_counterfactual_tick":next((t for t,x in stop_cfs if x=="STOP"),None),
                         "same_state_flat_host_choice_differences":flat_diffs,
                         "model_host_sampled_trail":metrics,"original_W0_only_controls":controls})
        return {"shard":shard,"runner":summary["runner_name"],"sampled_live_window":[alive[0]["epoch_ns"],alive[-1]["epoch_ns"]],
                "peak_two_model_sampled_RSS_kib":summary["peak_aggregate_resident_rss_kib"],"ZIP_members":len(names),"episodes":rows}

def audit(alpha,beta,worm=None,gallery=None):
    shards=[audit_shard(alpha,"alpha",worm,gallery),audit_shard(beta,"beta",worm,gallery)]
    rows=[row for s in shards for row in s["episodes"]]
    need(len(rows)==len({r["lease"] for r in rows})==len({r["seed"] for r in rows})==4, "not four independent seeds/leases")
    need(shards[0]["runner"]!=shards[1]["runner"], "not two independent four-vCPU VMs")
    overlap=min(r["last_forward_epoch_ns"] for r in rows)-max(r["first_forward_epoch_ns"] for r in rows)
    sampled=min(s["sampled_live_window"][1] for s in shards)-max(s["sampled_live_window"][0] for s in shards)
    need(overlap>0 and sampled>0, "no four live advancing producer model episode overlap")
    return {"schema":VERSION,"producer_run":RUN,"producer_commit":SOURCE,"W0_sha":W0,
            "original_zip_sha256":{k:v[0] for k,v in ARTIFACTS.items()},
            "four_independent_loaded_resident_models":True,"runner_count":2,
            "genuine_original_model_score_forwards":512,"original_W0_PNGs_checked":16,
            "independent_pinned_W0_png_byte_replay":worm is not None,
            "four_forward_episode_overlap_seconds":round(overlap/1e9,3),
            "sampled_four_process_live_overlap_seconds":round(sampled/1e9,3),
            "original_STOP_legal_but_masked_choices":sum(r["STOP_masked_host_choices"] for r in rows),
            "same_state_counterfactual_STOP_choices_NOT_rollouts":sum(r["STOP_restored_same_state_one_step_choices"] for r in rows),
            "same_state_flat_choices_different_NOT_rollouts":sum(r["same_state_flat_host_choice_differences"] for r in rows),
            "six_score_top_distribution":dict(sum((Counter(r["raw_score_top_counts"]) for r in rows),Counter())),
            "native_128_permission":False,"STOP_autonomy_qualified":False,
            "optimizer_updates_added":0,"useful_learned_freeze":False,
            "model_weights_independently_reexecuted":False,"shards":shards}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--alpha",required=True,type=Path)
    p.add_argument("--beta",required=True,type=Path)
    p.add_argument("--w0-dir",type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--gallery",type=Path)
    args=p.parse_args()
    worm=None
    if args.w0_dir is not None:
        sys.path.insert(0,str(args.w0_dir))
        import worm as pinned_w0
        worm=pinned_w0
    report=audit(args.alpha,args.beta,worm,args.gallery)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="shards"},sort_keys=True))

if __name__=="__main__":
    main()
