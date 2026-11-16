#!/usr/bin/env python3
"""Independent source-pinned F3 original-artifact audit; NO model, optimizer or native run.

Primary producer attests four separately resident trained-model PIDs. This offline
auditor checks the archived four-lease evidence, redoes host W0 simulations and
byte-exact raster PNGs, and recomputes every host-seeded six-logit sampled action.
Overlapping inference intervals do not prove simultaneous individual forwards.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import zipfile
import zlib
from pathlib import Path

ORIGINAL_ZIP_SHA = "5eb64e5f7d5335809f631aa11caf9edbfcf1a67f546601d5d14e795cc61f7af6"
SOURCE = "e56e0aac49b310cc1e00aafe3ffeddb90eca6e38"
MODEL = "kandivault/sprocket-500m"
REV = "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
W0_COMMIT = "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
W0_FILE_SHA = "81c3923cb38d7f196b99986806216388c34ebc27fd4976e168b3b370c25a323e"
CHECKPOINT_SHA = "7328159322adb831f2a1b758a70740fbd39ee17be79592c660e78d48c36df98a"
MODEL_DOWNLOAD_SHA = "fddfb2a03a05d9c4884309ca69f4abc691098923c10662663b9114d34953fdac"
WORLD = {(0,8111,"greebles"),(0,8123,"pipework"),(1,8147,"architecture"),(1,8161,"greebles")}
ACTIONS = ("STRAIGHT","LEFT","RIGHT","BRANCH","RETRACE","STOP")

def require(ok, reason):
    if not ok: raise ValueError(reason)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def load_w0():
    # The live shared branch evolved after the original F3 producer's pinned W0.
    # Never replay F3 using that altered source; CI downloads the original
    # immutable W0 commit and verifies its complete SHA-256 separately.
    source = Path(os.environ.get("SPROCKET_F3_PINNED_W0",
          str(Path(__file__).resolve().parent.parent / "common" / "worm.py")))
    require(source.is_file() and digest(source.read_bytes()) == W0_FILE_SHA,
            "pinned common W0 source mismatch")
    spec = importlib.util.spec_from_file_location("original_pinned_worm_w0", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(tuple(module.ACTIONS)==ACTIONS, "W0 legal action contract drift")
    return module

def sample(logits, legal, draw):
    require(set(logits)==set(ACTIONS) and all(math.isfinite(v) for v in logits.values()),"nonfinite/missing logits")
    require(len(legal)==len(set(legal)) and legal and not(set(legal)-set(ACTIONS)), "host mask invalid")
    active=[a for a in ACTIONS if a in legal]
    scores=[logits[a]/2.25 for a in active]
    exps=[math.exp(v-max(scores)) for v in scores]
    total=sum(exps)
    probs={a:0. for a in ACTIONS}
    for action,weight in zip(active,exps): probs[action]=weight/total
    cumulative=0.
    choice=active[-1]
    for action in active:
        cumulative+=probs[action]
        if draw<cumulative:
            choice=action
            break
    return choice,probs

def host_draw(seed,tick):
    payload=f"612:sprocket:score-crn-s2:{seed}:{tick}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8],"big")/(1<<64)

def audit(archive, expected=ORIGINAL_ZIP_SHA, w0=None):
    require(digest(archive)==expected,"outer original ZIP digest mismatch")
    w0 = w0 or load_w0()
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        names=z.namelist()
        require(len(names)==len(set(names))==65,"original inventory mismatch")
        require(all(not n.startswith("/") and ".." not in Path(n).parts for n in names),"unsafe ZIP member")
        require(all(z.getinfo(n).file_size<=500000 for n in names),"oversize ZIP member")
        require(all(z.getinfo(n).CRC==(zlib.crc32(z.read(n))&0xffffffff) for n in names),"ZIP CRC mismatch")
        def obj(name):
            require(name in names,"missing ZIP member "+name)
            return json.loads(z.read(name))
        checks=0
        for slot in (0,1):
            prefix=f"fleet-{slot}/"
            declared=set()
            for line in z.read(prefix+"SHA256SUMS.txt").decode().splitlines():
                match=re.fullmatch(r"([a-f0-9]{64})  \./(.+)",line)
                require(match is not None,"invalid shard checksum line")
                name=prefix+match[2]
                require(name in names and name not in declared,"missing/duplicate checksum member")
                declared.add(name)
                require(digest(z.read(name))==match[1],"shard member digest mismatch: "+name)
                checks+=1
            require(declared==set(n for n in names if n.startswith(prefix) and n!=prefix+"SHA256SUMS.txt"),
                    "shard checksum inventory incomplete")
        require(checks==62,"shard checksum count")
        master=obj("four-way-audit.json")
        require(master["schema"]=="cybersand.worm.sprocket-4-live-long128-evidence/1"
                and master["source_sha"]==SOURCE and master["decision_bound"]==128
                and master["not_native"] and master["optimizer_steps_this_run"]==0
                and master["producer_failure_preserved"],"master provenance/authority")
        attest=master["original_worm_processes"]
        require(len(attest)==4,"missing four original model witnesses")
        world_ids,pids,leases,periods,rows=set(),set(),set(),[],[]
        for slot in (0,1):
            fleet=obj(f"fleet-{slot}/results/fleet.json")
            require(fleet["slot"]==slot and fleet["source_sha"]==SOURCE
                    and fleet["run_id"]=="43769025429" and fleet["model_id"]==MODEL
                    and fleet["model_revision"]==REV and fleet["w0_commit"]==W0_COMMIT,
                    "runner provenance")
            require(fleet["F1_qualified"] and fleet["F2_qualified"] and fleet["decision_bound"]==128
                    and fleet["maximum_actually_live_pids"]==2 and fleet["completed"]==2
                    and fleet["failed"]==fleet["active"]==fleet["queued"]==fleet["cancelled"]==0
                    and fleet["optimizer_steps"]==0 and fleet["native"] is False
                    and fleet["sealed_final_opened"] is False and len(fleet["workers"])==2,
                    "runner budgets/authority/residency")
            for worker in fleet["workers"]:
                seed,family,pid,lease=(worker[k] for k in ("seed","family","pid","lease"))
                world=(slot,seed,family)
                require(world in WORLD and world not in world_ids,"duplicate/mismatched World")
                require((slot,pid) not in pids and pid>0,"PID mismatch")
                require(lease not in leases and re.fullmatch(r"[A-Za-z0-9_-]{12,160}",lease),
                        "lease mismatch")
                world_ids.add(world);pids.add((slot,pid));leases.add(lease)
                require(worker in [v for s,v in attest if s==slot], "master PID witness mismatch")
                require(worker["real_trained_six_score_forwards"]==128
                        and worker["checkpoint_inventory_sha256"]==CHECKPOINT_SHA
                        and worker["model_download_receipt_sha256"]==MODEL_DOWNLOAD_SHA,
                        "missing original model forwards/pin")
                require(worker["start_utc"] < worker["inference_start_utc"] <
                        worker["inference_end_utc"] <= worker["end_utc"],"process times invalid")
                periods.append((worker["inference_start_utc"],worker["inference_end_utc"]))
                root=f"fleet-{slot}/results/{lease}/"
                identity,complete=obj(root+"identity.json"),obj(root+"complete.json")
                require(identity["pid"]==complete["pid"]==pid
                        and identity["lease"]==complete["lease"]==lease
                        and identity["seed"]==complete["seed"]==seed
                        and identity["family"]==complete["family"]==family,
                        "worker identity drift")
                require(complete["revision"]==REV and complete["model_id"]==MODEL
                        and complete["checkpoint_inventory_sha256"]==CHECKPOINT_SHA
                        and complete["real_trained_six_score_forwards"]==128
                        and complete["steps_bound"]==128 and complete["optimizer_steps"]==0
                        and complete["native"] is False and complete["source_faithful_W0_not_native"],
                        "worker model/native budget")
                base=root+"original-png-and-json/"
                traces=sorted(n for n in names if n.startswith(base) and n.endswith(".json"))
                require(len(traces)==4 and len([n for n in names if n.startswith(base) and n.endswith(".png")])==4,
                        "missing original PNG/JSON pairs")
                trained_path=next((n for n in traces if "sprocket-f1-" in n),None)
                require(trained_path is not None,"missing original trained trace")
                cells={}
                for path in traces:
                    trace=obj(path)
                    require(trace["seed"]==seed and trace["family"]==family
                            and trace["size"]==96 and trace["step_budget"]==128,
                            "cross-World original trace")
                    actions=[event["raw_choice"] for event in trace["events"]]
                    if path==trained_path:
                        reproduction=w0.run(seed,"replay",family,steps=128,raw_actions=actions)
                    else:
                        require(trace["mode"] in ("random","wall","fidget"),"invalid controller label")
                        reproduction=w0.run(seed,trace["mode"],family,steps=128)
                        cells[trace["mode"]]=trace["metrics"]["unique_trail_cells"]
                        require(trace["metrics"]==complete["controls"][trace["mode"]]["metrics"],
                                "control metrics receipt mismatch")
                    for key in ("seed","family","obstacles","trajectory","events",
                                "branch_points","metrics","steps_executed","step_budget","size"):
                        require(reproduction[key]==trace[key],"W0 state/event exact replay: "+key)
                    require(trace["steps_executed"]==128,"not 128 applied W0 events")
                    require(path[:-5]+".png" in names and z.read(path[:-5]+".png")==w0.png_bytes(trace),
                            "original W0 pixel-exact PNG replay")
                require(set(cells)=={"random","wall","fidget"},"incomplete model-free controls")
                trained=obj(trained_path);receipts=trained["score_receipts"]
                require(len(receipts)==len(trained["events"])==128,"missing model score receipts")
                differing=0;actions_count={a:0 for a in ACTIONS}
                for tick,(r,event) in enumerate(zip(receipts,trained["events"])):
                    obs=event["observation"]
                    require(r["tick"]==obs["tick"]==tick
                            and r["observation_sha256"]==digest(json.dumps(obs,sort_keys=True,
                                                     separators=(",",":")).encode()),
                            "score observation/tick divergence")
                    legal=[a for a in obs["legal"] if a!="STOP"] if tick<126 and len(obs["legal"])>1 else obs["legal"]
                    require(r["original_W0_legal"]==obs["legal"] and r["host_policy_legal"]==legal
                            and r["host_stop_deferred"]==(tick<126 and len(obs["legal"])>1),
                            "host STOP masking divergence")
                    require(r["model_forward"] and r["trained_adapter_active"]
                            and r["is_model_logit_conditioned"] and r["conditioned_six_action_scores"]
                            and not r["host_flat_no_model_forward"]
                            and not r["raw_greedy_EOS_completion"] and not r["is_eos5_completion"],
                            "model/host attribution divergence")
                    draw=host_draw(seed,tick)
                    require(r["host_uniform_draw"]==draw and r["temperature"]==2.25,"seeded host RNG")
                    selected,probs=sample(r["full_six_token_logits"],legal,draw)
                    flat,_=sample({a:0. for a in ACTIONS},legal,draw)
                    require(selected==r["selected_action"]==r["host_selected"]==r["actual_W0_applied"]==event["applied"]
                            and r["selected_host_legal"] and r["same_observation_flat"]==flat
                            and r["differs_from_flat"]==(selected!=flat),"model score selection/flat paired")
                    require(all(abs(probs[a]-r["masked_probs"][a])<1e-9 for a in ACTIONS),
                            "masked score probability divergence")
                    require(r["raw_unmasked_top_action"]==max(ACTIONS,key=lambda a:r["full_six_token_logits"][a]),
                            "raw argmax attribution")
                    actions_count[selected]+=1
                    differing+=int(selected!=flat)
                require(differing==complete["trained"]["differs_same_observation_flat"]
                        and trained["metrics"]==complete["trained"]["metrics"]
                        and complete["trained"]["actions"]==[e["applied"] for e in trained["events"]],
                        "trained receipt summary mismatch")
                rows.append({"slot":slot,"pid":pid,"lease":lease,"seed":seed,"family":family,
                             "model_forwards_attested":128,"unique_cells":trained["metrics"]["unique_trail_cells"],
                             "controls":cells,"branches":actions_count["BRANCH"],"retraces":actions_count["RETRACE"],
                             "revisits":trained["metrics"]["total_revisit_contacts"],"differs_flat":differing,
                             "original_png_sha256":digest(z.read(trained_path[:-5]+".png"))})
        require(world_ids==WORLD and len(pids)==len(leases)==len(rows)==4,"four resident identities incomplete")
        overlap=max(0.,min(t[1] for t in periods)-max(t[0] for t in periods))
        require(overlap>1 and abs(overlap-master["actual_four_MODEL_INFERENCE_interval_overlap_seconds"])<.01,
                "four-way episode overlap mismatch")
        return {"schema":"cybersand.worm.sprocket-support-F3-original-independent/1",
                "original_source":SOURCE,"original_archive_sha256":digest(archive),
                "model_revision":REV,"w0_commit":W0_COMMIT,"original_archive_zip_crc":len(names),
                "internal_sha256_verified":checks,"source_replayed_original_png_json_pairs":16,
                "producer_attested_six_logit_model_forwards":sum(r["model_forwards_attested"] for r in rows),
                "mean_trained_unique_cells":sum(r["unique_cells"] for r in rows)/4,
                "mean_control_unique_cells":{p:sum(r["controls"][p] for r in rows)/4
                                             for p in ("random","wall","fidget")},
                "model_vs_flat_changed_actions":sum(r["differs_flat"] for r in rows),
                "four_model_episode_overlap_seconds":round(overlap,3),
                "simultaneous_individual_model_forward_calls_proven":False,
                "new_model_inference_performed":False,"native":False,"optimizer_updates":0,
                "streams":rows}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("original_zip",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    report=json.dumps(audit(args.original_zip.read_bytes()),indent=2,sort_keys=True)+"\n"
    if args.output:args.output.write_text(report)
    print(report)

if __name__=="__main__":
    main()
