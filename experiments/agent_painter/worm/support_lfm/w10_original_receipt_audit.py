#!/usr/bin/env python3
"""Original W10 source-fenced audit; no inference, optimizer or native work."""
import argparse, collections, hashlib, json, math, pathlib, zipfile
ZIP_SHA="d060b7f15555b5830b96501cfc7b216b7479c7bd0ed0c8cccf7c06a96685fa1b"
SOURCE="1e57a4df8d344b209a14df886dd898211dcc4918"
MODEL="fc6221ca597f3315e4f82fc2df606783267b34ba"
W0="4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
SUPPORT="cc27b3206cb61f98432fa95b799c9b9ff508c2bc"
ACTIONS=("STRAIGHT","LEFT","RIGHT","BRANCH","RETRACE","STOP")
def check(v,msg):
    if not v: raise ValueError(msg)
def sha(b): return hashlib.sha256(b).hexdigest()
def read(z,n): return json.loads(z.read("worm-w10/"+n))
def raw(r):
    check(r["origin"] in ("W0-observed","counterfactual-no-world-or-native-receipt"),"source")
    check(r["expected"] in r["legal"] and len(r["legal"])==len(set(r["legal"])),"legal teacher")
    tok=r["token_ids"]; action=r["decoded_action"]
    syntax=len(tok)==2 and tok[1]==7 and action in ACTIONS and tok[0]==525+ACTIONS.index(action) and r["raw_text"]==str(ACTIONS.index(action))
    legal=syntax and action in r["legal"]
    check(r["raw_syntactic_eos"]==syntax and r["raw_legal"]==legal and r["valid_first_pass"]==legal and r["eos_terminated"]==(bool(tok) and tok[-1]==7),"raw tokens/EOS/host legality")
    check(r["raw_action"]==(action if syntax else "MODEL_REFUSED"),"model refusal concealed")
    return syntax,legal,r["raw_action"]==r["expected"]
def probe(d):
    check(d["source"]=="novel-WORM-DEVELOPMENT-not-FINAL" and len(d["samples"])==len(d["raw_first_passes"])==d["denominator"]==18,"held-out denominator")
    s=collections.Counter(); tops=[]; projected=[]; ids=set(); worlds=set()
    for x,r in zip(d["samples"],d["raw_first_passes"]):
        check(x["id"]==r["id"] and x["id"] not in ids and x["family"]==r["family"] and x["legal"]==r["legal"],"paired development")
        ids.add(x["id"])
        if x["origin"]=="W0-observed": worlds.add(x["world_group"])
        vals=x["scores"];check(len(vals)==6 and all(type(v) in (int,float) and math.isfinite(v) for v in vals),"bad scores")
        top=ACTIONS[max(range(6),key=lambda i:vals[i])]
        mask=max((a for a in ACTIONS if a in x["legal"]),key=lambda a:vals[ACTIONS.index(a)])
        check(x["raw_score_top"]==top and x["legal_projected_top"]==mask and x["teacher_raw_match"]==(top==x["expected"]) and x["teacher_legal_projected_match"]==(mask==x["expected"]),"fake argmax/teacher")
        s["score_teacher"]+=top==x["expected"];s["masked_teacher"]+=mask==x["expected"]
        v=x["synthetic_exclusion"]
        if v:
            check(v["synthetic_not_W0_observed"] and v["excluded"]==x["expected"] and len(x["legal"])>1,"synthetic DEV provenance")
            al=[a for a in x["legal"] if a!=v["excluded"]];vv=v["scores"]
            check(len(vv)==6 and all(type(a) in (int,float) and math.isfinite(a) for a in vv),"paired scores")
            t=ACTIONS[max(range(6),key=lambda i:vv[i])]
            p=max((a for a in ACTIONS if a in al),key=lambda a:vv[ACTIONS.index(a)])
            check(t==v["raw_score_top"] and p==v["new_legal_top"] and v["unmasked_score_top_changed"]==(t!=top),"synthetic masked action masquerades as learned")
            s["paired"]+=1;s["changed"]+=t!=top
        syntax,legal,teacher=raw(r)
        s["EOS7"]+=syntax;s["legal"]+=legal;s["raw_teacher"]+=teacher
        tops.append(top);projected.append(mask)
    check(len(worlds)==3 and d["score_argmax"]==dict(collections.Counter(tops)) and d["score_legal_top"]==dict(collections.Counter(projected)),"world/score aggregation")
    check((d["score_teacher"],d["legal_score_teacher"],d["raw_true_EOS7_syntax"],d["raw_host_legal"],d["raw_teacher"],d["paired_exclusions"],d["paired_unmasked_changes"])==(s["score_teacher"],s["masked_teacher"],s["EOS7"],s["legal"],s["raw_teacher"],s["paired"],s["changed"]),"probe totals inconsistent")
    return dict(s,top_actions=dict(collections.Counter(tops)),raw_actions=dict(collections.Counter(r["raw_action"] for r in d["raw_first_passes"])),actual_W0_worlds=len(worlds))
def audit(path,out):
    check(not out.exists(),"output must be new")
    check(sha(path.read_bytes())==ZIP_SHA,"original W10 artifact digest mismatch")
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        check(len(names)==len(set(names)) and z.testzip() is None and all(not n.startswith("/") and ".." not in pathlib.PurePosixPath(n).parts for n in names),"unsafe ZIP/CRC")
        summ,lease,man,pre=[read(z,n) for n in ("summary.json","lease.json","corpus-manifest.json","preflight.json")]
        check(lease["source_sha"]==summ["source_sha"]==pre["config"]["source_sha"]==SOURCE and summ["revision"]==pre["config"]["model"]["revision"]==MODEL and lease["W0_source"]==man["W0_source"]==W0 and lease["support_source"]==man["support_source"]==SUPPORT,"source/model/host pin")
        check(lease["fresh_from_pinned_full_model"] and lease["not_W8_resume"] and lease["native_execution"] is False and lease["provider_changed"] is False and lease["sealed_FINAL_opened"] is False,"owner custody")
        check(summ["status"]=="complete" and summ["actual_optimizer_updates"]==summ["requested_updates"]==45 and summ["real_backward_passes"]==90 and summ["frozen_base_unchanged"] and summ["adapter_changed"] and summ["gallery_model_rollouts"]==0 and summ["sealed_FINAL_opened"] is False and man["TRAIN_W0_original"]==man["TRAIN_synthetic_exclusions"]==15 and man["TRAIN_repeated_epochs"]==3 and man["DEV_optimizer_rows"]==0 and man["TRAIN_DEV_prompt_overlap"]==0,"optimizer/split custody")
        b=z.read("worm-w10/optimizer-pairs.jsonl");check(sha(b)==man["pair_sha256"]==summ["pair_sha256"],"pair SHA")
        rows=[json.loads(r) for r in b.splitlines()]
        check(len(rows)==30 and len(set(r["prompt"] for r in rows))==30,"TRAIN duplicates")
        targets=collections.Counter()
        for i in range(0,30,2):
            a,c=rows[i:i+2];p,q=json.loads(a["prompt"]),json.loads(c["prompt"])
            check(a["origin"]=="W0-observed-TRAIN" and c["origin"]=="synthetic-TRAIN-not-W0" and a["pair_id"]==c["pair_id"] and a["family"]==c["family"],"TRAIN pairing")
            check({k:v for k,v in p.items() if k!="legal"}=={k:v for k,v in q.items() if k!="legal"} and q["legal"]==[v for v in p["legal"] if v!=a["completion"]] and c["excluded_action"]==a["completion"],"synthetic world drift")
            targets[a["family"],a["completion"]]+=1
        check(targets==collections.Counter({(f,a):1 for f in ("greebles","pipework","architecture") for a in ACTIONS if a!="STOP"}),"TRAIN not stratified")
        order=read(z,"sampler-receipt.json")
        check(order["steps"]==45 and collections.Counter(order["pair_order"])==collections.Counter({i:3 for i in range(15)}),"three epoch sampler")
        check(order["pair_ids"]==[[rows[2*i]["id"],rows[2*i+1]["id"]] for i in order["pair_order"]],"sampler row pairing")
        prog=[json.loads(b) for b in z.read("worm-w10/progress.jsonl").splitlines()]
        check(len(prog)==45,"actual optimizer progress")
        for step,(rec,index) in enumerate(zip(prog,order["pair_order"]),1):
            a,c=rows[2*index:2*index+2]
            check(rec["global_step"]==step and rec["original_id"]==a["id"] and rec["synthetic_id"]==c["id"] and rec["two_genuine_backward_passes"]==2 and all(type(rec[k]) in (int,float) and math.isfinite(rec[k]) and rec[k]>0 for k in ("gradient_norm","loss_original","loss_synthetic")),"actual gradient/step receipt")
        cks={}
        for step in (15,30,45):
            folder="worm-w10/checkpoint-%06d/"%step;inv=json.loads(z.read(folder+"inventory.json"))
            check(len(inv)==12 and set(inv)=={n[len(folder):] for n in names if n.startswith(folder)}-{"inventory.json"},"checkpoint inventory")
            for name,v in inv.items():
                data=z.read(folder+name)
                check(len(data)==v["bytes"] and sha(data)==v["sha256"],"original checkpoint checksum")
            cks[str(step)]=inv["adapter/adapter_model.safetensors"]["sha256"]
        check(len(set(cks.values()))==3,"unchanged optimizer adapters")
        before,after=[read(z,n) for n in ("untuned-probes.json","trained-probes.json")]
        b,a=probe(before),probe(after)
        check([x["id"] for x in before["samples"]]==[x["id"] for x in after["samples"]] and summ["strict_raw_first_passes"]==after["raw_first_passes"] and summ["untuned_score_teacher"]==b["score_teacher"] and summ["trained_score_teacher"]==a["score_teacher"] and summ["trained_paired_unmasked_changes"]==a["changed"],"before/after original evidence mismatch")
        train=read(z,"train-pair-probe-trained.json")
        check(train==summ["stratified_training_pair_probe_after"] and train["paired_raw_argmax_changes"]==train["correct_both_of_pair"]==0,"TRAIN overfit cannot count as held-out generalization")
        result={"schema":"worm-lfm-support-w10-original/1","source_SHA":SOURCE,"original_artifact_SHA256":ZIP_SHA,"dataset_version":man["schema"],"W0_SHA":W0,"support_SHA":SUPPORT,"genuine_AdamW_updates":45,"backwards":90,"checkpoint_members":36,"adapter_hashes":cks,"before":b,"after":a,"new_optimizer":0,"model_PNGs":0,"learned_art_qualified":False,"finding":"EOS7 and legal raw action substantially improve; teacher accuracy and paired unmasked sensitivity do not."}
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    (out/"SHA256SUMS").write_text(sha((out/"report.json").read_bytes())+"  report.json\n")
    return result
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--artifact",required=True,type=pathlib.Path);ap.add_argument("--out",required=True,type=pathlib.Path)
    v=ap.parse_args();print(json.dumps(audit(v.artifact,v.out),sort_keys=True))
