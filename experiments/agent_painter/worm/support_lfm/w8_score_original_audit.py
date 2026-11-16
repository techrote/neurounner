#!/usr/bin/env python3
"""Original W8 score-sampled W0 images: independent, no new inference/training/native."""
import argparse,copy,hashlib,importlib.util,json,math,subprocess,zipfile
from collections import Counter
from pathlib import Path,PurePosixPath

ZIP="027a7e92568a45f2276c19cf72edd4584a8cba56e837349fa5c7a3240ed0447c"
W0_SHA="81c3923cb38d7f196b99986806216388c34ebc27fd4976e168b3b370c25a323e"
W0_BLOB="8979b6f0296e81051194bacc907992d38e006605"
SOURCE="91ac02e3a3386774ba3ce1172bfc605c2896c58f"
PRODUCER="61730799eddfbd43eac5d7945da11d6f953b47e6"
MODEL="fc6221ca597f3315e4f82fc2df606783267b34ba"
ACTIONS=("STRAIGHT","LEFT","RIGHT","BRANCH","RETRACE","STOP")
FAMILIES=("greebles","pipework","architecture")
SEEDS=(3101,3109,3119)
PREFIX="w8-scored/"

def check(value,msg):
    if not value: raise ValueError(msg)
def sha(x): return hashlib.sha256(x).hexdigest()
def j(z,name): return json.loads(z.read(PREFIX+name))
def canonical(x): return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True)

def load_w0():
    path=Path(__file__).resolve().parents[1]/"common"/"worm.py"
    check(path.is_file() and not path.is_symlink(),"W0 file missing/symlink")
    check(sha(path.read_bytes())==W0_SHA,"W0 source bytes changed")
    check(subprocess.check_output(["git","hash-object",str(path)],text=True).strip()==W0_BLOB,"W0 blob changed")
    spec=importlib.util.spec_from_file_location("w8_score_audit_pinned_w0",path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def choose(p,legal,u):
    check(0<=u<1 and set(p)==set(legal) and len(legal)==len(set(legal)),"invalid host RNG or action support")
    v=0
    for a in legal:
        v+=p[a]
        if u<v: return a
    check(abs(v-1)<1e-6,"invalid probability mass")
    return legal[-1]

def check_step(row,event,policy):
    legal=row["legal"]; allowed=row["sampled_legal"]
    check(legal==event["observation"]["legal"] and allowed==
          ([a for a in legal if a!="STOP"] if len(legal)>1 else ["STOP"]),"illegal mask/STOP defer")
    check(row["sampler"]==policy and row["tick"]==event["observation"]["tick"],"wrong tick/scorer")
    check(row["model_fraction"]==.7 and row["temperature"]==2.5
          and row["STOP_deferred_unless_forced"] is True
          and row["W8_raw_greedy_not_overwritten"] is True,"wrong experiment identity")
    probs=row["sample_probs"]
    check(set(probs)==set(allowed) and all(math.isfinite(x) and x>=0 for x in probs.values())
          and abs(sum(probs.values())-1)<1e-6,"bad sample probabilities")
    uniform={a:1/len(allowed) for a in allowed}
    if policy=="learned":
        scores=row["scores"]
        check(scores["candidate_token_ids"]==dict(zip(ACTIONS,range(525,531))),"wrong LFM six digit tokens")
        logits=scores["candidate_logits"]
        check(set(logits)==set(ACTIONS) and all(math.isfinite(x) for x in logits.values()),"invalid logits")
        check(max(ACTIONS,key=lambda a:logits[a])==scores["six_code_top_action"]=="RETRACE",
              "unmasked W8 greedy is not RETRACE")
        high=max(logits[a] for a in allowed)
        weights={a:math.exp((logits[a]-high)/2.5) for a in allowed}
        norm=sum(weights.values())
        for a in allowed:
            expected=weights[a]/norm
            check(abs(expected-row["conditioned_model_probs"][a])<1e-6,"wrong temperature softmax")
            check(abs(.7*expected+.3*uniform[a]-probs[a])<1e-6,"wrong host/model mix")
    else:
        check(row["scores"] is None and all(abs(probs[a]-uniform[a])<1e-8 for a in allowed),
              "flat policy model-conditioned")
    check(choose(probs,allowed,row["host_u"])==row["choice"]==row["host_selected_action"]
          ==event["raw_choice"]==event["applied"] and event["raw_legal"] is True,
          "original sampled action differs from host and source CRN")
    return sum(abs(probs[a]-uniform[a]) for a in allowed)/2

def audit(path,out=None):
    check(sha(path.read_bytes())==ZIP,"wrong original scored-run ZIP SHA")
    worm=load_w0()
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        check(len(names)==65 and len(set(names))==65 and z.testzip() is None
              and all(n.startswith(("w8-scored/","diagnostics/")) and ".." not in PurePosixPath(n).parts
                      and not n.startswith("/") and "\\" not in n for n in names),"unsafe/invalid ZIP")
        lease=j(z,"lease.json"); summary=j(z,"summary.json"); readback=j(z,"model-readback.json")
        check(lease["experiment"]=="lfm-worm-w8-score-CRN-sampler-v1"
              and lease["evaluator_source"]==SOURCE and lease["producer_source"]==PRODUCER
              and lease["worlds"]==list(SEEDS) and lease["max_steps"]==16
              and lease["model_temperature"]==2.5 and lease["model_fraction"]==.7
              and lease["common_rng_independent_of_family"] is True
              and lease["new_optimizer_updates"]==0 and lease["native_execution"] is False
              and lease["sealed_FINAL_opened"] is False,"source/run/FINAL custody mismatch")
        c=readback["contract"]
        check(c["model_id"]=="LiquidAI/LFM2.5-VL-450M" and c["revision"]==MODEL
              and c["model_class"]=="Lfm2VlForConditionalGeneration","model revision substituted")
        check(summary["status"]=="complete" and summary["model_scored_decisions"]==144
              and summary["model_rollouts"]==summary["flat_rollouts"]==9
              and len(summary["results"])==18,"incomplete model-scored rollouts")
        summaries={(r["family"],r["seed"],r["policy"]):r for r in summary["results"]}
        check(len(summaries)==18 and set(summaries)==
              {(f,s,p) for f in FAMILIES for s in SEEDS for p in ("learned","flat")},
              "duplicated/missing family-world-policy result")
        sources={}; rows=[]; pngs=[]
        for family in FAMILIES:
            for seed in SEEDS:
                for policy in ("learned","flat"):
                    stem=f"gallery/{family}-{policy}-{seed}"
                    record=j(z,stem+".json"); score=j(z,stem+"-scored-actions.json")
                    png=z.read(PREFIX+stem+".png")
                    primary=summaries[(family,seed,policy)]
                    check(record["family"]==family and record["seed"]==seed
                          and record["mode"]=="callback" and record["size"]==96
                          and record["step_budget"]==record["steps_executed"]==16
                          and len(record["events"])==len(score)==16,"bad original W0 event structure")
                    check(sha(png)==primary["png_sha256"],"original PNG hash differs from producer receipt")
                    replay=worm.run(seed=seed,mode="replay",family=family,steps=16,
                                    raw_actions=[r["choice"] for r in score])
                    expected=dict(replay); observed=dict(record)
                    expected.pop("mode"); observed.pop("mode")
                    check(expected==observed,"W0 source-faithful trajectory mismatch")
                    check(worm.png_bytes(replay)==png,"W0 PNG pixels do not match original producer")
                    total_tv=sum(check_step(x,e,policy) for x,e in zip(score,record["events"]))
                    m=primary["metrics"]; h=record["metrics"]
                    check(m["unique_cells"]==h["unique_trail_cells"] and m["branches"]==h["branches"]
                          and m["retrace_events"]==h["retraces"]
                          and m["invalid_raw_count"]==h["invalid_raw_choices"],
                          "original metric/host receipt contradiction")
                    sources[(family,seed,policy)]=score
                    rows.append(dict(family=family,seed=seed,policy=policy,cells=m["unique_cells"],
                                     branches=m["branches"],revisits=m["revisits"],
                                     counts=dict(Counter(r["choice"] for r in score)),
                                     mean_TV_from_flat=total_tv/16,png_sha256=sha(png)))
                    pngs.append((f"{family}-{policy}-{seed}.png",png))
        matches=diff=conditional_diff=0
        for seed in SEEDS:
            for policy in ("learned","flat"):
                a=sources[("greebles",seed,policy)]
                for family in FAMILIES[1:]:
                    b=sources[(family,seed,policy)]
                    check([v["choice"] for v in a]==[v["choice"] for v in b]
                          and [v["host_u"] for v in a]==[v["host_u"] for v in b],
                          "family-dependent reported drawn policy")
            a=sources[("greebles",seed,"learned")]; b=sources[("greebles",seed,"flat")]
            for x,y in zip(a,b):
                check(x["host_u"]==y["host_u"],"nonmatched common random numbers")
                matches+=int(x["legal"]==y["legal"])
                diff+=int(x["choice"]!=y["choice"])
                conditional_diff+=int(x["legal"]==y["legal"] and x["choice"]!=y["choice"])
        check((matches,diff,conditional_diff)==(44,11,7),"three-world paired CRN attribution mismatch")
        lookup={(r["family"],r["seed"],r["policy"]):r for r in rows}
        cells=lambda p:[lookup[("greebles",s,p)]["cells"] for s in SEEDS]
        check(cells("learned")==[14,13,12] and cells("flat")==[13,11,12],"original coverage changed")
        report=dict(schema="worm-lfm-w8-score-CRN-independent-audit-v1",model_pin=MODEL,
                    original_run=43739243804,original_artifact_sha256=ZIP,producer_source=PRODUCER,
                    evaluator_source=SOURCE,W0_source_sha256=W0_SHA,original_pngs_replayed=18,
                    original_scorer_decisions=288,actual_model_scored_decisions=144,
                    independent_obstacle_worlds=3,worlds=list(SEEDS),family_labels=3,
                    per_world_model_cells=cells("learned"),per_world_flat_cells=cells("flat"),
                    per_world_coverage_delta=[1,2,0],matched_legal_ticks=matches,
                    different_actions_all_ticks=diff,different_actions_same_legal=conditional_diff,
                    model_raw_unmasked_top_RETRACE=144,model_pngs_distinct=3,
                    new_model_inferences=0,optimizer_steps_run=0,native_Godot_paintings=0,
                    sealed_FINAL_opened=False,
                    interpretation="Model-scored legal temperature/70:30 host stochastic policy, not greedy "
                                   "digit+EOS7 and not proven independent artistic gain; family labels duplicate worlds.")
        if out is not None:
            check(not out.exists(),"immutable output already exists")
            out.mkdir(parents=True); (out/"original-pngs").mkdir()
            for name,png in pngs: (out/"original-pngs"/name).write_bytes(png)
            (out/"report.json").write_text(canonical(report)+"\n")
            (out/"all-original-trials.json").write_text(canonical(rows)+"\n")
            paths=sorted(p for p in out.rglob("*") if p.is_file())
            (out/"SHA256SUMS.txt").write_text("".join(f"{sha(p.read_bytes())}  {p.relative_to(out)}\n"
                                                      for p in paths))
        return report

def self_test(path):
    with zipfile.ZipFile(path) as z:
        original=j(z,"gallery/greebles-learned-3101-scored-actions.json")[0]
        event=j(z,"gallery/greebles-learned-3101.json")["events"][0]
    check_step(original,event,"learned")
    mutations=(lambda a:a.update(choice="STOP"),
               lambda a:a["sample_probs"].update(STRAIGHT=.99),
               lambda a:a["scores"].update(six_code_top_action="LEFT"),
               lambda a:a.update(legal=["STOP"]),
               lambda a:a.update(host_u=1.2))
    for mutate in mutations:
        damaged=copy.deepcopy(original);mutate(damaged)
        try:check_step(damaged,event,"learned")
        except ValueError:pass
        else:raise AssertionError("adversarial sampler mutation escaped")
    print("5/5 sampler/legality adversarial tests PASS")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--artifact",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--self-test",action="store_true")
    opt=parser.parse_args()
    if opt.self_test:self_test(opt.artifact)
    print(canonical(audit(opt.artifact,opt.output)))
