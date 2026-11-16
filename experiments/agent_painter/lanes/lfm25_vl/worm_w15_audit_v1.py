#!/usr/bin/env python3
"""W15 independently replays original W0 PNGs and compares fixed tick-zero worlds."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from PIL import Image,ImageDraw

FAMILIES=("greebles","pipework","architecture")
ACTIONS=("STRAIGHT","LEFT","RIGHT","BRANCH","RETRACE","STOP")

def probabilities(scores,legal):
    logits=[scores["candidate_logits"][a]/2.5 for a in legal]
    peak=max(logits)
    exp=[math.exp(v-peak) for v in logits]
    return {a: .7*v/sum(exp)+.3/len(legal) for a,v in zip(legal,exp)}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--worm-root",type=Path,required=True)
    parser.add_argument("--seed",type=int,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.worm_root))
    import worm
    rows=[]
    first=[]
    images=[]
    for family in FAMILIES:
        d=args.root/family
        done=json.loads((d/"completed.json").read_text())
        trace=json.loads((d/"original-w0.json").read_text())
        moves=json.loads((d/"model-decisions.json").read_text())
        assert done["seed"]==args.seed and done["family"]==family
        assert done["status"]=="completed" and done["model_score_forwards"]==len(moves)
        assert done["model"]=="LiquidAI/LFM2.5-VL-450M"
        assert done["original_optimizer_steps"]==45 and done["new_optimizer_steps"]==0
        assert 1<=len(moves)<=24 and len(trace["events"])==len(moves)
        assert done["new_optimizer_steps"]==0 and not done["native_execution"]
        assert not done["sealed_FINAL_opened"] and done["synthetic_W0_24_turn_budget_only"]
        assert all("STOP" in x["legal"] and "STOP" in x["sampling_receipt"]["sampled_legal"]
                   and x["sampling_receipt"]["STOP_deferred_unless_forced"] is False for x in moves)
        assert all(e["raw_legal"] and e["raw_choice"]==x["raw_choice_selected_by_host_sampler"]
                   for e,x in zip(trace["events"],moves))
        png=(d/"original-w0.png").read_bytes()
        assert hashlib.sha256(png).hexdigest()==done["png_sha256"]
        replay=worm.run(seed=args.seed,mode="replay",family=family,steps=len(moves),
                        raw_actions=[e["raw_choice"] for e in trace["events"]])
        assert worm.png_bytes(replay)==png
        scores=moves[0]["model_six_code_scores"]
        first.append((family,trace["events"][0]["observation"],scores))
        images.append((family,Image.open(d/"original-w0.png").convert("RGB")))
        rows.append({"family":family,"seed":args.seed,"turns_executed":len(moves),
           "terminated_by_STOP":trace["metrics"]["terminated_by_stop"],
           "first_greedy_model_top":scores["six_code_top_action"],
           "first_raw_greedy_EOS":done["raw_greedy_probe"],
           "unique_cells":trace["metrics"]["unique_trail_cells"],
           "branches":trace["metrics"]["branches"],
           "png_sha256":done["png_sha256"]})
    fixed_keys=("x","y","heading","tick","local_5x5","legal","last_actions")
    freeze=lambda o:{k:o[k] for k in fixed_keys}
    assert len({json.dumps(freeze(obs),sort_keys=True) for _,obs,_ in first})==1, "worlds not paired"
    probs={family:probabilities(scores,obs["legal"]) for family,obs,scores in first}
    pairs=[]
    for a,b in itertools.combinations(FAMILIES,2):
        tvd=sum(abs(probs[a].get(k,0)-probs[b].get(k,0)) for k in ACTIONS)/2
        pairs.append({"families":[a,b],"TVD":round(tvd,8),
            "argmax_changed":next(x["first_greedy_model_top"] for x in rows if x["family"]==a)!=
                             next(x["first_greedy_model_top"] for x in rows if x["family"]==b)})
    canvas=Image.new("RGB",(1152,420),(18,23,31))
    d=ImageDraw.Draw(canvas)
    for i,(family,img) in enumerate(images):
        d.text((i*384+6,8),f"{family} / world {args.seed} / actual LFM",fill=(244,244,244))
        canvas.paste(img,(i*384,36))
    canvas.save(args.root/"w15-original-family-contact.png")
    report={"schema":"cybersand.worm.lfm-w15-first-state-family-audit/1",
       "status":"completed","world_seed":args.seed,"model":"full-LFM-W15-step45",
       "source_truth":"live-model-forward-to-host-W0-original-PNG-NOT-native",
       "stop_suppressed":False,"families":rows,"first_fixed_observation":freeze(first[0][1]),
       "paired_first_state_family_scores":pairs,
       "first_greedy_changes":sum(x["argmax_changed"] for x in pairs),
       "mean_TVD":round(sum(x["TVD"] for x in pairs)/len(pairs),8),
       "actual_model_score_forwards":sum(r["turns_executed"] for r in rows),
       "new_optimizer_updates":0,"native_execution":False}
    (args.root/"family-audit.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"seed":args.seed,"real_forwards":report["actual_model_score_forwards"],
          "family_argmax_changes":report["first_greedy_changes"],"mean_TVD":report["mean_TVD"]}))

if __name__=="__main__":
    main()
