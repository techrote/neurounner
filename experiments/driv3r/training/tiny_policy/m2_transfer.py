"""M2: frozen M1 policy heads on genuinely new synthetic CyberSand-like worlds.

Read-only M1 checkpoints; no optimizer, native action, or game-side edit.
"""
from __future__ import annotations
import argparse, hashlib, json, random, resource, statistics, time, traceback
from pathlib import Path
import numpy as np
from .core import ACTIONS, FAMILIES, World, teacher, choose_random, choose_wall, raster
from .pretrained_cpu import MODELS, SPLITS, text_observation

CASES = ((6101, "corridor"), (6131, "culdesac"), (6163, "protected_gate"), (6197, "branchyard"))
WINDOW = 48
ORIGIN = "7d3aadea30830ba5e44aea6bd6c71a8666923e7e"
RUN = 43790987589
assert not {s for s, _ in CASES}.intersection(sum(SPLITS.values(), []))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def world(seed, family, case):
    w = World(seed, family, budget=96)
    if case == "corridor":
        w.blocked[43:58,46] = True
        w.blocked[43:58,50] = True
        w.blocked[43,46:51] = w.blocked[57,46:51] = True
        w.blocked[51,46] = False
    elif case == "culdesac":
        w.blocked[46:50,46] = w.blocked[46:50,50] = True
        w.blocked[46,46:51] = w.blocked[49,46:51] = True
        w.blocked[49,48] = False
    elif case == "protected_gate":
        w.protected[46:52,50] = True
        w.protected[45,46:50] = True
        w.blocked[51,46:50] = True
        w.protected[51,47] = False
    elif case == "branchyard":
        w.blocked[44:56,45] = w.blocked[44:56,52] = True
        w.blocked[44,45:53] = w.blocked[55,45:53] = True
        w.blocked[48,45] = w.blocked[50,52] = False
        w.protected[49,51] = True
    else:
        raise ValueError("unknown case")
    w.blocked[48,48] = w.protected[48,48] = False
    assert w.legal() and 4 in w.legal()
    return w

def geometry_sha(w):
    return hashlib.sha256(w.blocked.tobytes() + w.protected.tobytes()).hexdigest()

def trace(w, case, name, selector):
    events = []
    rng = random.Random(w.seed * 451 + FAMILIES.index(w.family))
    initial = geometry_sha(w)
    for _ in range(WINDOW):
        legal = w.legal()
        if not legal: break
        record = {"before_sha256": w.state_sha256(),
                  "feature_sha256": hashlib.sha256(w.observe().tobytes()).hexdigest(),
                  "legal": [ACTIONS[i] for i in legal]}
        choice = int(selector(w, rng))
        if choice not in legal: raise ValueError("illegal selection")
        record.update(w.apply(choice))
        events.append(record)
        if w.stopped: break
    result = {"seed": w.seed, "family": w.family, "case": case, "policy": name,
              "budget": 96, "window": WINDOW, "geometry_sha256": initial,
              "events": events, "unique_cells": int(w.visited.sum()),
              "branches": w.branch_nodes(), "retraces": w.revisit_landings,
              "dead_end_ticks": w.dead_ends,
              "stop": bool(w.actions and w.actions[-1] == "STOP"),
              "window_exhausted": len(events) == WINDOW and not w.stopped,
              "final_sha256": w.state_sha256()}
    replay(result)
    return result

def replay(result):
    w = world(result["seed"], result["family"], result["case"])
    if geometry_sha(w) != result["geometry_sha256"] or result["budget"] != 96 or result["window"] != WINDOW:
        raise ValueError("wrong provenance or geometry")
    if len(result["events"]) > WINDOW: raise ValueError("unbounded evidence")
    for e in result["events"]:
        if w.state_sha256() != e["before_sha256"]: raise ValueError("state digest")
        if [ACTIONS[i] for i in w.legal()] != e["legal"]: raise ValueError("legal menu")
        if hashlib.sha256(w.observe().tobytes()).hexdigest() != e["feature_sha256"]:
            raise ValueError("observation digest")
        if e["action"] not in ACTIONS: raise ValueError("action")
        receipt = w.apply(ACTIONS.index(e["action"]))
        if receipt != {k: e[k] for k in receipt}: raise ValueError("action receipt")
    if (w.state_sha256() != result["final_sha256"] or
        int(w.visited.sum()) != result["unique_cells"] or
        w.branch_nodes() != result["branches"] or w.revisit_landings != result["retraces"] or
        result["stop"] != bool(w.actions and w.actions[-1] == "STOP") or
        result["window_exhausted"] != (len(result["events"]) == WINDOW and not w.stopped)):
        raise ValueError("terminal receipt")
    return w

def verify_original(p, key):
    report = json.loads((p/"report.json").read_text())
    hashes = json.loads((p/"manifest.json").read_text())
    assert (report["status"] == "COMPLETED" and
            report["protocol"] == "tiny-policy-frozen-pretrained-m1" and
            report["model_id"] == MODELS[key][0] and report["revision"] == MODELS[key][1] and
            report["head_optimizer_steps"] == 108 and report["backbone_optimizer_steps"] == 0 and
            report["split_seeds"] == SPLITS and report["action_vocabulary"] == list(ACTIONS))
    for rel, expected in hashes.items():
        if sha(p/rel) != expected: raise ValueError("M1 original manifest corruption: "+rel)
    return report, hashes

def gallery(out, case, receipts):
    from PIL import Image, ImageDraw
    policies = ("live", "frozen_initial", "swapped_theme", "teacher", "random", "wall")
    block = 96*3
    image = Image.new("RGB", (block*len(policies), (block+24)*len(FAMILIES)), "white")
    d = ImageDraw.Draw(image)
    for y, fam in enumerate(FAMILIES):
        for x, policy in enumerate(policies):
            r = receipts[(case,fam,policy)]
            at = x*block, y*(block+24)
            image.paste(raster(replay(r),3), (at[0],at[1]+24))
            d.text((at[0]+3,at[1]+4),f"{fam} {policy}: {r['unique_cells']}",fill="black")
    image.save(out/(case+"-comparison.png"))

def evaluate(key, original, out):
    import torch
    from transformers import AutoTokenizer, AutoModel
    from huggingface_hub import HfApi
    torch.set_num_threads(2)
    m1, hashes = verify_original(original, key)
    model_id, revision, license_text = MODELS[key]
    if HfApi().model_info(model_id,revision=revision).sha != revision:
        raise ValueError("HF revision drift")
    tokenizer = AutoTokenizer.from_pretrained(model_id,revision=revision,trust_remote_code=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModel.from_pretrained(model_id, revision=revision,
                   trust_remote_code=False, use_safetensors=key!="opt-125m")
    backbone.eval().requires_grad_(False)
    ck = torch.load(original/"trained-decision-head.pt", map_location="cpu",weights_only=True)
    if ck["model_id"] != model_id or ck["revision"] != revision or ck["identities"] != m1["splits"]:
        raise ValueError("M1 head provenance mismatch")
    mean, std = ck["mean"], ck["std"]
    dims = ck["head"]["weight"].shape
    assert dims[0] == 5 and dims[1] == mean.numel()
    head = torch.nn.Linear(dims[1],5)
    head.load_state_dict(ck["head"])
    head.eval().requires_grad_(False)
    timings = []
    def scores(w,swap=False):
        v = w.observe().copy()
        if swap:
            fi=(FAMILIES.index(w.family)+1)%3
            v[151:154]=0
            v[151+fi]=1
        tokens=tokenizer([text_observation(v,w.legal())],
                         return_tensors="pt",truncation=True,max_length=128)
        t=time.perf_counter()
        with torch.inference_mode():
            hidden=backbone(**tokens).last_hidden_state
            pos=int(tokens["attention_mask"].sum()-1)
            logits=head(((hidden[0,pos].float()-mean)/std).clamp(-8,8)).tolist()
        timings.append((time.perf_counter()-t)*1000)
        return logits
    def choose(w, scores_):
        return max(w.legal(), key=lambda a: (scores_[a], -a))
    receipts, rows = {}, []
    start=time.perf_counter()
    for seed, case in CASES:
        for family in FAMILIES:
            init = world(seed,family,case)
            frozen = scores(init)
            variants = (
                ("live",lambda w,r:choose(w,scores(w))),
                ("frozen_initial",lambda w,r:choose(w,frozen)),
                ("swapped_theme",lambda w,r:choose(w,scores(w,True))),
                ("teacher",teacher), ("random",choose_random), ("wall",choose_wall))
            for name, fn in variants:
                w=world(seed,family,case)
                evidence=trace(w,case,name,fn)
                receipts[(case,family,name)]=evidence
                path=out/"rollouts"
                path.mkdir(exist_ok=True)
                base=path/f"{case}-{seed}-{family}-{name}"
                base.with_suffix(".json").write_text(json.dumps(evidence,sort_keys=True)+"\n")
                raster(w,3).save(base.with_suffix(".png"))
                rows.append({"seed":seed,"case":case,"family":family,"policy":name,
                             "actions":len(evidence["events"]),"cells":evidence["unique_cells"],
                             "branches":evidence["branches"],"retraces":evidence["retraces"],
                             "deadends":evidence["dead_end_ticks"],"stop":evidence["stop"]})
        gallery(out,case,receipts)
    policies={}
    for p in ("live","frozen_initial","swapped_theme","teacher","random","wall"):
        group=[r for r in rows if r["policy"]==p]
        policies[p]={"n":len(group),
                     "mean_cells":round(statistics.mean(r["cells"] for r in group),2),
                     "mean_branches":round(statistics.mean(r["branches"] for r in group),2),
                     "mean_retraces":round(statistics.mean(r["retraces"] for r in group),2),
                     "early_stops":sum(r["stop"] for r in group),
                     "mean_actions":round(statistics.mean(r["actions"] for r in group),2)}
    result={"status":"COMPLETED","protocol":"tiny-policy-m2-stress-inference-v1",
            "source_M1":ORIGIN,"run_M1":RUN,"model":key,"model_id":model_id,
            "revision":revision,"license":license_text,"synthetic_only":True,
            "game_native":False,"optimizer_steps":0,"action_strings":list(ACTIONS),
            "cases":CASES,"host_budget":96,"bounded_prefix":WINDOW,
            "m1_checkpoint_sha256":sha(original/"trained-decision-head.pt"),
            "m1_report_sha256":sha(original/"report.json"),
            "original_files_verified":len(hashes),
            "single_forward_calls":len(timings),
            "single_forward_ms_median":round(statistics.median(timings),2),
            "single_forward_ms_p95":round(sorted(timings)[int((len(timings)-1)*.95)],2),
            "rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "seconds":round(time.perf_counter()-start,2),"policies":policies}
    (out/"rollout_rows.json").write_text(json.dumps(rows,indent=2,sort_keys=True)+"\n")
    (out/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True,choices=MODELS)
    ap.add_argument("--original",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()
    if a.out.exists(): ap.error("output already exists (no evidence overwrite)")
    a.out.mkdir(parents=True)
    try:
        print(json.dumps(evaluate(a.model,a.original,a.out),indent=2))
    except Exception:
        (a.out/"FAILED.txt").write_text(traceback.format_exc())
        raise
    finally:
        files={str(p.relative_to(a.out)):sha(p) for p in sorted(a.out.rglob("*")) if p.is_file()}
        (a.out/"manifest.json").write_text(json.dumps(files,indent=2,sort_keys=True)+"\n")

if __name__=="__main__": main()
