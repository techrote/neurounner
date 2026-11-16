#!/usr/bin/env python3
"""LFM350 game-policy research: genuine bounded rank-8 LoRA on synthetic CyberSand W0.
No game implementation changes, no native material writing, no production provider.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import time

MODEL = "LiquidAI/LFM2.5-350M"
REVISION = "9921cd4d82c0e3986887aa4b376f9446729ef8a3"
W0_COMMIT = "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
W0_BLOB = "8979b6f0296e81051194bacc907992d38e006605"
SCHEMA = "driv3r.lfm350.game-policy/1"
ACTIONS = ("STRAIGHT", "LEFT", "RIGHT", "BRANCH", "RETRACE", "STOP")
FAMILIES = ("greebles", "pipework", "architecture")
TRAIN_SEEDS = tuple(range(3101, 3113))
DEV_SEEDS = (9103, 9119, 9137)
EVAL_SEEDS = (9503, 9521)
MAX_STEPS = 24
STOP = False

def stop_signal(_sig, _frame):
    global STOP
    STOP = True

def check(ok, explanation):
    if not ok:
        raise ValueError(explanation)

def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def sha(path):
    return sha_bytes(Path(path).read_bytes())

def save_json(path, obj):
    path = Path(path)
    check(not path.exists(), "immutable evidence already exists: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n", encoding="utf-8")

def load_w0(path):
    path = Path(path)
    # Validate exact source *blob* as well as original source SHA in workflow.
    import subprocess
    blob = subprocess.check_output(["git", "hash-object", str(path)], text=True).strip()
    check(blob == W0_BLOB, "W0 original host blob mismatch")
    spec = importlib.util.spec_from_file_location("lfm350_game_policy_w0", path)
    check(spec is not None and spec.loader is not None, "W0 module absent")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check(tuple(module.ACTIONS) == ACTIONS, "W0 action vocabulary differs")
    return module

def nearby(obs, turn):
    # W0 x-east, y-south, 5x5 local pixels. '+'/ '#' represent visited / blocked.
    vectors = ((1, 0), (0, 1), (-1, 0), (0, -1))
    h = (obs["heading"] + turn) % 4
    dx, dy = vectors[h]
    return obs["local_5x5"][2 + dy][2 + dx]

def teacher(obs):
    """Synthetic local policy with obstacle/visited/protected-cell and motif habits."""
    legal = set(obs["legal"])
    forward = [a for a,t in (("STRAIGHT",0),("LEFT",-1),("RIGHT",1))
               if a in legal and nearby(obs,t) == "."]
    available = [a for a in ("STRAIGHT","LEFT","RIGHT") if a in legal]
    family, tick = obs["family"], obs["tick"]
    if family == "greebles":
        if "BRANCH" in legal and tick % 4 == 0:
            return "BRANCH"
        preference = ("RIGHT","LEFT","STRAIGHT") if tick % 2 == 0 else ("LEFT","RIGHT","STRAIGHT")
    elif family == "pipework":
        if "BRANCH" in legal and tick % 9 == 8:
            return "BRANCH"
        preference = ("STRAIGHT","RIGHT","LEFT")
    else:
        preference = ("STRAIGHT","LEFT","RIGHT") if tick % 8 < 6 else ("LEFT","STRAIGHT","RIGHT")
    return next((a for a in preference if a in forward),
                next((a for a in preference if a in available),
                     "RETRACE" if "RETRACE" in legal else "STOP"))

def prompt_for(obs):
    # Inference and supervision use the exact same one-digit boundary.
    clean = {k: obs[k] for k in ("x","y","heading","local_5x5","legal","last_actions","family","tick")}
    return ("CyberSand synthetic local worm controller. Obstacles #, old trail +, cursor @, free ."
            "\nOnly the host applies legal actions. Aim for " + clean["family"]
            + " lines, turns, branches and recovery."
            "\n0=STRAIGHT 1=LEFT 2=RIGHT 3=BRANCH 4=RETRACE 5=STOP"
            "\nstate=" + canonical(clean) + "\nchoice=")

def collect_world(w0, seed, family, steps=18):
    examples = []
    def choose(obs):
        act = teacher(obs)
        check(act in obs["legal"], "teacher attempted illegal action")
        examples.append({"seed": seed, "family": family, "tick": obs["tick"],
                         "observation": obs, "action": act, "origin": "W0-observed"})
        return act
    record = w0.run(seed, family=family, mode="callback", steps=steps, selector=choose)
    check(record["metrics"]["invalid_raw_choices"] == 0, "W0 teacher host mismatch")
    return examples

def dataset(w0):
    # Counterfactual themes keep the exact same local geometry, history and legal menu.
    train, dev = [], []
    for seeds, output in ((TRAIN_SEEDS, train), (DEV_SEEDS, dev)):
        for seed in seeds:
            for family in FAMILIES:
                for example in collect_world(w0, seed, family):
                    output.append(example)
                    if example["tick"] % 4 == 0:
                        for altered_family in FAMILIES:
                            if altered_family == family:
                                continue
                            obs = dict(example["observation"], family=altered_family)
                            output.append({"seed":seed,"family":altered_family,
                                           "tick":obs["tick"],"observation":obs,
                                           "action":teacher(obs),
                                           "origin":"explicit-synthetic-family-counterfactual"})
    # The upstream W0 generator may yield identical local states in different worlds.
    # World identity groups, not individual duplicate prompt strings, are split.
    check({e["seed"] for e in train}.isdisjoint({e["seed"] for e in dev}), "world leakage")
    for e in train + dev:
        check(e["action"] in e["observation"]["legal"], "illegal training target")
    pairs, paired = [], set()
    for e in train:
        if e["tick"] % 4 != 0:
            continue
        key = (e["seed"], e["tick"], canonical({k:v for k,v in e["observation"].items() if k != "family"}))
        if key in paired:
            continue
        a = dict(e["observation"], family="greebles")
        b = dict(e["observation"], family="pipework")
        ta, tb = teacher(a), teacher(b)
        if ta != tb and ta in a["legal"] and tb in b["legal"]:
            pairs.append(({"seed":e["seed"],"observation":a,"action":ta},
                          {"seed":e["seed"],"observation":b,"action":tb}))
            paired.add(key)
    check(len(pairs) >= 24, "not enough genuine W0 matched-geometry motif contrasts")
    return train, dev, pairs

def id_map(tokenizer):
    ids = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(6)]
    check(all(len(v) == 1 for v in ids) and len({v[0] for v in ids}) == 6,
          "six distinct one-token action digits required")
    return [v[0] for v in ids]

def score(model, tokenizer, ids, obs):
    import torch
    input_ids = tokenizer(prompt_for(obs), return_tensors="pt", add_special_tokens=False)["input_ids"]
    check(4 < input_ids.shape[1] <= 256, "observation token budget exceeded")
    logits = model(input_ids=input_ids, use_cache=False).logits[0,-1].float()
    selection = logits.index_select(0, torch.tensor(ids, dtype=torch.long))
    check(torch.isfinite(selection).all().item(), "nonfinite action scores")
    return selection

def model_probe(model, tokenizer, ids, examples, limit=18):
    import torch
    model.eval()
    answers = []
    with torch.inference_mode():
        for e in examples[::max(1,len(examples)//limit)][:limit]:
            logits = score(model, tokenizer, ids, e["observation"])
            raw = ACTIONS[int(logits.argmax().item())]
            valid = e["observation"]["legal"]
            chosen = max(valid, key=lambda a: float(logits[ACTIONS.index(a)]))
            answers.append({"world_seed":e["seed"],"family":e["family"],
                            "tick":e["tick"],"observed_origin":e["origin"],
                            "teacher":e["action"],"unmasked_argmax":raw,
                            "raw_legal":raw in valid,"legal_choice":chosen,
                            "teacher_match":chosen==e["action"],
                            "six_logits":[float(v) for v in logits]})
    return {"cases": answers,"teacher_matches":sum(x["teacher_match"] for x in answers),
            "unmasked_illegal":sum(not x["raw_legal"] for x in answers),
            "n":len(answers)}

def rollouts(w0, model, tokenizer, ids, output, stage):
    import torch
    model.eval()
    episodes=[]
    for seed, family in zip(EVAL_SEEDS, ("greebles","pipework")):
        scores=[]
        def choose(obs):
            with torch.inference_mode():
                z = score(model,tokenizer,ids,obs)
            raw = ACTIONS[int(z.argmax().item())]
            legal = obs["legal"] # STOP always remains available and sampleable
            selected = max(legal,key=lambda a:float(z[ACTIONS.index(a)]))
            scores.append({"tick":obs["tick"],"unmasked_argmax":raw,
                           "unmasked_legal":raw in legal,"selected":selected,
                           "candidate_logits":[float(x) for x in z]})
            return selected
        record=w0.run(seed,mode="callback",family=family,steps=16,selector=choose)
        check(len(record["events"])==len(scores), "score/host receipt mismatch")
        check(all(e["raw_legal"] and e["raw_choice"]==s["selected"]
                  for e,s in zip(record["events"],scores)), "host refused model action")
        # Exact deterministic replay of all model-produced legal actions.
        replay=w0.run(seed,mode="replay",family=family,steps=16,
                      raw_actions=[s["selected"] for s in scores])
        original=w0.png_bytes(record)
        check(w0.png_bytes(replay)==original, "W0 source replay pixel mismatch")
        file=output/f"{stage}-{family}-{seed}.png"
        check(not file.exists(), "original PNG path already exists")
        file.write_bytes(original)
        save_json(output/f"{stage}-{family}-{seed}.json", {"world":record,"scores":scores})
        episodes.append({"family":family,"seed":seed,"decisions":len(scores),
                         "png":file.name,"png_sha256":sha_bytes(original),
                         "unique_cells":record["metrics"]["unique_trail_cells"],
                         "branches":record["metrics"]["branches"],
                         "retraces":record["metrics"]["retraces"],
                         "stopped":record["metrics"]["terminated_by_stop"],
                         "raw_illegal":sum(not c["unmasked_legal"] for c in scores)})
    if stage=="stock":
        for seed,family in zip(EVAL_SEEDS,("greebles","pipework")):
            for control in ("random","wall","fidget"):
                rec=w0.run(seed,mode=control,family=family,steps=16)
                file=output/f"control-{control}-{family}-{seed}.png"
                file.write_bytes(w0.png_bytes(rec))
                save_json(output/f"control-{control}-{family}-{seed}.json",rec)
    return episodes

def base_fingerprint(model):
    """Fingerprint identical pretrained tensors in stock and PEFT-wrapped LFM350."""
    names=[]
    for n,p in model.named_parameters():
        if "lora_" in n:
            continue
        # PEFT prefixes the same full pretrained model by "base_model.model.".
        # Canonicalize its name so an unadapted stock control is also audited.
        stem=n.removeprefix("base_model.model.")
        if stem.endswith("weight") and (stem.startswith("model.embed_tokens")
                                       or stem.startswith("model.layers.0.")):
            names.append((stem,sha_bytes(p.detach().cpu().reshape(-1)[:1024].numpy().tobytes())))
    check(names, "frozen reference fingerprint missing")
    return names

def train(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from huggingface_hub import HfApi, snapshot_download
    torch.set_num_threads(2)
    random.seed(3500316)
    torch.manual_seed(3500316)
    started=time.monotonic()
    output=args.output
    check(not output.exists(), "immutable job lease output already exists")
    output.mkdir(parents=True)
    save_json(output/"lease.json",{"schema":SCHEMA,"source_sha":os.environ.get("GITHUB_SHA"),
        "run_id":os.environ.get("GITHUB_RUN_ID"),"attempt":os.environ.get("GITHUB_RUN_ATTEMPT"),
        "model":MODEL,"revision":REVISION,"optimizer_max_steps":args.steps,
        "native":False,"production_provider":False,"d1_control":"not-run-matched"})
    w0=load_w0(args.w0)
    tr,dev,pairs=dataset(w0)
    corpus=[{"seed":e["seed"],"family":e["family"],"tick":e["tick"],
             "target":e["action"],"prompt":prompt_for(e["observation"]),
             "origin":e["origin"]} for e in tr]
    raw=("\n".join(canonical(e) for e in corpus)+"\n").encode()
    (output/"train.jsonl").write_bytes(raw)
    save_json(output/"data.json",{"train_worlds":list(TRAIN_SEEDS),
        "dev_worlds":list(DEV_SEEDS),"eval_worlds":list(EVAL_SEEDS),
        "train_rows":len(tr),"dev_rows":len(dev),"different_target_pairs":len(pairs),
        "train_sha256":sha_bytes(raw),"heldout_world_overlap":0,
        "synthetic_labels_not_native_or_human":True})
    info=HfApi().model_info(MODEL,revision=REVISION)
    check(info.sha==REVISION, "upstream pin resolved to different hash")
    snapshot=args.model_cache
    snapshot_download(MODEL,revision=REVISION,local_dir=snapshot,max_workers=2,
        allow_patterns=["*.safetensors","*.safetensors.index.json","config.json",
                        "generation_config.json","tokenizer*","special_tokens_map.json",
                        "added_tokens.json","vocab.*","merges.txt","README.md","LICENSE*"])
    inv={str(p.relative_to(snapshot)):sha(p) for p in sorted(snapshot.rglob("*"))
         if p.is_file() and ".cache" not in p.parts}
    check(any(k.endswith(".safetensors") for k in inv),"no pinned full model weights")
    save_json(output/"download.json",{"model":MODEL,"revision":info.sha,"sha256":inv})
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    code_ids=id_map(tokenizer)
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,
                                               trust_remote_code=False,dtype=torch.float32)
    check(model.config.model_type=="lfm2" and model.config.num_hidden_layers==16,
          "unexpected LFM350 architecture")
    # Only full-attention projections are trainable; conv, FFN, embeddings and LM head frozen.
    targets=[n for n,m in model.named_modules()
             if isinstance(m,torch.nn.Linear) and ".self_attn." in n
             and n.endswith((".q_proj",".v_proj"))]
    check(len(targets)==12 and len(set(targets))==12,
          "expected six LFM2 attention layers x q/v projection")
    model=get_peft_model(model,LoraConfig(task_type="CAUSAL_LM",r=8,lora_alpha=16,
               lora_dropout=0.0,bias="none",target_modules=targets))
    trainable=[p for p in model.parameters() if p.requires_grad]
    check(trainable and all("lora_" in n for n,p in model.named_parameters() if p.requires_grad),
          "non-LoRA parameter unexpectedly trainable")
    check(not any(p.requires_grad for n,p in model.named_parameters() if "lora_" not in n),
          "base model improperly unfrozen")
    frozen_before=base_fingerprint(model)
    adapter_before={n:sha_bytes(p.detach().cpu().numpy().tobytes())
                    for n,p in model.named_parameters() if "lora_" in n}
    save_json(output/"preflight.json",{"targets":targets,"rank":8,"alpha":16,
        "trainable_parameters":sum(p.numel() for p in trainable),
        "model_class":type(model).__name__,"token_ids":code_ids,
        "torch":torch.__version__,"precision":"float32","threads":2})
    pre=model_probe(model,tokenizer,code_ids,dev)
    save_json(output/"probe-stock.json",pre)
    stock=rollouts(w0,model,tokenizer,code_ids,output,"stock")
    optimizer=torch.optim.AdamW(trainable,lr=0.0007,weight_decay=0.01,betas=(0.9,0.95))
    rnd=random.Random(3500317)
    order=list(range(len(pairs)));rnd.shuffle(order)
    progress=[]
    for index in range(args.steps):
        if STOP or time.monotonic()-started>args.deadline-420:
            break
        row0,row1=pairs[order[index%len(order)]]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses=[]
        for row in (row0,row1):
            z=score(model,tokenizer,code_ids,row["observation"])
            target=ACTIONS.index(row["action"])
            loss=F.cross_entropy(z.unsqueeze(0),torch.tensor([target]))
            check(torch.isfinite(loss).item(),"nonfinite CE")
            (loss/2).backward()
            losses.append(float(loss.item()))
        grads=[p.grad for p in trainable if p.grad is not None]
        check(grads and any(torch.count_nonzero(g).item()>0 for g in grads),
              "no actual nonzero LoRA gradient")
        norm=float(torch.nn.utils.clip_grad_norm_(trainable,1.0).item())
        check(math.isfinite(norm),"nonfinite gradient norm")
        optimizer.step()
        record={"step":index+1,"pair_world":row0["seed"],"target_a":row0["action"],
                "target_b":row1["action"],"losses":losses,
                "actual_backward_passes":2,"grad_norm":norm}
        progress.append(record)
        with (output/"optimizer.jsonl").open("a") as f:f.write(canonical(record)+"\n")
        print(canonical({"real_adamw":record}),flush=True)
        if (index+1)%8==0 or index+1==args.steps:
            model.save_pretrained(output/f"adapter-step-{index+1:02d}",safe_serialization=True)
    check(progress,"no real optimizer updates")
    after_frozen=base_fingerprint(model)
    check(frozen_before==after_frozen,"frozen LFM350 base changed")
    adapter_after={n:sha_bytes(p.detach().cpu().numpy().tobytes())
                   for n,p in model.named_parameters() if "lora_" in n}
    check(any(adapter_before[k]!=adapter_after[k] for k in adapter_before),
          "LoRA numerically unchanged")
    if len(progress)%8 and len(progress)!=args.steps:
        model.save_pretrained(output/f"adapter-step-{len(progress):02d}",
                              safe_serialization=True)
    probe=model_probe(model,tokenizer,code_ids,dev)
    save_json(output/"probe-trained.json",probe)
    trained=rollouts(w0,model,tokenizer,code_ids,output,"trained")
    adapter_files={str(p.relative_to(output)):sha(p) for p in output.rglob("adapter-step-*/*")
                   if p.is_file()}
    check(adapter_files,"no actual adapter checkpoint artifact")
    result={"schema":SCHEMA,"source_sha":os.environ.get("GITHUB_SHA"),
        "status":"completed" if len(progress)==args.steps else "partial",
        "updates":len(progress),"backward_passes":2*len(progress),
        "requested_steps":args.steps,"train_rows":len(tr),"dev_rows":len(dev),
        "matched_contrast_pairs":len(pairs),
        "stock_probe_matches":pre["teacher_matches"],"trained_probe_matches":probe["teacher_matches"],
        "probe_n":probe["n"],"stock":stock,"trained":trained,
        "adapter_files_sha256":adapter_files,"model_revision":REVISION,
        "host":"synthetic-W0-NOT-native-Godot","native_material_receipts":0,
        "wall_seconds":round(time.monotonic()-started,3),
        "process_cpu_seconds":round(time.process_time(),3),
        "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "frozen_reference_unchanged":True,"adapter_changed":True,
        "d1_matched_control":"NOT EXECUTED; original Stage B d1 evidence is separate"}
    save_json(output/"summary.json",result)
    check(len(progress)==args.steps,"deadline stopped before full update budget")
    return result

def self_test(w0):
    assert teacher({"family":"pipework","tick":0,"heading":0,
        "local_5x5":[".....",".....","..@..",".....","....."],
        "legal":list(ACTIONS),"last_actions":[]})=="STRAIGHT"
    assert teacher({"family":"greebles","tick":0,"heading":0,
        "local_5x5":[".....",".....","..@..",".....","....."],
        "legal":list(ACTIONS),"last_actions":[]})=="BRANCH"
    mod=load_w0(w0)
    tr,dev,pairs=dataset(mod)
    assert len(tr)>100 and len(dev)>10 and len(pairs)>=24
    assert {r["seed"] for r in tr}.isdisjoint({r["seed"] for r in dev})
    assert all(row["action"] in row["observation"]["legal"] for row in tr+dev)
    print(canonical({"self_test":"PASS","train":len(tr),"dev":len(dev),"pairs":len(pairs)}))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--w0",type=Path,required=True)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--output",type=Path)
    p.add_argument("--model-cache",type=Path)
    p.add_argument("--steps",type=int,default=MAX_STEPS)
    p.add_argument("--deadline",type=int,default=1900)
    a=p.parse_args()
    if a.self_test:
        self_test(a.w0)
        return
    check(a.output is not None and a.model_cache is not None,"missing output/cache")
    check(a.steps==MAX_STEPS and 1500<=a.deadline<=2200,
          "this immutable wave uses exactly 24 AdamW steps")
    try:
        train(a)
    except BaseException as exc:
        a.output.mkdir(parents=True,exist_ok=True)
        save_json(a.output/"failure.json",{"kind":type(exc).__name__,
                  "message":str(exc),"source_sha":os.environ.get("GITHUB_SHA"),
                  "no_success_claim":True})
        raise

if __name__=="__main__":
    signal.signal(signal.SIGTERM,stop_signal)
    signal.signal(signal.SIGINT,stop_signal)
    main()
