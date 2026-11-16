#!/usr/bin/env python3
"""MZ-001: actual Pythia-70M LoRA training and host-legal CyberSand-style synthetic game policy.
The target is explicitly synthetic, not native Workbench, and never edits game files.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import resource
import struct
import time
import zlib
from pathlib import Path

MODEL_ID = "EleutherAI/pythia-70m-deduped"
MODEL_REV = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
ACTIONS = ("STRAIGHT", "LEFT", "RIGHT", "RETRACE", "BRANCH", "STOP")
CODES = (" A", " B", " C", " D", " E", " F")
DIRS = ((0,-1),(1,0),(0,1),(-1,0))

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda:f.read(1048576), b""):
            h.update(part)
    return h.hexdigest()

def save_png(path, world, scale=12):
    colours = {"EMPTY":(245,238,211),"ROCK":(75,79,88),"PROTECTED":(204,74,83),
               "INK":(38,88,139),"CURSOR":(246,186,52)}
    raw = bytearray()
    for y in range(world.size):
        row = bytearray()
        for x in range(world.size):
            p = (x,y)
            tile = "CURSOR" if p == world.pos else "INK" if p in world.ink else world.grid[y][x]
            row.extend(bytes(colours[tile]) * scale)
        for _ in range(scale):
            raw.extend(b"\x00" + row)
    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I",zlib.crc32(tag+body)&0xffffffff)
    image = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">2I5B",world.size*scale,world.size*scale,8,2,0,0,0))
    image += chunk(b"IDAT",zlib.compress(bytes(raw),9)) + chunk(b"IEND",b"")
    Path(path).write_bytes(image)

class World:
    def __init__(self, seed, size=13):
        self.seed, self.size = seed, size
        rng = random.Random(seed)
        self.grid = [["EMPTY"]*size for _ in range(size)]
        for y in range(size):
            for x in range(size):
                z = rng.random()
                self.grid[y][x] = "ROCK" if z < .10 else "PROTECTED" if z < .135 else "EMPTY"
        self.pos = (size//2,size//2)
        for y in range(size//2-1,size//2+2):
            for x in range(size//2-1,size//2+2): self.grid[y][x] = "EMPTY"
        self.facing = rng.randrange(4)
        self.path = [self.pos]
        self.seen = {self.pos}
        self.ink = {self.pos}
        self.branches = 0
        self.motif = ("CORRIDOR","ROOTS","TEXTURE")[seed%3]
        self.stopped = False
        self.trace = []

    def neighbor(self, delta):
        dx,dy = DIRS[(self.facing+delta)%4]
        return self.pos[0]+dx, self.pos[1]+dy

    def status(self, p):
        x,y = p
        if not (0<=x<self.size and 0<=y<self.size): return "WALL"
        if self.grid[y][x] != "EMPTY": return self.grid[y][x]
        return "VISITED" if p in self.seen else "FREE"

    def legal(self):
        if self.stopped: return ("STOP",)
        options = [name for delta,name in ((0,"STRAIGHT"),(-1,"LEFT"),(1,"RIGHT"))
                   if self.status(self.neighbor(delta)) in ("FREE","VISITED")]
        if len(self.path)>=2: options.append("RETRACE")
        if len(self.ink)>=4 and sum(self.status(self.neighbor(d))=="FREE" for d in (0,-1,1))>=2:
            options.append("BRANCH")
        options.append("STOP")
        return tuple(options)

    def prompt(self, step):
        ahead,left,right = [self.status(self.neighbor(d)) for d in (0,-1,1)]
        offered = " ".join(f"{CODES[ACTIONS.index(a)].strip()}={a}" for a in self.legal())
        return (f"Game worm motif={self.motif}; turn={min(step,32)}; ink={min(len(self.ink),32)}; "
                f"ahead={ahead}; left={left}; right={right}; branches={min(self.branches,9)}. "
                f"Legal: {offered}. Choice:")

    def teacher(self, step):
        legal = self.legal()
        free = {name:self.status(self.neighbor(delta))=="FREE"
                for delta,name in ((0,"STRAIGHT"),(-1,"LEFT"),(1,"RIGHT"))}
        if step>=22 or len(self.ink)>=30: return "STOP"
        if "BRANCH" in legal and step%6==5 and self.motif!="CORRIDOR": return "BRANCH"
        if "STRAIGHT" in legal and free["STRAIGHT"] and (step%5 not in (3,4) or self.motif=="CORRIDOR"):
            return "STRAIGHT"
        for name in (("LEFT","RIGHT") if step%2==0 else ("RIGHT","LEFT")):
            if name in legal and free[name]: return name
        if "STRAIGHT" in legal and free["STRAIGHT"]: return "STRAIGHT"
        return "RETRACE" if "RETRACE" in legal else "STOP"

    def apply(self, a):
        if a not in self.legal(): raise ValueError("Host-illegal action: "+a)
        prev = self.pos
        if a=="STOP": self.stopped=True
        elif a=="RETRACE":
            self.path.pop()
            self.pos = self.path[-1]
        else:
            if a=="BRANCH":
                offsets = [d for d in (-1,1,0) if self.status(self.neighbor(d))=="FREE"]
                if not offsets: raise AssertionError("Invalid host branch")
                delta=offsets[self.branches%len(offsets)]
                self.branches+=1
            else: delta={"STRAIGHT":0,"LEFT":-1,"RIGHT":1}[a]
            self.facing=(self.facing+delta)%4
            self.pos=self.neighbor(0)
            if self.status(self.pos) not in ("FREE","VISITED"):
                raise AssertionError("Wrote onto rock/protected/outside")
            self.path.append(self.pos)
            self.seen.add(self.pos)
            self.ink.add(self.pos)
        self.trace.append({"action":a,"from":prev,"to":self.pos,"painted_cells":len(self.ink)})

def episodes(seeds):
    result=[]
    for seed in seeds:
        w=World(seed)
        for step in range(24):
            prompt=w.prompt(step)
            a=w.teacher(step)
            if a not in w.legal(): raise AssertionError("Teacher selected illegal action")
            result.append((prompt,a,seed))
            w.apply(a)
            if w.stopped: break
    return result

def self_test():
    train=episodes(range(1000,1048))
    held=episodes(range(9000,9012))
    assert train and held and not set(x[2] for x in train).intersection(x[2] for x in held)
    w=World(9)
    try: w.apply("MAGIC")
    except ValueError: pass
    else: raise AssertionError("Illegal host action accepted")
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
        p=Path(tmp)/"sample.png"
        save_png(p,w)
        assert p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    print(json.dumps({"self_test":"PASS","training_examples":len(train),"heldout_examples":len(held)}))

def train(args):
    import torch
    from torch.nn import functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    started=time.monotonic()
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REV)
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=MODEL_REV,torch_dtype=torch.float32)
    code_ids=[tokenizer.encode(c,add_special_tokens=False) for c in CODES]
    if any(len(c)!=1 for c in code_ids): raise AssertionError(f"Action tokens must be atomic: {code_ids}")
    ids=torch.tensor([x[0] for x in code_ids],dtype=torch.long)
    if len(set(ids.tolist()))!=6: raise AssertionError("Action tokens collide")
    training=episodes(range(1000,1048))
    heldout=episodes(range(9000,9012))
    if set(x[2] for x in training).intersection(x[2] for x in heldout):
        raise AssertionError("Synthetic train/heldout world identity leakage")
    features=tokenizer([x[0] for x in training],padding=True,truncation=True,max_length=128,return_tensors="pt")
    labels=torch.tensor([ACTIONS.index(x[1]) for x in training])
    def permitted(prompt):
        menu=prompt.split("Legal: ",1)[1].split(". Choice:",1)[0]
        flags=[f"{CODES[i].strip()}={name}" in menu for i,name in enumerate(ACTIONS)]
        if not flags or not flags[-1]: raise AssertionError("Expected STOP in host menu")
        return flags
    train_allowed=torch.tensor([permitted(x[0]) for x in training],dtype=torch.bool)
    if not all(train_allowed[i,labels[i]] for i in range(len(training))):
        raise AssertionError("Synthetic teacher escaped host legal actions")
    model=get_peft_model(model,LoraConfig(r=4,lora_alpha=8,lora_dropout=0,
                          target_modules=["query_key_value"],task_type="CAUSAL_LM",bias="none"))
    def select(w,step,stock):
        ctx=model.disable_adapter() if stock else contextlib.nullcontext()
        encoded=tokenizer(w.prompt(step),return_tensors="pt",truncation=True,max_length=128)
        token_count=int(encoded["input_ids"].numel())
        t0=time.perf_counter()
        with torch.no_grad(),ctx:
            model.eval()
            scores=model(**encoded).logits[0,-1,ids]
        duration_ms=(time.perf_counter()-t0)*1000
        legal=w.legal()
        allowed=torch.tensor([a in legal for a in ACTIONS],dtype=torch.bool)
        chosen=int(scores.masked_fill(~allowed,-1e9).argmax())
        raw=int(scores.argmax())
        return ACTIONS[chosen],ACTIONS[raw] not in legal,duration_ms,token_count
    def rollouts(tag,stock):
        runs=[]; illegal_preferences=0; count=0; latencies=[]; input_tokens=[]
        for seed in range(9000,9006):
            w=World(seed)
            for step in range(24):
                if w.stopped: break
                a,illegal,ms,ntokens=select(w,step,stock)
                latencies.append(ms)
                input_tokens.append(ntokens)
                illegal_preferences+=int(illegal)
                count+=1
                w.apply(a)
            png=output/f"{tag}-seed{seed}.png"
            save_png(png,w)
            runs.append({"seed":seed,"decisions":len(w.trace),"painted_cells":len(w.ink),
                         "actions":[t["action"] for t in w.trace],"stopped":w.stopped,
                         "png":png.name,"png_sha256":sha(png)})
        ordered=sorted(latencies)
        def percentile(p):
            return round(ordered[min(len(ordered)-1,max(0,math.ceil(p*len(ordered))-1))],3)
        return {"episodes":runs,"decision_count":count,"unmasked_illegal_preferences":illegal_preferences,
                "host_executed_illegal_actions":0,
                "p50_forward_ms":percentile(.50),"p95_forward_ms":percentile(.95),
                "mean_input_tokens":round(sum(input_tokens)/len(input_tokens),2),
                "input_token_per_second":round(sum(input_tokens)/(sum(latencies)/1000),2),
                "mean_painted":sum(x["painted_cells"] for x in runs)/len(runs)}
    baseline=rollouts("stock",True)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr)
    counts=torch.bincount(labels,minlength=6)
    weights=(len(labels)/(6*counts.clamp_min(1))).pow(args.balance_gamma).clamp(max=1.6).float()
    losses=[]
    for step in range(args.steps):
        model.train()
        idx=torch.randint(0,len(training),(args.batch,))
        batch={k:v[idx] for k,v in features.items()}
        last=batch["attention_mask"].sum(1)-1
        logits=model(**batch).logits[torch.arange(args.batch),last][:,ids]
        scores=logits.masked_fill(~train_allowed[idx],-1e9)
        loss=F.cross_entropy(scores,labels[idx],weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step+1)%8==0: print(json.dumps({"step":step+1,"loss":round(losses[-1],5)}),flush=True)
    adapted=rollouts("adapted",False)
    def score_heldout(stock):
        correct=0; legal_correct=0; per_action={a:{"correct":0,"n":0,"legal_correct":0} for a in ACTIONS}
        for prompt,target,_ in heldout:
            ctx=model.disable_adapter() if stock else contextlib.nullcontext()
            with torch.no_grad(),ctx:
                model.eval()
                inputs=tokenizer(prompt,return_tensors="pt",truncation=True,max_length=128)
                scores=model(**inputs).logits[0,-1,ids]
            hit=ACTIONS[int(scores.argmax())]==target
            allowed=torch.tensor(permitted(prompt),dtype=torch.bool)
            legal_hit=ACTIONS[int(scores.masked_fill(~allowed,-1e9).argmax())]==target
            correct+=hit
            legal_correct+=legal_hit
            per_action[target]["n"]+=1
            per_action[target]["correct"]+=int(hit)
            per_action[target]["legal_correct"]+=int(legal_hit)
        balanced=sum(row["correct"]/row["n"] for row in per_action.values() if row["n"])/sum(row["n"]>0 for row in per_action.values())
        legal_balanced=sum(row["legal_correct"]/row["n"] for row in per_action.values() if row["n"])/sum(row["n"]>0 for row in per_action.values())
        return {"unmasked_action_accuracy":correct/len(heldout),"host_legal_accuracy":legal_correct/len(heldout),
                "balanced_action_accuracy":balanced,"host_legal_balanced_accuracy":legal_balanced,
                "per_action":per_action,"n":len(heldout)}
    heldscores={"stock":score_heldout(True),"adapted":score_heldout(False)}
    model.save_pretrained(output/"adapter",safe_serialization=True)
    files=[{"path":str(p.relative_to(output)),"sha256":sha(p),"size":p.stat().st_size}
           for p in sorted((output/"adapter").rglob("*")) if p.is_file()]
    result={"experiment":"MZ-001","source_sha":os.getenv("GITHUB_SHA","local"),
        "model":MODEL_ID,"model_revision":MODEL_REV,
        "architecture":"GPTNeoXForCausalLM","weights_training":"real LoRA qkv; cross-entropy over host-legal atomic action IDs; weighted by gamma",
        "synthetic_world":"13x13 generated toy grid, NOT Godot/native Workbench",
        "legal_actions":"host constrained; model original illegal preferences separately reported",
        "train_examples":len(training),"heldout_examples":len(heldout),
        "train_steps":args.steps,"batch":args.batch,"seed":args.seed,
        "learning_rate":args.lr,"balance_gamma":args.balance_gamma,
        "class_counts":dict(zip(ACTIONS,counts.tolist())),
        "class_weights":dict(zip(ACTIONS,weights.tolist())),
        "loss_first8_mean":sum(losses[:8])/min(len(losses),8),
        "loss_last8_mean":sum(losses[-8:])/min(len(losses),8),
        "action_tokens":dict(zip(ACTIONS,ids.tolist())),"loss_first":losses[0],"loss_last":losses[-1],
        "heldout":heldscores,"stock":baseline,"adapted":adapted,
        "adapter_files":files,"duration_seconds":round(time.monotonic()-started,2),
        "maxrss_kb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (output/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":"COMPLETE","source":result["source_sha"],
        "loss":[losses[0],losses[-1]],"heldout":heldscores,
        "seconds":result["duration_seconds"]}),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--output",default="mz001-out")
    ap.add_argument("--seed",type=int,default=13007)
    ap.add_argument("--steps",type=int,default=48)
    ap.add_argument("--batch",type=int,default=4)
    ap.add_argument("--lr",type=float,default=0.00025)
    ap.add_argument("--balance-gamma",type=float,default=0.0)
    arguments=ap.parse_args()
    self_test() if arguments.self_test else train(arguments)
