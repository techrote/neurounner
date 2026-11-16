"""Real pinned Mamba-130M -> frozen features -> trained game-action probe.

Only model-side research. World is synthetic, not native Godot.
No mocked inference: all features use the real pretrained Mamba weights.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import statistics
import time

from .world import ACTIONS,N,World,snapshots

MODEL_ID='state-spaces/mamba-130m-hf'
REV_PREFIX='1e76775'  # upstream main on 2024-03-06; full hash resolved and enforced

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='zoo-output')
    ap.add_argument('--train-seeds',type=int,default=8)
    ap.add_argument('--steps',type=int,default=14)
    ap.add_argument('--episode-steps',type=int,default=16)
    args=ap.parse_args()
    import torch
    from huggingface_hub import model_info
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from PIL import Image,ImageDraw
    torch.manual_seed(2007)
    torch.set_num_threads(2)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    source_sha=os.getenv('GITHUB_SHA','local-unverified')
    info=model_info(MODEL_ID)
    sha=info.sha
    if not sha.startswith(REV_PREFIX):
        raise RuntimeError(f'upstream model moved: expected {REV_PREFIX}, observed {sha}; refuse silent change')
    t0=time.monotonic()
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=sha,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=sha,
                                                 torch_dtype=torch.float32,
                                                 trust_remote_code=False)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    hidden=int(model.config.hidden_size)
    model_load_s=round(time.monotonic()-t0,3)
    feature_ms=[];cache={}
    def features(prompt):
        if prompt in cache:return cache[prompt]
        inputs=tokenizer(prompt,return_tensors='pt',truncation=True,max_length=128)
        start=time.perf_counter()
        with torch.inference_mode():
            state=model.backbone(input_ids=inputs['input_ids'],use_cache=False)
            ft=state.last_hidden_state[0,-1].float().detach().clone()
        feature_ms.append((time.perf_counter()-start)*1000)
        cache[prompt]=ft
        return ft
    training=snapshots(range(100,100+args.train_seeds),args.steps)
    dev=snapshots(range(900,903),args.steps)
    eval_seeds=list(range(1700,1703))
    X=torch.stack([features(p) for p,_,_,_ in training]); Y=torch.tensor([y for _,y,_,_ in training])
    V=torch.stack([features(p) for p,_,_,_ in dev]); T=torch.tensor([y for _,y,_,_ in dev])
    head=torch.nn.Sequential(torch.nn.LayerNorm(hidden),torch.nn.Linear(hidden,64),
                             torch.nn.GELU(),torch.nn.Linear(64,len(ACTIONS)))
    def accuracy(logits,labels):
        return round((logits.argmax(-1)==labels).float().mean().item(),4)
    with torch.no_grad():
        initial_acc=accuracy(head(V),T)
    head_initial=copy.deepcopy(head).eval()
    opt=torch.optim.AdamW(head.parameters(),lr=0.002,weight_decay=0.01)
    train_loss=[]
    head.train()
    for epoch in range(14):
        opt.zero_grad(set_to_none=True)
        logits=head(X)
        loss=torch.nn.functional.cross_entropy(logits,Y)
        loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.0)
        opt.step();train_loss.append(round(loss.item(),5))
    head.eval()
    with torch.inference_mode():
        train_acc=accuracy(head(X),Y); dev_acc=accuracy(head(V),T)
    ckpt=out/'mamba130_game_head.pt'
    torch.save(dict(model_id=MODEL_ID,revision=sha,source_sha=source_sha,
                    actions=ACTIONS,head_state_dict=head.state_dict(),
                    train_seeds=args.train_seeds,steps=args.steps,seed=2007),ckpt)
    ckpt_sha=hashlib.sha256(ckpt.read_bytes()).hexdigest()
    def trace(seed,method):
        w=World.make(seed);r=random.Random(seed+9099)
        actions=[];invalid=0;timing=[]
        for i in range(args.episode_steps):
            legal=w.legal()
            if method=='teacher': pick=w.teacher(r)
            elif method=='random':pick=r.choice(legal)
            elif method=='random_nonstop':
                candidates=[a for a in legal if a!='STOP']
                pick=r.choice(candidates or ['STOP'])
            else:
                start=time.perf_counter()
                with torch.inference_mode():
                    policy=head_initial if method=='untrained' else head
                    scores=policy(features(w.prompt()).unsqueeze(0))[0]
                    raw=ACTIONS[int(torch.argmax(scores))]
                    invalid+=int(raw not in legal)
                    legal_indices=[ACTIONS.index(a) for a in legal]
                    pick=ACTIONS[max(legal_indices,key=lambda k:float(scores[k]))]
                timing.append((time.perf_counter()-start)*1000)
            actions.append(w.apply(pick))
            if w.stopped:break
        return w,dict(seed=seed,mode=method,steps=len(actions),
                      painted=len(w.painted),stopped=w.stopped,
                      raw_invalid=invalid,actions=actions,
                      p50_decision_ms=round(statistics.median(timing),3) if timing else None)
    gallery=Image.new('RGB',(N*19*5,N*19*3+24*3),(242,242,242))
    gd=ImageDraw.Draw(gallery)
    traces=[]
    palette={'wall':(44,46,53),'protected':(203,109,52),'paint':(61,134,205),
             'goal':(51,165,110),'cursor':(216,61,76)}
    for row,seed in enumerate(eval_seeds):
        for col,method in enumerate(('trained','untrained','random_nonstop','random','teacher')):
            w,ev=trace(seed,method);traces.append(ev)
            ox=col*N*19;oy=row*(N*19+24)
            for y in range(N):
                for x in range(N):
                    pt=(x,y);fill=(235,235,235)
                    if pt in w.walls:fill=palette['wall']
                    if pt in w.protected:fill=palette['protected']
                    if pt in w.painted:fill=palette['paint']
                    if pt==w.goal:fill=palette['goal']
                    if pt==w.pos:fill=palette['cursor']
                    gd.rectangle((ox+x*19,oy+y*19,ox+x*19+17,oy+y*19+17),fill=fill)
            gd.text((ox+2,oy+N*19+3),f'{method} seed={seed} ink={len(w.painted)}',fill=(20,20,20))
    gallery.save(out/'heldout_worlds.png')
    model_traces=[e for e in traces if e['mode']=='trained']
    manifest=dict(schema='MZ-MAMBA130-v1',source_sha=source_sha,generated_utc=datetime.now(timezone.utc).isoformat(),
                  model_id=MODEL_ID,model_revision=sha,model_load_seconds=model_load_s,
                  backbone_parameters=sum(p.numel() for p in model.parameters()),
                  trained_head_parameters=sum(p.numel() for p in head.parameters()),
                  torch_version=torch.__version__,checkpoint_sha256=ckpt_sha,
                  train_count=len(training),dev_count=len(dev),
                  initial_dev_acc=initial_acc,dev_acc=dev_acc,train_acc=train_acc,
                  losses=train_loss,feature_calls=len(feature_ms),
                  feature_p50_ms=round(statistics.median(feature_ms),3),
                  feature_p95_ms=round(sorted(feature_ms)[int(.95*(len(feature_ms)-1))],3),
                  peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1),
                  heldout_model_raw_invalid_choices=sum(e['raw_invalid'] for e in model_traces),
                  heldout_model_painted_total=sum(e['painted'] for e in model_traces),
                  heldout_random_painted_total=sum(e['painted'] for e in traces if e['mode']=='random'),
                  heldout_random_nonstop_painted_total=sum(e['painted'] for e in traces if e['mode']=='random_nonstop'),
                  heldout_untrained_painted_total=sum(e['painted'] for e in traces if e['mode']=='untrained'),
                  traces=traces,train_seed_range=[100,100+args.train_seeds-1],
                  dev_seeds=[900,901,902],eval_seeds=eval_seeds,
                  limitations=['synthetic micro-world, not native CyberSand',
                   'frozen pretrained Mamba backbone; trained head only',
                   'host-masked legality does not prove raw model legal choices',
                   'teacher and random are not trained-model baselines',
                   'random legal may stop early; random non-STOP is fairer continuation control'])
    (out/'results.json').write_text(json.dumps(manifest,indent=2))
    md=['# Mamba 130M Game-Policy Onboarding — actual pretrained forward + trained head',
        f'Exact upstream revision: `{sha}`',f'Git source: `{source_sha}`',
        f'Pretrained backbone parameters: {manifest["backbone_parameters"]:,}; trained policy-head parameters: {manifest["trained_head_parameters"]:,}',
        f'TRAIN/DEV samples: {len(training)}/{len(dev)}; pre-training head DEV accuracy {initial_acc:.1%}, trained head DEV accuracy {dev_acc:.1%}',
        f'Feature forward p50/p95: {manifest["feature_p50_ms"]}/{manifest["feature_p95_ms"]} ms; peak RSS {manifest["peak_rss_mb"]} MiB',
        f'Held-out raw illegal argmax: {manifest["heldout_model_raw_invalid_choices"]}; trained-painted {manifest["heldout_model_painted_total"]}; random-painted {manifest["heldout_random_painted_total"]}',
        f'Seed-matched controls: untrained-head-painted {manifest["heldout_untrained_painted_total"]}; random non-STOP-painted {manifest["heldout_random_nonstop_painted_total"]}',
        f'Checkpoint sha256: `{ckpt_sha}`',
        'The model selects legal candidates only after host masking; raw illegal argmax is retained separately.',
        'All outputs synthetic game-like trajectories; **no native Godot/Workbench claims**.',
        'See results.json, heldout_worlds.png, and mamba130_game_head.pt in the run artifact.']
    (out/'REPORT.md').write_text('\n\n'.join(md)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('traces','losses')},indent=2),flush=True)

if __name__=='__main__':main()
