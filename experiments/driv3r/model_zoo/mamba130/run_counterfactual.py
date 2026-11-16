"""MZ-002: frozen pretrained Mamba versus explicit game-feature control.

Counterfactual goal swaps are source-fenced synthetic game experiments,
not native CyberSand simulation or a game-code change.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,os,random,resource,statistics,time
from pathlib import Path
from .counterfactual import samples,paired_switch_metrics,raw_features,class_histogram
from .world import ACTIONS,N,World

MODEL='state-spaces/mamba-130m-hf'
PIN='1e76775'

def measure(rows,preds):
    truth=[r[1] for r in rows]
    if len(truth)!=len(preds):raise ValueError('prediction length')
    per={};f1=[];recall=[]
    for i,a in enumerate(ACTIONS):
        n=sum(y==i for y in truth)
        if not n:continue
        tp=sum(y==i and p==i for y,p in zip(truth,preds))
        q=sum(p==i for p in preds)
        r=tp/n;pr=tp/q if q else 0
        f=2*r*pr/(r+pr) if r+pr else 0
        recall.append(r);f1.append(f)
        per[a]=dict(support=n,recall=round(r,4),precision=round(pr,4),f1=round(f,4))
    legal=[set(r[0].split('legal=')[1].split(' choose:')[0].split(',')) for r in rows]
    return dict(accuracy=round(sum(y==p for y,p in zip(truth,preds))/len(rows),4),
                balanced_accuracy=round(sum(recall)/len(recall),4),
                macro_f1=round(sum(f1)/len(f1),4),
                raw_invalid=sum(ACTIONS[p] not in s for p,s in zip(preds,legal)),
                paired=paired_switch_metrics(rows,preds),per_class=per)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='mz002-output')
    ap.add_argument('--train-seeds',type=int,default=24)
    ap.add_argument('--steps',type=int,default=12)
    ap.add_argument('--epochs',type=int,default=60)
    args=ap.parse_args()
    if args.train_seeds<4 or args.steps<4 or args.epochs<1:raise ValueError('invalid budget')
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from huggingface_hub import model_info
    from PIL import Image,ImageDraw
    torch.manual_seed(3309);torch.set_num_threads(2)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    src=os.environ.get('GITHUB_SHA','local-unverified')
    rev=model_info(MODEL).sha
    if not rev.startswith(PIN):raise RuntimeError('upstream revision moved: '+rev)
    t0=time.monotonic()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=rev,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=rev,torch_dtype=torch.float32,trust_remote_code=False)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    load_seconds=round(time.monotonic()-t0,3)
    cache={};fwd_ms=[]
    def embed(prompt):
        if prompt not in cache:
            ids=tokenizer(prompt,return_tensors='pt',truncation=True,max_length=128)['input_ids']
            t=time.perf_counter()
            with torch.inference_mode():
                state=model.backbone(input_ids=ids,use_cache=False)
                cache[prompt]=state.last_hidden_state[0,-1].detach().float().clone()
            fwd_ms.append((time.perf_counter()-t)*1000)
        return cache[prompt]
    train=samples(range(300,300+args.train_seeds),args.steps)
    dev=samples(range(900,908),args.steps)
    if set(r[2] for r in train)&set(r[2] for r in dev):raise AssertionError('split leak')
    if paired_switch_metrics(dev,[r[1] for r in dev])['changed_target_pairs']<50:
        raise AssertionError('insufficient goal-dependent teacher signal')
    Y=torch.tensor([r[1] for r in train])
    X=torch.stack([embed(r[0]) for r in train])
    V=torch.stack([embed(r[0]) for r in dev])
    H=torch.tensor([raw_features(r[0]) for r in train],dtype=torch.float32)
    J=torch.tensor([raw_features(r[0]) for r in dev],dtype=torch.float32)
    hist=torch.bincount(Y,minlength=len(ACTIONS)).float()
    weights=torch.zeros(len(ACTIONS));mask=hist>0
    weights[mask]=(hist[mask].sum()/hist[mask]).sqrt()
    weights[mask]/=weights[mask].mean()
    def fit(name,x,v):
        head=torch.nn.Sequential(torch.nn.LayerNorm(x.shape[-1]),
              torch.nn.Linear(x.shape[-1],64),torch.nn.GELU(),
              torch.nn.Linear(64,len(ACTIONS)))
        opt=torch.optim.AdamW(head.parameters(),lr=.002,weight_decay=.01)
        losses=[]
        for _ in range(args.epochs):
            opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(head(x),Y,weight=weights)
            if not torch.isfinite(loss):raise RuntimeError(name+' nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            opt.step();losses.append(round(float(loss),5))
        head.eval()
        with torch.inference_mode():prediction=head(v).argmax(-1).tolist()
        return head,dict(training_losses=losses,dev=measure(dev,prediction))
    mhead,mres=fit('Mamba',X,V)
    hhead,hres=fit('HandFeature',H,J)
    majority=int(hist.argmax())
    maj=measure(dev,[majority]*len(dev))
    checkpoint=out/'mamba130_counterfactual_heads.pt'
    torch.save(dict(model=MODEL,revision=rev,source_sha=src,actions=ACTIONS,
                    mamba_head=mhead.state_dict(),cheap_head=hhead.state_dict(),
                    train_seeds=args.train_seeds,steps=args.steps,epochs=args.epochs,seed=3309),checkpoint)
    cksha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    def rollout(seed,kind):
        w=World.make(seed);rng=random.Random(seed*101+17);events=[];illegal=0
        for _ in range(20):
            menu=w.legal()
            if kind=='teacher':pick=w.teacher(rng);raw=pick
            elif kind=='random-nonstop':
                pick=rng.choice([a for a in menu if a!='STOP'] or ['STOP']);raw=pick
            else:
                ft=embed(w.prompt()) if kind=='Mamba' else torch.tensor(raw_features(w.prompt()),dtype=torch.float32)
                head=mhead if kind=='Mamba' else hhead
                with torch.inference_mode():scores=head(ft.unsqueeze(0))[0]
                raw=ACTIONS[int(scores.argmax())];illegal+=int(raw not in menu)
                indices=[ACTIONS.index(a) for a in menu]
                pick=ACTIONS[max(indices,key=lambda j:float(scores[j]))]
            events.append(dict(raw=raw,selected=pick,legal=menu,turn=w.turns))
            w.apply(pick)
            if w.stopped:break
        return w,dict(seed=seed,policy=kind,painted=len(w.painted),
                      steps=len(events),raw_invalid=illegal,events=events)
    worlds=[];traces=[]
    for seed in (1707,1708,1709):
        for kind in ('Mamba','HandFeature','random-nonstop','teacher'):
            w,tr=rollout(seed,kind);worlds.append(w);traces.append(tr)
    px=18
    img=Image.new('RGB',(4*N*px,3*(N*px+24)),(245,245,245))
    draw=ImageDraw.Draw(img)
    for i,w in enumerate(worlds):
        ox=i%4*N*px;oy=i//4*(N*px+24)
        for y in range(N):
            for x in range(N):
                p=(x,y);fill=(236,236,236)
                if p in w.walls:fill=(44,46,52)
                if p in w.protected:fill=(203,110,52)
                if p in w.painted:fill=(61,134,205)
                if p==w.goal:fill=(51,165,110)
                if p==w.pos:fill=(216,61,76)
                draw.rectangle((ox+x*px,oy+y*px,ox+x*px+px-2,oy+y*px+px-2),fill=fill)
        draw.text((ox+2,oy+N*px+3),traces[i]['policy']+' ink='+str(traces[i]['painted']),fill=(25,25,25))
    img.save(out/'paired_rollouts.png')
    result=dict(schema='MZ-MAMBA130-COUNTERFACTUAL-v2',source_sha=src,
       generated_utc=datetime.now(timezone.utc).isoformat(),model=MODEL,model_revision=rev,
       load_seconds=load_seconds,backbone_parameters=sum(p.numel() for p in model.parameters()),
       train_examples=len(train),dev_examples=len(dev),train_class_histogram=class_histogram(train),
       dev_class_histogram=class_histogram(dev),epochs=args.epochs,
       Mamba=mres,hand_feature_control=hres,
       majority_control=dict(action=ACTIONS[majority],dev=maj),
       forward_calls=len(fwd_ms),forward_p50_ms=round(statistics.median(fwd_ms),3),
       forward_p95_ms=round(sorted(fwd_ms)[int(.95*(len(fwd_ms)-1))],3),
       peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1),
       checkpoint_sha256=cksha,traces=traces,
       limitations=['External synthetic micro-world, not native CyberSand.',
                    'Only the policy head is trained; pretrained Mamba remains frozen.',
                    'Targets are scripted and goal-swapped; no human labels.',
                    'STOP is not taught when an alternative action exists.',
                    'Host masks illegal actions; raw illegal decisions measured.'])
    (out/'results-v2.json').write_text(json.dumps(result,indent=2)+'\n')
    report=[
      '# MZ-002 Goal-swapped Mamba game-policy evidence',
      'Upstream SHA: '+rev+' | Git source: '+src+' | Checkpoint SHA256: '+cksha,
      'TRAIN / DEV: '+str(len(train))+' / '+str(len(dev)),
      'Mamba DEV '+str(mres['dev']),
      'Cheap structured game-feature DEV '+str(hres['dev']),
      'Constant majority '+ACTIONS[majority]+' DEV '+str(maj),
      'Forward p50/p95 ms: '+str(result['forward_p50_ms'])+'/'+str(result['forward_p95_ms']),
      'Synthetic game-like traces and PNG only. No native Godot PAINT.',
      'No CyberSand game code/content changed.'
    ]
    (out/'REPORT.md').write_text('\n\n'.join(report)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('traces','Mamba','hand_feature_control')},indent=2),flush=True)
    print('\n'.join(report),flush=True)

if __name__=='__main__':
    main()
