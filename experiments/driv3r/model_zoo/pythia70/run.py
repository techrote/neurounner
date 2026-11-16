"""MZ-004: genuinely pretrained GPT-NeoX 70M game-policy head and partial tuning.

Paired frozen-feature and *actual last-block backpropagation* on identical
synthetic CyberSand-style worlds. Research only; zero game implementation edits.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import statistics
import time

from experiments.driv3r.model_zoo.mamba130.counterfactual import (
    samples, raw_features, paired_switch_metrics, class_histogram,
)
from experiments.driv3r.model_zoo.mamba130.run_counterfactual import measure
from experiments.driv3r.model_zoo.mamba130.world import World, ACTIONS, N
from experiments.driv3r.model_zoo.rwkv7.run import fit, safe_scores

MODEL = 'EleutherAI/pythia-70m-deduped'
PIN = 'e93a9faa9c77e5d09219f6c868bfc7a1bd65593c'
SCHEMA = 'MZ004-PYTHIA70-GAME-v1'


def last_block_trainable_names(names, layers):
    """Limit all backbone changes to final attention output and layer norm."""
    if not isinstance(layers, int) or layers < 2:
        raise ValueError('invalid transformer layer count')
    prefix = f'gpt_neox.layers.{layers-1}.'
    selected = tuple(n for n in names if n.startswith(prefix) and (
        '.attention.dense.' in n or '.post_attention_layernorm.' in n))
    if not selected or len(selected) > 4:
        raise RuntimeError('unexpected final-block trainable parameter names')
    if not any('.attention.dense.' in n for n in selected):
        raise RuntimeError('no final attention output parameters')
    if not any('.post_attention_layernorm.' in n for n in selected):
        raise RuntimeError('no final layernorm parameters')
    return selected


def minibatches(length, steps, size, seed):
    if length < size or steps < 1 or size < 1:
        raise ValueError('invalid minibatch budget')
    rng = random.Random(seed)
    return [rng.sample(range(length), size) for _ in range(steps)]


def make_head(torch, dim):
    return torch.nn.Sequential(
        torch.nn.LayerNorm(dim),
        torch.nn.Linear(dim, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, len(ACTIONS)),
    )


def class_weights(torch, y):
    hist = torch.bincount(y, minlength=len(ACTIONS)).float()
    w = torch.zeros(len(ACTIONS))
    present = hist > 0
    if not present.any():
        raise RuntimeError('empty target support')
    w[present] = (hist[present].sum() / hist[present]).sqrt()
    w[present] /= w[present].mean()
    return w


def render_gallery(Image, ImageDraw, records, out):
    px=14
    kinds=5
    if len(records) % kinds:
        raise RuntimeError('nonrectangular rollout gallery')
    img=Image.new('RGB', (kinds*N*px, (len(records)//kinds)*(N*px+24)), (247,247,247))
    d=ImageDraw.Draw(img)
    for i,(world,trace) in enumerate(records):
        ox=(i%kinds)*N*px
        oy=(i//kinds)*(N*px+24)
        for y in range(N):
            for x in range(N):
                pt=(x,y)
                color=(234,234,234)
                if pt in world.walls: color=(44,46,52)
                if pt in world.protected: color=(207,119,54)
                if pt in world.painted: color=(56,125,209)
                if pt == world.goal: color=(39,169,100)
                if pt == world.pos: color=(213,62,79)
                d.rectangle((ox+x*px,oy+y*px,ox+x*px+px-2,oy+y*px+px-2),fill=color)
        d.text((ox+2,oy+N*px+3),trace['policy']+':'+str(trace['painted']),fill=(12,12,12))
    img.save(out)


def run(args):
    if args.train_seeds < 5 or args.dev_seeds < 2 or args.epochs < 1:
        raise ValueError('invalid dataset or optimizer budget')
    if args.tune_steps < 1 or args.batch_size < 1 or not (0 < args.head_extra_lr <= .002) or not (0 < args.backbone_lr <= .001):
        raise ValueError('invalid last-block tuning budget')
    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import model_info

    torch.set_num_threads(2)
    torch.manual_seed(42017)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    source_sha=os.environ.get('GITHUB_SHA','LOCAL-UNVERIFIED')
    info=model_info(MODEL,revision=PIN)
    if info.sha != PIN:
        raise RuntimeError('pinned pretrained revision mismatch')
    t0=time.monotonic()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=PIN,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=PIN,torch_dtype=torch.float32,trust_remote_code=False,
        use_safetensors=True)
    if not hasattr(model,'gpt_neox') or not hasattr(model.gpt_neox,'layers'):
        raise RuntimeError('unexpected architecture; requires GPT-NeoX backbone')
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    parameters=sum(p.numel() for p in model.parameters())
    if not 20_000_000 <= parameters <= 150_000_000:
        raise RuntimeError(f'unexpected pretrained model size: {parameters}')
    num_layers=len(model.gpt_neox.layers)
    selected=last_block_trainable_names(
        (name for name,_ in model.named_parameters()), num_layers)
    load_seconds=round(time.monotonic()-t0,3)

    train=samples(range(400,400+args.train_seeds),args.steps)
    dev=samples(range(900,900+args.dev_seeds),args.steps)
    test=samples(range(1200,1200+args.dev_seeds),args.steps)
    sets=[{row[2] for row in chunk} for chunk in (train,dev,test)]
    if sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]:
        raise RuntimeError('TRAIN/DEV/TEST seed leakage')
    if paired_switch_metrics(test,[r[1] for r in test])['changed_target_pairs']<20:
        raise RuntimeError('missing held-out counterfactual challenge')

    feature_ms=[]
    frozen_features={}
    def feature(prompt,track=True):
        if prompt not in frozen_features:
            encoded=tok(prompt,return_tensors='pt',truncation=True,max_length=128)
            if not 2 <= encoded['input_ids'].shape[1] <= 128:
                raise RuntimeError('bad tokenized game percept')
            tic=time.perf_counter()
            with torch.inference_mode():
                hidden=model.gpt_neox(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask'],
                    use_cache=False,
                    return_dict=True).last_hidden_state
                vector=hidden[0,-1,:].detach().float().cpu().clone()
            if track: feature_ms.append((time.perf_counter()-tic)*1000)
            frozen_features[prompt]=vector
        return frozen_features[prompt]

    try:
        X=torch.stack([feature(r[0]) for r in train])
        V=torch.stack([feature(r[0]) for r in dev])
        T=torch.stack([feature(r[0]) for r in test])
        Y=torch.tensor([r[1] for r in train],dtype=torch.long)
        H=torch.tensor([raw_features(r[0]) for r in train],dtype=torch.float32)
        HD=torch.tensor([raw_features(r[0]) for r in dev],dtype=torch.float32)
        HT=torch.tensor([raw_features(r[0]) for r in test],dtype=torch.float32)

        # Two fully matched 140-update head-only probes.
        frozen_head,frozen_losses=fit(torch,X,Y,args.epochs,42017)
        structured_head,structured_losses=fit(torch,H,Y,args.epochs,42017)
        # Extra optimizer budget is matched: same starting head and same batches.
        frozen_extra=copy.deepcopy(frozen_head)
        tuned_head=copy.deepcopy(frozen_head)
        for name,param in model.named_parameters():
            if name in selected:
                param.requires_grad_(True)
        tuned_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
        if not 100_000 <= tuned_parameters < 1_000_000:
            raise RuntimeError('unexpected selected last-block trainable count')

        weight=class_weights(torch,Y)
        batch_ids=minibatches(len(train),args.tune_steps,args.batch_size,8111)
        extra_opt=torch.optim.AdamW(frozen_extra.parameters(),lr=args.head_extra_lr,weight_decay=.01)
        tune_opt=torch.optim.AdamW([
            {'params':[p for p in model.parameters() if p.requires_grad], 'lr':args.backbone_lr},
            {'params':list(tuned_head.parameters()),'lr':args.head_extra_lr}
        ],weight_decay=.01)
        extra_losses=[]
        tuned_losses=[]
        frozen_extra.train()
        tuned_head.train()
        for indices in batch_ids:
            labels=Y[indices]
            extra_opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(frozen_extra(X[indices]),labels,weight=weight)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite head-extra loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(frozen_extra.parameters(),1.)
            extra_opt.step()
            extra_losses.append(round(float(loss.detach()),5))

            batch=[train[i][0] for i in indices]
            encoded=tok(batch,padding=True,truncation=True,max_length=128,
                        return_tensors='pt')
            mask=encoded['attention_mask']
            tune_opt.zero_grad(set_to_none=True)
            hidden=model.gpt_neox(
                input_ids=encoded['input_ids'],attention_mask=mask,
                use_cache=False,return_dict=True).last_hidden_state
            last=mask.sum(dim=-1)-1
            vectors=hidden[torch.arange(len(indices)),last,:]
            tuned_loss=torch.nn.functional.cross_entropy(
                tuned_head(vectors),labels,weight=weight)
            if not torch.isfinite(tuned_loss):
                raise RuntimeError('nonfinite last-block tuning loss')
            tuned_loss.backward()
            grads=[p.grad for p in model.parameters() if p.requires_grad]
            if any(g is None or not torch.isfinite(g).all() for g in grads):
                raise RuntimeError('missing/nonfinite backbone gradients')
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad]+list(tuned_head.parameters()),1.)
            tune_opt.step()
            tuned_losses.append(round(float(tuned_loss.detach()),5))
        model.eval()
        tuned_head.eval()
        frozen_extra.eval()

        tune_ms=[]
        def tuned_feature(prompt):
            encoded=tok(prompt,return_tensors='pt',truncation=True,max_length=128)
            t=time.perf_counter()
            with torch.inference_mode():
                hidden=model.gpt_neox(
                    input_ids=encoded['input_ids'],attention_mask=encoded['attention_mask'],
                    use_cache=False,return_dict=True).last_hidden_state
            tune_ms.append((time.perf_counter()-t)*1000)
            return hidden[0,-1,:].float()

        with torch.inference_mode():
            predictions={
              'frozen_head':frozen_head(T).argmax(-1).tolist(),
              'frozen_extra':frozen_extra(T).argmax(-1).tolist(),
              'structured':structured_head(HT).argmax(-1).tolist(),
            }
            predictions['tuned_last_block']=torch.stack(
                [tuned_head(tuned_feature(r[0])) for r in test]).argmax(-1).tolist()
            dev_predictions={
              'frozen_head':frozen_head(V).argmax(-1).tolist(),
              'frozen_extra':frozen_extra(V).argmax(-1).tolist(),
              'structured':structured_head(HD).argmax(-1).tolist(),
            }
            dev_predictions['tuned_last_block']=torch.stack(
                [tuned_head(tuned_feature(r[0])) for r in dev]).argmax(-1).tolist()
        majority=int(torch.bincount(Y,minlength=len(ACTIONS)).argmax())
        score={key:{'dev':measure(dev,dev_predictions[key]),
                    'test':measure(test,predictions[key])} for key in predictions}
        score['majority']={'action':ACTIONS[majority],
                           'test':measure(test,[majority]*len(test))}

        gallery=[]
        traces=[]
        for seed in (1811,1812,1813,1814):
            for kind in ('tuned','frozen','structured','random','teacher'):
                w=World.make(seed)
                rng=random.Random(seed*101+17)
                events=[]
                invalid=0
                for _ in range(args.rollout_steps):
                    legal=w.legal()
                    if kind=='teacher':
                        choice=w.teacher(rng)
                        raw=choice
                    elif kind=='random':
                        choice=rng.choice([a for a in legal if a!='STOP'] or ['STOP'])
                        raw=choice
                    else:
                        prompt=w.prompt()
                        with torch.inference_mode():
                            if kind=='tuned':
                                logits=tuned_head(tuned_feature(prompt).unsqueeze(0))[0]
                            elif kind=='frozen':
                                logits=frozen_extra(feature(prompt).unsqueeze(0))[0]
                            else:
                                ft=torch.tensor(raw_features(prompt),dtype=torch.float32)
                                logits=structured_head(ft.unsqueeze(0))[0]
                        raw=ACTIONS[int(logits.argmax())]
                        invalid+=int(raw not in legal)
                        choice=safe_scores(logits,legal)
                    events.append({'turn':w.turns,'raw':raw,'legal':legal,'selected':choice})
                    w.apply(choice)
                    if w.stopped:
                        break
                trace={'seed':seed,'policy':kind,'painted':len(w.painted),
                       'turns':len(events),'raw_invalid':invalid,'events':events}
                gallery.append((w,trace))
                traces.append(trace)
        render_gallery(Image,ImageDraw,gallery,out/'rollouts.png')

        ck=out/'pythia70_game_adapter.pt'
        torch.save({'model':MODEL,'revision':PIN,'source_sha':source_sha,
                    'schema':SCHEMA,'actions':ACTIONS,'selected_names':selected,
                    'last_block_delta':{k:p.detach().cpu().clone()
                        for k,p in model.named_parameters() if k in selected},
                    'frozen_head':frozen_head.state_dict(),
                    'frozen_extra':frozen_extra.state_dict(),
                    'structured_head':structured_head.state_dict(),
                    'tuned_head':tuned_head.state_dict(),
                    'train_seeds':args.train_seeds,'dev_seeds':args.dev_seeds,
                    'tune_steps':args.tune_steps},ck)
        corpus=json.dumps({'train':train,'dev':dev,'test':test},sort_keys=True).encode()
        result={'schema':SCHEMA,'source_sha':source_sha,
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'model':MODEL,'revision':PIN,'backbone_parameters':parameters,
            'selected_trainable_names':selected,'trainable_backbone_parameters':tuned_parameters,
            'trained_head_parameters':sum(p.numel() for p in frozen_head.parameters()),
            'train_examples':len(train),'dev_examples':len(dev),'test_examples':len(test),
            'train_histogram':class_histogram(train),'epochs':args.epochs,
            'extra_updates':args.tune_steps,'batch_size':args.batch_size,
            'head_extra_lr':args.head_extra_lr,'backbone_lr':args.backbone_lr,
            'frozen_head_losses':frozen_losses,'structured_losses':structured_losses,
            'frozen_extra_losses':extra_losses,'tuned_losses':tuned_losses,
            'score':score,'load_seconds':load_seconds,
            'forward_calls':len(feature_ms),'forward_p50_ms':round(statistics.median(feature_ms),3),
            'forward_p95_ms':round(sorted(feature_ms)[int(.95*(len(feature_ms)-1))],3),
            'tuned_forward_calls':len(tune_ms),
            'tuned_forward_p50_ms':round(statistics.median(tune_ms),3),
            'peak_rss_mib':round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1),
            'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),
            'corpus_sha256':hashlib.sha256(corpus).hexdigest(),
            'traces':traces,
            'limitations':[
              'Only attention output dense and post-attention norm in final GPTNeoX block are changed.',
              'Frozen-head and partly-tuned arms share initial head and identical extra minibatches.',
              'Pretrained model is a base language model, not instruction tuned.',
              'Synthetic teacher/World, NOT native CyberSand PAINT or independent human evaluation.',
              'Host legal-choice masking must not be conflated with raw action legality.',
              'Train/DEV/TEST are disjoint by world seed, not by general language pretraining data.'
            ]}
        (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        report=[
           '# MZ-004 Pythia-70M: frozen game head vs real last-block tuning',
           'Exact pretrained revision '+PIN+'; source '+source_sha,
           'TRAIN/DEV/TEST '+str((len(train),len(dev),len(test))),
           'Pretrained frozen '+json.dumps(score['frozen_head']['test']),
           'Frozen + extra minibatches '+json.dumps(score['frozen_extra']['test']),
           'Last-block tuned + same minibatches '+json.dumps(score['tuned_last_block']['test']),
           'Cheap structured '+json.dumps(score['structured']['test']),
           'Majority '+json.dumps(score['majority']['test']),
           'Frozen p50/p95 '+str((result['forward_p50_ms'],result['forward_p95_ms'])),
           'Tuned p50 '+str(result['tuned_forward_p50_ms'])+' ms',
           'Peak RSS '+str(result['peak_rss_mib'])+' MiB',
           'Checkpoint SHA256 '+result['checkpoint_sha256'],
           'No CyberSand game edits; external synthetic research only. Draft/HOLD.'
        ]
        (out/'REPORT.md').write_text('\n\n'.join(report)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in (
            'traces','frozen_head_losses','structured_losses','frozen_extra_losses','tuned_losses')},
            indent=2),flush=True)
    finally:
        for name,param in model.named_parameters():
            if name in selected:
                param.requires_grad_(False)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='mz004-pythia70-evidence')
    ap.add_argument('--train-seeds',type=int,default=14)
    ap.add_argument('--dev-seeds',type=int,default=4)
    ap.add_argument('--steps',type=int,default=9)
    ap.add_argument('--epochs',type=int,default=140)
    ap.add_argument('--tune-steps',type=int,default=96)
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--head-extra-lr',type=float,default=.0002)
    ap.add_argument('--backbone-lr',type=float,default=.00004)
    ap.add_argument('--rollout-steps',type=int,default=16)
    run(ap.parse_args())


if __name__=='__main__':
    main()
