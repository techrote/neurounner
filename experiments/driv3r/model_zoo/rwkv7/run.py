"""MZ-003: pretrained RWKV-7 recurrent encoder and genuine policy-head training.

External synthetic game environment; NEVER modifies CyberSand game code.
Pinned custom remote code undergoes limited import/exec static screen before
loading in an isolated CPU-only GitHub runner. Not a full security audit.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, os, random, resource, statistics, time, tempfile, shutil
from datetime import datetime, timezone
from pathlib import Path
from experiments.driv3r.model_zoo.mamba130.counterfactual import samples,raw_features,paired_switch_metrics,class_histogram
from experiments.driv3r.model_zoo.mamba130.world import ACTIONS,N,World
from experiments.driv3r.model_zoo.mamba130.run_counterfactual import measure

MODEL='Hakureirm/rwkv7-0.1b-hf'
PIN='16136bb51c0cf16356f410956d38b48f249c73e1'
CODE_FILES=('configuration_rwkv7.py','hf_rwkv_tokenizer.py','modeling_rwkv7.py')
FORBIDDEN=frozenset(('subprocess','socket','requests','urllib','ftplib','paramiko','pickle','multiprocessing','ctypes'))

def audit_source(contents):
    """Limited AST surface scan; not a guarantee that remote code is safe."""
    if set(contents)!=set(CODE_FILES):
        raise RuntimeError('custom-code source manifest incomplete')
    result={}
    for file in CODE_FILES:
        code=contents[file]
        tree=ast.parse(code,filename=file)
        imports=set()
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                imports.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node,ast.ImportFrom) and node.module:
                imports.add(node.module.split('.')[0])
            elif isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('eval','exec','compile','__import__'):
                raise RuntimeError(file+': dynamic code execution expression')
        if imports & FORBIDDEN:
            raise RuntimeError(file+': prohibited imported modules '+str(sorted(imports&FORBIDDEN)))
        result[file]={'sha256':hashlib.sha256(code.encode()).hexdigest(),
                      'imports':sorted(imports),'chars':len(code)}
    return result

def prepare_pinned_snapshot(snapshot_download):
    """Locally rewrite one unsafe tokenizer eval to ast.literal_eval.

    Source data/weights remain at immutable revision PIN. The patch is confined
    to the ephemeral experiment runner; never modify the user/game repository.
    Any unexpected upstream source fails closed before custom-code execution.
    """
    cache=Path(snapshot_download(repo_id=MODEL,revision=PIN,
        allow_patterns=['*.py','*.json','*.safetensors','*.txt']))
    sources={p:(cache/p).read_text() for p in CODE_FILES}
    tokenizer=sources['hf_rwkv_tokenizer.py']
    needle='x = eval(l[l.index(" ") : l.rindex(" ")])'
    if tokenizer.count(needle)!=1:
        raise RuntimeError('unexpected unsafe tokenizer eval location')
    tokenizer=tokenizer.replace(needle,'x = ast.literal_eval(l[l.index(" ") : l.rindex(" ")])')
    tokenizer='import ast\n'+tokenizer if not tokenizer.startswith('from __future__') else tokenizer
    # Python allows 'import ast' following the header comment/docstring only,
    # but upstream has no future imports in this pinned tokenizer.
    sources['hf_rwkv_tokenizer.py']=tokenizer
    manifest=audit_source(sources)
    manifest['tokenizer_patch']='eval -> ast.literal_eval for pinned vocab entries only'
    # Copy pinned weights/source out of the Hub cache before changing custom code.
    scratch=Path(tempfile.mkdtemp(prefix='mz003-rwkv7-'))
    for file in cache.iterdir():
        if file.is_file():
            shutil.copy2(file,scratch/file.name)
    (scratch/'hf_rwkv_tokenizer.py').write_text(tokenizer)
    return scratch,manifest

def safe_scores(scores,menu):
    indices=[ACTIONS.index(a) for a in menu]
    if not indices: raise ValueError('host legal menu empty')
    return ACTIONS[max(indices,key=lambda i:float(scores[i]))]

def fit(torch,x,y,steps,seed):
    torch.manual_seed(seed)
    net=torch.nn.Sequential(torch.nn.LayerNorm(x.shape[-1]),
          torch.nn.Linear(x.shape[-1],64),torch.nn.GELU(),
          torch.nn.Linear(64,len(ACTIONS)))
    opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.01)
    hist=torch.bincount(y,minlength=len(ACTIONS)).float()
    weight=torch.zeros(len(ACTIONS))
    mask=hist>0
    weight[mask]=(hist[mask].sum()/hist[mask]).sqrt()
    weight[mask]/=weight[mask].mean()
    losses=[]
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(net(x),y,weight=weight)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.step()
        losses.append(round(float(loss.detach()),5))
    net.eval()
    return net,losses

def render_gallery(Image,ImageDraw,worlds,traces,path):
    px=12;cols=4
    assert len(worlds)==len(traces) and len(worlds)%cols==0
    img=Image.new('RGB',(cols*N*px,len(worlds)//cols*(N*px+22)),(247,247,247))
    draw=ImageDraw.Draw(img)
    for k,(w,t) in enumerate(zip(worlds,traces)):
        ox=k%cols*N*px;oy=k//cols*(N*px+22)
        for y in range(N):
            for x in range(N):
                p=x,y;fill=(233,233,233)
                if p in w.walls:fill=(42,44,50)
                if p in w.protected:fill=(200,110,50)
                if p in w.painted:fill=(55,128,205)
                if p==w.goal:fill=(45,166,105)
                if p==w.pos:fill=(211,58,76)
                draw.rectangle((ox+x*px,oy+y*px,ox+x*px+px-2,oy+y*px+px-2),fill=fill)
        draw.text((ox+2,oy+N*px+2),t['policy']+' ink='+str(t['painted']),fill=(20,20,20))
    img.save(path)

def run(args):
    if args.train_seeds<4 or args.dev_seeds<2 or args.epochs<1:raise ValueError('invalid budget')
    import torch
    from PIL import Image,ImageDraw
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from huggingface_hub import snapshot_download,model_info
    torch.set_num_threads(2)
    torch.manual_seed(42017)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    source_sha=os.environ.get('GITHUB_SHA','LOCAL-UNVERIFIED')
    model_meta=model_info(MODEL,revision=PIN)
    if model_meta.sha!=PIN:raise RuntimeError('upstream SHA mismatch')
    model_root,code_audit=prepare_pinned_snapshot(snapshot_download)
    start=time.monotonic()
    tokenizer=AutoTokenizer.from_pretrained(str(model_root),trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(str(model_root),
                                             trust_remote_code=True,torch_dtype=torch.float32)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    params=sum(p.numel() for p in model.parameters())
    if not 100_000_000<=params<=300_000_000:
        raise RuntimeError('unexpected pretrained parameter count '+str(params))
    load_sec=round(time.monotonic()-start,3)
    output_layer=model.get_output_embeddings()
    if output_layer is None:raise RuntimeError('output projection not exposed')
    seen=[];cache={};costs=[]
    def capture(_layer,args):
        if not args or args[0].ndim!=3:raise RuntimeError('hidden feature shape invalid')
        seen.append(args[0][0,-1,:].detach().float().cpu().clone())
    handle=output_layer.register_forward_pre_hook(capture)
    def embed(prompt):
        if prompt not in cache:
            # RWKV requires the token-0 state initializer before the percept.
            encoded=tokenizer('<|rwkv_tokenizer_end_of_text|>'+prompt,
                              return_tensors='pt',truncation=True,max_length=128)
            ids=encoded['input_ids']
            if ids.shape[1]<2:raise RuntimeError('empty tokenized game percept')
            seen.clear();t=time.perf_counter()
            with torch.inference_mode():model(input_ids=ids,use_cache=False)
            costs.append((time.perf_counter()-t)*1000)
            if len(seen)!=1:raise RuntimeError('true pretrained hidden forward was not captured')
            cache[prompt]=seen[0]
        return cache[prompt]
    try:
        train=samples(range(400,400+args.train_seeds),args.steps)
        dev=samples(range(900,900+args.dev_seeds),args.steps)
        test=samples(range(1200,1200+args.dev_seeds),args.steps)
        s0={r[2] for r in train};s1={r[2] for r in dev};s2={r[2] for r in test}
        if s0&s1 or s0&s2 or s1&s2:raise RuntimeError('TRAIN/DEV/TEST leakage')
        if paired_switch_metrics(test,[r[1] for r in test])['changed_target_pairs']<20:
            raise RuntimeError('insufficient goal-swap pairs')
        y=torch.tensor([r[1] for r in train],dtype=torch.long)
        x=torch.stack([embed(r[0]) for r in train])
        xd=torch.stack([embed(r[0]) for r in dev])
        xt=torch.stack([embed(r[0]) for r in test])
        h=torch.tensor([raw_features(r[0]) for r in train],dtype=torch.float32)
        hd=torch.tensor([raw_features(r[0]) for r in dev],dtype=torch.float32)
        ht=torch.tensor([raw_features(r[0]) for r in test],dtype=torch.float32)
        trained,losses=fit(torch,x,y,args.epochs,42017)
        cheap,cheapest=fit(torch,h,y,args.epochs,42017)
        with torch.inference_mode():
            rd=trained(xd).argmax(-1).tolist()
            rt=trained(xt).argmax(-1).tolist()
            cd=cheap(hd).argmax(-1).tolist()
            ct=cheap(ht).argmax(-1).tolist()
        maj=int(torch.bincount(y,minlength=len(ACTIONS)).argmax())
        scores={'rwkv':{'dev':measure(dev,rd),'test':measure(test,rt)},
                'structured':{'dev':measure(dev,cd),'test':measure(test,ct)},
                'majority':{'action':ACTIONS[maj],
                            'test':measure(test,[maj]*len(test))}}
        worlds=[];traces=[]
        for seed in (1807,1808,1809,1810):
            for kind in ('RWKV7','structured','random-nonstop','teacher'):
                w=World.make(seed)
                rng=random.Random(seed*101+17)
                events=[];invalid=0
                for _ in range(args.rollout_steps):
                    menu=w.legal()
                    if kind=='teacher':
                        raw=w.teacher(rng);pick=raw
                    elif kind=='random-nonstop':
                        raw=rng.choice([a for a in menu if a!='STOP'] or ['STOP']);pick=raw
                    else:
                        ft=embed(w.prompt()) if kind=='RWKV7' else torch.tensor(raw_features(w.prompt()),dtype=torch.float32)
                        net=trained if kind=='RWKV7' else cheap
                        with torch.inference_mode():logits=net(ft.unsqueeze(0))[0]
                        raw=ACTIONS[int(logits.argmax())]
                        invalid+=int(raw not in menu)
                        pick=safe_scores(logits,menu)
                    events.append({'turn':w.turns,'raw':raw,'legal':menu,'selected':pick})
                    w.apply(pick)
                    if w.stopped:break
                worlds.append(w)
                traces.append({'seed':seed,'policy':kind,'painted':len(w.painted),
                               'turns':len(events),'raw_illegal':invalid,'events':events})
        render_gallery(Image,ImageDraw,worlds,traces,out/'rollouts.png')
        checkpoint=out/'rwkv7_game_head.pt'
        torch.save({'upstream':MODEL,'revision':PIN,'source_sha':source_sha,
                    'code_sha':code_audit,'actions':ACTIONS,
                    'rwkv_head':trained.state_dict(),'structured_head':cheap.state_dict(),
                    'train_seeds':args.train_seeds,'epochs':args.epochs},checkpoint)
        corpus=json.dumps({'train':train,'dev':dev,'test':test},sort_keys=True).encode()
        result={'schema':'MZ003-RWKV7-GAME-v1','source_sha':source_sha,
          'model':MODEL,'revision':PIN,'created_utc':datetime.now(timezone.utc).isoformat(),
          'backbone_parameters':params,'trained_head_parameters':sum(p.numel() for p in trained.parameters()),
          'code_audit':code_audit,'load_seconds':load_sec,
          'train_examples':len(train),'dev_examples':len(dev),'test_examples':len(test),
          'train_histogram':class_histogram(train),'epochs':args.epochs,
          'rwkv_losses':losses,'structured_losses':cheapest,'score':scores,
          'forward_calls':len(costs),'forward_p50_ms':round(statistics.median(costs),3),
          'forward_p95_ms':round(sorted(costs)[int(.95*(len(costs)-1))],3),
          'peak_rss_mib':round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1),
          'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
          'corpus_sha256':hashlib.sha256(corpus).hexdigest(),'traces':traces,
          'limitations':['RWKV backbone frozen: only new action head is optimized.',
             'Custom code uses limited static screening; no full security audit.',
             'External synthetic game research, NOT native Godot PAINT.',
             'Host legal-action masking is independent of model raw legality.',
             'Scripted synthetic targets are not evidence of autonomous gameplay.']}
        (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        lines=['# MZ-003 RWKV7 pretrained frozen-backbone game-action probe',
               'Model '+MODEL+'@'+PIN,'Exact source '+source_sha,
               'TRAIN/DEV/TEST '+str((len(train),len(dev),len(test))),
               'RWKV TEST '+json.dumps(scores['rwkv']['test']),
               'Structured TEST '+json.dumps(scores['structured']['test']),
               'Majority TEST '+json.dumps(scores['majority']['test']),
               'Pretrained p50/p95 milliseconds '+str((result['forward_p50_ms'],result['forward_p95_ms'])),
               'Peak RSS MiB '+str(result['peak_rss_mib']),
               'Head checkpoint SHA256 '+result['checkpoint_sha256'],
               'Synthetic, not native. Draft/HOLD; no CyberSand game edits.']
        (out/'REPORT.md').write_text('\n\n'.join(lines)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('traces','rwkv_losses','structured_losses')},indent=2),flush=True)
    finally:
        handle.remove()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='mz003-rwkv7-evidence')
    p.add_argument('--train-seeds',type=int,default=14)
    p.add_argument('--dev-seeds',type=int,default=4)
    p.add_argument('--steps',type=int,default=9)
    p.add_argument('--epochs',type=int,default=140)
    p.add_argument('--rollout-steps',type=int,default=16)
    run(p.parse_args())
if __name__=='__main__':main()
