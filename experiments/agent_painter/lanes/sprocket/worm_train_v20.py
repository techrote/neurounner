#!/usr/bin/env python3
"""W20: action-local material router, source-faithful W0 legal-prefix tests.

External synthetic-game policy only. Root Sprocket is PINNED but never loaded.
Unlike W19's rewritten arbitrary observation, each paired intervention changes
one unprotected cell in a persistent synthetic world and preserves an actually
replayable host-legal W0 action prefix. This is NOT native physics or proof of
autonomous-paired reachability.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, random, resource, time
from collections import Counter
from pathlib import Path
import worm_train_v18 as w18
import worm_train_v19 as w19

VERSION='sprocket-w20-action-local-world-counterfactual-v1'
W19_SHA='0358824dd721118fe9e9bbad7b19d0bfd48be36ce038a18bf7b952c7f9eee01b'
W18_SHA=w19.W18_SHA
W17_SHA=w18.W17_SHA
W0_SHA=w18.W0_SHA
TRAIN=(457,461,463,467)
HELDOUT=(3401,3413,3433)
UPDATES=160

def source(torch,path,version):
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert state['version']==version and state['root_forwards']==0
    model=w18.model_class(torch)()
    model.load_state_dict(state['model'],strict=True)
    model.eval()
    return model

def policy_class(torch):
    class Policy(torch.nn.Module):
        def __init__(self,source_model):
            super().__init__()
            self.parent=w18.model_class(torch)()
            self.parent.load_state_dict(source_model.state_dict(),strict=True)
            for p in self.parent.parameters():p.requires_grad_(False)
            # Explicit heading-rotated candidate material features: S,L,R.
            self.register_buffer('neighbor_cells',torch.tensor([13,17,11,7],dtype=torch.long),persistent=False)
            self.delta=torch.nn.Sequential(torch.nn.Linear(160,64),torch.nn.Tanh(),
                torch.nn.Linear(64,6))
            torch.nn.init.zeros_(self.delta[-1].weight)
            torch.nn.init.zeros_(self.delta[-1].bias)
        def forward(self,x,m):
            n=x.shape[0]
            heading=x[:,100:104].argmax(1)
            mats=m.reshape(n,25,6)
            candidate=[]
            for rotation in (0,-1,1):
                index=self.neighbor_cells[(heading+rotation)%4]
                candidate.append(mats[torch.arange(n,device=m.device),index])
            action_local=torch.cat(candidate,dim=1)
            return self.parent(x,m)+self.delta(torch.cat([x,action_local],dim=1))
    return Policy

def snapshot(torch,model,source_model):
    q=policy_class(torch)(source_model)
    q.load_state_dict(model.state_dict(),strict=True)
    q.eval()
    return q

def patched(original,seed,cell,material):
    """One source-world coordinate changes; geometry and host legality do not."""
    x,y=cell
    assert not w18.protected(seed,x,y)
    before=w18.material(seed,x,y)
    assert material!=before and material in w18.MATERIALS[:4]
    out=copy.deepcopy(original)
    dx=x-original['x'];dy=y-original['y']
    assert -2<=dx<=2 and -2<=dy<=2
    assert original['material_5x5'][dy+2][dx+2]==before
    out['material_5x5'][dy+2][dx+2]=material
    assert out['local_5x5']==original['local_5x5'] and out['legal']==original['legal']
    assert out['last_actions']==original['last_actions']
    assert sum(out['material_5x5'][j][i]!=original['material_5x5'][j][i]
        for j in range(5) for i in range(5))==1
    return out

def augment_world(raw,seed,overrides):
    """Spatially persistent override map: same world cell at every observation."""
    o=w18.augment(raw,seed)
    for (x,y),material in overrides.items():
        if abs(x-o['x'])<=2 and abs(y-o['y'])<=2:
            o=patched(o,seed,(x,y),material)
    return o

def interventions(w0,torch,source_model,seeds):
    ordinary=[]; paired=[]; receipts=[]; examples=[]
    for seed in seeds:
        for family in w18.FAMILIES:
            for name,p,temp in [('teacher',None,0),('random',None,0),
                                ('W19-greedy',source_model,0),('W19-t07',source_model,.7)]:
                rec=w18.rollout(torch,w0,p,seed,family,name,temp)
                ordinary.extend(w18.to_rows([rec],seed))
                local_count=0
                for tick,event in enumerate(rec['events']):
                    raw=event['observation']
                    obs=w18.augment(raw,seed)
                    a=w18.teacher(obs)
                    target={'greebles':'metal','pipework':'water','architecture':'stone'}[family]
                    for action,rotation in [('STRAIGHT',0),('LEFT',-1),('RIGHT',1)]:
                        if action not in obs['legal']:continue
                        dx,dy=w18.DIRECTIONS[(obs['heading']+rotation)%4]
                        cell=(obs['x']+dx,obs['y']+dy)
                        if w18.material(seed,*cell)==target or w18.protected(seed,*cell):continue
                        other=augment_world(raw,seed,{cell:target})
                        b=w18.teacher(other)
                        if a==b:continue
                        paired.append(((obs,a),(other,b)))
                        local_count+=1
                        if len(examples)<192:
                            examples.append({'seed':seed,'family':family,'controller':name,'tick':tick,
                                'cell':list(cell),'material_before':w18.material(seed,*cell),
                                'material_after':target,'first_teacher':a,'second_teacher':b,
                                'legal_prefix':[e['applied'] for e in rec['events'][:tick]],
                                'original_x':obs['x'],'original_y':obs['y'],
                                'original_heading':obs['heading']})
                receipts.append({'seed':seed,'family':family,'controller':name,
                    'steps':rec['steps_executed'],'generated_pairs':local_count,
                    'head_forwards':rec['original_head_forwards'],
                    'host_legal_filters':rec['additional_host_legal_filters']})
    return ordinary,paired,receipts,examples

def verify_prefix_replay(w0,examples):
    """Replay actual original legal prefixes, then expose the changed cell.
    This is reachable in the external synthetic host under that fixed action
    prefix, not proof that a free-running policy reaches both automatically.
    """
    witnessed=[]
    seen=set()
    for v in examples:
        key=(v['seed'],v['family'],v['controller'],v['tick'])
        if key in seen:continue
        seen.add(key)
        tape=v['legal_prefix']
        captured=[];seen_prefix=[]
        override={tuple(v['cell']):v['material_after']}
        def select(o):
            # The same single-cell material-world intervention is present
            # throughout the replay, including before it enters the 5x5 window.
            before=w18.augment(o,v['seed'])
            after=augment_world(o,v['seed'],override)
            assert before['legal']==after['legal']
            assert before['local_5x5']==after['local_5x5']
            seen_prefix.append({'tick':o['tick'],'visible_change':before['material_5x5']!=after['material_5x5']})
            if o['tick']>=len(tape):
                captured.append(o)
                return 'STOP'
            action=tape[o['tick']]
            assert action in o['legal']
            return action
        w0.run(v['seed'],'callback',v['family'],len(tape)+1,selector=select)
        assert len(captured)==1 and len(seen_prefix)==len(tape)+1
        raw=captured[0]
        assert (raw['x'],raw['y'],raw['heading'])==(v['original_x'],v['original_y'],v['original_heading'])
        before=w18.augment(raw,v['seed'])
        after=augment_world(raw,v['seed'],override)
        assert w18.teacher(before)==v['first_teacher']
        assert w18.teacher(after)==v['second_teacher']
        witnessed.append({'seed':v['seed'],'family':v['family'],'tick':v['tick'],
            'world_patch_cell':v['cell'],'world_patch_material':v['material_after'],
            'prefix_SHA256':hashlib.sha256(json.dumps(tape,separators=(',',':')).encode()).hexdigest(),
            'legal_host_prefix':True,'one_world_cell_only':True,
            'persistent_world_material_observations':len(seen_prefix),
            'prefix_observations_noticed_patch':sum(v['visible_change'] for v in seen_prefix)})
        if len(witnessed)>=36:break
    assert len(witnessed)>=12
    return witnessed

def pair_eval(torch,model,paired):
    if not paired:return {'pairs':0}
    a,b=zip(*paired)
    xa,ma,ya,la=w18.tensorize(torch,a)
    xb,mb,yb,lb=w18.tensorize(torch,b)
    with torch.no_grad():
        pa=model(xa,ma).masked_fill(~la,-1e4).argmax(1)
        pb=model(xb,mb).masked_fill(~lb,-1e4).argmax(1)
    return {'pairs':len(paired),'top_action_switches':int((pa!=pb).sum()),
        'both_teacher_targets_correct':int(((pa==ya)&(pb==yb)).sum()),
        'individual_teacher_accuracy':round(float(((pa==ya).sum()+(pb==yb).sum())/(2*len(paired))),6)}

def checkpoint(torch,model,opt,rng,out,name,assessment):
    file=out/(name+'.pt')
    torch.save({'version':VERSION,'stage':name,'model':model.state_dict(),
        'optimizer':opt.state_dict(),'rng':rng.getstate(),
        'torch_rng_state':torch.random.get_rng_state(),'assessment':assessment,
        'root_model_loaded':False,'root_forwards':0,'root_updates':0,
        'LoRA_updates':0,'native':False},file)
    return {'file':file.name,'sha256':w18.sha(file),'bytes':file.stat().st_size}

def main(args):
    assert set(TRAIN).isdisjoint(HELDOUT)
    assert (w18.ROOT,w18.REVISION,w18.EOS)==('kandivault/sprocket-500m',
        'eb157d4ddddc4619e661d4a2426bb2dd000fc668',5)
    assert not (set(TRAIN+HELDOUT)&set(w18.TRAIN+w18.HELDOUT+w19.TRAIN+w19.HELDOUT))
    if args.smoke_only:
        print('W20 source-only pinned, unique-seed, owner-boundary preflight PASS')
        return
    begin=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=False)
    import torch
    torch.manual_seed(202020);torch.set_num_threads(1)
    rng=random.Random(202020)
    assert w18.sha(args.w19)==W19_SHA and w18.sha(args.w18)==W18_SHA
    assert w18.sha(args.w17)==W17_SHA and w18.sha(args.w0)==W0_SHA
    w0=w18.w0_module(args.w0)
    w19_frozen=source(torch,args.w19,w19.VERSION)
    w18_frozen=source(torch,args.w18,w18.VERSION)
    w17_frozen=w19.source_w17(torch,args.w17)
    Policy=policy_class(torch)
    model=Policy(w19_frozen)
    frozen=snapshot(torch,model,w19_frozen)
    w18.save(args.output/'provenance.json',{'version':VERSION,
        'head':os.getenv('GITHUB_SHA','local'),'run_id':os.getenv('GITHUB_RUN_ID','local'),
        'root':w18.ROOT,'revision':w18.REVISION,'eos':w18.EOS,
        'W19_checkpoint_sha256':W19_SHA,'W18_checkpoint_sha256':W18_SHA,
        'W17_checkpoint_sha256':W17_SHA,'W0_source_sha256':W0_SHA,
        'script_sha256':w18.sha(__file__),'root_model_loaded':False,
        'root_forwards':0,'root_updates':0,'LoRA_updates':0,'draft_hold':True,
        'synthetic_world':True,'native':False,'sealed_FINAL':False,
        'counterfactual_reachability':'fixed-host-legal-prefix-only',
        'intervention_persists_in_synthetic_world':True})
    train,pairs,train_receipts,_=interventions(w0,torch,w19_frozen,TRAIN)
    hold,dev,heldout_receipts,proof_examples=interventions(w0,torch,w19_frozen,HELDOUT)
    assert len(pairs)>=80 and len(dev)>=40,(len(pairs),len(dev))
    witness=verify_prefix_replay(w0,proof_examples)
    w18.save(args.output/'replayability-witnesses.json',witness)
    w18.save(args.output/'data-receipts.json',{'train_rows':len(train),
        'train_world_pairs':len(pairs),'heldout_rows':len(hold),
        'heldout_world_pairs':len(dev),'train_seeds':TRAIN,'heldout_seeds':HELDOUT,
        'train_source_receipts':train_receipts,'heldout_source_receipts':heldout_receipts,
        'replayed_proof_cases':len(witness),'native':False,
        'autonomous_counterfactual_reachability_unproven':True})
    xa,ma,ya,la=w18.tensorize(torch,train)
    (x0,m0,y0,l0),(x1,m1,y1,l1)=w19.pair_tensors(torch,pairs)
    optimizer=torch.optim.AdamW(model.delta.parameters(),lr=.001,weight_decay=.004)
    stages=[{'name':'frozen-W19-in-W20-routing','ordinary':w18.evaluate(torch,frozen,hold),
             'material_pairs':pair_eval(torch,frozen,dev)}]
    history=[];intermediate=None
    for wave,name in enumerate(('candidate-local-contrast','onpolicy-history-branch-retention')):
        if wave:
            intermediate=snapshot(torch,model,w19_frozen)
            replay,_,replay_receipts,_=interventions(w0,torch,intermediate,TRAIN)
            rx,rm,ry,rl=w18.tensorize(torch,replay)
            w18.save(args.output/'onpolicy-replay.json',{'rows':len(replay),
                'receipts':replay_receipts,'synthetic':True})
        for step in range(UPDATES):
            optimizer.zero_grad(set_to_none=True)
            ix=[rng.randrange(len(pairs)) for _ in range(64)]
            left=model(x0[ix],m0[ix]).masked_fill(~l0[ix],-1e4)
            right=model(x1[ix],m1[ix]).masked_fill(~l1[ix],-1e4)
            va=y0[ix];vb=y1[ix]
            ce=(torch.nn.functional.cross_entropy(left,va)+torch.nn.functional.cross_entropy(right,vb))/2
            delta_a=left.gather(1,va[:,None]).flatten()-left.gather(1,vb[:,None]).flatten()
            delta_b=right.gather(1,vb[:,None]).flatten()-right.gather(1,va[:,None]).flatten()
            contrast=(torch.nn.functional.softplus(1.0-delta_a)+torch.nn.functional.softplus(1.0-delta_b)).mean()/2
            oi=[rng.randrange(len(train)) for _ in range(64)]
            ordinary=torch.nn.functional.cross_entropy(
                model(xa[oi],ma[oi]).masked_fill(~la[oi],-1e4),ya[oi])
            if wave:
                ri=[rng.randrange(len(replay)) for _ in range(64)]
                replay_ce=torch.nn.functional.cross_entropy(
                    model(rx[ri],rm[ri]).masked_fill(~rl[ri],-1e4),ry[ri])
                with torch.no_grad():
                    old=w19_frozen(xa[oi],ma[oi]).masked_fill(~la[oi],-1e4)
                new=model(xa[oi],ma[oi]).masked_fill(~la[oi],-1e4)
                branch=w18.ACTIONS.index('BRANCH')
                retain=torch.nn.functional.mse_loss(torch.softmax(new,1)[:,branch],
                                                    torch.softmax(old,1)[:,branch])
                loss=.33*ce+.30*contrast+.15*ordinary+.19*replay_ce+.03*retain
            else:loss=.45*ce+.35*contrast+.20*ordinary
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite loss')
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.delta.parameters(),3.0)
            if not bool(torch.isfinite(grad)) or float(grad)<=0:raise ValueError('invalid gradients')
            optimizer.step()
            if (step+1)%20==0:
                history.append({'stage':name,'stage_step':step+1,'genuine_step':wave*UPDATES+step+1,
                    'loss':round(float(loss.detach()),6),'grad_norm':round(float(grad),6)})
                w18.save(args.output/'partial-optimizer-history.json',history)
        assessment={'ordinary':w18.evaluate(torch,model,hold),
                    'material_pairs':pair_eval(torch,model,dev)}
        cp=checkpoint(torch,model,optimizer,rng,args.output,name,assessment)
        stages.append({'name':name,'genuine_updates':UPDATES,'assessment':assessment,'checkpoint':cp})
        w18.save(args.output/'stage-progress.json',stages)
    gallery=args.output/'original-material-png-json'
    gallery.mkdir(exist_ok=True)
    output=[]
    controllers=[('W17',w17_frozen,0),('W18',w18_frozen,0),
        ('W19',w19_frozen,0),('W20-stage1',intermediate,0),
        ('W20-final',model,0),('W20-final-T07',model,.7),
        ('W20-final-T11',model,1.1),('teacher',None,0),
        ('random',None,0),('wall',None,0)]
    for seed in HELDOUT:
        for family in w18.FAMILIES:
            for name,policy,temp in controllers:
                rec=w18.rollout(torch,w0,policy,seed,family,name,temp)
                basename=f'{family}-{name}-{seed}'
                js=gallery/(basename+'.json');png=gallery/(basename+'.png')
                w18.save(js,rec)
                png.write_bytes(w18.material_png(w0,rec))
                output.append({'seed':seed,'family':family,'controller':name,
                    'metrics':rec['metrics'],'head_forwards':rec['original_head_forwards'],
                    'host_filtered':rec['additional_host_legal_filters'],
                    'sha_json':w18.sha(js),'sha_png':w18.sha(png),
                    'trajectory_sha256':hashlib.sha256(json.dumps(rec['trajectory'],
                        separators=(',',':')).encode()).hexdigest()})
        w18.save(args.output/'partial-gallery.json',output)
    assert len(output)==90 and all(v['metrics']['invalid_raw_choices']==0 for v in output)
    for stage in stages:
        if 'checkpoint' in stage:
            assert w18.sha(args.output/stage['checkpoint']['file'])==stage['checkpoint']['sha256']
    w18.save(args.output/'summary.json',{'status':'complete','version':VERSION,
        'genuine_aux_adamw_steps':2*UPDATES,'root_forwards':0,'root_updates':0,
        'LoRA_updates':0,'original_png_json_pairs':len(output),
        'actual_head_forwards':sum(v['head_forwards'] for v in output),
        'stages':stages,'results':output,'prefix_replay_witnesses':len(witness),
        'wall_seconds':time.monotonic()-begin,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'synthetic_not_native':True,'counterfactual_free_policy_reachability_unproven':True})
    print(json.dumps({'W20':'complete','optimizer_steps':2*UPDATES,
        'original_paintings':len(output),'prefix_witnesses':len(witness)}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for f in ('w19','w18','w17','w0','output'):parser.add_argument('--'+f,type=Path)
    parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args()
    try:
        if not args.smoke_only and not all((args.w19,args.w18,args.w17,args.w0,args.output)):
            parser.error('source-pinned inputs required')
        main(args)
    except BaseException as err:
        if args.output and args.output.exists():
            w18.save(args.output/'failure.json',{'type':type(err).__name__,'error':str(err)})
        raise
