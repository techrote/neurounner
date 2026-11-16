#!/usr/bin/env python3
"""W19 explicit one-cell material counterfactual training, synthetic W0.

Warm-start an exact SHA-admitted W18 auxiliary head. Model-side only: pinned
Sprocket 500M root and EOS5 are provenance anchors, not loaded/updated. The
material mask and teacher are synthetic and cannot claim native physics.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,os,random,resource,time
from collections import Counter
from pathlib import Path
import worm_train_v18 as w18

VERSION='sprocket-w19-single-cell-material-contrast-v1'
W18_SHA='13189f3e71b0db9d6d0cb961221e649cf7d2e868ed0c66eb18a93d8378fd015b'
W17_SHA=w18.W17_SHA
W0_SHA=w18.W0_SHA
TRAIN=(433,439,443,449)
HELDOUT=(3001,3007,3011)
UPDATES=160


def warmstart(torch,path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    assert payload['version']==w18.VERSION and payload['root_updates']==0
    policy=w18.model_class(torch)()
    policy.load_state_dict(payload['model'],strict=True)
    return policy


def snapshot(torch,policy):
    clone=w18.model_class(torch)()
    clone.load_state_dict(policy.state_dict(),strict=True)
    clone.eval()
    return clone


def source_w17(torch,path):
    raw=torch.load(path,map_location='cpu',weights_only=True)
    assert raw['version']=='sprocket-w17-hard-negative-branch-history-calibration-v1'
    m=w18.model_class(torch)()
    m.base.load_state_dict({k.removeprefix('layers.'):v for k,v in raw['head'].items()},strict=True)
    m.material[-1].weight.data.zero_()
    m.material[-1].bias.data.zero_()
    m.eval()
    return m


def gather(torch,w0,policy,seeds):
    rows=[];receipts=[]
    for seed in seeds:
        for family in w18.FAMILIES:
            for name,p,t in [('teacher',None,0),('random',None,0),('W18-t07',policy,.7)]:
                rec=w18.rollout(torch,w0,p,seed,family,name,t)
                added=w18.to_rows([rec],seed)
                rows.extend(added)
                receipts.append({'seed':seed,'family':family,'controller':name,
                    'events':len(added),'host_filter_count':rec['additional_host_legal_filters']})
    return rows,receipts


def pairs(rows):
    """Single-cell material intervention; no geometry/legality/last-action edits."""
    result=[]
    for obs,_ in rows:
        movements=[]
        for name,rotation in [('STRAIGHT',0),('LEFT',-1),('RIGHT',1)]:
            if name in obs['legal']:
                dx,dy=w18.DIRECTIONS[(obs['heading']+rotation)%4]
                movements.append((name,2+dy,2+dx))
        if len(movements)<2:continue
        neutral=copy.deepcopy(obs)
        for _,y,x in movements:neutral['material_5x5'][y][x]='sand'
        a=w18.teacher(neutral)
        desired={'greebles':'metal','pipework':'water','architecture':'stone'}[obs['family']]
        for _,y,x in movements:
            altered=copy.deepcopy(neutral)
            altered['material_5x5'][y][x]=desired
            b=w18.teacher(altered)
            if b!=a:
                assert neutral['legal']==altered['legal']
                assert neutral['local_5x5']==altered['local_5x5']
                assert neutral['last_actions']==altered['last_actions']
                diffs=sum(neutral['material_5x5'][yy][xx]!=altered['material_5x5'][yy][xx]
                          for yy in range(5) for xx in range(5))
                assert diffs==1 and a in neutral['legal'] and b in altered['legal']
                result.append(((neutral,a),(altered,b)))
    return result


def pair_tensors(torch,pairs_):
    aa,bb=zip(*pairs_)
    return w18.tensorize(torch,aa),w18.tensorize(torch,bb)


def pair_eval(torch,model,pairs_):
    if not pairs_:return {'pairs':0}
    (xa,ma,ya,la),(xb,mb,yb,lb)=pair_tensors(torch,pairs_)
    with torch.no_grad():
        pa=model(xa,ma).masked_fill(~la,-1e4).argmax(1)
        pb=model(xb,mb).masked_fill(~lb,-1e4).argmax(1)
    return {'pairs':len(pairs_),'top_action_switches':int((pa!=pb).sum()),
        'both_teacher_targets_correct':int(((pa==ya)&(pb==yb)).sum()),
        'individual_teacher_accuracy':round(float(((pa==ya).sum()+(pb==yb).sum())/(2*len(pairs_))),6)}


def checkpoint(torch,model,opt,rng,out,name,assessment):
    p=out/(name+'.pt')
    torch.save({'version':VERSION,'stage':name,'model':model.state_dict(),
        'optimizer':opt.state_dict(),'rng':rng.getstate(),
        'torch_rng':torch.random.get_rng_state(),'assessment':assessment,
        'root_updates':0,'root_forwards':0,'LoRA_updates':0},p)
    return {'file':p.name,'sha256':w18.sha(p),'bytes':p.stat().st_size}


def main(args):
    assert not set(TRAIN)&set(HELDOUT)
    assert (w18.ROOT,w18.REVISION,w18.EOS)==('kandivault/sprocket-500m','eb157d4ddddc4619e661d4a2426bb2dd000fc668',5)
    if args.smoke_only:
        assert all(s not in set(w18.TRAIN+w18.HELDOUT) for s in TRAIN+HELDOUT)
        print('W19 source-only seed, parent and EOS5 preflight PASS (0 optimizer steps)')
        return
    start=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=False)
    import torch
    torch.manual_seed(191919);torch.set_num_threads(1)
    rng=random.Random(191919)
    assert w18.sha(args.w18)==W18_SHA and w18.sha(args.w17)==W17_SHA and w18.sha(args.w0)==W0_SHA
    w0=w18.w0_module(args.w0)
    parent=warmstart(torch,args.w18)
    frozen=snapshot(torch,parent)
    w17=source_w17(torch,args.w17)
    w18.save(args.output/'provenance.json',{
        'version':VERSION,'source_head':os.getenv('GITHUB_SHA','local'),
        'run_id':os.getenv('GITHUB_RUN_ID','local'),'attempt':os.getenv('GITHUB_RUN_ATTEMPT','local'),
        'root':w18.ROOT,'revision':w18.REVISION,'eos':5,
        'W18_full_state_sha256':W18_SHA,'W17_full_state_sha256':W17_SHA,
        'W0_source_sha256':W0_SHA,'script_sha256':w18.sha(__file__),
        'real_AdamW_updates_planned':320,'root_model_loaded':False,
        'root_forwards':0,'root_updates':0,'LoRA_updates':0,
        'counterfactual_material_reachability_verified':False,'synthetic_host':True,
        'native':False,'sealed_FINAL':False,'draft_hold':True})
    rows,train_receipts=gather(torch,w0,frozen,TRAIN)
    hold,hold_receipts=gather(torch,w0,frozen,HELDOUT)
    contrast=pairs(rows);dev=pairs(hold)
    assert len(contrast)>=128 and len(dev)>=128,(len(contrast),len(dev))
    w18.save(args.output/'datasets.json',{'train_rows':len(rows),'heldout_rows':len(hold),
       'train_material_contrast_pairs':len(contrast),'heldout_material_contrast_pairs':len(dev),
       'train_seeds':TRAIN,'heldout_seeds':HELDOUT,'training_receipts':train_receipts,
       'heldout_receipts':hold_receipts,'one_cell_difference_per_pair':True,
       'synthetic_interventions_not_guaranteed_reachable':True,
       'teacher_actions':dict(Counter(a for _,a in rows)),'native':False})
    xa,ma,ya,la=w18.tensorize(torch,rows)
    (x0,m0,y0,l0),(x1,m1,y1,l1)=pair_tensors(torch,contrast)
    model=warmstart(torch,args.w18)
    opt=torch.optim.AdamW(model.material.parameters(),lr=.0009,weight_decay=.004)
    stages=[{'name':'frozen-W18','ordinary':w18.evaluate(torch,frozen,hold),
             'material_pairs':pair_eval(torch,frozen,dev)}]
    history=[];stage1=None
    for wave,name in enumerate(('one-cell-material-supervision','onpolicy-material-replay-and-branch')):
        if wave:
            stage1=snapshot(torch,model)
            for p in model.base[-1].parameters():p.requires_grad_(True)
            opt.add_param_group({'params':list(model.base[-1].parameters()),'lr':.00005})
            replay,receipts=gather(torch,w0,stage1,TRAIN)
            rx,rm,ry,rl=w18.tensorize(torch,replay)
            w18.save(args.output/'onpolicy-replay.json',{'rows':len(replay),'receipts':receipts,
                'real_model_head_forwards':sum(r['events'] for r in receipts if r['controller']=='W18-t07'),
                'synthetic':True})
        for step in range(UPDATES):
            opt.zero_grad(set_to_none=True)
            ids=[rng.randrange(len(contrast)) for _ in range(48)]
            sa=model(x0[ids],m0[ids]).masked_fill(~l0[ids],-1e4)
            sb=model(x1[ids],m1[ids]).masked_fill(~l1[ids],-1e4)
            va=y0[ids];vb=y1[ids]
            ce=(torch.nn.functional.cross_entropy(sa,va)+torch.nn.functional.cross_entropy(sb,vb))/2
            ga=sa.gather(1,va[:,None]).flatten()-sa.gather(1,vb[:,None]).flatten()
            gb=sb.gather(1,vb[:,None]).flatten()-sb.gather(1,va[:,None]).flatten()
            contrast_loss=(torch.nn.functional.softplus(1.25-ga)+torch.nn.functional.softplus(1.25-gb)).mean()/2
            idx=[rng.randrange(len(rows)) for _ in range(48)]
            ordinary=torch.nn.functional.cross_entropy(model(xa[idx],ma[idx]).masked_fill(~la[idx],-1e4),ya[idx])
            if wave:
                rid=[rng.randrange(len(replay)) for _ in range(64)]
                replay_ce=torch.nn.functional.cross_entropy(model(rx[rid],rm[rid]).masked_fill(~rl[rid],-1e4),ry[rid])
                # Source-W18 head is read-only; rehearse its BRANCH preferences on actual W0 observations.
                with torch.no_grad():
                    old=frozen(xa[idx],ma[idx]).masked_fill(~la[idx],-1e4)
                recent=model(xa[idx],ma[idx]).masked_fill(~la[idx],-1e4)
                branch=w18.ACTIONS.index('BRANCH')
                retain=torch.nn.functional.mse_loss(torch.softmax(recent,1)[:,branch],torch.softmax(old,1)[:,branch])
                loss=.30*ce+.34*contrast_loss+.12*ordinary+.18*replay_ce+.06*retain
            else:loss=.42*ce+.43*contrast_loss+.15*ordinary
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite loss')
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(filter(lambda p:p.requires_grad,model.parameters()),3.)
            if not bool(torch.isfinite(grad)) or float(grad)<=0:raise ValueError('invalid gradient')
            opt.step()
            if (step+1)%20==0:
                history.append({'stage':name,'stage_step':step+1,'genuine_step':wave*UPDATES+step+1,
                    'loss':round(float(loss.detach()),6),'grad_norm':round(float(grad),6)})
                w18.save(args.output/'partial-optimizer-history.json',history)
        stats={'ordinary':w18.evaluate(torch,model,hold),'material_pairs':pair_eval(torch,model,dev)}
        cp=checkpoint(torch,model,opt,rng,args.output,name,stats)
        stages.append({'name':name,'genuine_adamw_steps':UPDATES,'metrics':stats,'checkpoint':cp})
        w18.save(args.output/'stage-progress.json',stages)
    gallery=args.output/'original-material-png-json';gallery.mkdir(exist_ok=True)
    results=[]
    controllers=[('W17',w17,0),('W18-frozen',frozen,0),('W19-stage1',stage1,0),
        ('W19-final',model,0),('W19-final-t07',model,.7),
        ('random',None,0),('wall',None,0),('teacher',None,0)]
    for seed in HELDOUT:
        for family in w18.FAMILIES:
            for mode,policy,temp in controllers:
                rec=w18.rollout(torch,w0,policy,seed,family,mode,temp)
                file=f'{family}-{mode}-{seed}'
                jp=gallery/(file+'.json');pp=gallery/(file+'.png')
                w18.save(jp,rec)
                pp.write_bytes(w18.material_png(w0,rec))
                results.append({'seed':seed,'family':family,'controller':mode,
                    'metrics':rec['metrics'],'actual_head_forwards':rec['original_head_forwards'],
                    'filtered_actions':rec['additional_host_legal_filters'],
                    'json_sha256':w18.sha(jp),'png_sha256':w18.sha(pp),
                    'trajectory_sha256':hashlib.sha256(json.dumps(rec['trajectory'],separators=(',',':')).encode()).hexdigest()})
        w18.save(args.output/'partial-gallery.json',results)
    assert len(results)==72 and all(r['metrics']['invalid_raw_choices']==0 for r in results)
    assert all(w18.sha(args.output/x['checkpoint']['file'])==x['checkpoint']['sha256'] for x in stages if 'checkpoint' in x)
    w18.save(args.output/'summary.json',{'status':'complete','version':VERSION,
        'genuine_aux_AdamW_updates':320,'root_model_optimizer_steps':0,'root_forwards':0,
        'LoRA_updates':0,'original_png_json_pairs':len(results),
        'real_head_forwards':sum(r['actual_head_forwards'] for r in results),
        'stages':stages,'results':results,'wall_seconds':time.monotonic()-start,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'host_enforced_protected_violations':0,'native':False})
    print(json.dumps({'W19':'complete','steps':320,'original_pngs':len(results)}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('w18','w17','w0','output'):p.add_argument('--'+key,type=Path)
    p.add_argument('--smoke-only',action='store_true')
    a=p.parse_args()
    try:
        if not a.smoke_only and not all((a.w18,a.w17,a.w0,a.output)):p.error('all original SHA-pinned paths required')
        main(a)
    except BaseException as e:
        if a.output and a.output.exists():w18.save(a.output/'failure.json',{'type':type(e).__name__,'error':str(e)})
        raise
