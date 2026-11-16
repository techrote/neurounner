#!/usr/bin/env python3
"""W21: trajectory-recovery calibration of the SHA-pinned W20 material policy.

All actions are selected in an external W0 synthetic host with host-owned
legality/protection. This does not alter CyberSand and does not load Sprocket
root weights. W21 seeks material sensitivity WITHOUT W20's blind RETRACE bias.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, resource, time
from collections import Counter
from pathlib import Path
import worm_train_v18 as w18
import worm_train_v19 as w19
import worm_train_v20 as w20

VERSION = 'sprocket-w21-trajectory-recovery-material-retention-v1'
W20_SHA = '5562ccae559f466ac4efd7532f37820b192a7689df631a79aef3f3b886e41632'
W19_SHA = w20.W19_SHA
W17_SHA = w18.W17_SHA
W0_SHA = w18.W0_SHA
TRAIN = (601, 607, 613, 617)
HELDOUT = (4001, 4013, 4021)
WAVES = 160

def load_w20(torch, path, w19_source):
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert state['version'] == w20.VERSION
    assert state['root_model_loaded'] is False and state['root_forwards'] == 0
    assert state['root_updates'] == state['LoRA_updates'] == 0
    policy = w20.policy_class(torch)(w19_source)
    policy.load_state_dict(state['model'], strict=True)
    policy.eval()
    return policy

def clone(torch, model, w19_source):
    p = w20.policy_class(torch)(w19_source)
    p.load_state_dict(model.state_dict(), strict=True)
    p.eval()
    return p

def focused(obs):
    """Host legal and physically free next-step frontier, not merely a logit."""
    count = 0
    for action, rotation in (('STRAIGHT', 0), ('LEFT', -1), ('RIGHT', 1)):
        if action not in obs['legal']:
            continue
        dx, dy = w18.DIRECTIONS[(obs['heading'] + rotation) % 4]
        if obs['local_5x5'][dy+2][dx+2] == '.':
            count += 1
    return count

def gather(torch, w0, models, seeds):
    rows = []
    receipts = []
    for seed in seeds:
        for family in w18.FAMILIES:
            for label, model, temp in models:
                rec = w18.rollout(torch, w0, model, seed, family, label, temp)
                these = w18.to_rows([rec], seed)
                rows += these
                receipts.append({
                    'seed': seed, 'family': family, 'controller': label,
                    'events': len(these), 'actual_head_forwards': rec['original_head_forwards'],
                    'host_protection_filters': rec['additional_host_legal_filters']})
    return rows, receipts

def select_sets(torch, rows, frozen, require_coverage=True):
    x, m, y, mask = w18.tensorize(torch, rows)
    R = w18.ACTIONS.index('RETRACE')
    B = w18.ACTIONS.index('BRANCH')
    with torch.no_grad():
        preds = frozen(x, m).masked_fill(~mask, -1e4).argmax(1).tolist()
    hard = [i for i, ((o, action), pred) in enumerate(zip(rows, preds))
            if 'RETRACE' in o['legal'] and action != 'RETRACE'
            and focused(o) > 0 and pred == R]
    progress = [i for i, (o, action) in enumerate(rows)
                if 'RETRACE' in o['legal'] and action != 'RETRACE' and focused(o) > 0]
    recovery = [i for i, (_, action) in enumerate(rows) if action == 'RETRACE']
    branch = [i for i, (_, action) in enumerate(rows) if action == 'BRANCH']
    if require_coverage:
        assert len(progress) >= 16
        assert recovery and branch
    return (x,m,y,mask), {'hard': hard, 'progress': progress,
                           'recovery': recovery, 'branch': branch,
                           'frozen_false_retrace': len(hard)}

def retrace_eval(torch, model, rows):
    x,m,y,legal = w18.tensorize(torch, rows)
    with torch.no_grad():
        p = model(x,m).masked_fill(~legal,-1e4).argmax(1)
    retrace = w18.ACTIONS.index('RETRACE')
    eligible = torch.tensor([focused(o)>0 and 'RETRACE' in o['legal'] for o,_ in rows],
                            dtype=torch.bool)
    return {'rows': len(rows), 'frontier_eligible': int(eligible.sum()),
            'false_retrace_with_free_frontier': int(((p==retrace)&(y!=retrace)&eligible).sum()),
            'retrace_teacher_positive': int((y==retrace).sum()),
            'retrace_true_positive': int(((p==retrace)&(y==retrace)).sum()),
            'teacher_agreement': round(float((p==y).float().mean()), 6),
            'teacher_counts': dict(Counter(a for _,a in rows))}

def snapshot(torch, model, opt, rng, path, stage, stats):
    target = path / (stage+'.pt')
    torch.save({'version': VERSION, 'stage':stage,'model':model.state_dict(),
                'optimizer':opt.state_dict(),'random_state':rng.getstate(),
                'torch_rng':torch.random.get_rng_state(),'assessment':stats,
                'W20_original_sha256':W20_SHA,'root_model_loaded':False,
                'root_forwards':0,'root_optimizer_steps':0,'LoRA_updates':0,
                'native':False},target)
    return {'file':target.name,'sha256':w18.sha(target),'bytes':target.stat().st_size}

def main(args):
    assert (w18.ROOT,w18.REVISION,w18.EOS) == (
        'kandivault/sprocket-500m','eb157d4ddddc4619e661d4a2426bb2dd000fc668',5)
    assert not set(TRAIN) & set(HELDOUT)
    assert not set(TRAIN+HELDOUT) & set(w18.TRAIN+w18.HELDOUT+w19.TRAIN+w19.HELDOUT+w20.TRAIN+w20.HELDOUT)
    if args.smoke_only:
        assert WAVES==160
        assert all(s not in w20.HELDOUT for s in TRAIN+HELDOUT)
        print('W21 source preflight PASS: W20 continuation, distinct seeds, root frozen')
        return
    start = time.monotonic()
    args.output.mkdir(parents=True,exist_ok=False)
    import torch
    torch.manual_seed(212121)
    torch.set_num_threads(1)
    rng = random.Random(212121)
    assert w18.sha(args.w20)==W20_SHA and w18.sha(args.w19)==W19_SHA
    assert w18.sha(args.w17)==W17_SHA and w18.sha(args.w0)==W0_SHA
    w0 = w18.w0_module(args.w0)
    frozen_w19 = w20.source(torch,args.w19,w19.VERSION)
    frozen_w17 = w19.source_w17(torch,args.w17)
    frozen_w20 = load_w20(torch,args.w20,frozen_w19)
    model = clone(torch,frozen_w20,frozen_w19)
    for p in model.parent.parameters():
        assert not p.requires_grad
    assert sum(p.requires_grad for p in model.parameters())==4
    w18.save(args.output/'provenance.json',{
        'version':VERSION,'source_head':os.getenv('GITHUB_SHA','local'),
        'run_id':os.getenv('GITHUB_RUN_ID','local'),
        'attempt':os.getenv('GITHUB_RUN_ATTEMPT','local'),
        'script_sha256':w18.sha(__file__),
        'W20_checkpoint_sha256':W20_SHA,'W19_checkpoint_sha256':W19_SHA,
        'W17_checkpoint_sha256':W17_SHA,'W0_source_sha256':W0_SHA,
        'root':w18.ROOT,'revision':w18.REVISION,'EOS':w18.EOS,
        'root_loaded':False,'root_forwards':0,'root_optimizer_steps':0,
        'LoRA_updates':0,'synthetic_host':True,'native':False,
        'FINAL_opened':False,'owner_gate':'Draft/HOLD'})
    train, train_receipts = gather(torch,w0,[
        ('teacher',None,0),('random',None,0),
        ('W19-frozen',frozen_w19,0),('W20-frozen',frozen_w20,0),
        ('W20-sampled',frozen_w20,0.7)],TRAIN)
    hold, hold_receipts = gather(torch,w0,[
        ('teacher',None,0),('random',None,0),
        ('W19-frozen',frozen_w19,0),('W20-frozen',frozen_w20,0),
        ('W20-sampled',frozen_w20,0.7)],HELDOUT)
    # World-persistent one-cell contrasts from actual W0 host-legal prefixes.
    _, pairs, pair_train_receipts, _ = w20.interventions(w0,torch,frozen_w19,TRAIN)
    _, dev, pair_hold_receipts, examples = w20.interventions(w0,torch,frozen_w19,HELDOUT)
    witnesses = w20.verify_prefix_replay(w0,examples)
    assert len(pairs)>80 and len(dev)>40 and len(witnesses)>=12
    w18.save(args.output/'legal-prefix-material-witnesses.json',witnesses)
    (x,m,y,legal), sets = select_sets(torch,train,frozen_w20)
    px0,px1=w19.pair_tensors(torch,pairs)
    (x0,m0,y0,l0),(x1,m1,y1,l1)=px0,px1
    w18.save(args.output/'dataset.json',{
        'train_worlds':TRAIN,'heldout_worlds':HELDOUT,
        'train_rows':len(train),'heldout_rows':len(hold),
        'material_training_pairs':len(pairs),'material_heldout_pairs':len(dev),
        'train_receipts':train_receipts,'heldout_receipts':hold_receipts,
        'material_train_receipts':pair_train_receipts,
        'material_heldout_receipts':pair_hold_receipts,
        'hard_negative_counts':{k:len(v) if isinstance(v,list) else v for k,v in sets.items()},
        'host_legal_material_witnesses':len(witnesses),
        'synthetic_host':True,'native':False})
    opt=torch.optim.AdamW(model.delta.parameters(),lr=.00045,weight_decay=.005)
    stages=[{'name':'frozen-W20','ordinary':w18.evaluate(torch,frozen_w20,hold),
             'retrace':retrace_eval(torch,frozen_w20,hold),
             'material_pairs':w20.pair_eval(torch,frozen_w20,dev)}]
    history=[]
    snapshots=[]
    replay=None
    for wave,stage in enumerate(('progress-negative-recovery',
                                 'material-counterfactual-rehearsal',
                                 'onpolicy-replay-retention')):
        if wave==2:
            replay,rr=gather(torch,w0,[
                ('W21-stage2-greedy',clone(torch,model,frozen_w19),0),
                ('W21-stage2-T07',clone(torch,model,frozen_w19),0.7)],TRAIN)
            (rx,rm,ry,rl),replay_sets=select_sets(torch,replay,frozen_w20,require_coverage=False)
            w18.save(args.output/'onpolicy-replay.json',{
                'rows':len(replay),'receipts':rr,'hard_count':len(replay_sets['hard']),
                'real_head_forwards':sum(v['actual_head_forwards'] for v in rr),
                'source':'W21-stage2, frozen W20 queried for hard negatives'})
        for step in range(WAVES):
            opt.zero_grad(set_to_none=True)
            def take(space,n):
                return [space[rng.randrange(len(space))] for _ in range(n)]
            def ce_on(ids,tensors=(x,m,y,legal)):
                a,b,c,d=tensors
                return torch.nn.functional.cross_entropy(
                    model(a[ids],b[ids]).masked_fill(~d[ids],-1e4),c[ids])
            regular=ce_on(take(list(range(len(train))),64))
            difficult=ce_on(take(sets['hard'] or sets['progress'],64))
            recover=ce_on(take(sets['recovery'],32))
            branching=ce_on(take(sets['branch'],32))
            hi=take(sets['progress'],48)
            logits=model(x[hi],m[hi]).masked_fill(~legal[hi],-1e4)
            retract=w18.ACTIONS.index('RETRACE')
            progress_margin=torch.nn.functional.softplus(
                logits[:,retract]-logits.gather(1,y[hi,None]).flatten()+0.5).mean()
            ip=take(list(range(len(pairs))),48)
            sa=model(x0[ip],m0[ip]).masked_fill(~l0[ip],-1e4)
            sb=model(x1[ip],m1[ip]).masked_fill(~l1[ip],-1e4)
            pair_ce=(torch.nn.functional.cross_entropy(sa,y0[ip])+
                     torch.nn.functional.cross_entropy(sb,y1[ip]))/2
            contrast=(torch.nn.functional.softplus(
                0.8-(sa.gather(1,y0[ip,None]).flatten()-
                      sa.gather(1,y1[ip,None]).flatten())).mean()+
                torch.nn.functional.softplus(
                0.8-(sb.gather(1,y1[ip,None]).flatten()-
                      sb.gather(1,y0[ip,None]).flatten())).mean())/2
            if wave==0:
                loss=.23*regular+.31*difficult+.15*recover+.10*branching+\
                     .16*progress_margin+.05*pair_ce
            elif wave==1:
                loss=.15*regular+.19*difficult+.10*recover+.10*branching+\
                     .14*progress_margin+.21*pair_ce+.11*contrast
            else:
                rids=take(list(range(len(replay))),64)
                replay_loss=ce_on(rids,(rx,rm,ry,rl))
                rh=take(replay_sets['hard'] or replay_sets['progress'] or list(range(len(replay))),48)
                replay_hard=ce_on(rh,(rx,rm,ry,rl))
                loss=.11*regular+.12*difficult+.10*recover+.09*branching+\
                     .10*progress_margin+.17*pair_ce+.09*contrast+\
                     .14*replay_loss+.08*replay_hard
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.delta.parameters(),3.0)
            if not bool(torch.isfinite(norm)) or float(norm)<=0:
                raise ValueError('invalid optimizer gradient')
            opt.step()
            if (step+1)%20==0:
                history.append({'stage':stage,'stage_step':step+1,
                    'completed_updates':wave*WAVES+step+1,
                    'loss':round(float(loss.detach()),6),
                    'grad_norm':round(float(norm),6)})
                w18.save(args.output/'partial-optimizer-history.json',history)
        assessment={'ordinary':w18.evaluate(torch,model,hold),
                    'retrace':retrace_eval(torch,model,hold),
                    'material_pairs':w20.pair_eval(torch,model,dev)}
        cp=snapshot(torch,model,opt,rng,args.output,stage,assessment)
        stages.append({'name':stage,'genuine_adamw_updates':WAVES,
                       'assessment':assessment,'checkpoint':cp})
        snapshots.append(clone(torch,model,frozen_w19))
        w18.save(args.output/'stage-progress.json',stages)
    out=args.output/'original-material-png-json'
    out.mkdir(exist_ok=True)
    results=[]
    ctrls=[
        ('W17',frozen_w17,0),('W19',frozen_w19,0),
        ('W20-frozen',frozen_w20,0),
        ('W21-stage1',snapshots[0],0),('W21-stage2',snapshots[1],0),
        ('W21-final',model,0),('W21-final-T07',model,0.7),
        ('W21-final-T11',model,1.1),
        ('teacher',None,0),('random',None,0),('wall',None,0)]
    for seed in HELDOUT:
        for family in w18.FAMILIES:
            for name,policy,temp in ctrls:
                rec=w18.rollout(torch,w0,policy,seed,family,name,temp)
                filename=f'{family}-{name}-{seed}'
                jp=out/(filename+'.json');pp=out/(filename+'.png')
                w18.save(jp,rec)
                pp.write_bytes(w18.material_png(w0,rec))
                results.append({'seed':seed,'family':family,'controller':name,
                    'metrics':rec['metrics'],'head_forwards':rec['original_head_forwards'],
                    'host_filtered':rec['additional_host_legal_filters'],
                    'sha_json':w18.sha(jp),'sha_png':w18.sha(pp),
                    'trajectory_sha256':hashlib.sha256(
                        json.dumps(rec['trajectory'],separators=(',',':')).encode()).hexdigest()})
        w18.save(args.output/'partial-gallery.json',results)
    assert len(results)==99 and all(r['metrics']['invalid_raw_choices']==0 for r in results)
    assert all(w18.sha(args.output/s['checkpoint']['file'])==s['checkpoint']['sha256']
               for s in stages if 'checkpoint' in s)
    w18.save(args.output/'summary.json',{
        'status':'complete','version':VERSION,'genuine_aux_AdamW_updates':3*WAVES,
        'root_forwards':0,'root_updates':0,'LoRA_updates':0,
        'original_png_json_pairs':len(results),
        'real_head_forwards':sum(r['head_forwards'] for r in results),
        'stages':stages,'results':results,
        'wall_seconds':time.monotonic()-start,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'native':False,'material_physics_synthetic':True,
        'autonomous_material_counterfactual_reachability_unproven':True})
    print(json.dumps({'W21':'complete','genuine_adamw_steps':3*WAVES,
                      'original_paintings':len(results)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('w20','w19','w17','w0','output'):
        p.add_argument('--'+k,type=Path)
    p.add_argument('--smoke-only',action='store_true')
    a=p.parse_args()
    try:
        if not a.smoke_only and not all((a.w20,a.w19,a.w17,a.w0,a.output)):
            p.error('four SHA-admitted sources and output path required')
        main(a)
    except BaseException as exc:
        if a.output and a.output.exists():
            w18.save(a.output/'failure.json',
                     {'type':type(exc).__name__,'message':str(exc),'status':'failed'})
        raise
