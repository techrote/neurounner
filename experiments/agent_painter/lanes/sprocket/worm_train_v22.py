#!/usr/bin/env python3
"""W22: bounded lookahead/frontier distillation after W21's dead-end diagnosis.

Research-only synthetic W0 obstacle/protection overlay. The SHA-pinned W0 host
still owns legality, moves, Stop and replay. No native CyberSand modification,
root-model load, LoRA training, production provider update or FINAL access.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, resource, time
from collections import Counter, deque
from pathlib import Path
import worm_train_v18 as w18
import worm_train_v19 as w19
import worm_train_v20 as w20
import worm_train_v21 as w21

VERSION='sprocket-w22-frontier-trap-anticipation-v1'
W21_SHA='d2508b0cda096e571da53abb579d38f9ad385248e1f70bc646b9dffee8f7401e'
W20_SHA=w21.W20_SHA
W19_SHA=w20.W19_SHA
W17_SHA=w18.W17_SHA
W0_SHA=w18.W0_SHA
TRAIN=(701,709,719,727)
HELDOUT=(5003,5011,5021)
WAVES=160
MOVES={'STRAIGHT':0,'LEFT':-1,'RIGHT':1}

def overlay_host(w0):
    """Deterministic external obstacle fixtures, never edit W0 source or game."""
    original=w0.initial_world
    def make(seed,size=96):
        cells=set(original(seed,size))
        # C-shaped pockets with open west mouths, and partial corridor baffles.
        # Keep start at (7,48) free. Geometry varies with seed; no final split.
        dx=seed%4
        for cx,cy in ((15+dx,48),(21+dx,39+(seed%3)),(25+dx,57-(seed%3))):
            for x in range(cx,cx+7):
                cells.add((x,cy-3));cells.add((x,cy+3))
            for y in range(cy-3,cy+4):cells.add((cx+6,y))
            for y in (cy-1,cy,cy+1):cells.discard((cx,y))
        for yy in range(44,54):
            if yy not in (46+(seed%3),50+(seed%2)):
                cells.add((33+dx,yy))
        cells.discard((7,size//2))
        return cells
    w0.initial_world=make
    return {'kind':'external_seeded_C_pockets_and_corridor_baffles',
            'w0_source_unmodified':True,'width':96,
            'parameters':{'pockets':3,'wall_span':7,'baffle_x_base':33}}

def analysis(obs,action):
    """Censored 5x5 1-3-hop frontier: a predictor, NOT a true map solver."""
    if action not in MOVES or action not in obs['legal']:return None
    direction=(obs['heading']+MOVES[action])%4
    dx,dy=w18.DIRECTIONS[direction]
    cells=obs['local_5x5']
    if cells[dy+2][dx+2]!='.':return None
    q=deque([(dx,dy,0)]);seen={(dx,dy)};continuations=0;edge=0
    while q:
        x,y,depth=q.popleft()
        if abs(x)==2 or abs(y)==2:edge+=1
        if depth>=3:continue
        for vx,vy in w18.DIRECTIONS:
            nx,ny=x+vx,y+vy
            if not (-2<=nx<=2 and -2<=ny<=2):continue
            if (nx,ny) in seen or cells[ny+2][nx+2]!='.':continue
            if depth==0:continuations+=1
            seen.add((nx,ny));q.append((nx,ny,depth+1))
    # Edge sightings are censored unknown space, not known dead ends.
    return {'reachable':len(seen),'exits':continuations,'censored':edge,
            'score':len(seen)+2*continuations+1.5*min(edge,2)}

def plan(obs):
    """Choose legal action by material preference plus short frontier potential."""
    candidates={a:analysis(obs,a) for a in MOVES}
    fresh={a:v for a,v in candidates.items() if v is not None}
    orig=w18.teacher(obs)
    if not fresh:return orig,False,{'best_frontier':0,'teacher_frontier':0}
    desired={'greebles':'metal','pipework':'water','architecture':'stone'}[obs['family']]
    scores={}
    for a,v in fresh.items():
        direction=(obs['heading']+MOVES[a])%4
        dx,dy=w18.DIRECTIONS[direction]
        mat=obs['material_5x5'][dy+2][dx+2]
        # Do not infer unseen cells beyond the 5x5 observation.
        scores[a]=(0.55*v['score']+1.65*float(mat==desired)
                   +0.12*float(a=='STRAIGHT')
                   -1.8*float(v['exits']==0 and v['censored']==0))
        if obs['family']=='architecture' and mat=='water':scores[a]-=0.8
    best=max(scores,key=scores.get)
    # Branch is kept conditionally for motif scaffolding rather than erased.
    if orig=='BRANCH' and 'BRANCH' in obs['legal'] and len(fresh)>=2:
        ordered=sorted((v['score'] for v in fresh.values()),reverse=True)
        if ordered[1]>=3 and 'BRANCH' not in obs['last_actions'][-3:]:
            return 'BRANCH',False,{'best_frontier':round(ordered[0],3),'teacher_frontier':round(ordered[1],3)}
    if orig in fresh:
        current=scores[orig]
        # Hysteresis: change teacher choice only for a measurable trap disparity.
        if scores[best]-current<1.6:
            return orig,False,{'best_frontier':round(scores[best],3),'teacher_frontier':round(current,3)}
    return best,best!=orig,{'best_frontier':round(scores[best],3),
                            'teacher_frontier':round(scores.get(orig,-1),3)}

def supervision(torch,rows):
    labels=[];changes=0;counts=Counter()
    for obs,_ in rows:
        target,changed,_=plan(obs)
        assert target in obs['legal']
        labels.append(w18.ACTIONS.index(target))
        if changed:changes+=1;counts[target]+=1
    return torch.tensor(labels,dtype=torch.long),changes,dict(counts)

def frontier_eval(torch,model,rows):
    x,m,teacher,mask=w18.tensorize(torch,rows)
    targets,nchange,_=supervision(torch,rows)
    with torch.no_grad():logit=model(x,m).masked_fill(~mask,-1e4)
    pred=logit.argmax(1).tolist()
    raw=Counter();avoidable=0;choice=0
    for (obs,_),idx in zip(rows,pred):
        chosen=w18.ACTIONS[idx]
        known={a:analysis(obs,a) for a in MOVES}
        good={a:z for a,z in known.items() if z is not None}
        if not good:continue
        choice+=1
        cur=known.get(chosen)
        top=max(v['score'] for v in good.values())
        if cur and top-cur['score']>=4 and cur['censored']==0:
            avoidable+=1
            raw[chosen]+=1
    return {'rows':len(rows),'planner_changes':nchange,
            'planner_agreement':round(sum(a==b for a,b in zip(pred,targets.tolist()))/max(1,len(rows)),6),
            'teacher_agreement':round(sum(a==b for a,b in zip(pred,teacher.tolist()))/max(1,len(rows)),6),
            'known_frontier_decisions':choice,
            'high_margin_avoidable_local_trap_choices':avoidable,
            'trap_by_action':dict(raw)}

def loader(torch,path,frozen_w19):
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert state['version']==w21.VERSION
    assert state['root_model_loaded'] is False and state['root_forwards']==0
    assert state['LoRA_updates']==0
    model=w20.policy_class(torch)(frozen_w19)
    model.load_state_dict(state['model'],strict=True)
    model.eval()
    return model

def checkpoint(torch,model,opt,rng,out,name,assessment):
    target=out/(name+'.pt')
    torch.save({'version':VERSION,'stage':name,'model':model.state_dict(),
                'optimizer':opt.state_dict(),'random_state':rng.getstate(),
                'torch_rng':torch.random.get_rng_state(),'assessment':assessment,
                'W21_original_sha256':W21_SHA,
                'root_loaded':False,'root_forwards':0,
                'root_optimizer_steps':0,'LoRA_updates':0,'native':False},target)
    return {'file':target.name,'sha256':w18.sha(target),'bytes':target.stat().st_size}

def train(args):
    assert (w18.ROOT,w18.REVISION,w18.EOS)==(
        'kandivault/sprocket-500m','eb157d4ddddc4619e661d4a2426bb2dd000fc668',5)
    prior=set(w18.TRAIN+w18.HELDOUT+w19.TRAIN+w19.HELDOUT+
              w20.TRAIN+w20.HELDOUT+w21.TRAIN+w21.HELDOUT)
    assert not prior.intersection(TRAIN+HELDOUT) and not set(TRAIN)&set(HELDOUT)
    if args.smoke_only:
        assert WAVES==160 and W21_SHA!='' and len(HELDOUT)==3
        sample={'x':7,'y':48,'heading':0,'tick':0,'family':'greebles',
                'last_actions':[],'legal':list(w18.ACTIONS),
                'local_5x5':['.....','.....','..@..','.....','.....'],
                'material_5x5':[['sand']*5 for _ in range(5)]}
        assert plan(sample)[0] in sample['legal']
        print('W22 source preflight PASS: independent seeds and local-frontier legal target')
        return
    start=time.monotonic(); args.output.mkdir(parents=True,exist_ok=False)
    import torch
    torch.manual_seed(222222);torch.set_num_threads(1)
    rng=random.Random(222222)
    for file,digest in ((args.w21,W21_SHA),(args.w19,W19_SHA),
                        (args.w20,W20_SHA),(args.w17,W17_SHA),(args.w0,W0_SHA)):
        assert w18.sha(file)==digest,(file,'SHA mismatch')
    w0=w18.w0_module(args.w0)
    source_w19=w20.source(torch,args.w19,w19.VERSION)
    source_w17=w19.source_w17(torch,args.w17)
    source_w20=w21.load_w20(torch,args.w20,source_w19)
    frozen=loader(torch,args.w21,source_w19)
    model=w21.clone(torch,frozen,source_w19)
    assert sum(p.requires_grad for p in model.parameters())==4
    w18.save(args.output/'provenance.json',{
        'version':VERSION,'source_head':os.getenv('GITHUB_SHA','local'),
        'script_sha256':w18.sha(__file__),'W21_checkpoint_sha256':W21_SHA,
        'W20_checkpoint_sha256':W20_SHA,'W19_checkpoint_sha256':W19_SHA,
        'W17_checkpoint_sha256':W17_SHA,'W0_source_sha256':W0_SHA,
        'root':w18.ROOT,'revision':w18.REVISION,'EOS':w18.EOS,
        'root_loaded':False,'root_forwards':0,'LoRA_updates':0,
        'FINAL_opened':False,'native':False,'owner_gate':'Draft/HOLD'})
    # SHA-admitted material contrasts retain old host, then independently
    # install obstacle fixture overlay for W22 train/evaluation worlds.
    _,pairs,train_witnesses,_=w20.interventions(w0,torch,source_w19,TRAIN)
    _,held_pairs,held_witnesses,_=w20.interventions(w0,torch,source_w19,HELDOUT)
    fixture=overlay_host(w0)
    w18.save(args.output/'obstacle-fixture.json',fixture)
    train_rows,train_receipts=w21.gather(torch,w0,[
        ('teacher',None,0),('random',None,0),
        ('W21-frozen',frozen,0),('W21-T07',frozen,0.7)],TRAIN)
    held_rows,held_receipts=w21.gather(torch,w0,[
        ('teacher',None,0),('random',None,0),
        ('W21-frozen',frozen,0),('W21-T07',frozen,0.7)],HELDOUT)
    x,m,ty,mask=w18.tensorize(torch,train_rows)
    planner,changes,targets=supervision(torch,train_rows)
    hp,hchanges,_=supervision(torch,held_rows)
    px0,px1=w19.pair_tensors(torch,pairs)
    (a0,b0,y0,l0),(a1,b1,y1,l1)=px0,px1
    assert changes>=0 and len(held_rows)>100
    w18.save(args.output/'dataset.json',{
        'train_seeds':TRAIN,'heldout_seeds':HELDOUT,'train_rows':len(train_rows),
        'heldout_rows':len(held_rows),'planner_changes_train':changes,
        'planner_changes_heldout':hchanges,'targets_changed':targets,
        'material_pair_train':len(pairs),'material_pair_heldout':len(held_pairs),
        'material_train_witnesses':train_witnesses,
        'material_heldout_witnesses':held_witnesses,
        'receipts_train':train_receipts,'receipts_heldout':held_receipts,
        'source_w0_unmodified':True,'obstacle_fixture_overlay':True})
    opt=torch.optim.AdamW(model.delta.parameters(),lr=.00028,weight_decay=.008)
    stages=[{'stage':'frozen-W21','frontier':frontier_eval(torch,frozen,held_rows),
             'ordinary':w18.evaluate(torch,frozen,held_rows),
             'material':w20.pair_eval(torch,frozen,held_pairs)}]
    history=[];snapshots=[];replay=None
    for wave,name in enumerate(('frontier-potential-distillation',
                                'closed-loop-trap-replay',
                                'motif-material-history-retention')):
        if wave in (1,2):
            policies=[(f'W22-stage{wave}-greedy',w21.clone(torch,model,source_w19),0),
                      (f'W22-stage{wave}-T07',w21.clone(torch,model,source_w19),0.7)]
            replay,receipts=w21.gather(torch,w0,policies,TRAIN)
            rx,rm,ry,rl=w18.tensorize(torch,replay)
            rp,_,_=supervision(torch,replay)
            w18.save(args.output/f'onpolicy-replay-stage{wave}.json',
                     {'rows':len(replay),'receipts':receipts,'source_model_stage':wave})
        with torch.no_grad():
            freeze_logits=frozen(x,m).masked_fill(~mask,-1e4)
        changed_ids=(planner!=ty).nonzero().flatten().tolist()
        regular_ids=list(range(len(train_rows)))
        branch_ids=(ty==w18.ACTIONS.index('BRANCH')).nonzero().flatten().tolist()
        for step in range(WAVES):
            opt.zero_grad(set_to_none=True)
            def take(items,n):
                return [items[rng.randrange(len(items))] for _ in range(n)]
            ix=take(regular_ids,64)
            log=model(x[ix],m[ix]).masked_fill(~mask[ix],-1e4)
            planner_loss=torch.nn.functional.cross_entropy(log,planner[ix])
            original_loss=torch.nn.functional.cross_entropy(log,ty[ix])
            hix=take(changed_ids or regular_ids,48)
            hard=torch.nn.functional.cross_entropy(
                model(x[hix],m[hix]).masked_fill(~mask[hix],-1e4),planner[hix])
            i=take(list(range(len(pairs))),48)
            mat_loss=(torch.nn.functional.cross_entropy(model(a0[i],b0[i]).masked_fill(~l0[i],-1e4),y0[i])+
                      torch.nn.functional.cross_entropy(model(a1[i],b1[i]).masked_fill(~l1[i],-1e4),y1[i]))/2
            bi=take(branch_ids or regular_ids,24)
            branch_loss=torch.nn.functional.cross_entropy(
                model(x[bi],m[bi]).masked_fill(~mask[bi],-1e4),ty[bi])
            frozen_probs=torch.nn.functional.softmax(freeze_logits[ix],dim=1)
            kl=torch.nn.functional.kl_div(torch.nn.functional.log_softmax(log,dim=1),
                 frozen_probs,reduction='batchmean')
            if wave==0:loss=.51*planner_loss+.23*hard+.10*mat_loss+.09*original_loss+.04*branch_loss+.03*kl
            elif wave==1:
                ri=take(list(range(len(replay))),64)
                rlog=model(rx[ri],rm[ri]).masked_fill(~rl[ri],-1e4)
                replay_loss=torch.nn.functional.cross_entropy(rlog,rp[ri])
                loss=.34*planner_loss+.20*hard+.16*replay_loss+.12*mat_loss+.08*original_loss+.06*branch_loss+.04*kl
            else:
                ri=take(list(range(len(replay))),64)
                rlog=model(rx[ri],rm[ri]).masked_fill(~rl[ri],-1e4)
                replay_loss=torch.nn.functional.cross_entropy(rlog,rp[ri])
                loss=.25*planner_loss+.14*hard+.13*replay_loss+.17*mat_loss+.15*original_loss+.10*branch_loss+.06*kl
            assert bool(torch.isfinite(loss))
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.delta.parameters(),3.0)
            if not bool(torch.isfinite(grad)) or float(grad)<=0:raise ValueError('non-finite or zero optimizer gradient')
            opt.step()
            if (step+1)%20==0:
                history.append({'stage':name,'step':step+1,'updates':wave*WAVES+step+1,
                                'loss':round(float(loss.detach()),6),
                                'gradient_norm':round(float(grad),6)})
                w18.save(args.output/'partial-optimizer-history.json',history)
        assessment={'frontier':frontier_eval(torch,model,held_rows),
                    'ordinary':w18.evaluate(torch,model,held_rows),
                    'material':w20.pair_eval(torch,model,held_pairs)}
        cp=checkpoint(torch,model,opt,rng,args.output,name,assessment)
        stages.append({'stage':name,'actual_optimizer_updates':WAVES,
                       'assessment':assessment,'checkpoint':cp})
        snapshots.append(w21.clone(torch,model,source_w19))
        w18.save(args.output/'stage-progress.json',stages)
    out=args.output/'original-synthetic-png-json';out.mkdir()
    controllers=[
        ('W17',source_w17,0),('W20',source_w20,0),
        ('W21-frozen',frozen,0),
        ('W22-stage1',snapshots[0],0),('W22-stage2',snapshots[1],0),
        ('W22-final',model,0),('W22-final-T07',model,0.7),
        ('W22-final-T11',model,1.1),
        ('teacher',None,0),('random',None,0),('wall',None,0)]
    all_results=[]
    for seed in HELDOUT:
        for family in w18.FAMILIES:
            for label,controller,temp in controllers:
                rec=w18.rollout(torch,w0,controller,seed,family,label,temp)
                jp=out/f'{family}-{label}-{seed}.json'
                pp=out/f'{family}-{label}-{seed}.png'
                w18.save(jp,rec);pp.write_bytes(w18.material_png(w0,rec))
                all_results.append({'seed':seed,'family':family,'controller':label,
                    'metrics':rec['metrics'],'head_forwards':rec['original_head_forwards'],
                    'protected_filtered':rec['additional_host_legal_filters'],
                    'sha_json':w18.sha(jp),'sha_png':w18.sha(pp),
                    'trajectory_sha256':hashlib.sha256(
                        json.dumps(rec['trajectory'],separators=(',',':')).encode()).hexdigest()})
        w18.save(args.output/'partial-gallery.json',all_results)
    assert len(all_results)==99
    assert all(r['metrics']['invalid_raw_choices']==0 for r in all_results)
    assert all(w18.sha(args.output/s['checkpoint']['file'])==s['checkpoint']['sha256']
               for s in stages if 'checkpoint' in s)
    w18.save(args.output/'summary.json',{
        'status':'complete','version':VERSION,'genuine_aux_AdamW_updates':3*WAVES,
        'root_forwards':0,'root_updates':0,'LoRA_updates':0,
        'heldout_trajectory_pairs':len(all_results),'results':all_results,
        'stages':stages,'wall_seconds':time.monotonic()-start,
        'actual_head_forwards':sum(r['head_forwards'] for r in all_results),
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'native':False,'external_synthetic_obstacle_fixtures':True,
        'censored_5x5_lookahead_not_global_pathfinding':True})
    print(json.dumps({'W22':'complete','actual_updates':3*WAVES,
                     'original_png_json_pairs':len(all_results)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('w21','w20','w19','w17','w0','output'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--smoke-only',action='store_true')
    args=p.parse_args()
    try:
        if not args.smoke_only and not all((args.w21,args.w20,args.w19,args.w17,args.w0,args.output)):
            p.error('SHA-admitted original W21/W20/W19/W17 checkpoint and W0 source required')
        train(args)
    except BaseException as exc:
        if args.output and args.output.exists():
            w18.save(args.output/'failure.json',{'status':'failed','error_type':type(exc).__name__,'message':str(exc)})
        raise
