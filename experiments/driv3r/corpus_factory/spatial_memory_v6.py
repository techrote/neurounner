#!/usr/bin/env python3
"""CyberSand-style synthetic host: controlled spatial-memory action-ranking ablation.

Experimental and external. Only synthetic TRAIN world trajectories may fit the two
CPU models; DEV and held-out worlds are scored after the models are frozen.
Both arms share exactly the same host-applied state/candidate-action examples,
labels, optimizer, epoch count, and legal-action mask. The only ablation is
host-observable long-history features. No native/game modification.
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import hashlib
import io
import json
import random
import subprocess
import zipfile

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier

from recovery_curriculum import (ACTIONS, DIRS, FAMILIES, apply, distance_field,
                                  fit, legal_state, make_world, observation,
                                  oracle, teacher_data, sha, png)
from on_policy_recovery_v2 import make_cross_world, decide_model
from domain_recovery_v3 import make_slalom_world, validate_partitions, world_digest
from transfer_orientation_v5 import make_vertical_comb_world

SCHEMA = 'cybersand/synthetic-spatial-memory-action-ranking-v6'
SOURCE = tuple(range(1000,1012))
TRAIN_ORIGINAL = tuple(range(2600,2608))
TRAIN_HORIZONTAL = tuple(range(7600,7608))
TRAIN_VERTICAL = tuple(range(8600,8608))
DEV_ORIGINAL = tuple(range(4600,4604))
DEV_HORIZONTAL = tuple(range(6600,6604))
HELD_ORIGINAL = tuple(range(4700,4704))
HELD_HORIZONTAL = tuple(range(6700,6704))
HELD_SLALOM = tuple(range(9700,9712))
BUDGET = 192
EPOCHS = 24
SOURCE_PINS = {
    'recovery_curriculum.py': 'fcb1c1f5edfd5724e7cbe73fffdd9365e918b0d4',
    'closed_loop_transfer_probe.py': '928537a3f02d227c51e36ba61815315d4a2c4ff7',
    'learnability_probe_scoped.py': '78d8c899e73d692ffe0540f6ac24dd0e173c38e0',
    'on_policy_recovery_v2.py': '3804065af7432b57a52b524de4f4140c54f49e92',
    'domain_recovery_v3.py': 'c284f09de420f05f95746b141e5d990dddc22cc1',
    'transfer_orientation_v5.py': '499f5eb99f187e01ae08697897923faf07ea5b00',
}
MAKERS = {'original': make_world, 'horizontal':make_cross_world,
          'vertical_train': make_vertical_comb_world, 'unseen_slalom': make_slalom_world}


def verify_sources(path):
    for file, expected in SOURCE_PINS.items():
        actual=subprocess.run(['git','hash-object',str(Path(path)/file)],
              capture_output=True,text=True,check=True).stdout.strip()
        if actual!=expected: raise ValueError(f'failed source pin {file}: {actual}')


def difficulty_band(world,oriented_minimum):
    """Heuristic synthetic path/obstacle challenge, not empirical game difficulty."""
    blocked=len(world['obstacles'])+len(world['protected'])
    score=oriented_minimum+blocked//16
    return 'B1-short' if score<90 else 'B2-medium' if score<150 else 'B3-long'


def partitions(source=SOURCE, train_original=TRAIN_ORIGINAL,
               train_horizontal=TRAIN_HORIZONTAL, train_vertical=TRAIN_VERTICAL,
               dev_original=DEV_ORIGINAL, dev_horizontal=DEV_HORIZONTAL,
               held_original=HELD_ORIGINAL,held_horizontal=HELD_HORIZONTAL,
               held_slalom=HELD_SLALOM,budget=BUDGET):
    if not 136<=budget<=256:raise ValueError('slalom minimum route budget requires >=136 and <=256')
    group_specs={
      'TRAIN_source':(source,make_world),
      'TRAIN_original':(train_original,make_world),
      'TRAIN_horizontal':(train_horizontal,make_cross_world),
      'TRAIN_vertical':(train_vertical,make_vertical_comb_world),
      'DEV_original':(dev_original,make_world),
      'DEV_horizontal':(dev_horizontal,make_cross_world),
      'HELDOUT_original':(held_original,make_world),
      'HELDOUT_horizontal':(held_horizontal,make_cross_world),
      'HELDOUT_unseen_slalom':(held_slalom,make_slalom_world),
    }
    seen=set();sets={};identities=[]
    for group,(seeds,maker) in group_specs.items():
        if not seeds:raise ValueError('empty split')
        if len(set(seeds))!=len(seeds) or set(seeds)&seen:
            raise ValueError('seed collision across TRAIN/DEV/HELDOUT')
        seen|=set(seeds)
        worlds=[maker(s) for s in seeds]
        for w in worlds:
            d=distance_field(w).get((*w['start'],0))
            if d is None: raise ValueError('unreachable world')
            if group.startswith('DEV_') or group.startswith('HELDOUT_'):
                if d>budget:raise ValueError(f'impossible evaluation under budget {group}:{w["seed"]}:{d}')
            identities.append({'split':group.split('_')[0], 'group':group,
                'id':f'{w.get("generator","original")}:{w["seed"]}',
                'world_sha256':world_digest(w), 'oriented_minimum':d,
                'difficulty_band':difficulty_band(w,d),
                'generator':w.get('generator','original')})
        sets[group]=worlds
    validate_partitions(sets)
    if len(set(x['world_sha256'] for x in identities))!=len(identities):
        raise ValueError('world geometry leakage')
    return sets,identities


def candidate_targets(w,pos,heading,visited,history,legal):
    moves,fresh,relegal=legal_state(w,pos,heading,visited,history)
    if list(relegal)!=list(legal):raise AssertionError('legal menu drift')
    output={}
    for a in legal:
        if a in moves: output[a]=moves[a]
        elif a=='BRANCH':
            primary=next(z for z in ('RIGHT','LEFT','STRAIGHT') if z in fresh)
            output[a]=moves[primary]
        elif a=='RETRACE':output[a]=history[-1]
        elif a=='STOP':output[a]=(pos,heading)
        else:raise AssertionError('unknown legal action')
    return output


def memory_for(targets,path,oriented_path):
    """Use host-observed positions/headings only; no privileged world geometry."""
    full=Counter(path)
    oriented=Counter(oriented_path)
    last={p:i for i,p in enumerate(path)}
    prior=path[-2] if len(path)>=2 else None
    recent8=Counter(path[-8:]);recent32=Counter(path[-32:])
    result={}
    for action,(target,newhead) in targets.items():
        age=len(path)-1-last[target] if target in last else 2048
        result[action]={
          'seen_total':min(16,full[target]),
          'seen_oriented':min(8,oriented[(target,newhead)]),
          'seen_8':min(8,recent8[target]),
          'seen_32':min(16,recent32[target]),
          'age':min(2048,age),
          'is_two_step_return':int(prior is not None and target==prior),
        }
    return result


def observation_pack(w,pos,heading,visited,history,past,path,oriented,tick,budget,family):
    moves,fresh,legal=legal_state(w,pos,heading,visited,history)
    obs=observation(w,pos,heading,visited,past,legal,tick,budget,family)
    targets=candidate_targets(w,pos,heading,visited,history,legal)
    memory=memory_for(targets,path,oriented)
    return obs, targets,memory


def candidate_features(obs,action,target,memory=None):
    """Candidate-action features for true binary action ranking.

    Baseline only sees the ordinary 5x5 game observation and next target
    determined by the host's public legal menu. Memory adds visit count,
    recency and oriented-loop statistics available from prior executed actions.
    """
    head=obs['heading'];pos=(obs['x'],obs['y'])
    dx,dy=obs['goal_delta'];tx,ty=target[0]
    olddist=abs(dx)+abs(dy)
    nxtdist=abs(pos[0]+dx-tx)+abs(pos[1]+dy-ty)
    f={'act:'+action:1.0,
       'act|heading:'+action+':'+str(head):1.0,
       'act|budget:'+action+':'+str(obs['budget_remaining']//32):1.0,
       'act|progress_sign:'+action+':'+str((olddist>nxtdist)-(olddist<nxtdist)):1.0,
       'act|progress:'+action:float(max(-1,min(1,olddist-nxtdist))),
       'act|goal_reached:'+action+':'+str(obs['goal_reached']):1.0,
       'act|family:'+action+':'+obs['family']:1.0}
    for h,val in [('forward',dx*DIRS[head][0]+dy*DIRS[head][1]),
                  ('right',dx*DIRS[(head+1)%4][0]+dy*DIRS[(head+1)%4][1])]:
        f['act|'+h+'sign:'+action+':'+str((val>0)-(val<0))]=1.0
        f['act|'+h+'bin:'+action+':'+str(min(12,abs(val)//8))]=1.0
    for i,a in enumerate(obs['last_actions'][-3:]):
        f['act|last:'+action+':'+str(i)+':'+a]=1.0
    for a in obs['legal']:f['act|legal:'+action+':'+a]=1.0
    # Entire local 5x5 context, conditioned on candidate; no future state/grid.
    for y in range(5):
        for x in range(5):
            f['act|local:'+action+':'+str(x)+':'+str(y)+':'+obs['local_5x5'][y][x]]=1.0
    # Since STOP and RETRACE may not map to adjacent cells, their target is
    # never silently treated as an adjacent local occupancy observation.
    ox,oy=tx-pos[0],ty-pos[1]
    if -2<=ox<=2 and -2<=oy<=2:
        f['act|target_local:'+action+':'+obs['local_5x5'][oy+2][ox+2]]=1.0
    if memory is not None:
        seen=memory['seen_total'];age=memory['age']
        f['mem|seen_bin:'+action+':'+str(min(5,seen))]=1.0
        f['mem|oriented_bin:'+action+':'+str(min(4,memory['seen_oriented']))]=1.0
        f['mem|recent8:'+action+':'+str(min(4,memory['seen_8']))]=1.0
        f['mem|recent32:'+action+':'+str(min(6,memory['seen_32']//2))]=1.0
        f['mem|age:'+action+':'+str(min(8,age//4))]=1.0
        f['mem|two_step:'+action+':'+str(memory['is_two_step_return'])]=1.0
        f['mem|loop_and_progress:'+action+':'+str(bool(seen))+':'+str((olddist>nxtdist)-(olddist<nxtdist))]=1.0
    return f


def state_key(obs,memory):
    # Exclude family labels and absolute tick, as near-identical family replicas
    # with identical state/action histories are not independent evidence.
    norm={k:v for k,v in obs.items() if k not in ('family','tick','budget_remaining')}
    norm['memory']=memory
    return sha(json.dumps(norm,sort_keys=True,separators=(',',':')).encode())


def collect_train(worldsets,source,budget=BUDGET):
    # Use a separately fitted TRAIN-only behavioural student to enter the
    # actual failure states; neither evaluation split is observed here.
    source_rows,_,hard,labels,counters=teacher_data(source,budget)
    vec,student,meta=fit(source_rows,epochs=EPOCHS)
    rows=[];drop=Counter();global_signatures=set();stats=Counter();worldmeta=[];trajectories=[]
    for domain, worlds in worldsets:
        for w in worlds:
            dist=distance_field(w)
            worldmeta.append({'split':'TRAIN','world_group_id':f'{domain}:{w["seed"]}',
              'world_sha256':world_digest(w),'generator':domain,
              'difficulty_band':difficulty_band(w,distance_field(w).get((*w['start'],0)))})
            for family in FAMILIES:
                pos=w['start'];head=0;visited={pos};history=[];past=[]
                path=[pos];oriented=[(pos,head)];legal_receipts=[];end_reason="budget"
                for tick in range(budget):
                    obs,targets,memory=observation_pack(w,pos,head,visited,
                                  history,past,path,oriented,tick,budget,family)
                    teacher,negs=oracle(w,pos,head,visited,history,dist,obs['legal'])
                    action=decide_model(obs,vec,student)
                    if action not in obs['legal'] or teacher not in obs['legal']:
                        raise AssertionError('illegal correction/behavior action')
                    key=state_key(obs,memory)
                    if key not in global_signatures:
                        global_signatures.add(key)
                        rows.append({'obs':obs,'label':teacher,
                          'targets':targets,'memory':memory,
                          'world_group_id':f'{domain}:{w["seed"]}',
                          'state_sha256':key,
                          'negative_actions':{a:negs.get(a,{'host_legal_distractor':True}) for a in obs['legal'] if a!=teacher},
                          'behavior_action':action,'split':'TRAIN'})
                    else:drop['near_duplicate_state']+=1
                    stats['observed']+=1
                    stats['student_teacher_disagreements']+=int(action!=teacher)
                    legal_receipts.append({'step':tick,'legal':obs['legal'],'action':action,
                      'observation_sha256':key})
                    if action=='STOP':
                        end_reason='stop';stats['early_stops']+=int(pos!=w['goal']);break
                    pos,head=apply(w,pos,head,visited,history,action)
                    if pos in w['obstacles'] or pos in w['protected']:
                        raise AssertionError('host entered protected/obstacle cell')
                    visited.add(pos);past.append(action);path.append(pos);oriented.append((pos,head))
                    if pos==w['goal']:
                        stats['student_goal_arrivals']+=1;end_reason='goal';break
                trajectories.append({'world_group_id':f'{domain}:{w["seed"]}',
                   'family':family,'split':'TRAIN','world_sha256':world_digest(w),
                   'action_receipts':legal_receipts,'path':path,
                   'oriented_path':oriented,'termination':end_reason,
                   'native_receipt':False,'execution':'synthetic-host-applied'})
                stats['episodes']+=1
    if len(rows)<1000:raise ValueError('insufficient unique correction states')
    if len({r['world_group_id'] for r in rows}) < len(worldmeta)//2:
        raise ValueError('too few represented world groups')
    return rows,{'source_fit':meta,'source_teacher_episodes':counters,
                  'source_rows':len(source_rows),'source_hard_negatives':hard,
                  'sampled_state_stats':dict(stats),'dedup':dict(drop),
                  'train_worlds':worldmeta,'trajectories':trajectories}


def fit_ranker(rows,include_memory,epochs=EPOCHS):
    if not 1<=epochs<=64: raise ValueError('bounded optimizer epochs')
    # Matched state/candidate instances, no held-out participation.
    x=[];y=[];weight=[]
    for r in rows:
        obs=r['obs'];legal=obs['legal']
        if r['label'] not in legal:raise ValueError('illegal label')
        for a in legal:
            features=candidate_features(obs,a,r['targets'][a],
                              r['memory'][a] if include_memory else None)
            x.append(features)
            y.append(int(a==r['label']))
            # Each decision contributes equal class-balanced total weight.
            weight.append(.5 if a==r['label'] else .5/max(1,len(legal)-1))
    vec=DictVectorizer(sort=True)
    X=vec.fit_transform(x)
    mdl=SGDClassifier(loss='log_loss',alpha=.0001,max_iter=epochs,
          tol=None,random_state=17,shuffle=True,learning_rate='optimal')
    mdl.fit(X,y,sample_weight=weight)
    return vec,mdl,{'training_decisions':len(rows),'weighted_candidate_instances':len(y),
         'teacher_positive_instances':sum(y),'teacher_negative_instances':len(y)-sum(y),
         'feature_count':X.shape[1], 'epochs':int(mdl.n_iter_),
         'optimizer_samples_seen':int(mdl.t_),'include_long_history':bool(include_memory)}


_MODEL_CACHE={}

def model_action(obs,targets,memory,vec,model,include_memory):
    """Source-equivalent sparse binary linear scorer; avoids per-tick refitting
    or expensive DictVectorizer rematerialization during long rollouts."""
    key=(id(vec),id(model))
    if key not in _MODEL_CACHE:
        weights={f:float(model.coef_[0,i]) for f,i in vec.vocabulary_.items()}
        _MODEL_CACHE[key]=(weights,float(model.intercept_[0]))
    weights,bias=_MODEL_CACHE[key]
    scored=[]
    for a in obs['legal']:
        feats=candidate_features(obs,a,targets[a],memory[a] if include_memory else None)
        score=bias+sum(weights.get(k,0.0)*v for k,v in feats.items())
        scored.append((a,score))
    return max(scored,key=lambda s:(s[1],-ACTIONS.index(s[0])))[0]


def run_policy(w,family,vec,model,with_memory,budget):
    pos=w['start'];head=0;visited={pos};history=[];past=[]
    path=[pos];orient=[(pos,head)];actions=Counter();illegal=0
    for tick in range(budget):
        obs,targets,mem=observation_pack(w,pos,head,visited,history,past,path,orient,tick,budget,family)
        action=model_action(obs,targets,mem,vec,model,with_memory)
        if action not in obs['legal']:illegal+=1;raise AssertionError('illegal action')
        actions[action]+=1
        if action=='STOP':break
        pos,head=apply(w,pos,head,visited,history,action)
        if pos in w['protected']|w['obstacles']:raise AssertionError('protection violation')
        visited.add(pos);past.append(action);path.append(pos);orient.append((pos,head))
        if pos==w['goal']:break
    goal_tick=len(past) if pos==w['goal'] else None
    return {'seed':w['seed'],'family':family,
        'goal_tick':goal_tick,'turns':len(past),'unique_cells':len(visited),
        'two_step_returns':sum(path[i]==path[i-2] for i in range(2,len(path))),
        'max_cell_visits':max(Counter(path).values()),
        'retrace_count':actions['RETRACE'],
        'illegal_actions':illegal,'actions':dict(sorted(actions.items())),'path':path}


def evaluate(sets,policies,budget,png_limit=2):
    report={};images=[]
    for split,worlds in sets.items():
        records=[];identities=[]
        for w in worlds:
            identities.append({'seed':w['seed'],'world_sha256':world_digest(w),
                 'generator':w.get('generator','original'),
                 'oriented_minimum':distance_field(w).get((*w['start'],0))})
            for family in FAMILIES:
                for name,(vec,model,mem) in policies.items():
                    roll=run_policy(w,family,vec,model,mem,budget)
                    roll.update({'policy':name,'world_group_id':f'{split}:{w["seed"]}'})
                    if family==FAMILIES[0] and w['seed'] in [z['seed'] for z in identities[:png_limit]]:
                        pic=png(w,{'path':roll['path']})
                        filename=f'{split}--{w["seed"]}--{name}.png'
                        images.append((filename,pic))
                    del roll['path'];records.append(roll)
        summary={}
        for name in policies:
            rs=[r for r in records if r['policy']==name]
            summary[name]={
              'rollouts':len(rs),'worlds':len(worlds),
              'worlds_solved':sum(any(r['goal_tick'] is not None for r in rs if r['seed']==w['seed']) for w in worlds),
              'episodes_solved':sum(r['goal_tick'] is not None for r in rs),
              'mean_unique_cells':round(sum(r['unique_cells'] for r in rs)/len(rs),3),
              'mean_two_step_returns':round(sum(r['two_step_returns'] for r in rs)/len(rs),3),
              'mean_retrace':round(sum(r['retrace_count'] for r in rs)/len(rs),3),
              'mean_steps':round(sum(r['turns'] for r in rs)/len(rs),3),
              'illegal_actions':sum(r['illegal_actions'] for r in rs)}
        report[split]={'worlds':identities,'summary':summary,'rollouts':records}
    return report,images


def deterministic_zip(path,members):
    if path.exists(): raise FileExistsError(path)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,payload in sorted(members.items()):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o644<<16
            z.writestr(info,payload,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    with zipfile.ZipFile(path) as z:
        if z.testzip():raise AssertionError('bad zip CRC')


def export_train(rows,worldmeta,trajectories,path):
    if any(r['split']!='TRAIN' or r['label'] not in r['obs']['legal'] for r in rows):
        raise ValueError('attempt to export non-TRAIN/illegal examples')
    message=[];prov=[];negs=[]
    for i,r in enumerate(rows):
        obs=r['obs'];memory=r['memory']
        # Explicit observable route counters; no teacher BFS field.
        state={'observation':obs,'host_observed_action_memory':memory,
               'note':'Memory counters are from executed prior route only'}
        message.append(json.dumps({'messages':[
          {'role':'system','content':'CyberSand-style synthetic host. Choose one legal action; immutable protected cells. Goal-directed movement.'},
          {'role':'user','content':json.dumps(state,sort_keys=True,separators=(',',':'))},
          {'role':'assistant','content':r['label']} ]},sort_keys=True,separators=(',',':')))
        prov.append(json.dumps({'index':i,'split':'TRAIN','world_group_id':r['world_group_id'],
          'state_sha256':r['state_sha256'],
          'evidence_class':'synthetic-host-executed-predecision-state',
          'oracle_label_is_synthetic':True,'native_receipt':False,
          'teacher':'privileged-BFS-label-only-not-in-observation'},sort_keys=True,separators=(',',':')))
        negs.append(json.dumps({'index':i,'positive':r['label'],
          'negative_actions':r['negative_actions']},sort_keys=True,separators=(',',':')))
    if any(t['split']!='TRAIN' for t in trajectories):
        raise ValueError('non-TRAIN trajectory export')
    for t in trajectories:
        for receipt in t['action_receipts']:
            if receipt['action'] not in receipt['legal']:
                raise ValueError('illegal host action in trajectory')
    trace_lines=[json.dumps(t,sort_keys=True,separators=(',',':')) for t in trajectories]
    items={'train.trajectories.jsonl':('\n'.join(trace_lines)+'\n').encode(),
           'train.messages.jsonl':('\n'.join(message)+'\n').encode(),
           'train.provenance.jsonl':('\n'.join(prov)+'\n').encode(),
           'train.hard_negatives.jsonl':('\n'.join(negs)+'\n').encode()}
    member_hashes={k:sha(v) for k,v in items.items()}
    manifest={'schema':SCHEMA,'split':'TRAIN','train_decisions':len(rows),
        'world_groups':worldmeta,'full_trajectories':len(trajectories),
        'difficulty_band_method':'oriented minimum + floor(blocked/16); bands under 90 / under 150 / else',
        'normalization':'strip family and tick; preserve host-observed action memory',
        'member_sha256':member_hashes,
        'restrictions':['no native receipts','no DEV/HELDOUT training','no privileged distance in prompts',
                        'same candidate targets for both representation arms']}
    items['manifest.json']=(json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n').encode()
    deterministic_zip(path,items)
    with zipfile.ZipFile(path) as z:
        if any(sha(z.read(k))!=h for k,h in member_hashes.items()):raise AssertionError('corrupted member')
    return {'sha256':sha(path.read_bytes()),'rows':len(rows),
       'world_groups':len(worldmeta),'hard_negative_rows':len(rows),
       'hard_negative_candidates':sum(len(r['negative_actions']) for r in rows),
       'member_sha256':member_hashes,'trajectories':len(trajectories)}


def run(out,budget=BUDGET,source=SOURCE,train_orig=TRAIN_ORIGINAL,
        train_h=TRAIN_HORIZONTAL,train_v=TRAIN_VERTICAL,
        dev_orig=DEV_ORIGINAL,dev_h=DEV_HORIZONTAL,
        held_orig=HELD_ORIGINAL,held_h=HELD_HORIZONTAL,held_s=HELD_SLALOM,
        epochs=EPOCHS):
    out=Path(out)
    if out.exists():raise FileExistsError('immutable output destination')
    verify_sources(Path(__file__).parent)
    worlds,identities=partitions(source,train_orig,train_h,train_v,
              dev_orig,dev_h,held_orig,held_h,held_s,budget)
    train_sets=[('original',worlds['TRAIN_original']),
                ('horizontal',worlds['TRAIN_horizontal']),
                ('vertical_train',worlds['TRAIN_vertical'])]
    trainrows,provenance=collect_train(train_sets,source,budget)
    base_vec,base_m,base_meta=fit_ranker(trainrows,False,epochs)
    mem_vec,mem_m,mem_meta=fit_ranker(trainrows,True,epochs)
    policies={'baseline_5x5':(base_vec,base_m,False),
              'long_history':(mem_vec,mem_m,True)}
    evalsets={k:v for k,v in worlds.items() if k.startswith(('DEV_','HELDOUT_'))}
    results,images=evaluate(evalsets,policies,budget)
    out.mkdir(parents=True);(out/'png').mkdir()
    train_export=export_train(trainrows,provenance['train_worlds'],
                              provenance['trajectories'],out/'train_only_memory_v6.zip')
    images_meta=[]
    for name,pixels in images:
        (out/'png'/name).write_bytes(pixels)
        images_meta.append({'name':name,'sha256':sha(pixels)})
    report={'schema':SCHEMA,'source_pins':SOURCE_PINS,
      'evidence_class':'synthetic-real-CPU-optimizer-and-host-applied-world-disjoint-evaluation',
      'training':{'source_behavior':{k:v for k,v in provenance.items() if k!='trajectories'},
        'ranker_baseline':base_meta,
        'ranker_long_history':mem_meta,'export':train_export},
      'split_worlds':identities,'evaluation':results,'images':images_meta,
      'caveats':['Three family replicas within each world are correlated',
        'Synthetic external host, not native/official DEV/heldout',
        'No original model checkpoint or transformer trained',
        'Training world geometry cannot be claimed to transfer to native gameplay'],
      'native_events':0,'illegal_actions':sum(s['illegal_actions'] for v in results.values() for s in v['summary'].values()),
      'game_files_modified':0,'production_promotions':0,'paid_spend':0}
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(); r=run(args.out)
    print(json.dumps({k:v['summary'] for k,v in r['evaluation'].items()},indent=2,sort_keys=True))
