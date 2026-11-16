"""P13: real W16 auxiliary-head gameplay, outcome-proxy preference and goal-conditioned stopping.

All outcomes are algorithmic, not human aesthetic judgements. Source-pinned W0
and frozen W16 model are read from immutable P11 hosted evidence, with no native
CyberSand or production gameplay modification. P12 is used as a pinned harness.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json
import math
import torch
import torch.nn.functional as F
import p12_recovery_preference as base

GOALS=('coverage48','sculpture96_branch10','recovery6_new10','landmark')
HOLDOUT=base.HELDOUT
TRAIN=base.DISCOVERY
VALIDATION=base.VALIDATION
POLICIES=('baseline_late','unrestricted','oracle','trained_raw','trained_shielded')


def position_in_landmark(pos):
    return 30<=pos[0]<=35 and 45<=pos[1]<=51


def make_states(rec):
    """State just BEFORE each action; nothing from future trajectory enters features."""
    pos=rec['trajectory']; events=rec['events']
    seen={tuple(pos[0])};nov=br=retr=after_retr=0; had_retr=False
    entered=position_in_landmark(pos[0]); recent_n=[];recent_b=[];recent_r=[]
    states=[]
    for t in range(len(events)+1):
        states.append({'tick':t,'novel':nov,'branches':br,'retraces':retr,
            'post_retrace_new':after_retr,'landmark':entered,
            'x':pos[t][0],'y':pos[t][1],
            'recent_n':sum(recent_n[-8:]),'recent_b':sum(recent_b[-8:]),
            'recent_r':sum(recent_r[-8:])})
        if t==len(events):break
        a=events[t]['applied']; p=tuple(pos[t+1]);isnew=int(p not in seen)
        seen.add(p);nov+=isnew;br+=int(a=='BRANCH');retr+=int(a=='RETRACE')
        recent_n.append(isnew);recent_b.append(int(a=='BRANCH'));recent_r.append(int(a=='RETRACE'))
        if had_retr:after_retr+=isnew
        if a=='RETRACE':had_retr=True
        entered|=position_in_landmark(p)
    return states


def accomplished(s,goal):
    if goal=='coverage48':return s['novel']>=48
    if goal=='sculpture96_branch10':return s['novel']>=96 and s['branches']>=10
    if goal=='recovery6_new10':return s['retraces']>=6 and s['post_retrace_new']>=10
    if goal=='landmark':return s['landmark']
    raise ValueError(f'unknown goal: {goal}')


def feat(s,goal,obs=None):
    # Individual features are present/prefix-derived; no goal seed identifiers.
    n,b,r,a=s['novel'],s['branches'],s['retraces'],s['post_retrace_new']
    dx=max(30-s['x'],0,s['x']-35);dy=max(45-s['y'],0,s['y']-51)
    v=[float(goal==g) for g in GOALS]+[
      min(n/128,1),min(b/40,1),min(r/40,1),min(a/128,1),
      min(n/48,2),min(n/96,2),min(b/10,2),min(r/6,2),min(a/10,2),
      float(s['landmark']),s['recent_n']/8,s['recent_b']/8,s['recent_r']/8,
      s['x']/96,s['y']/96, (dx+dy)/96, s['tick']/128,
    ]
    assert len(v)==21
    return v


def rows_of(rec,family,seed,cohort):
    states=make_states(rec);rows=[]
    for goal in GOALS:
      for t in range(0,len(states),4):
        s=states[t]
        rows.append({'cohort':cohort,'family':family,'seed':seed,'goal':goal,
          'tick':t,'x':feat(s,goal),'stop_proxy':int(accomplished(s,goal)),
          'source_events':len(rec['events'])})
    return rows


def to_tensors(rows):
    return (torch.tensor([r['x'] for r in rows],dtype=torch.float32),
      torch.tensor([r['stop_proxy'] for r in rows],dtype=torch.float32))


def train(rows,valrows):
    # A learned outcome-preference proxy; comparator exact goal oracle remains available.
    torch.set_num_threads(1);torch.manual_seed(13016)
    x,y=to_tensors(rows);vx,vy=to_tensors(valrows)
    m=torch.nn.Sequential(torch.nn.Linear(21,24),torch.nn.Tanh(),
                           torch.nn.Linear(24,12),torch.nn.Tanh(),torch.nn.Linear(12,1))
    opt=torch.optim.Adam(m.parameters(),lr=.013,weight_decay=.0008)
    pos=float(y.sum());weights=torch.tensor(min(15,(len(y)-pos)/max(pos,1)))
    best=float('inf');beststep=0;state=None
    for t in range(1,601):
        opt.zero_grad();loss=F.binary_cross_entropy_with_logits(m(x).squeeze(1),y,pos_weight=weights)
        loss.backward();opt.step()
        if t%20==0:
            with torch.no_grad():
                v=F.binary_cross_entropy_with_logits(m(vx).squeeze(1),vy).item()
            if v<best:
                best=v;beststep=t;state={k:a.detach().clone() for k,a in m.state_dict().items()}
    m.load_state_dict(state);m.eval()
    return m,{'updates':600,'best_step':beststep,'validation_bce':best,'train_rows':len(rows),
            'validation_rows':len(valrows),'train_positive':int(pos),'val_positive':int(vy.sum())}


def evaluate(m,rows):
    x,y=to_tensors(rows)
    with torch.no_grad():
        p=torch.sigmoid(m(x).squeeze(1))
        labels=y.tolist();scores=p.tolist();binary=[int(s>=.5) for s in scores]
    return {'rows':len(rows),'positive':int(y.sum()),
       'accuracy':sum(a==b for a,b in zip(binary,labels))/len(rows),
       'false_positive':sum(a and not b for a,b in zip(binary,labels)),
       'false_negative':sum(b and not a for a,b in zip(binary,labels)),
       'brier':float(((p-y)**2).mean()),
       'average_precision':base.average_precision(scores,labels)}


def live_rollout(w0,head,seed,family,goal,policy,model,source):
    # Existing W16 movement samples are retained across interventions.
    # Never access future trajectory; only the source's existing movement
    # check, which is used to audit and reject unexpected divergence.
    track={'state':{'tick':0,'novel':0,'branches':0,'retraces':0,'post_retrace_new':0,
         'landmark':position_in_landmark((7,48)),'x':7,'y':48,'recent_n':0,
         'recent_b':0,'recent_r':0},'seen':{(7,48)},'had_retr':False,
         'recent_n':[],'recent_b':[],'recent_r':[],'prev_action':None,'decisions':[]}
    def choose(obs):
        t=obs['tick'];s=track['state']
        if t:
            x,y=obs['x'],obs['y'];p=(x,y);isnew=int(p not in track['seen']);track['seen'].add(p)
            a=track['prev_action'];s['novel']+=isnew;s['branches']+=int(a=='BRANCH');s['retraces']+=int(a=='RETRACE')
            if track['had_retr']:s['post_retrace_new']+=isnew
            if a=='RETRACE':track['had_retr']=True
            track['recent_n'].append(isnew);track['recent_b'].append(int(a=='BRANCH'))
            track['recent_r'].append(int(a=='RETRACE'))
            s['recent_n']=sum(track['recent_n'][-8:]);s['recent_b']=sum(track['recent_b'][-8:]);s['recent_r']=sum(track['recent_r'][-8:])
            s['landmark']|=position_in_landmark(p)
            s['x']=x;s['y']=y;s['tick']=t
        with torch.no_grad():
            logits=head(torch.tensor([base.model_features(obs)],dtype=torch.float32))[0].tolist()
            prob=float(torch.sigmoid(model(torch.tensor([feat(s,goal)],dtype=torch.float32)))[0,0])
        complete=accomplished(s,goal)
        moving=[a for a in obs['legal'] if a!='STOP']
        move=base.choice(logits,moving,.7,base.uniform(seed,family,t,'w11-CRN'))
        pstop=base.stop_prob(logits,obs['legal'])
        original_stop=t>=124 and base.uniform(seed,family,t,'p10-independent-STOP-CRN-v1')<pstop
        unrestricted_stop=base.uniform(seed,family,t,'p10-independent-STOP-CRN-v1')<pstop
        if policy=='baseline_late':stop=original_stop
        elif policy=='unrestricted':stop=unrestricted_stop
        elif policy=='oracle':stop=complete
        elif policy=='trained_raw':stop=prob>=.5
        elif policy=='trained_shielded':stop=complete and prob>=.5
        else:raise ValueError(policy)
        raw='STOP' if stop else move
        if t<len(source['events']) and raw!='STOP' and source['events'][t]['applied']!='STOP' and raw!=source['events'][t]['applied']:
            raise AssertionError('frozen W16 non-STOP movement must be exact original prefix')
        if raw not in obs['legal']:raise AssertionError('host illegal action')
        track['prev_action']=raw
        track['decisions'].append({'tick':t,'move':move,'stop_probability':pstop,
         'learned_complete_probability':prob,'goal_completed':complete,'applied':raw,
         'head_logits':logits,'legal':list(obs['legal'])})
        return raw
    r=w0.run(seed,'callback',family,128,selector=choose)
    if not all(x['applied']==e['applied'] and e['raw_legal'] for x,e in zip(track['decisions'],r['events'])):
        raise AssertionError('host action receipt failed')
    if policy=='baseline_late':
        if r['trajectory']!=source['trajectory'] or r['metrics']!=source['metrics']:
            raise AssertionError('frozen W16 original baseline mismatch')
    s=make_states(r)[-1];r['goal']=goal;r['goal_completed']=accomplished(s,goal)
    r['stop_policy']=policy;r['head_decisions']=track['decisions'];r['model_boundary']='frozen W16 aux head, synthetic W0'
    return r


def do_experiment(source_bytes,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    z,w0,head,cksha=base.open_pinned(source_bytes)
    traces={};rows=[];source_hashes={}
    for fam in base.FAMILIES:
      for seed in TRAIN+VALIDATION+HOLDOUT:
        rec,trh,pngh=base.extract_trace(z,fam,seed)
        traces[fam,seed]=rec;source_hashes[f'{fam}/{seed}']={'trace':trh,'png':pngh}
        cohort='train' if seed in TRAIN else 'validation' if seed in VALIDATION else 'heldout'
        rows+=rows_of(rec,fam,seed,cohort)
    m,training=train([r for r in rows if r['cohort']=='train'],
                     [r for r in rows if r['cohort']=='validation'])
    heldout=[r for r in rows if r['cohort']=='heldout']
    trainmetrics=evaluate(m,heldout)
    torch.save({'state_dict':m.state_dict(),'goal_names':GOALS,'training':training,
     'source_p11':base.P11_SHA,'source_w16_checkpoint':cksha,'human_ratings':0},out/'P13-goal-preference-CPU.pt')
    experiments=[];pairs=[];images=[]
    for fam in base.FAMILIES:
      for seed in HOLDOUT:
        src=traces[fam,seed]
        for goal in GOALS:
          variants={}
          for policy in POLICIES:
            rec=live_rollout(w0,head,seed,fam,goal,policy,m,src)
            stem=f'{fam}-{seed}-{goal}-{policy}'
            png=w0.png_bytes(rec);text=(json.dumps(rec,sort_keys=True,separators=(',',':'))+'\n').encode()
            (out/f'{stem}.json').write_bytes(text);(out/f'{stem}.png').write_bytes(png)
            info={'family':fam,'seed':seed,'goal':goal,'policy':policy,
              'success':rec['goal_completed'],'steps':len(rec['events']),
              'stop':rec['metrics']['terminated_by_stop'],
              'unique_cells':rec['metrics']['unique_trail_cells'],
              'branches':rec['metrics']['branches'],'retraces':rec['metrics']['retraces'],
              'illegal':rec['metrics']['invalid_raw_choices'],
              'source_trace_sha256':source_hashes[f'{fam}/{seed}']['trace'],
              'trace_sha256':base.sha(text),'png_sha256':base.sha(png)}
            if not info['success'] and policy in ('oracle','trained_shielded') and info['stop']:
                raise AssertionError('shield allowed incomplete STOP')
            if policy=='baseline_late' and png!=z.read(f'rollouts/{fam}-{seed}-intact-late.png'):
                raise AssertionError('original PNG byte mismatch')
            variants[policy]=info;experiments.append(info)
          for policy in POLICIES:
            if policy=='baseline_late':continue
            candidate=variants[policy];baseinfo=variants['baseline_late']
            # Explicit task-completion-first proxy, then action parsimony, no aesthetics.
            score=lambda x: 100*int(x['success'])-x['steps']/128
            diff=score(candidate)-score(baseinfo)
            pairs.append({'family':fam,'seed':seed,'goal':goal,'candidate':policy,
               'baseline':'baseline_late','outcome_proxy_preference':
                    'candidate' if diff>0 else 'baseline' if diff<0 else 'tie',
               'algorithmic_reward_delta':diff,'candidate_success':candidate['success'],
               'baseline_success':baseinfo['success'],'human_owner_rating':None})
    aggregates={}
    for policy in POLICIES:
        r=[x for x in experiments if x['policy']==policy]
        aggregates[policy]={'count':len(r),'goal_completed':sum(x['success'] for x in r),
           'stopped':sum(x['stop'] for x in r),
           'failed_goal_stops':sum(x['stop'] and not x['success'] for x in r),
           'mean_steps':sum(x['steps'] for x in r)/len(r),
           'mean_unique_cells':sum(x['unique_cells'] for x in r)/len(r),
           'illegal_actions':sum(x['illegal'] for x in r)}
    result={'schema':'driv3r-preference-P13-goal-conditional-stop/v1',
        'truth':'W16 auxiliary PyTorch head and W0 synthetic host; goal labels algorithmic, no human aesthetics',
        'provenance':{'P11_sha256':base.P11_SHA,'W16_sha256':base.W16_SHA,
          'W16_head_sha256':cksha,'P11_source_HEAD':base.HEAD_SHA,'original_traces':source_hashes},
        'goals':list(GOALS),'cohorts':{'train':TRAIN,'validation':VALIDATION,'heldout':HOLDOUT},
        'training':training,'heldout_preference_fit':trainmetrics,
        'new_model_inference_rollouts':len(experiments),
        'new_model_forward_passes':sum(x['steps'] for x in experiments),
        'comparison_pairs':len(pairs),'summary':aggregates,
        'owner_ratings':0,'human_ratings':0,
        'limitations':['Goal rewards are transparent synthetic rules, not human preference.',
          'A learned goal recogniser cannot exceed an exact rule-based completion oracle.',
          'Host does not simulate CyberSand paints, real materials or native physics.']}
    (out/'P13-results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (out/'P13-pairs.json').write_text(json.dumps(pairs,indent=2,sort_keys=True)+'\n')
    (out/'P13-rollout-index.json').write_text(json.dumps(experiments,indent=2,sort_keys=True)+'\n')
    (out/'P13-SHA256SUMS.txt').write_text('\n'.join(f'{base.sha(p.read_bytes())}  {p.name}'
                   for p in sorted(out.iterdir()) if p.is_file())+'\n')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--p11-artifact',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();r=do_experiment(args.p11_artifact.read_bytes(),args.out)
    print(json.dumps({'training':r['training'],'heldout_fit':r['heldout_preference_fit'],
      'new_rollouts':r['new_model_inference_rollouts'],'summary':r['summary']},indent=2))
