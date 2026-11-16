"""P9: source-pinned test of W16 archived STOP propensity: clock vs local state.

Inputs are original per-turn logits/observations from a real PyTorch auxiliary
model head acting in *synthetic W0* with STOP masked until tick 124. The target
is a counterfactual instantaneous STOP probability along the ORIGINAL path,
not a decision, outcome, human rating, STOP-open rollout, or causal ablation.

No game imports, state edits, model downloads, or production changes. This tiny
stdlib Adam probe is descriptive, not a new gameplay reward model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from p8_stop_retention import ACTIONS, FAMILIES, SEEDS, archive, choice, stop_probability

MODES = ('W15-t07', 'W16-final-t07', 'W16-mixed-t07')
T = 0.7
EPOCHS = 90
LR = .045
L2 = .006
CLOCK_CENTERS = tuple(range(0, 128, 4))


def sigmoid(x):
    if x >= 0:
        z = math.exp(-min(700.0, x))
        return 1.0/(1.0+z)
    z = math.exp(max(-700.0, x))
    return z/(1.0+z)


def clip(p):
    return max(1e-9, min(1.0-1e-9, p))


def cross_entropy(y,p):
    p=clip(p)
    return -y*math.log(p)-(1.0-y)*math.log1p(-p)


def local_features(observation, previous_actions, seen, novelty, stalls):
    """Strictly prefix-based; no source logits or post-choice/future information."""
    rows=observation['local_5x5']
    if len(rows)!=5 or any(len(x)!=5 for x in rows):raise ValueError('bad local view')
    if any(ch not in '#+.@' for row in rows for ch in row):raise ValueError('unknown local symbol')
    if sum(row.count('@') for row in rows)!=1:raise ValueError('cursor absent or duplicated')
    n=max(1,len(previous_actions))
    heading=observation['heading']
    if heading not in (0,1,2,3):raise ValueError('invalid heading')
    p=[observation['x']/96,observation['y']/96,
       min(1.0,len(seen)/128), # cumulative progress leaks some elapsed time; report explicitly
       sum(a=='BRANCH' for a in previous_actions)/n,
       sum(a=='RETRACE' for a in previous_actions)/n,
       sum(a in ('LEFT','RIGHT') for a in previous_actions)/n,
       sum(novelty[-8:])/max(1,len(novelty[-8:])),
       min(1.0,stalls/12),
       sum(row.count('#') for row in rows)/24,
       sum(row.count('+') for row in rows)/24,
       len(observation['legal'])/6]
    p.extend(float(heading==i) for i in range(4))
    return p


def clock_features(t):
    if not isinstance(t,int) or not 0<=t<128:raise ValueError('tick outside W0')
    # Fixed, preregistered radial basis grid; no hand-picked 88/89/90/123 pulse indicators.
    return [t/127, (t/127)**2]+[math.exp(-.5*((t-k)/2.2)**2) for k in CLOCK_CENTERS]


def source_episodes(data):
    z,manifest=archive(data)   # checks original producer bytes, 213 internal SHAs, CRC, HEAD
    episodes=[]
    for family in FAMILIES:
      for seed in SEEDS:
       for mode in MODES:
        stem=f'campaign/original-png-json/{family}-{mode}-{seed}'
        blob=z.read(stem+'.json')
        png=z.read(stem+'.png')
        if hashlib.sha256(blob).hexdigest()!=manifest[stem+'.json'] or hashlib.sha256(png).hexdigest()!=manifest[stem+'.png']:
            raise ValueError('original trace/png identity conflict')
        obj=json.loads(blob)
        if (obj['family'],obj['seed'],obj['mode'])!=(family,seed,mode) or obj['engine']!='SHA-pinned-synthetic-W0-PyTorch-head-not-native':
            raise ValueError('original source identity mismatch')
        events=obj['events'];heads=obj['head_decisions'];path=obj['trajectory']
        if not (len(events)==len(heads)==len(path)-1 and 1<=len(events)<=128 and obj['step_budget']==128):
            raise ValueError('original trajectory contract mismatch')
        seen={tuple(path[0])}; actions=[]; novelty=[];stalls=0;rows=[]
        for t,(event,head) in enumerate(zip(events,heads)):
            obs=event['observation']; legal=obs['legal']; restricted=head['host_legal'];scores=head['raw_six_scores']; draw=head['host_draw']
            if (obs['tick'],obs['x'],obs['y'])!=(t,*path[t]) or head['tick']!=t or tuple(event['position_after'])!=tuple(path[t+1]):
                raise ValueError('original position/tick receipt mismatch')
            if legal!=[a for a in ACTIONS if a in legal] or 'STOP' not in legal or restricted!=[a for a in legal if a!='STOP' or t>=124]:
                raise ValueError('host legal/STOP gate violation')
            if not (head['actual_head_forward'] and not head['root_forward'] and head['temperature']==T and 0<=draw<1 and event['raw_legal']):
                raise ValueError('not a genuine sampled head choice')
            if choice(scores,restricted,T,draw)!=head['selected'] or not (head['selected']==event['applied']==event['raw_choice']):
                raise ValueError('chosen head action does not reproduce original host receipt')
            if any(not math.isfinite(v) for v in scores):raise ValueError('nonfinite scores')
            if len(path[t])!=2 or len(path[t+1])!=2:raise ValueError('invalid original path')
            # Only source ticks with a masked STOP enter training/evaluation.
            if t<124:
                p=stop_probability(scores,legal,T)
                rows.append({'tick':t,'stop_p':p,'clock':clock_features(t),
                             'space':local_features(obs,actions,seen,novelty,stalls)})
            new=int(tuple(path[t+1]) not in seen and event['applied']!='STOP')
            novelty.append(new)
            stalls=0 if new else stalls+1
            seen.add(tuple(path[t+1]));actions.append(event['applied'])
        episodes.append({'family':family,'seed':seed,'mode':mode,
                         'original_sha256':hashlib.sha256(blob).hexdigest(),
                         'png_sha256':hashlib.sha256(png).hexdigest(),
                         'original_head_decisions':len(events),'rows':rows})
    if len(episodes)!=27:raise ValueError('missing sampled original episodes')
    return episodes


def vector(row,model):
    mode = MODES.index(row['mode'])
    x=[float(mode==i) for i in range(3)]
    if model in ('clock','combined'):x.extend(row['clock'])
    if model in ('space','combined'):x.extend(row['space'])
    if model not in ('mode','clock','space','combined'):raise ValueError('unknown model')
    return x


def fit(data,model,epochs=EPOCHS):
    """Deterministic full-batch Adam for soft target Bernoulli log loss."""
    if not data:raise ValueError('empty training data')
    x=[vector(d,model) for d in data];y=[d['stop_p'] for d in data]
    if any(not 0<=p<=1 or not math.isfinite(p) for p in y):raise ValueError('invalid target')
    n=len(x);dim=len(x[0]);w=[0.0]*dim
    # A separately trainable scalar intercept avoids initial 50% STOP artefact.
    b=math.log(clip(sum(y)/n)/(1-clip(sum(y)/n)))
    m=[0.0]*(dim+1);v=[0.0]*(dim+1)
    for step in range(1,epochs+1):
        g=[0.0]*(dim+1)
        for row,yi in zip(x,y):
            pi=sigmoid(b+sum(wi*xi for wi,xi in zip(w,row)))
            delta=pi-yi
            for j,xi in enumerate(row):g[j]+=delta*xi
            g[-1]+=delta
        for j in range(dim):g[j]=g[j]/n + L2*w[j]
        g[-1]/=n
        for j in range(dim+1):
            m[j]=.9*m[j]+.1*g[j]
            v[j]=.999*v[j]+.001*g[j]*g[j]
            moment=m[j]/(1-.9**step)
            variance=v[j]/(1-.999**step)
            delta=LR*moment/(math.sqrt(variance)+1e-8)
            if j==dim:b-=delta
            else:w[j]-=delta
    return {'model':model,'weights':w,'bias':b,'updates':epochs,
            'train_rows':len(data),'feature_count':dim}


def predict(head,row):
    x=vector(row,head['model'])
    if len(x)!=len(head['weights']):raise ValueError('head feature count mismatch')
    return clip(sigmoid(head['bias']+sum(wi*xi for wi,xi in zip(head['weights'],x))))


def evaluate(episodes,fold_field):
    if fold_field not in ('seed','family'):raise ValueError('invalid split')
    categories=SEEDS if fold_field=='seed' else FAMILIES
    by_model={m:[] for m in ('mode','clock','space','combined')}
    fits=[];per_episode=[]
    for held in categories:
        train=[e for e in episodes if e[fold_field]!=held]
        test=[e for e in episodes if e[fold_field]==held]
        # Strict group isolation: no same seed/family, not individual-turn random split.
        assert not ({e[fold_field] for e in train}&{e[fold_field] for e in test})
        train_rows=[dict(r,mode=e['mode']) for e in train for r in e['rows']]
        for model in by_model:
            head=fit(train_rows,model)
            fits.append({'held_group':held,**head})
            for e in test:
                rows=[dict(r,mode=e['mode']) for r in e['rows']]
                ll=sum(cross_entropy(r['stop_p'],predict(head,r)) for r in rows)/len(rows)
                brier=sum((r['stop_p']-predict(head,r))**2 for r in rows)/len(rows)
                rec={'fold_type':fold_field,'held_group':held,'family':e['family'],'seed':e['seed'],
                     'mode':e['mode'],'model':model,'rows':len(rows),'mean_soft_logloss':ll,
                     'mean_soft_brier':brier,'source_trace_sha256':e['original_sha256']}
                per_episode.append(rec);by_model[model].append(rec)
    summary={model:{'episode_macro_logloss':sum(e['mean_soft_logloss'] for e in rows)/len(rows),
                    'episode_macro_brier':sum(e['mean_soft_brier'] for e in rows)/len(rows),
                    'episodes':len(rows),'ticks':sum(e['rows'] for e in rows)} for model,rows in by_model.items()}
    # Identical episodes/coverage, paired macro comparisons.
    def delta_vs(m,b):
        ids=lambda r:(r['family'],r['seed'],r['mode'])
        aa={ids(r):r for r in by_model[m]};bb={ids(r):r for r in by_model[b]}
        assert aa.keys()==bb.keys()
        deltas=[aa[k]['mean_soft_logloss']-bb[k]['mean_soft_logloss'] for k in aa]
        return {'better_episodes':sum(x < -1e-8 for x in deltas),
                'worse_episodes':sum(x > 1e-8 for x in deltas),
                'ties':sum(abs(x)<=1e-8 for x in deltas),
                'mean_logloss_delta':sum(deltas)/len(deltas)}
    return {'fold_type':fold_field,'metrics':summary,
            'paired':{'clock_vs_mode':delta_vs('clock','mode'),
                      'space_vs_mode':delta_vs('space','mode'),
                      'combined_vs_clock':delta_vs('combined','clock'),
                      'clock_vs_space':delta_vs('clock','space')},
            'per_episode':per_episode,'trained_heads':fits}


def run(data):
    episodes=source_episodes(data)
    analyzed=sum(len(e['rows']) for e in episodes)
    return {'schema':'driv3r-preference-P9-source-clock-vs-spatial-STOP-probe/v1',
            'original_w16_archive_sha256':hashlib.sha256(data).hexdigest(),
            'producer_commit':'6005c19df9d99819cc36dc9e0a9fd37343965146',
            'producer_workflow':43851411911,'producer_artifact':14217467236,
            'sampled_model_head_trajectories':len(episodes),
            'original_masked_STOP_ticks':analyzed,
            'original_model_head_decisions':sum(e['original_head_decisions'] for e in episodes),
            'optimizer':{'algorithm':'CPU stdlib full-batch Adam','epochs_per_fit':EPOCHS,
                         'training_labels':'source-logit-derived STOP probability, not human preference',
                         'no_checkpoint_promotion':True},
            'split_seed':evaluate(episodes,'seed'),
            'split_family':evaluate(episodes,'family'),
            'limitations':['The target is source-logit propensity on a STOP-masked historical path. Not actual STOP-unmasked behavior.',
                           'Same checkpoint and only three material families/three seeds; no cross-checkpoint or real-world generalization.',
                           'Space-only history variables can proxy elapsed time: this is predictive discrimination, not causal clock ablation.',
                           'Clock radial-basis positions are fixed before model fitting; optimizer outputs are logit-propensity probes, not aesthetic preferences.',
                           'No game change, native Godot, fresh worm inference, human ratings, or owner-valued ranker.']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();report=run(Path(a.original).read_bytes())
    Path(a.output).write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print('Source original:',report['original_masked_STOP_ticks'],'masked sampled-model decisions')
    for key in ('split_seed','split_family'):
        print(key, {k:round(v['episode_macro_logloss'],5) for k,v in report[key]['metrics'].items()})
        print('paired:',report[key]['paired'])


if __name__=='__main__':main()
