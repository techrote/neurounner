"""P12 / DRIV3R prospective recovery and stopping research (model-side only).

Train light CPU rankers on *future outcomes of genuine W16 auxiliary-head*
trajectories in the immutable P11 source. Compare distinct clock/history feature
sets on untouched worlds, then exercise the trained stop option in the pinned
synthetic W0 host with fresh frozen-head forwards and exact legal action receipts.

All labels are algorithmic FUTURE-OUTCOME proxies, never human aesthetic votes.
No CyberSand game, root Sprocket model, provider, or native simulation is edited.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import types
import zipfile
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

P11_SHA='9a9f61038bb1a4f68395816a8296fb8957a8a8f166610921c2a9db6502308d9c'
W16_SHA='a7b756a3b0c2d639e5e48dad36d518dfd0e6b752deadca8b04ac11326717eafb'
W0_SHA='81c3923cb38d7f196b99986806216388c34ebc27fd4976e168b3b370c25a323e'
HEAD_SHA='e9c7d9ca2a018c2da591b3fcb3429991c907fdfa'
DISCOVERY=(2609,2617,2621,2633,2647,2657)
VALIDATION=(2663,2683,2693)
HELDOUT=(2711,2729,2741,2753,2767,2789,2801,2819,2833)
FAMILIES=('architecture','greebles','pipework')
ACTIONS=('STRAIGHT','LEFT','RIGHT','BRANCH','RETRACE','STOP')
HORIZON=24
SEED=12016
EPS=1e-8

def sha(data):return hashlib.sha256(data).hexdigest()
def uniform(seed,family,t,salt):
    return int.from_bytes(hashlib.sha256(f'{seed}/{family}/{t}/{salt}'.encode()).digest()[:8],'big')/2**64

def choice(scores,legal,temp,u):
    indices=[ACTIONS.index(a) for a in legal]
    if not indices:return 'STOP'
    logits=[scores[i]/temp for i in indices]; mx=max(logits)
    weights=[math.exp(v-mx) for v in logits]; s=sum(weights); target=u*s
    cum=0.
    for a,w in zip(legal,weights):
        cum+=w
        if target<cum:return a
    return legal[-1]

def stop_prob(scores,legal,temp=.7):
    if 'STOP' not in legal:return 0.
    vals=[scores[ACTIONS.index(a)]/temp for a in legal];m=max(vals)
    return math.exp(scores[5]/temp-m)/sum(math.exp(v-m) for v in vals)


def model_features(obs):
    f=[float(ch==s) for line in obs['local_5x5'] for ch in line for s in '.#+@']
    f += [float(obs['heading']==k) for k in range(4)]
    f += [float(obs['family']==s) for s in ('greebles','pipework','architecture')]
    t=obs['tick']; periods=(6,9,12,13,15,17,32)
    f += [math.sin(2*math.pi*t/p) for p in periods]
    f += [math.cos(2*math.pi*t/p) for p in periods]
    f += [min(t/128,1.),obs['x']/96,obs['y']/96]
    f += [float(a in obs['legal']) for a in ACTIONS]
    recent=obs['last_actions']
    f += [sum(a==s for a in recent)/4 for s in ACTIONS]
    f += [float(bool(recent) and recent[-1]==s) for s in ACTIONS]
    if len(f)!=142:raise ValueError('W16 feature contract drift')
    return f

def open_pinned(data):
    if sha(data)!=P11_SHA:raise ValueError('unexpected P11 source archive hash')
    z=zipfile.ZipFile(io.BytesIO(data)); names=z.namelist()
    if len(names)!=len(set(names)) or z.testzip() is not None:
        raise ValueError('corrupt or duplicated P11 archive entries')
    if z.read('EXECUTED_SOURCE_HEAD.txt').decode().strip()!=HEAD_SHA:raise ValueError('P11 source HEAD mismatch')
    original=z.read('original-W16.zip')
    if sha(original)!=W16_SHA:raise ValueError('W16 producer SHA mismatch')
    orig=zipfile.ZipFile(io.BytesIO(original))
    if orig.testzip() is not None or sha(orig.read('pinned-W0.py'))!=W0_SHA:
        raise ValueError('W0 source CRC/SHA mismatch')
    manifest={}
    for line in orig.read('SHA256SUMS.txt').decode().splitlines():
        if not line.strip():continue
        digest,name=line.split(None,1);name=name.lstrip(' *').removeprefix('./');manifest[name]=digest
    for name,digest in manifest.items():
        if sha(orig.read(name))!=digest:raise ValueError(f'original producer member mismatch {name}')
    w0=types.ModuleType('pinned_w0_p12')
    exec(compile(orig.read('pinned-W0.py'),'pinned-W0.py','exec'),w0.__dict__)
    if tuple(w0.ACTIONS)!=ACTIONS:raise ValueError('host action set mismatch')
    ck=orig.read('campaign/onpolicy-branch-history-replay.pt')
    payload=torch.load(io.BytesIO(ck),map_location='cpu',weights_only=True)
    if payload['version']!='sprocket-w16-branch-history-dual-retention-v1':raise ValueError('W16 checkpoint stage mismatch')
    head=torch.nn.Sequential(torch.nn.Linear(142,64),torch.nn.Tanh(),
        torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,6))
    head.load_state_dict({k.removeprefix('layers.'):v for k,v in payload['head'].items()},strict=True)
    head.eval()
    return z,w0,head,sha(ck)

def extract_trace(z,family,seed):
    name=f'rollouts/{family}-{seed}-intact-late.json'
    raw=z.read(name);rec=json.loads(raw)
    if len(rec['events'])!=len(rec['head_decisions']) or len(rec['trajectory'])!=len(rec['events'])+1:
        raise ValueError('broken source trace event alignment')
    if not all(e['raw_legal'] and e['applied']==d['action'] and
                e['observation']['tick']==d['tick']==i and
                e['observation']['legal']==d['host_legal']
                for i,(e,d) in enumerate(zip(rec['events'],rec['head_decisions']))):
        raise ValueError('not host-legal model decisions')
    png=z.read(name.removesuffix('.json')+'.png')
    return rec,sha(raw),sha(png)

def outcomes(rec):
    visited={tuple(rec['trajectory'][0])};novel=[];branch=[];retr=[]
    for i,e in enumerate(rec['events']):
        p=tuple(rec['trajectory'][i+1]); novel.append(int(p not in visited));visited.add(p)
        branch.append(int(e['applied']=='BRANCH'))
        retr.append(int(e['applied']=='RETRACE'))
    return novel,branch,retr

def features(rec,t,novel,branch,retr):
    """Strictly prefix-observable features. No future trajectory / seed identity."""
    obs=rec['events'][t]['observation']
    def recent(xs,n):return sum(xs[max(0,t-n):t])/min(n,max(t,1))
    spatial=[recent(novel,4),recent(novel,8),recent(novel,16),recent(novel,32),
        recent(branch,8),recent(branch,24),recent(retr,8),recent(retr,24),
        sum(novel[:t])/max(1,t),sum(branch[:t])/max(1,t),
        obs['local_5x5'][0].count('#')/5,
        sum(row.count('#') for row in obs['local_5x5'])/24,
        sum(row.count('+') for row in obs['local_5x5'])/24,
        obs['x']/96,obs['y']/96,
        float('BRANCH' in obs['legal']),float('RETRACE' in obs['legal']),
        float(obs['heading'] in (0,2))]
    clock=[t/128]+[math.sin(2*math.pi*t/p) for p in (13,17,32)]+[
        math.cos(2*math.pi*t/p) for p in (13,17,32)]
    if len(spatial)!=18 or len(clock)!=7:raise AssertionError('feature drift')
    return spatial,clock

def rows_from(rec,cohort,family,seed):
    novel,branch,retr=outcomes(rec);r=[]
    for t in range(24,min(104,len(novel)-HORIZON)+1):
        future16=sum(novel[t:t+16]);future24=sum(novel[t:t+24]);
        sf,tf=features(rec,t,novel,branch,retr)
        r.append({'cohort':cohort,'family':family,'seed':seed,'tick':t,
                  'spatial':sf,'clock':tf,'future16_new':future16,'future24_new':future24,
                  'future24_branch':sum(branch[t:t+24]),
                  'low_prospect':int(future16<=3 and future24<=6)})
    return r

def arrays(rows,feature):
    if feature=='spatial':x=[r['spatial'] for r in rows]
    elif feature=='clock':x=[r['clock'] for r in rows]
    elif feature=='combined':x=[r['spatial']+r['clock'] for r in rows]
    else:raise ValueError('bad feature set')
    y=[r['low_prospect'] for r in rows]
    return torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.float32)

def train(trainrows,valrows,feature):
    """Two-layer lightweight prospect ranker; real Adam updates and seed-heldout selection."""
    torch.manual_seed(SEED);torch.set_num_threads(1)
    xt,yt=arrays(trainrows,feature);xv,yv=arrays(valrows,feature)
    model=torch.nn.Sequential(torch.nn.Linear(xt.shape[1],20),torch.nn.Tanh(),torch.nn.Linear(20,1))
    optimizer=torch.optim.Adam(model.parameters(),lr=.016,weight_decay=.002)
    positives=int(yt.sum().item());negatives=len(yt)-positives
    if not positives or not negatives:raise ValueError('cannot fit a one-class preference target')
    wt=min(20.,negatives/positives)
    best_state=None;bestloss=float('inf');beststep=None
    for step in range(1,801):
        optimizer.zero_grad();out=model(xt).squeeze(1)
        loss=F.binary_cross_entropy_with_logits(out,yt,pos_weight=torch.tensor(wt))
        loss.backward();optimizer.step()
        if step%20==0:
            with torch.no_grad():
                logits=model(xv).squeeze(1)
                # Balanced validation BCE to prevent majority-class trivial solution
                pos=yv.sum().item();neg=len(yv)-pos
                vwt=min(20.,neg/max(1.,pos))
                value=float(F.binary_cross_entropy_with_logits(logits,yv,
                            pos_weight=torch.tensor(vwt)))
                if value<bestloss:
                    bestloss=value;beststep=step
                    best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state);model.eval()
    return model,{'feature':feature,'updates':800,'best_step':beststep,
                  'val_weighted_bce':bestloss,'train_positives':positives,
                  'val_positives':int(yv.sum().item())}

def predict(model,features):
    with torch.no_grad():return float(torch.sigmoid(model(torch.tensor([features],dtype=torch.float32))[0,0]))

def average_precision(scores,labels):
    # Tie-aware area under PR step curve: each identical score is a single
    # threshold. Do not spuriously sort positive labels before negatives.
    bins=defaultdict(lambda:[0,0]);n=sum(labels)
    if n==0:return 0.
    for score,y in zip(scores,labels):
        bins[float(score)][0]+=int(y);bins[float(score)][1]+=1
    tp=seen=0;result=0.
    for score in sorted(bins,reverse=True):
        positives,total=bins[score];tp+=positives;seen+=total
        result+=(positives/n)*(tp/seen)
    return result

def scoring(model,rows,feature):
    x,y=arrays(rows,feature)
    with torch.no_grad():p=torch.sigmoid(model(x).squeeze(1))
    brier=float(((p-y)**2).mean());bce=float(F.binary_cross_entropy(p.clamp(1e-6,1-1e-6),y))
    return {'count':len(rows),'positives':int(y.sum()),'brier':brier,
       'logloss':bce,'average_precision':average_precision(p.tolist(),y.tolist()),
       'base_rate':float(y.mean())}

def baseline_rankings(rows):
    labels=[r['low_prospect'] for r in rows]
    return {
        'constant_base_rate':sum(labels)/len(labels),
        'recent_4_novelty':average_precision([-r['spatial'][0] for r in rows],labels),
        'recent_8_novelty':average_precision([-r['spatial'][1] for r in rows],labels),
        'local_visited_density':average_precision([r['spatial'][12] for r in rows],labels),
        'recent_8_retracing':average_precision([r['spatial'][6] for r in rows],labels),
    }

def first_stop(model,rec,feature,threshold):
    novel,branch,retr=outcomes(rec)
    for t in range(24,min(104,len(novel)-24)+1):
        s,c=features(rec,t,novel,branch,retr)
        v=s if feature=='spatial' else c if feature=='clock' else s+c
        p=predict(model,v)
        if p>=threshold:return t,p
    return None,None

def simple_stop(rec,window=12,maximum=2):
    novel,br,retr=outcomes(rec)
    for t in range(24,min(104,len(novel)-HORIZON)+1):
        if sum(novel[t-window:t])<=maximum:return t,None
    return None,None

def score_stop(original,stoptick):
    nov,br,retr=outcomes(original);t=stoptick if stoptick is not None else len(nov)
    if t>len(nov):raise ValueError('non-prefix STOP')
    # Conservative world-agnostic proxy: one novel cell = 1, one branch = 0.3,
    # each action consumes 0.2 resource units. Unrated by humans.
    return {'score':sum(nov[:t])+.3*sum(br[:t])-.2*(t+int(stoptick is not None)),
            'novel':sum(nov[:t]),'branches':sum(br[:t]),'steps':t+int(stoptick is not None),
            'lost_novel':sum(nov[t:]),'lost_branches':sum(br[t:]),
            'future24_novel':sum(nov[t:min(len(nov),t+24)]) if stoptick is not None else None,
            'future_remaining_novel':sum(nov[t:]) if stoptick is not None else None}

def feature_by_observation(obs,nov,branch,retr):
    # In callback only prefix event positions are available; exact recent counts
    # tracked by host-observed last position and command history.
    t=obs['tick']
    fake={'events':[None]*t+[{'observation':obs}]}
    return features(fake,t,nov,branch,retr)

def inference_rollout(w0,head,seed,family,original,mode,model=None,feature=None,threshold=None):
    # Require fresh, exact frozen model forwards, not list-of-actions replay alone.
    # Actual prefix metrics are exposed only through previously applied host actions.
    state={'nov':[],'br':[],'retr':[],'seen':{(7,48)},'pos':(7,48),'last':None,'decisions':[]}
    def selector(obs):
        t=obs['tick']
        if state['last'] is not None:
            # Exact state update from previous legal action via the source-pinned
            # original trajectory; reject any movement divergence.
            past=original['events'][t-1]
            if obs['x']!=past['position_after'][0] or obs['y']!=past['position_after'][1]:
                raise ValueError('new model movement diverged from original')
            pos=(obs['x'],obs['y'])
            state['nov'].append(int(pos not in state['seen']));state['seen'].add(pos)
            a=state['last'];state['br'].append(int(a=='BRANCH'));state['retr'].append(int(a=='RETRACE'))
        with torch.no_grad():
            logits=head(torch.tensor([model_features(obs)],dtype=torch.float32))[0].tolist()
        move=choice(logits,[a for a in obs['legal'] if a!='STOP'],.7,
                    uniform(seed,family,t,'w11-CRN'))
        pstop=stop_prob(logits,obs['legal'])
        normal=t>=124 and uniform(seed,family,t,'p10-independent-STOP-CRN-v1')<pstop
        prospect=None
        if mode=='learned' and t>=24 and t<=104:
            s,c=feature_by_observation(obs,state['nov'],state['br'],state['retr'])
            vec=s if feature=='spatial' else c if feature=='clock' else s+c
            prospect=predict(model,vec)
        if mode=='simple' and t>=24 and t<=104:
            prospect=float(sum(state['nov'][max(0,t-12):t])<=2)
        stop=normal or (prospect is not None and prospect>=threshold)
        action='STOP' if stop else move
        if t<len(original['events']) and action!='STOP' and action!=original['events'][t]['applied']:
            raise ValueError('original trained-head prefix not reproduced')
        state['last']=action
        state['decisions'].append({'tick':t,'action':action,'legal':obs['legal'],
            'head_logits':logits,'host_stop_p':pstop,'prospect':prospect,
            'frozen_head':True,'trained_aux_ranker':mode=='learned'})
        return action
    r=w0.run(seed,'callback',family,128,selector=selector)
    if not all(e['raw_legal'] and e['applied']==d['action'] for e,d in zip(r['events'],state['decisions'])):
        raise ValueError('nonlegal action submitted')
    r['head_decisions']=state['decisions'];r['mode']=f'P12-{mode}-{feature or "history"}'
    return r

def execution(source_bytes,out):
    torch.set_num_threads(1)
    z,w0,head,cksha=open_pinned(source_bytes)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    allrows=[];traces={};source_hashes={}
    for family in FAMILIES:
        for seed in DISCOVERY+VALIDATION+HELDOUT:
            rec,trace_sha,png_sha=extract_trace(z,family,seed)
            cohort='train' if seed in DISCOVERY else 'validation' if seed in VALIDATION else 'heldout'
            allrows.extend(rows_from(rec,cohort,family,seed))
            traces[family,seed]=rec
            source_hashes[f'{family}/{seed}']={'trace_sha256':trace_sha,'png_sha256':png_sha}
    tr=[r for r in allrows if r['cohort']=='train'];va=[r for r in allrows if r['cohort']=='validation'];ho=[r for r in allrows if r['cohort']=='heldout']
    models={};reports={}
    for feature in ('clock','spatial','combined'):
        models[feature],rr=train(tr,va,feature)
        rr['validation']=scoring(models[feature],va,feature)
        rr['heldout']=scoring(models[feature],ho,feature)
        reports[feature]=rr
    # Pre-select thresholds exclusively by validation policies. Explicit
    # utility is a declared *cost proxy*, not an aesthetic preference.
    thresholds=[.55,.65,.75,.85,.92,.97]
    selection={}
    for feature,model in models.items():
        options=[]
        for th in thresholds:
            rr=[]
            for family in FAMILIES:
                for seed in VALIDATION:
                    rec=traces[family,seed];t,p=first_stop(model,rec,feature,th)
                    rr.append(score_stop(rec,t)['score'])
            options.append({'threshold':th,'val_proxy_utility':sum(rr)/len(rr)})
        # favour fewer false STOPs on ties
        chosen=max(options,key=lambda x:(x['val_proxy_utility'],x['threshold']))
        selection[feature]={'threshold':chosen['threshold'],'all_options':options}
    original_matches=0;tested=[];paired=[];snapshot=[]
    for family in FAMILIES:
      for seed in HELDOUT:
        original=traces[family,seed]
        variants={'baseline':('baseline',None,None,None),
                  'simple-stall':('simple',None,None,1.)}
        variants.update({f'learned-{f}':('learned',models[f],f,selection[f]['threshold'])
                         for f in models})
        outputs={}
        for name,(mode,model,feature,th) in variants.items():
            rec=inference_rollout(w0,head,seed,family,original,mode,model,feature,th)
            s=[e['applied'] for e in rec['events']]
            base=[e['applied'] for e in original['events']]
            if s!=base and not(len(s)<=len(base) and s[-1]=='STOP' and s[:-1]==base[:len(s)-1]):
                raise ValueError('prefix-only comparison invalid')
            if name=='baseline':
                if rec['trajectory']!=original['trajectory'] or rec['metrics']!=original['metrics']:
                    raise ValueError('model original W16 late-gate replay mismatch')
                if w0.png_bytes(rec)!=z.read(f'rollouts/{family}-{seed}-intact-late.png'):
                    raise ValueError('model base PNG mismatch')
                original_matches+=1
            stoptick=(len(s)-1 if s!=base else None)
            score=score_stop(original,stoptick)
            png=w0.png_bytes(rec)
            stem=f'{family}-{seed}-{name}'
            output=json.dumps(rec,sort_keys=True,separators=(',',':')).encode()+b'\n'
            (out/(stem+'.json')).write_bytes(output);(out/(stem+'.png')).write_bytes(png)
            outputs[name]={'stop_tick':stoptick,**score,'trace_sha256':sha(output),'png_sha256':sha(png)}
            tested.append({'family':family,'seed':seed,'mode':name,'stop_tick':stoptick,**score,
                           'original_source_hash':source_hashes[f'{family}/{seed}'],
                           'trace_sha256':sha(output),'png_sha256':sha(png)})
        for name in outputs:
            if name=='baseline':continue
            paired.append({'family':family,'seed':seed,'against':'baseline',
                'method':name,'utility_delta':outputs[name]['score']-outputs['baseline']['score'],
                'novel_lost':outputs[name]['lost_novel'],
                'branch_lost':outputs[name]['lost_branches'],
                'stopped_at':outputs[name]['stop_tick']})
    summaries={}
    for name in ('baseline','simple-stall','learned-clock','learned-spatial','learned-combined'):
        rr=[x for x in tested if x['mode']==name]
        summaries[name]={'worlds':len(rr),'stopped_early':sum(x['stop_tick'] is not None for x in rr),
            'mean_proxy_utility':sum(x['score'] for x in rr)/len(rr),
            'mean_novel':sum(x['novel'] for x in rr)/len(rr),
            'mean_steps':sum(x['steps'] for x in rr)/len(rr),
            'mean_lost_novel':sum(x['lost_novel'] for x in rr)/len(rr),
            'false_stops_future24_gt6':sum(x['future24_novel'] is not None and x['future24_novel']>6 for x in rr)}
    for f,m in models.items():
        torch.save({'state_dict':m.state_dict(),'feature':f,'source_P11_SHA':P11_SHA,
                    'source_W16_checkpoint_sha':cksha,'selection':selection[f]},out/f'P12-{f}-CPU-ranker.pt')
    report={'schema':'DRIV3R-P12-W16-real-prospect-preference/v1',
        'provenance':{'p11_actions_sha256':P11_SHA,'w16_producer_sha256':W16_SHA,
           'w16_checkpoint_sha256':cksha,'p11_source_HEAD':HEAD_SHA,'source_traces':source_hashes},
        'truth_boundary':'Real frozen 142-input W16 auxiliary MLP and trained 2-layer prospect rankers in exact synthetic W0; NOT Sprocket root SLM nor native; all labels are algorithmic outcomes, NO human ratings',
        'cohorts':{'train':DISCOVERY,'validation':VALIDATION,'heldout':HELDOUT},
        'labelling':'Low prospective progress means <=3 novel moves in next 16 AND <=6 in next 24; never available to model at decision time',
        'utility_proxy':'new positions + 0.3 * branch actions - 0.2 * action count; not human/artistic preference',
        'model_training':reports,'validation_thresholds':selection,
        'new_live_frozen_head_replays':len(tested),'original_replays_exact':original_matches,
        'new_frozen_head_forwards':sum(x['steps'] for x in tested),
        'adam_updates_total':sum(x['updates'] for x in reports.values()),'human_ratings':0,
        'heldout_binary_labels':{'total':len(ho),'low_prospect_count':sum(x['low_prospect'] for x in ho)},
        'simple_baseline_ranking':baseline_rankings(ho),
        'heldout_results':tested,'heldout_pairs':paired,'summary':summaries}
    (out/'P12-results.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    (out/'P12-pair-comparisons.json').write_text(json.dumps(paired,sort_keys=True,indent=2)+'\n')
    checks=[f'{sha(f.read_bytes())}  {f.name}' for f in sorted(out.iterdir()) if f.is_file()]
    (out/'P12-SHA256SUMS.txt').write_text('\n'.join(checks)+'\n')
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--p11-artifact',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();r=execution(args.p11_artifact.read_bytes(),args.out)
    print(json.dumps({'training':r['model_training'],'summary':r['summary'],
        'new_frozen_head_forwards':r['new_frozen_head_forwards']},indent=2))
if __name__=='__main__':main()
