"""P10: real frozen W16 PyTorch auxiliary-head inference in source-pinned W0.

2x3 matched factorial: STOP offered at tick zero vs >=124; clock intact,
zeroed or frozen at tick zero. A separate STOP RNG preserves conditional
movement before stopping. NOT Sprocket root-LLM inference, native Godot,
human preference or trained reward. Writes complete host receipts and PNGs.
"""
import argparse
import hashlib
import io
import json
import math
import types
from pathlib import Path

import torch
from p8_stop_retention import ACTIONS, FAMILIES, archive, choice, stop_probability, digest

SEEDS=(2609,2617,2621,2633,2647,2657,2663,2683,2693)
W0_SHA='81c3923cb38d7f196b99986806216388c34ebc27fd4976e168b3b370c25a323e'
PT='campaign/onpolicy-branch-history-replay.pt'
CLOCKS=('intact','zeroed','frozen_t0')


def model_features(obs,clock='intact'):
    if clock not in CLOCKS or len(obs['local_5x5'])!=5 or obs['local_5x5'][2][2]!='@':
        raise ValueError('invalid model observation/clock')
    if set(obs['legal'])-set(ACTIONS) or 'STOP' not in obs['legal']:
        raise ValueError('untrusted W0 legal menu')
    f=[float(ch==s) for line in obs['local_5x5'] for ch in line for s in '.#+@']
    f += [float(obs['heading']==k) for k in range(4)]
    f += [float(obs['family']==s) for s in ('greebles','pipework','architecture')]
    tick=0 if clock=='frozen_t0' else obs['tick']
    basis=([math.sin(2*math.pi*tick/p) for p in (6,9,12,13,15,17,32)]+
           [math.cos(2*math.pi*tick/p) for p in (6,9,12,13,15,17,32)])
    f += [0.]*14 if clock=='zeroed' else basis
    f += [0. if clock!='intact' else min(obs['tick']/128,1.),obs['x']/96,obs['y']/96]
    f += [float(a in obs['legal']) for a in ACTIONS]
    recent=obs['last_actions']
    f += [sum(a==s for a in recent)/4 for s in ACTIONS]
    f += [float(bool(recent) and recent[-1]==s) for s in ACTIONS]
    if len(f)!=142:raise ValueError('original W9 feature contract drift')
    return f


def deterministic_uniform(seed,family,tick,salt):
    return int.from_bytes(hashlib.sha256(f'{seed}/{family}/{tick}/{salt}'.encode()).digest()[:8],'big')/2**64


def open_original(data):
    z,manifest=archive(data)  # immutable W16 SHA, all 213 member SHA, 214 CRCs, producer HEAD
    b=z.read('pinned-W0.py')
    if digest(b)!=W0_SHA or manifest['pinned-W0.py']!=W0_SHA:raise ValueError('W0 source mismatch')
    w0=types.ModuleType('pinned_original_w0')
    exec(compile(b,'pinned-W0.py','exec'),w0.__dict__)
    if w0.ACTIONS!=ACTIONS:raise ValueError('W0 action contract mismatch')
    ck=z.read(PT)
    if digest(ck)!=manifest[PT]:raise ValueError('model checkpoint hash mismatch')
    payload=torch.load(io.BytesIO(ck),map_location='cpu',weights_only=True)
    if payload['version']!='sprocket-w16-branch-history-dual-retention-v1' or payload['stage']!='onpolicy-branch-history-replay':
        raise ValueError('wrong W16 trained checkpoint')
    head=torch.nn.Sequential(torch.nn.Linear(142,64),torch.nn.Tanh(),
             torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,6))
    head.load_state_dict({k.removeprefix('layers.'):v for k,v in payload['head'].items()},strict=True)
    head.eval()
    return z,manifest,w0,head,digest(ck)


def scores(head,obs,clock):
    with torch.no_grad():return head(torch.tensor([model_features(obs,clock)],dtype=torch.float32))[0].tolist()


def replay_original(z,manifest,w0,head):
    """Source-match nine actual archived trained-head trajectories and PNG bytes."""
    checks=[]
    for family in FAMILIES:
      for seed in SEEDS[:3]:
        stem=f'campaign/original-png-json/{family}-W16-final-t07-{seed}'
        src=z.read(stem+'.json');png=z.read(stem+'.png')
        if digest(src)!=manifest[stem+'.json'] or digest(png)!=manifest[stem+'.png']:
            raise ValueError('original source identity mismatch')
        old=json.loads(src);pred=[]
        def selector(obs):
            logits=scores(head,obs,'intact')
            menu=[a for a in obs['legal'] if a!='STOP' or obs['tick']>=124]
            a=choice(logits,menu,.7,deterministic_uniform(seed,family,obs['tick'],'w11-CRN'))
            pred.append((a,logits));return a
        rec=w0.run(seed,'callback',family,128,selector=selector)
        if rec['trajectory']!=old['trajectory'] or rec['metrics']!=old['metrics'] or w0.png_bytes(rec)!=png:
            raise ValueError('original W16 model/host replay diverged')
        if len(pred)!=len(old['head_decisions']) or any(
              a!=h['selected'] or max(abs(v-w) for v,w in zip(logits,h['raw_six_scores']))>1e-5
              for (a,logits),h in zip(pred,old['head_decisions'])):
            raise ValueError('original forward/selected action mismatch')
        checks.append({'family':family,'seed':seed,'decisions':len(pred),
                       'original_trace_sha256':digest(src),'original_png_sha256':digest(png)})
    return checks


def rollout(w0,head,seed,family,gate,clock):
    decisions=[]
    def selector(obs):
        logits=scores(head,obs,clock)
        move_u=deterministic_uniform(seed,family,obs['tick'],'w11-CRN')
        stop_u=deterministic_uniform(seed,family,obs['tick'],'p10-independent-STOP-CRN-v1')
        stop_p=stop_probability(logits,obs['legal'],.7)
        if (gate=='open' or obs['tick']>=124) and stop_u<stop_p:chosen='STOP'
        else:chosen=choice(logits,[a for a in obs['legal'] if a!='STOP'],.7,move_u)
        decisions.append({'tick':obs['tick'],'raw_six_scores':logits,
            'host_legal':obs['legal'],'model_selected':chosen,'stop_probability':stop_p,
            'movement_draw':move_u,'stop_draw':stop_u,'root_forward':False})
        return chosen
    rec=w0.run(seed,'callback',family,128,selector=selector)
    if len(rec['events'])!=len(decisions) or any(not e['raw_legal'] or e['applied']!=d['model_selected']
         for e,d in zip(rec['events'],decisions)):
        raise ValueError('non-host-legal action or receipt mismatch')
    rec['head_decisions']=decisions;rec['mode']=f'W16-final-{gate}-{clock}'
    rec['engine']='real-frozen-W16-aux-MLP-in-synthetic-W0-not-native'
    return rec


def pair(late,open_):
    ea=[e['applied'] for e in late['events']]
    eb=[e['applied'] for e in open_['events']]
    if eb==ea:return {'early_stop':False,'first_stop_tick':None,'lost_novel':0}
    if len(eb)>len(ea) or eb[-1]!='STOP' or eb[:-1]!=ea[:len(eb)-1] or (
       open_['trajectory'][:-1]!=late['trajectory'][:len(eb)]):
        raise ValueError('STOP intervention changed conditional movement before stopping')
    return {'early_stop':len(eb)-1<124,'first_stop_tick':len(eb)-1,
            'lost_novel':late['metrics']['new_steps']-open_['metrics']['new_steps']}


def run(source,out):
    torch.set_num_threads(1)
    z,manifest,w0,head,cksha=open_original(source)
    checks=replay_original(z,manifest,w0,head)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    results=[];paired=[]
    for family in FAMILIES:
      for seed in SEEDS:
        for clock in CLOCKS:
          group={}
          for gate in ('late','open'):
            rec=rollout(w0,head,seed,family,gate,clock)
            stem=f'{family}-{seed}-{gate}-{clock}'
            trace=json.dumps(rec,sort_keys=True,separators=(',',':')).encode()+b'\n'
            png=w0.png_bytes(rec)
            (out/(stem+'.json')).write_bytes(trace);(out/(stem+'.png')).write_bytes(png)
            group[gate]=rec
            results.append({'family':family,'seed':seed,'gate':gate,'clock':clock,
                'steps':rec['steps_executed'],'stop':rec['metrics']['terminated_by_stop'],
                'new':rec['metrics']['new_steps'],'branches':rec['metrics']['branches'],
                'retraces':rec['metrics']['retraces'],'invalid':rec['metrics']['invalid_raw_choices'],
                'trace_sha256':digest(trace),'png_sha256':digest(png)})
          paired.append({'family':family,'seed':seed,'clock':clock,**pair(group['late'],group['open'])})
    summary=[]
    for clock in CLOCKS:
      for gate in ('late','open'):
        rr=[r for r in results if r['clock']==clock and r['gate']==gate]
        summary.append({'clock':clock,'gate':gate,'n':len(rr),
            'early_stop':sum(r['stop'] and r['steps']<=124 for r in rr),
            'mean_new':sum(r['new'] for r in rr)/len(rr),
            'mean_branch':sum(r['branches'] for r in rr)/len(rr)})
    report={'schema':'driv3r-preference-P10-genuine-frozen-head-factorial/v1',
            'original_producer_SHA256':digest(source),'checkpoint_SHA256':cksha,
            'original_sources_reproduced':checks,
            'source_class':'trained W16 auxiliary PyTorch MLP, synthetic W0, not Sprocket root model/native',
            'training_updates':0,'human_ratings':0,
            'new_head_forwards':sum(r['steps'] for r in results),
            'runs':results,'paired':paired,'summary':summary}
    (out/'P10-results.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--original',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args()
    r=run(Path(a.original).read_bytes(),Path(a.out))
    print(json.dumps({'source_replays':len(r['original_sources_reproduced']),
         'new_forwards':r['new_head_forwards'],'summary':r['summary']}))
