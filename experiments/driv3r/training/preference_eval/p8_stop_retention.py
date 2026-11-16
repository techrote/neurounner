"""P8 original-path STOP-prefix structure exposure; NOT new model inference.

Input must be SHA-pinned W16 PyTorch-head synthetic-W0 producer ZIP.
Output is source-path-conditional expectation, NOT a STOP-open rollout,
Godot material receipt, human preference, or aesthetic rating.
"""
import argparse
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

SHA = 'a7b756a3b0c2d639e5e48dad36d518dfd0e6b752deadca8b04ac11326717eafb'
HEAD = '6005c19df9d99819cc36dc9e0a9fd37343965146'
ACTIONS = ('STRAIGHT','LEFT','RIGHT','BRANCH','RETRACE','STOP')
MODES = ('W14-greedy','W15-greedy','W15-t07','W16-final-greedy',
         'W16-final-t07','W16-mixed-greedy','W16-mixed-t07','W16-stage1-greedy')
FAMILIES = ('architecture','greebles','pipework')
SEEDS = (2609,2617,2621)
METRICS = ('new_positions','branches','retraces','moves')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def archive(data):
    if digest(data)!=SHA: raise ValueError('not the original pinned W16 producer')
    z=zipfile.ZipFile(io.BytesIO(data))
    if len(z.namelist())!=214 or z.testzip() is not None:raise ValueError('ZIP CRC/inventory failure')
    manifest={}
    for line in z.read('SHA256SUMS.txt').decode().splitlines():
        h,name=line.split('  ',1);name=name.removeprefix('./')
        if name in manifest or digest(z.read(name))!=h:raise ValueError('original member SHA mismatch')
        manifest[name]=h
    if len(manifest)!=213 or json.loads(z.read('campaign/provenance.json'))['source_head']!=HEAD:
        raise ValueError('manifest/source mismatch')
    return z,manifest


def choice(scores,menu,temp,draw):
    if temp==0:return max(menu,key=lambda a:scores[ACTIONS.index(a)])
    weights=[math.exp((scores[ACTIONS.index(a)]-max(scores[ACTIONS.index(b)] for b in menu))/temp) for a in menu]
    total=sum(weights);acc=0
    for action,w in zip(menu,weights):
        acc+=w/total
        if draw<acc:return action
    return menu[-1]


def stop_probability(scores,host_menu,temp):
    if len(scores)!=6 or any(not math.isfinite(x) for x in scores) or temp<0 or not math.isfinite(temp) or 'STOP' not in host_menu:
        raise ValueError('invalid scores/host menu/temperature')
    if temp==0:return float(choice(scores,host_menu,0,0)=='STOP')
    m=max(scores[ACTIONS.index(a)]/temp for a in host_menu)
    weights=[math.exp(scores[ACTIONS.index(a)]/temp-m) for a in host_menu]
    return weights[host_menu.index('STOP')]/sum(weights)


def expected_prefix(hazards,increments):
    if len(hazards)!=len(increments):raise ValueError('length disagreement')
    survival=1.;actual={k:0 for k in METRICS};expected={k:0. for k in METRICS}
    for p,inc in zip(hazards,increments):
        if not math.isfinite(p) or not 0<=p<=1 or set(inc)!=set(METRICS) or any(type(v)!=int or v not in (0,1) for v in inc.values()):
            raise ValueError('invalid probability/movement increment')
        survival*=1-p
        for k in METRICS:
            actual[k]+=inc[k];expected[k]+=survival*inc[k]
    return actual,expected,1-survival


def inspect(data):
    z,manifest=archive(data);episodes=[];count=0
    for family in FAMILIES:
      for seed in SEEDS:
       for mode in MODES:
        stem=f'campaign/original-png-json/{family}-{mode}-{seed}'
        t=z.read(stem+'.json');p=z.read(stem+'.png')
        if digest(t)!=manifest[stem+'.json'] or digest(p)!=manifest[stem+'.png']:
            raise ValueError('original JSON/PNG identity invalid')
        obj=json.loads(t)
        if (obj['family'],obj['seed'],obj['mode'],obj['engine'])!=(family,seed,mode,'SHA-pinned-synthetic-W0-PyTorch-head-not-native'):
            raise ValueError('original class identity invalid')
        events=obj['events'];heads=obj['head_decisions'];path=obj['trajectory']
        if len(events)!=len(heads) or len(path)!=len(events)+1 or obj['step_budget']!=128 or obj['size']!=96:
            raise ValueError('original source length mismatch')
        seen={tuple(path[0])};probabilities=[];incs=[]
        for i,(e,h) in enumerate(zip(events,heads)):
            obs=e['observation'];menu=obs['legal'];limited=h['host_legal'];scores=h['raw_six_scores'];temp=h['temperature'];draw=h['host_draw']
            if menu != [a for a in ACTIONS if a in menu] or 'STOP' not in menu or limited!=[a for a in menu if a!='STOP' or i>=124]:
                raise ValueError('host legal menu or historical STOP gate invalid')
            if obs['tick']!=i or (obs['x'],obs['y'])!=tuple(path[i]) or tuple(e['position_after'])!=tuple(path[i+1]):
                raise ValueError('original event path/tick receipt mismatch')
            if h['tick']!=i or not h['actual_head_forward'] or h['root_forward'] or not e['raw_legal'] or e['applied']!=h['selected'] or e['raw_choice']!=h['selected']:
                raise ValueError('actual model-forward choice receipt invalid')
            if not math.isfinite(draw) or not 0<=draw<1 or any(not math.isfinite(v) for v in scores) or not math.isfinite(temp) or temp<0:
                raise ValueError('sampling receipt invalid')
            if choice(scores,limited,temp,draw)!=h['selected']:
                raise ValueError('original model-head chooser replay diverged')
            pr=0. if 'STOP' in limited else stop_probability(scores,menu,temp)
            probabilities.append(pr)
            xy=tuple(path[i+1]);new=int(e['applied']!='STOP' and xy not in seen);seen.add(xy)
            incs.append({'new_positions':new,'branches':int(e['applied']=='BRANCH'),
                         'retraces':int(e['applied']=='RETRACE'),'moves':int(e['applied']!='STOP')})
            count+=1
        actual,exp,risk=expected_prefix(probabilities,incs)
        episodes.append({'family':family,'seed':seed,'mode':mode,'source_png_sha256':digest(p),'source_trace_sha256':digest(t),
                         'original':actual,'expected_original_path_prefix':exp,'early_stop_risk':risk,
                         'retained_new_positions':exp['new_positions']/actual['new_positions'] if actual['new_positions'] else None})
    if len(episodes)!=72 or count!=9005:raise ValueError('missing original model-head source receipts')
    summaries={}
    for mode in MODES:
        rows=[x for x in episodes if x['mode']==mode]
        summaries[mode]={'original_path_early_stop_risk':sum(x['early_stop_risk'] for x in rows)/9,
                         'mean_original_new_positions':sum(x['original']['new_positions'] for x in rows)/9,
                         'mean_expected_original_new_positions':sum(x['expected_original_path_prefix']['new_positions'] for x in rows)/9,
                         'mean_retained_new_positions_fraction':sum(x['retained_new_positions'] for x in rows)/9}
    return {'schema':'W16-P8-source-prefix-structure-exposure/v1','producer_sha256':SHA,'producer_source_head':HEAD,
            'producer_run':43851411911,'producer_artifact':14217467236,'verified_model_head_decisions':count,
            'original_model_episodes':len(episodes),'source_class':'real PyTorch-head choices in synthetic W0, not native Godot',
            'scope':'original-path conditional STOP-vs-continue analytical prefix; never a STOP-unmasked rollout',
            'human_ratings':0,'training_updates':0,'modes':summaries,'episodes':episodes}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--original',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();report=inspect(Path(a.original).read_bytes())
    Path(a.output).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    for mode,v in report['modes'].items():
        print(mode,round(v['original_path_early_stop_risk'],4),round(v['mean_retained_new_positions_fraction'],4))

if __name__=='__main__':main()
