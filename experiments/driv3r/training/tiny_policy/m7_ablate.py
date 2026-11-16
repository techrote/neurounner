"""M7: bounded, synthetic-only causal-observation ablation of real M5 heads.

M5 EVAL seeds here are *known research worlds*, not an unseen promotion set.
No optimizer, no native access, no model checkpoint written to Git.
"""
from __future__ import annotations
import argparse, hashlib, json, random, statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
from .core import ACTIONS, FAMILIES, choose_random, raster
from .m3_curriculum import models
from .m5_topology_train import EVAL_SEEDS, PROTOCOL as M5_PROTOCOL, maze_world, novel

PROTOCOL = 'tiny-policy-m7-observation-ablation-v1'
MODES = ('live', 'no-history', 'no-obstacles', 'frozen-input', 'swap-family')
ARCH = ('mlp', 'conv')
WINDOW = 64
# Original M5 local archived checkpoint identities, independently verified before loading.
SHA256 = {'mlp': '55f0ea469278086bef2c91b533a8451e1938d1739d596dd3f0ed8542a7267940',
          'conv': '39d0c2eb067eb0a12cb454f859de8f6434294704f7e3494af1999d25b484f636'}
TRAIN_IDENT = '2728c8ae225cc122dd39102d0a80f7866a31922fd3f6b39868aa038f60ea6f4c'

def change_observation(obs, mode):
    """Only local 162 floats may change; current host legal actions are never changed."""
    v=obs.copy()
    if mode=='no-history':
        v[49:98]=0   # 7x7 visited mask
        v[155:157]=0 # dead-end and revisit counters
        v[161]=0     # mean local visitation
    elif mode=='no-obstacles':
        v[0:49]=0    # blocked cells
        v[98:147]=0 # protected cells
        v[160]=0    # mean local blocking
    elif mode=='swap-family':
        v[151:154]=np.roll(v[151:154],1)
    elif mode in ('live','frozen-input'):
        pass
    else:
        raise ValueError('Unknown ablation mode')
    return v

def load_heads(directory):
    import torch
    torch.set_num_threads(2)
    nets=models(torch)
    for arch in ARCH:
        path=directory/(arch+'.pt')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=SHA256[arch]:
            raise ValueError('Checkpoint digest mismatch: '+arch)
        data=torch.load(path,map_location='cpu',weights_only=True)
        if (data['architecture']!=arch or data['protocol']!=M5_PROTOCOL
                or data['train_identity']!=TRAIN_IDENT):
            raise ValueError('Checkpoint model / train identity mismatch')
        nets[arch].load_state_dict(data['state_dict'],strict=True)
        nets[arch].eval()
    return torch,nets

def evaluate(seed,family,policy,torch,nets):
    w=maze_world(seed,family)
    rng=random.Random(seed*1733+FAMILIES.index(family)*149)
    initial=w.observe().copy()
    history=[]
    for tick in range(WINDOW):
        legal=w.legal()
        if not legal:break
        before=w.state_sha256()
        obs=w.observe()
        if policy=='random':a=choose_random(w,rng)
        elif policy=='novel':a=novel(w,rng)
        else:
            arch,mode=policy.split(':')
            source=initial if mode=='frozen-input' else obs
            tensor=torch.from_numpy(change_observation(source,mode)[None,:])
            with torch.inference_mode():logits=nets[arch](tensor)[0]
            a=max(legal,key=lambda i:(float(logits[i]),-i))
        if a not in legal:raise ValueError('Illegal host action')
        event={'before_sha256':before,'observation_sha256':hashlib.sha256(obs.tobytes()).hexdigest(),
               'legal':[ACTIONS[x] for x in legal]}
        event.update(w.apply(int(a)))
        history.append(event)
        if w.stopped:break
    receipt={'protocol':PROTOCOL,'seed':seed,'family':family,'policy':policy,
             'synthetic_only':True,'actions':history,'cells':int(w.visited.sum()),
             'branches':w.branch_nodes(),'retraces':w.revisit_landings,'dead_ends':w.dead_ends,
             'steps':len(history),'early_stop':bool(w.actions and w.actions[-1]=='STOP'),
             'end_sha256':w.state_sha256(),'window':WINDOW}
    return receipt,w

def replay(receipt):
    if (receipt['protocol']!=PROTOCOL or receipt['seed'] not in EVAL_SEEDS
            or receipt['family'] not in FAMILIES or receipt['window']!=WINDOW
            or receipt['synthetic_only'] is not True):raise ValueError('Receipt identity invalid')
    w=maze_world(receipt['seed'],receipt['family'])
    if not 1<=len(receipt['actions'])<=WINDOW:raise ValueError('Unbounded history')
    for e in receipt['actions']:
        if w.state_sha256()!=e['before_sha256']:raise ValueError('Tampered state')
        if hashlib.sha256(w.observe().tobytes()).hexdigest()!=e['observation_sha256']:
            raise ValueError('Tampered observation')
        if e['legal']!=[ACTIONS[a] for a in w.legal()] or e['action'] not in e['legal']:
            raise ValueError('Tampered legal menu')
        got=w.apply(ACTIONS.index(e['action']))
        if got!={k:e[k] for k in got}:raise ValueError('Tampered action receipt')
    expected={'end_sha256':w.state_sha256(),'cells':int(w.visited.sum()),
              'branches':w.branch_nodes(),'retraces':w.revisit_landings,
              'dead_ends':w.dead_ends,'steps':len(w.actions),
              'early_stop':bool(w.actions and w.actions[-1]=='STOP')}
    if any(receipt.get(k)!=v for k,v in expected.items()):raise ValueError('Wrong terminal aggregate')
    return w

def experiment(checkpoints,output):
    if output.exists():raise FileExistsError('Evidence destination already exists')
    output.mkdir(parents=True)
    root=output/'rollouts';root.mkdir()
    torch,nets=load_heads(checkpoints)
    policies=('novel','random')+tuple(a+':'+m for a in ARCH for m in MODES)
    rows=[];gallery={}
    for seed in EVAL_SEEDS:
        for family in FAMILIES:
            for policy in policies:
                receipt,w=evaluate(seed,family,policy,torch,nets)
                check=replay(receipt)
                if not np.array_equal(w.visited,check.visited):raise ValueError('Pixel replay mismatch')
                name=f'{seed}-{family}-{policy.replace(":","_")}'
                (root/(name+'.json')).write_text(json.dumps(receipt,sort_keys=True,separators=(',',':'))+'\n')
                raster(w,2).save(root/(name+'.png'))
                rows.append({k:receipt[k] for k in ('seed','family','policy','cells','branches','retraces','dead_ends','steps','early_stop')})
                if seed==EVAL_SEEDS[0]:gallery[(family,policy)]=w
    from PIL import Image,ImageDraw
    canvas=Image.new('RGB',(len(policies)*192,len(FAMILIES)*216),(248,248,245))
    draw=ImageDraw.Draw(canvas)
    for y,family in enumerate(FAMILIES):
        for x,policy in enumerate(policies):
            draw.text((x*192+3,y*216+3),family[:4]+' '+policy,fill=(0,0,0))
            canvas.paste(raster(gallery[(family,policy)],2),(x*192,y*216+22))
    canvas.save(output/'comparison.png')
    aggregate={}
    for policy in policies:
        subset=[r for r in rows if r['policy']==policy]
        aggregate[policy]={'n':len(subset), 'cells':round(statistics.mean(r['cells'] for r in subset),3),
          'branches':round(statistics.mean(r['branches'] for r in subset),3),
          'retraces':round(statistics.mean(r['retraces'] for r in subset),3),
          'early_stop':sum(r['early_stop'] for r in subset)}
    report={'protocol':PROTOCOL,'synthetic_only':True,'source_caveat':'local reconstructed W0; not exact GitHub HEAD',
            'evaluation_seeds':EVAL_SEEDS,'evaluation_seeds_previously_examined':True,
            'checkpoint_sha256':SHA256,'train_identity_sha256':TRAIN_IDENT,
            'policies':policies,'models':ARCH,'optimizer_updates':0,
            'applied_illegal_actions':0,'rollouts':len(rows),'aggregate':aggregate}
    (output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    manifest={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return report

def main():
    a=argparse.ArgumentParser();a.add_argument('--checkpoints',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    args=a.parse_args()
    print(json.dumps(experiment(args.checkpoints,args.output),indent=2))
if __name__=='__main__':main()
