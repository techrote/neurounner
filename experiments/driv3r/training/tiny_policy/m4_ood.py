"""M4: topology-out-of-distribution *inference-only* stress test for M3 tiny heads.

Research-only 96x96 synthetic legal-action host. Loads immutable M3 small-head
checkpoints; does not train, modify core, or claim native CyberSand authority.
M3 shared four obstacle templates; M4 uses six graph-derived, one-cell-wide
corridor topologies to test actual generalization rather than new random seeds
on the same four fixtures. STOP is never suppressed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import deque
from pathlib import Path

import numpy as np

from .core import ACTIONS, DELTA, FAMILIES, World, choose_random, choose_wall, raster
from .m3_curriculum import BUDGET, oracle, models

TOPOLOGIES = ("comb", "ladder", "serpentine", "spoked_ring", "courtyard", "braided")
SEEDS = tuple(range(11040, 11052))  # two independent worlds per topology
WINDOW = 64
CHECKPOINT_SHA256 = {
    "linear": "a5d1ec8f5f956c05968f7b5b538952e68d89b861d337a48559e0de759fc32435",
    "mlp": "1b2c9cea0b99fffde0b3ebd39942835540880d93a6e70ce1518ff39c508ceeb5",
    "conv": "b44a97d0a6af84010280bf83e22c481afa15546d8b1575ef4c855d5bb6661d6d",
}


def geometry_hash(w):
    return hashlib.sha256(w.blocked.tobytes()+w.protected.tobytes()).hexdigest()


def world(seed: int, family: str) -> World:
    """Graph-carved, seeded topological obstacles independent of family label."""
    if seed not in SEEDS:
        raise ValueError("Unregistered M4 evaluation seed")
    w = World(seed, family, budget=BUDGET)
    w.blocked[29:68,29:68] = True
    w.protected[29:68,29:68] = False
    variant=(seed-SEEDS[0])%6
    rnd=random.Random(seed*9709 + variant)
    edges=set()
    nodes=set()

    def edge(x1,y1,x2,y2):
        a=(x1,y1);b=(x2,y2)
        if abs(x1-x2)+abs(y1-y2)!=1:raise ValueError("Nonadjacent graph nodes")
        edges.add(tuple(sorted((a,b))));nodes.add(a);nodes.add(b)

    def h(y,x0,x1):
        for x in range(min(x0,x1),max(x0,x1)): edge(x,y,x+1,y)

    def v(x,y0,y1):
        for y in range(min(y0,y1),max(y0,y1)): edge(x,y,x,y+1)

    if variant == 0:  # Alternating teeth with genuinely terminal branches
        h(6,1,11)
        for x in (2,4,6,8,10):
            tip=2+rnd.randrange(0,2) if x%4==2 else 9+rnd.randrange(0,3)
            v(x,6,tip)
        h(3,2,4); h(9,6,8)
    elif variant == 1:  # Parallel channels with sparing cross links
        v(3,1,11); v(9,1,11);h(6,3,9)
        for y in (2,9):h(y,3,9)
        h(6,6,9);v(6,6,2)
        if seed%2: v(6,6,10)
    elif variant == 2:  # Long serpentine, hard turn memory
        for y in (2,4,6,8,10):h(y,1,11)
        for y in (2,4,6,8):v(11 if y%4==2 else 1,y,y+2)
        v(6,6,7)
        if seed%2:v(6,4,6)
    elif variant == 3:  # Cyclic ring and dead-end spokes
        h(2,2,10);h(10,2,10);v(2,2,10);v(10,2,10)
        h(6,2,10);v(6,2,10)
        for x in (4,8):v(x,6,8 if rnd.randrange(2) else 4)
    elif variant == 4:  # Nested courtyard, conflicting protected islands
        h(3,3,9);h(9,3,9);v(3,3,9);v(9,3,9)
        h(6,3,9);v(6,3,9)
        h(1,4,8);v(4,1,3);v(8,1,3)
        h(11,3,9);v(3,9,11);v(9,9,11)
    else:  # Braided: multiple loops plus dangling stubs
        for y in (2,6,10):h(y,2,10)
        for x in (2,6,10):v(x,2,10)
        v(4,6,9);v(8,3,6);h(4,6,8)
        for x in (4,8):h(8,6,x)
    # Node lattice spacing is three real pixels; only traversable edges are carved.
    def xy(node):return (48+(node[0]-6)*3,48+(node[1]-6)*3)
    for n in nodes:
        x,y=xy(n);w.blocked[y,x]=False
    for a,b in edges:
        ax,ay=xy(a); bx,by=xy(b)
        if ax==bx:w.blocked[min(ay,by):max(ay,by)+1,ax]=False
        else:w.blocked[ay,min(ax,bx):max(ax,bx)+1]=False
    # Proper protected cells block otherwise valid corridors, but keep origin free.
    gates={0:[(4,3)],1:[(9,2)],2:[(11,3)],3:[(2,6)],4:[(6,3)],5:[(2,2)]}
    for node in gates[variant]:
        x,y=xy(node)
        if (x,y)!=(48,48):
            w.blocked[y,x]=False
            w.protected[y,x]=True
    w.blocked[48,48]=False;w.protected[48,48]=False
    if not any(a!=4 for a in w.legal()):raise ValueError("Initial world trap")
    if reachable(w)<30:raise ValueError("World too small for useful OOD evaluation")
    return w


def reachable(w):
    """Audit-only global reachability; never provided to policies."""
    queue=deque([(w.x,w.y)]);seen={(w.x,w.y)}
    while queue:
        x,y=queue.popleft()
        for dx,dy in DELTA[:4]:
            nx,ny=x+dx,y+dy
            if (nx,ny) not in seen and 0<=nx<96 and 0<=ny<96 and not (w.blocked[ny,nx] or w.protected[ny,nx]):
                seen.add((nx,ny));queue.append((nx,ny))
    return len(seen)


def novel(w,_rng):
    legal=[a for a in w.legal() if a!=4]
    if not legal:return 4
    unseen=[a for a in legal if not w.visited[w.y+DELTA[a][1],w.x+DELTA[a][0]]]
    return min(unseen or legal,key=lambda a:(a!=w.heading,a))


def model_policy(torch,model):
    model.eval()
    def decision(w,_rng):
        with torch.inference_mode():
            logits=model(torch.from_numpy(w.observe()[None,:]))[0]
        return max(w.legal(),key=lambda a:(float(logits[a]),-a))
    return decision


def load_heads(directory:Path):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(730)
    heads=models(torch)
    untrained=__import__('copy').deepcopy(heads['conv'])
    ids=set()
    for name,model in heads.items():
        file=directory/(name+'.pt')
        if hashlib.sha256(file.read_bytes()).hexdigest()!=CHECKPOINT_SHA256[name]:
            raise ValueError(f"M3 {name} checkpoint digest mismatch")
        p=torch.load(file,map_location='cpu',weights_only=True)
        if p['architecture']!=name or p['protocol']!='tiny-policy-m3-curriculum-cpu-v1':
            raise ValueError("M3 checkpoint identity mismatch")
        ids.add(p['train_identity'])
        model.load_state_dict(p['state_dict'],strict=True)
        model.eval()
    if len(ids)!=1:raise ValueError('Mixed M3 training identities')
    untrained.eval()
    return torch,heads,untrained,next(iter(ids))


def rollout(seed,family,selector,policy):
    w=world(seed,family)
    start_hash=geometry_hash(w)
    rng=random.Random(seed*7919+FAMILIES.index(family)*17)
    events=[]
    for _ in range(WINDOW):
        legal=w.legal()
        if not legal:break
        obs=w.observe()
        event={'before_sha256':w.state_sha256(),
               'legal':[ACTIONS[a] for a in legal],
               'feature_sha256':hashlib.sha256(obs.tobytes()).hexdigest()}
        choice=int(selector(w,rng))
        if choice not in legal:raise ValueError('Illegal action proposed')
        event.update(w.apply(choice));events.append(event)
        if w.stopped:break
    receipt={'seed':seed,'family':family,'policy':policy,'topology':TOPOLOGIES[(seed-SEEDS[0])%6],
             'synthetic_only':True,'budget':BUDGET,'window':WINDOW,
             'geometry_sha256':start_hash,'events':events,
             'final_sha256':w.state_sha256(),
             'steps':len(events),'cells':int(w.visited.sum()),
             'branches':w.branch_nodes(),'retraces':w.revisit_landings,'dead_ends':w.dead_ends,
             'stopped':bool(w.actions and w.actions[-1]=='STOP'),
             'reachable':reachable(world(seed,family))}
    if receipt['steps']>WINDOW:raise ValueError('Exceeded window')
    return receipt,w


def replay(receipt):
    w=world(receipt['seed'],receipt['family'])
    if receipt['geometry_sha256']!=geometry_hash(w) or receipt['budget']!=BUDGET or receipt['window']!=WINDOW:
        raise ValueError('World/budget provenance mismatch')
    if len(receipt['events'])>WINDOW or not receipt['events']:
        raise ValueError('Bad history length')
    for event in receipt['events']:
        if w.state_sha256()!=event['before_sha256']:raise ValueError('State hash mismatch')
        if hashlib.sha256(w.observe().tobytes()).hexdigest()!=event['feature_sha256']:
            raise ValueError('Observation hash mismatch')
        if [ACTIONS[a] for a in w.legal()]!=event['legal']:raise ValueError('Host legality mismatch')
        if event['action'] not in ACTIONS:raise ValueError('Unknown action')
        got=w.apply(ACTIONS.index(event['action']))
        if got!={k:event[k] for k in got}:raise ValueError('Host receipt mismatch')
    actual={'final_sha256':w.state_sha256(),'steps':len(w.actions),'cells':int(w.visited.sum()),
            'branches':w.branch_nodes(),'retraces':w.revisit_landings,'dead_ends':w.dead_ends,
            'stopped':bool(w.actions and w.actions[-1]=='STOP')}
    if any(receipt[k]!=v for k,v in actual.items()):raise ValueError('Final metrics mismatch')
    return w


def manifest(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and p.name!='manifest.json'}


def experiment(out,ckdir):
    from PIL import Image,ImageDraw
    if out.exists():raise FileExistsError('Refuse overwrite immutable M4 outputs')
    out.mkdir(parents=True)
    torch,heads,untrained,training_id=load_heads(ckdir)
    policies={'oracle':oracle,'random':choose_random,'wall':choose_wall,'novel-greedy':novel,
              'conv-untrained':model_policy(torch,untrained)}
    policies.update({name:model_policy(torch,model) for name,model in heads.items()})
    report={'status':'STARTED','protocol':'tiny-policy-m4-ood-v1','synthetic_only':True,
            'native_game':False,'optimizer_updates':0,'action_strings':list(ACTIONS),
            'trained_checkpoint_sha256':CHECKPOINT_SHA256,
            'm3_training_identity_sha256':training_id,
            'world_size':96,'budget':BUDGET,'window':WINDOW,'seeds':list(SEEDS),
            'topologies':list(TOPOLOGIES),'policies':{},'worlds':{},'illegal_applied_actions':0}
    rows=[];proofs={};items=out/'rollouts';items.mkdir()
    try:
        for seed in SEEDS:
            for family in FAMILIES:
                w=world(seed,family)
                gh=geometry_hash(w)
                if family==FAMILIES[0]: report['worlds'][str(seed)]={'geometry_sha256':gh,'reachable':reachable(w),'topology':TOPOLOGIES[(seed-SEEDS[0])%6]}
                elif gh!=report['worlds'][str(seed)]['geometry_sha256']:
                    raise ValueError('Family labels changed obstacle geometry')
                for name,selector in policies.items():
                    receipt,paint=rollout(seed,family,selector,name)
                    roundtrip=replay(receipt)
                    if not np.array_equal(roundtrip.visited,paint.visited):raise ValueError('Replay pixels differ')
                    key=(seed,family,name)
                    proofs[key]=receipt
                    fname=f'{seed}-{family}-{name}'
                    (items/(fname+'.json')).write_text(json.dumps(receipt,sort_keys=True)+'\n')
                    raster(paint,3).save(items/(fname+'.png'))
                    rows.append({'seed':seed,'family':family,'topology':receipt['topology'],
                                 'policy':name,**{k:receipt[k] for k in ('steps','cells','branches','retraces','dead_ends','stopped','reachable')}})
        for topology_index in range(6):
            seed=SEEDS[topology_index]
            tile=288;header=20
            sheet=Image.new('RGB',(tile*len(policies),(tile+header)*3),'white')
            draw=ImageDraw.Draw(sheet)
            for i,family in enumerate(FAMILIES):
                for j,name in enumerate(policies):
                    r=proofs[(seed,family,name)]
                    sheet.paste(raster(replay(r),3),(j*tile,i*(tile+header)+header))
                    draw.text((j*tile+3,i*(tile+header)+3),f'{family} {name} {r["cells"]}/{r["reachable"]}',fill='black')
            sheet.save(out/f'm4-original-{TOPOLOGIES[topology_index]}-{seed}.png')
        for name in policies:
            group=[r for r in rows if r['policy']==name]
            report['policies'][name]={'n':len(group),
                'mean_cells':round(statistics.mean(r['cells'] for r in group),2),
                'mean_coverage_pct':round(100*statistics.mean(r['cells']/r['reachable'] for r in group),2),
                'mean_branches':round(statistics.mean(r['branches'] for r in group),2),
                'mean_retraces':round(statistics.mean(r['retraces'] for r in group),2),
                'mean_dead_ends':round(statistics.mean(r['dead_ends'] for r in group),2),
                'early_stops':sum(r['stopped'] for r in group),
                'mean_steps':round(statistics.mean(r['steps'] for r in group),2)}
        report['by_topology']={top:{name:round(statistics.mean(r['cells'] for r in rows if r['topology']==top and r['policy']==name),2)
                                    for name in policies} for top in TOPOLOGIES}
        (out/'rollout_rows.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
        report['status']='COMPLETED'
    except BaseException as exc:
        report['status']='FAILED';report['error']=repr(exc)
        raise
    finally:
        (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        (out/'manifest.json').write_text(json.dumps(manifest(out),indent=2,sort_keys=True)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoints',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    r=experiment(args.out,args.checkpoints)
    print(json.dumps({'status':r['status'],'policies':r['policies'],'by_topology':r['by_topology']},indent=2))

if __name__=='__main__':main()
