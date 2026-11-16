"""Counterfactual-goal supervision for external CyberSand-style model probes.

A paired sample has identical position, facing, obstacles, painted cells and
host-offered legal actions; only the target goal changes. No game imports.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import random
import re
from .world import ACTIONS, DIRS, World

def directional_goals(world):
    goals=[]
    for dx,dy in DIRS:
        for length in (5,4,3,2,1):
            p=(world.pos[0]+dx*length,world.pos[1]+dy*length)
            if world.safe(p):
                goals.append(p)
                break
    return tuple(goals)

def target(world):
    """A deterministic goal-aware teacher, not an optimal game planner."""
    ds=world.destinations()
    actions=[a for a in ACTIONS if a in ds and a!='STOP']
    if not actions:
        return 'STOP'
    old=abs(world.goal[0]-world.pos[0])+abs(world.goal[1]-world.pos[1])
    def utility(a):
        p=ds[a]
        dist=abs(world.goal[0]-p[0])+abs(world.goal[1]-p[1])
        progress=old-dist
        novelty=0.6 if p not in world.painted else -0.6
        bonus={'BRANCH':.3 if (world.seed+world.turns)%3==0 else -.35,
               'RETRACE':-.75,'PENUP':-.45}.get(a,0.)
        return 3.0*progress+novelty+bonus
    return max(actions,key=utility)

def samples(seeds,steps=12):
    """(prompt, class id, seed, turn, pair-id), disjoint by seed."""
    out=[]
    for seed in seeds:
        world=World.make(seed)
        rng=random.Random(seed*101+17)
        for t in range(steps):
            original=world.goal
            for goal in directional_goals(world):
                world.goal=goal
                out.append((world.prompt(),ACTIONS.index(target(world)),seed,t,(seed,t)))
            world.goal=original
            act=world.teacher(rng)
            if act=='STOP':
                break
            world.apply(act)
    return out

def paired_switch_metrics(rows,predictions):
    """Only goal-swapped pairs where the teacher target actually changes."""
    if len(rows)!=len(predictions):
        raise ValueError('prediction length mismatch')
    groups=defaultdict(list)
    for row,pred in zip(rows,predictions):
        groups[row[4]].append((row[1],int(pred)))
    pairs=both=switched=0
    for rec in groups.values():
        for i in range(len(rec)):
            for j in range(i+1,len(rec)):
                ya,pa=rec[i];yb,pb=rec[j]
                if ya==yb:
                    continue
                pairs+=1
                both+=int(ya==pa and yb==pb)
                switched+=int(pa!=pb)
    return dict(changed_target_pairs=pairs,both_correct_pairs=both,
                pair_accuracy=round(both/pairs,4) if pairs else None,
                predicted_switch_rate=round(switched/pairs,4) if pairs else None)

def raw_features(prompt):
    """Cheap structured-game feature control, not a pretrained model forward."""
    m=re.fullmatch(r'WORM d=(\d+) goal=(-?\d+),(-?\d+) t=(\d+) map=([.#P*\/]+) legal=([A-Z,]+) choose:',prompt)
    if not m:
        raise ValueError(f'unsupported synthetic prompt: {prompt!r}')
    facing,dx,dy,turn,grid,menu=m.groups()
    cells=grid.replace('/','')
    if len(cells)!=25:
        raise ValueError('wrong percept size')
    v=[int(facing)==i for i in range(4)]
    v += [int(dx)/10,int(dy)/10,int(turn)/16]
    for c in cells:
        v.extend(float(c==s) for s in ('.','#','P','*'))
    allowed=set(menu.split(','))
    v.extend(float(a in allowed) for a in ACTIONS)
    return [float(x) for x in v]

def class_histogram(rows):
    c=Counter(r[1] for r in rows)
    return {a:c[i] for i,a in enumerate(ACTIONS)}
