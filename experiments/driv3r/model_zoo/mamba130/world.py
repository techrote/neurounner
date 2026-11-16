"""Deterministic model-independent micro-worlds for game-policy onboarding.

This is an external research projector, NOT a CyberSand native simulator.
Host owns legal actions; no game code/assets are imported or modified.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import random

ACTIONS = ('STRAIGHT', 'LEFT', 'RIGHT', 'BRANCH', 'RETRACE', 'PENUP', 'STOP')
DIRS = ((0,-1),(1,0),(0,1),(-1,0))
N=11

@dataclass
class World:
    seed: int
    pos: tuple[int,int]=(5,5)
    facing: int=0
    goal: tuple[int,int]=(9,9)
    walls: set=field(default_factory=set)
    protected: set=field(default_factory=set)
    painted: set=field(default_factory=set)
    history: list=field(default_factory=list)
    turns: int=0
    stopped: bool=False

    @classmethod
    def make(cls, seed:int) -> World:
        r=random.Random(seed)
        walls=set(); protected=set()
        for y in range(N):
            for x in range(N):
                if (x,y)==(5,5): continue
                v=r.random()
                if v<0.13: walls.add((x,y))
                elif v<0.20: protected.add((x,y))
        available=[(x,y) for y in range(N) for x in range(N)
                   if (x,y) not in walls|protected and abs(x-5)+abs(y-5)>4]
        goal=r.choice(available) if available else (5,5)
        w=cls(seed=seed, goal=goal, walls=walls, protected=protected,
              facing=r.randrange(4))
        w.painted.add(w.pos); w.history.append(w.pos)
        return w

    def ahead(self, rotate=0, steps=1):
        d=DIRS[(self.facing+rotate)%4]
        return (self.pos[0]+d[0]*steps,self.pos[1]+d[1]*steps)

    def safe(self,p):
        x,y=p
        return 0<=x<N and 0<=y<N and p not in (self.walls|self.protected)

    def destinations(self):
        """Host-authoritative offered destinations; STOP always legal."""
        ds={'STOP':self.pos}
        for a,rot in [('STRAIGHT',0),('LEFT',-1),('RIGHT',1),('BRANCH',-1 if self.turns%2==0 else 1)]:
            dest=self.ahead(rot)
            if self.safe(dest): ds[a]=dest
        if len(self.history)>1: ds['RETRACE']=self.history[-2]
        p1=self.ahead(0)
        p2=self.ahead(0,2)
        if self.safe(p1) and self.safe(p2): ds['PENUP']=p2
        return ds

    def legal(self):
        return tuple(a for a in ACTIONS if a in self.destinations())

    def apply(self,action):
        if self.stopped: raise ValueError('already stopped')
        ds=self.destinations()
        if action not in ds: raise ValueError(f'illegal {action}')
        if action=='STOP': self.stopped=True
        else:
            if action=='LEFT' or action=='BRANCH' and self.turns%2==0: self.facing=(self.facing-1)%4
            elif action=='RIGHT' or action=='BRANCH' and self.turns%2==1: self.facing=(self.facing+1)%4
            self.pos=ds[action]
            if action=='RETRACE':
                self.history.pop()
            elif action!='PENUP':
                self.history.append(self.pos)
                self.painted.add(self.pos)
            # PENUP preserves historical painted route; creates no paint.
        self.turns+=1
        return dict(action=action,pos=self.pos,painted=len(self.painted),turn=self.turns)

    def prompt(self):
        """Small fixed-radius percept with goal and host legal action names."""
        gx,gy=self.goal; x,y=self.pos
        glyphs=[]
        for j in range(y-2,y+3):
            for i in range(x-2,x+3):
                p=i,j
                if not (0<=i<N and 0<=j<N): glyphs.append('#')
                elif p in self.protected: glyphs.append('P')
                elif p in self.walls: glyphs.append('#')
                elif p in self.painted: glyphs.append('*')
                else: glyphs.append('.')
        local='/'.join(''.join(glyphs[k:k+5]) for k in range(0,25,5))
        return f'WORM d={self.facing} goal={gx-x},{gy-y} t={self.turns%16} map={local} legal={",".join(self.legal())} choose:'

    def teacher(self, rng:random.Random):
        choices=[a for a in self.legal() if a!='STOP']
        if not choices: return 'STOP'
        ds=self.destinations()
        def utility(a):
            p=ds[a]; dist=abs(self.goal[0]-p[0])+abs(self.goal[1]-p[1])
            novelty=2.1 if p not in self.painted else -2.0
            special={'BRANCH':0.7,'RETRACE':-0.4,'PENUP':-0.8}.get(a,0)
            if a=='RETRACE' and len([n for n in choices if n not in ('RETRACE','PENUP') and ds[n] not in self.painted])==0:
                special+=3.5
            return novelty -0.35*dist +special + rng.uniform(-1.25,1.25)
        return max(choices,key=utility)

def snapshots(seeds,steps=18):
    out=[]
    for seed in seeds:
        w=World.make(seed); r=random.Random(seed*101+17)
        for t in range(steps):
            label=w.teacher(r)
            out.append((w.prompt(), ACTIONS.index(label), seed, t))
            act=label
            if t%7==5:
                others=[a for a in w.legal() if a!='STOP']
                if others: act=r.choice(others)
            w.apply(act)
            if w.stopped: break
    return out
