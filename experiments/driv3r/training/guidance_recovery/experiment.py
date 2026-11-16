"""CyberSand-like worm guidance/recovery: entirely external, synthetic, CPU-only.

No import from game/runtime. The only material authority is World.apply() after
host legality checks. This is not a d1/System One implementation.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random, time
from dataclasses import dataclass
from pathlib import Path
from safetensors.torch import save_file
from collections import Counter
import numpy as np
import torch
from torch import nn

SCHEMA = "guidance-recovery-synthetic/1"
N = 15
DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
ARMS = (0.0, 0.35, 1.0)
FAMILIES_TRAIN = ("bars", "islands", "clumps")
FAMILIES_FINAL = ("switchback", "forks")
FEATURE_COUNT = 25
ACTIONS = tuple((kind, *direction) for kind in ("INK", "LIFT") for direction in DIRECTIONS) + (("BACKTRACK", 0, 0), ("STOP", 0, 0))


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def blocked(seed: int, family: str):
    rng = random.Random(seed + 104729)
    cells = set()
    if family == "bars":
        for x in (4, 9):
            gaps = {rng.randrange(2, 13), rng.randrange(2, 13)}
            cells.update((x, y) for y in range(1, 14) if y not in gaps)
    elif family == "switchback":
        for x, gap in ((4, 11), (8, 3), (11, 11)):
            cells.update((x, y) for y in range(1, 14) if abs(y-gap) > 1)
    elif family == "forks":
        for y in (5, 9):
            gap = rng.randrange(3, 12)
            cells.update((x, y) for x in range(2, 13) if abs(x-gap) > 1)
        cells.update((7, y) for y in (2, 3, 11, 12))
    elif family == "clumps":
        for x, y in ((5, 4), (9, 9), (4, 10)):
            cells.update((i,j) for i in range(x-1,x+2) for j in range(y-1,y+2))
    elif family == "islands":
        for _ in range(12):
            x, y = rng.randrange(3, 13), rng.randrange(2, 13)
            cells.add((x,y))
    else:
        raise ValueError(family)
    return cells - {(1,1),(13,13),(13,1)}


@dataclass
class Candidate:
    action: tuple
    target: tuple
    legal: bool
    features: list[float]


class World:
    def __init__(self, seed: int, family: str, personality: int):
        if personality not in (-1,1): raise ValueError("personality")
        self.seed, self.family, self.personality = seed, family, personality
        self.protected = blocked(seed, family)
        self.pos, self.goal = (1,1), (13,13)
        self.inked = set()
        self.history = []
        self.last_dir = (1,0)
        self.steps, self.stuck = 0, 0
        self.previous_distance = self.distance()
        self.stopped = False
        self.events = []
        self.goals_seen = 1
        self.goal_change = False
        self.invalid_attempts = 0
        self.recovery_cue_ttl = 0

    def distance(self, p=None):
        p = self.pos if p is None else p
        return abs(p[0]-self.goal[0]) + abs(p[1]-self.goal[1])

    def material_at(self, p):
        if not (0 <= p[0] < N and 0 <= p[1] < N): return "OUTSIDE"
        if p in self.protected: return "FORBIDDEN_GLASS" if (p[0]+p[1])%3==0 else "SOLID_ROCK"
        if p in self.inked: return "INK"
        return "AIR"

    def allowed(self, p):
        return self.material_at(p) in ("AIR", "INK")

    def change_goal(self):
        self.goal = (13, 1)
        self.goal_change = True
        self.goals_seen += 1
        self.stuck = 0

    def issue_recovery_cue(self, turns=3):
        """A host suggestion only: legality and material permission stay fixed."""
        self.recovery_cue_ttl = turns

    def hint(self, p):
        # A deliberately misleading local maximum close to the initial route.
        # It is *soft information* and never changes legal permissions.
        d = abs(p[0]-self.goal[0]) + abs(p[1]-self.goal[1])
        attraction = max(0., 1. - d / 25.)
        trap = math.exp(-((p[0]-6)**2+(p[1]-4)**2)/12.)
        return min(1., max(0., attraction*.67 + trap*.33))

    def hint_u16(self, p):
        return int(round(self.hint(p)*65535))

    def candidates(self, guidance: float):
        result = []
        for action in ACTIONS:
            kind, dx, dy = action
            p = (self.pos[0]+dx, self.pos[1]+dy)
            if kind == "BACKTRACK": p = self.history[-1] if self.history else self.pos
            if kind == "STOP": p = self.pos
            legal = kind == "STOP" or (bool(self.history) if kind=="BACKTRACK" else self.allowed(p))
            if kind == "INK" and p in self.protected: legal = False
            progress = (self.distance()-self.distance(p))/N
            fresh = p not in self.inked and kind=="INK"
            turn = self.last_dir[0]*dy - self.last_dir[1]*dx
            moving = kind in ("INK", "LIFT")
            correction = 1.0 if kind=="BACKTRACK" else progress*3
            adjacent_blocked = sum(not self.allowed((p[0]+ex,p[1]+ey)) for ex,ey in DIRECTIONS)/4
            fv = [
                1.0 if kind=="INK" else 0.0, 1.0 if kind=="LIFT" else 0.0,
                1.0 if kind=="BACKTRACK" else 0.0, 1.0 if kind=="STOP" else 0.0,
                float(legal), dx, dy, progress, self.distance(p)/28.,
                float(fresh), float(p in self.inked),
                (self.hint_u16(p)/65535.)*guidance, guidance,
                float(turn > 0), float(turn < 0), float(turn==0 and moving),
                correction*(1. if self.recovery_cue_ttl else 0.), min(1., self.stuck/6.),
                float(self.goal_change), self.steps/24.,
                float(self.personality), float(turn*self.personality > 0),
                adjacent_blocked, float(self.recovery_cue_ttl > 0),
                float(self.distance(p)<=1),
            ]
            assert len(fv)==FEATURE_COUNT
            result.append(Candidate(action, p, legal, fv))
        return result

    def apply(self, candidate: Candidate):
        """Host-only executor: no illegal action or permission change is possible."""
        if self.stopped: return False
        # Recompute safety independently: do not trust caller's legal field.
        kind, dx, dy = candidate.action
        expected = (self.pos[0]+dx,self.pos[1]+dy)
        if kind in ("INK", "LIFT") and (dx,dy) not in DIRECTIONS:
            self.invalid_attempts+=1; return False
        if kind in ("BACKTRACK", "STOP") and (dx,dy)!=(0,0):
            self.invalid_attempts+=1; return False
        if kind == "BACKTRACK":
            if not self.history: self.invalid_attempts+=1; return False
            expected=self.history[-1]
        elif kind == "STOP": expected=self.pos
        elif kind not in ("INK", "LIFT") or not self.allowed(expected):
            self.invalid_attempts+=1; return False
        if expected != candidate.target:
            self.invalid_attempts+=1; return False
        oldpos, olddist = self.pos, self.distance()
        if kind=="STOP": self.stopped=True
        elif kind=="BACKTRACK": self.history.pop(); self.pos=expected
        else:
            self.history.append(self.pos);self.pos=expected
            if kind=="INK": self.inked.add(self.pos)
            self.last_dir=(dx,dy)
        self.steps+=1
        self.stuck = self.stuck+1 if self.distance()>=olddist else 0
        self.events.append((kind,oldpos,self.pos,self.goal))
        self.recovery_cue_ttl=max(0,self.recovery_cue_ttl-1)
        self.goal_change=False
        return True


def teacher_logits(world: World, candidates, guidance):
    vals = []
    for c in candidates:
        k,dx,dy = c.action
        if not c.legal:
            vals.append(-16.);continue
        progress = (world.distance()-world.distance(c.target))
        turn = world.last_dir[0]*dy-world.last_dir[1]*dx
        score = 1.1*progress + .35*(c.target not in world.inked)
        score += .55*world.personality*turn
        score += guidance*7.0*(world.hint(c.target)-world.hint(world.pos))
        if k=="INK": score += .42
        if k=="LIFT": score -= .75
        if k=="BACKTRACK": score = -2. + world.stuck*.72 + (2.2 if world.recovery_cue_ttl and world.stuck>=2 else .2 if world.recovery_cue_ttl else 0.)
        if k=="STOP": score = -4. + (2.5 if world.distance()<=1 else 0.) + (world.steps/24.)*1.3
        if c.target in world.inked and k=="INK": score -= 1.1
        if k=="INK" and c.features[22]>=.75: score -= .25
        vals.append(score)
    return np.asarray(vals,dtype=np.float32)


def select_teacher(w,c,arm,rng):
    logits = teacher_logits(w,c,arm)/.67
    prob = np.exp(logits-logits.max());prob/=prob.sum()
    idx = int(rng.choice(len(c),p=prob))
    return idx,prob


def make_data(episodes=420, horizon=24, seed=20261116):
    rng=np.random.default_rng(seed)
    X=[];Y=[];groups=[]
    for i in range(episodes):
        family=FAMILIES_TRAIN[i%len(FAMILIES_TRAIN)]
        base=2000+i*67
        personality=(-1 if i%2 else 1)
        for arm in ARMS:
            w=World(base,family,personality)
            for t in range(horizon):
                if t==16: w.change_goal()
                c=w.candidates(arm)
                # Randomised presentation order; action semantics are unchanged.
                order=rng.permutation(len(c));c=[c[j] for j in order]
                idx,prob=select_teacher(w,c,arm,rng)
                X.append([v.features for v in c]);Y.append(prob);groups.append((family,base,arm,t))
                if t in (8,9):
                    # Two real host-legal, material-preserving, potentially counterproductive turns.
                    bad=[(j,v) for j,v in enumerate(c) if v.legal and v.action[0]=="LIFT" and w.distance(v.target)>w.distance()]
                    if bad:idx=bad[0][0]
                if t==10 and i%2==0: w.issue_recovery_cue(3)
                w.apply(c[idx])
                if w.stopped:break
    return np.asarray(X,np.float32),np.asarray(Y,np.float32),groups


class Scorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(FEATURE_COUNT,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
    def forward(self,x):return self.layers(x).squeeze(-1)


def fit(X,Y,seed=73,epochs=9,lr=.0025):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model=Scorer()
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.002)
    XT=torch.from_numpy(X);YT=torch.from_numpy(Y)
    rng=np.random.default_rng(seed)
    losses=[];steps=0
    for e in range(epochs):
        idx=rng.permutation(len(X));running=0.;count=0
        for a in range(0,len(X),192):
            sl=idx[a:a+192];logits=model(XT[sl]); target=YT[sl]
            loss=-(target*torch.log_softmax(logits,dim=1)).sum(dim=1).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            running+=float(loss.detach())*len(sl);count+=len(sl);steps+=1
        losses.append(round(running/count,5))
    return model,losses,steps


def policy_index(model,w,c,arm,rng,mode):
    if mode=="teacher":
        i,_=select_teacher(w,c,arm,rng)
        return i,0
    if mode=="random":
        ids=[i for i,v in enumerate(c) if v.legal and v.action[0]!="STOP"]
        return (int(rng.choice(ids)) if ids else next(i for i,v in enumerate(c) if v.legal)),0
    with torch.no_grad():
        logits=model(torch.tensor([v.features for v in c],dtype=torch.float32)).numpy()
    raw_bad = int(not c[int(np.argmax(logits))].legal)
    # Host authority: mask illegal candidates even when the neural head is wrong.
    logits=np.where([v.legal for v in c],logits,-1000.)
    if mode=="greedy":return int(np.argmax(logits)),raw_bad
    probs=np.exp((logits-logits.max())/.72);probs/=probs.sum()
    return int(rng.choice(len(c),p=probs)),raw_bad


def evaluation(model, seeds=36, horizon=24, mode="student"):
    records=[]; samples=[]
    for family in FAMILIES_FINAL:
        for s in range(seeds):
            base=71000+s*719+FAMILIES_FINAL.index(family)*170003
            for personality in (-1,1):
                for arm in ARMS:
                    for cue in (False,True):
                        w=World(base,family,personality)
                        rng=np.random.default_rng(base+4103+personality*17)
                        turns=[]; recovery_before=None;recovery_steps=None;induced=0
                        raw_bad=0; first_actions=[];cue_backtracks=0
                        for t in range(horizon):
                            if t==16:w.change_goal()
                            if t==10 and cue:w.issue_recovery_cue(3)
                            c=w.candidates(arm)
                            idx,bad=policy_index(model,w,c,arm,rng,mode)
                            raw_bad+=bad
                            if t<3:first_actions.append(c[idx].action)
                            if t in (8,9):
                                options=[v for v in c if v.legal and v.action[0]=="LIFT" and w.distance(v.target)>w.distance()]
                                if options:
                                    if t==8: recovery_before=w.distance()
                                    idx=c.index(options[0]);induced+=1
                            choice=c[idx]
                            last_dir=w.last_dir
                            accepted=w.apply(choice)
                            if not accepted:raise AssertionError("host rejected policy selection")
                            if choice.action[0] in ("INK","LIFT"):
                                turn=last_dir[0]*choice.action[2]-last_dir[1]*choice.action[1]
                                turns.append(turn)
                            if 10<=t<14 and choice.action[0]=="BACKTRACK":cue_backtracks+=1
                            if recovery_before is not None and 10<=t<16 and recovery_steps is None and w.distance()<=recovery_before:
                                recovery_steps=t-9
                            if w.stopped:break
                        k=Counter(turns)
                        turn_pref=(k[1]-k[-1])/max(1,len(turns))
                        r={"family":family,"seed":base,"personality":personality,"arm":arm,"mode":mode,"cue":cue,
                           "ink_cells":len(w.inked),"steps":w.steps,"final_distance":w.distance(),
                           "invalid_native":w.invalid_attempts,"raw_illegal_top1":raw_bad,
                           "induced_detour_steps":induced,"recovered":int(recovery_steps is not None),
                           "recovery_steps":recovery_steps,"cue_backtracks":cue_backtracks,
                           "turn_preference":turn_pref,"turn_counts":dict(k),
                           "first_actions":[str(a) for a in first_actions],"geometry_sha":sha(sorted(w.protected)),
                           "world_material_sha":sha(sorted(w.inked))}
                        records.append(r)
                        if s<2:samples.append({**r,"trace":w.events})
    return records,samples


def aggregate(records):
    out={}
    for mode in sorted({r["mode"] for r in records}):
        for arm in ARMS:
            for cue in (False,True):
                arr=[r for r in records if r["mode"]==mode and r["arm"]==arm and r["cue"]==cue]
                corr=[r for r in arr if r["induced_detour_steps"]>0]
                recovery=[r["recovery_steps"] for r in corr if r["recovered"]]
                persona={p:np.mean([r["turn_preference"] for r in arr if r["personality"]==p]) for p in (-1,1)}
                out[f"{mode}/{arm:.2f}/cue={int(cue)}"]={
                    "n_episodes":len(arr),"n_induced":len(corr),
                    "mean_ink_cells":round(float(np.mean([r["ink_cells"] for r in arr])),3),
                    "mean_final_goal_distance":round(float(np.mean([r["final_distance"] for r in arr])),3),
                    "recovery_within_six_decisions":round(sum(r["recovered"] for r in corr)/max(1,len(corr)),4),
                    "mean_recovery_steps_if_recovered":round(float(np.mean(recovery)),3) if recovery else None,
                    "backtracks_first_four_after_cue":round(float(np.mean([r["cue_backtracks"] for r in arr])),3),
                    "raw_illegal_top1_per_episode":round(sum(r["raw_illegal_top1"] for r in arr)/len(arr),3),
                    "native_illegal_edits":sum(r["invalid_native"] for r in arr),
                    "personality_signed_turn_gap":round(float(persona[1]-persona[-1]),4),
                }
    # First matched-state choice is not conflated by later world divergence.
    for mode in sorted({r["mode"] for r in records}):
        pairs={}
        for r in records:
            if r["mode"]==mode and r["cue"] is False:
                key=(r["family"],r["seed"],r["personality"])
                pairs.setdefault(key,{})[r["arm"]]=r["first_actions"][0] if r["first_actions"] else None
        pairs=[x for x in pairs.values() if len(x)==3]
        out[f"{mode}/same_state_first_choice"]={
            "n_matched":len(pairs),
            "weak_diff_vs_none":round(sum(p[0.35]!=p[0.0] for p in pairs)/max(1,len(pairs)),4),
            "strong_diff_vs_none":round(sum(p[1.0]!=p[0.0] for p in pairs)/max(1,len(pairs)),4),
        }
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--episodes",type=int,default=420);ap.add_argument("--eval-seeds",type=int,default=36)
    ap.add_argument("--epochs",type=int,default=9);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    t=time.perf_counter();X,Y,groups=make_data(episodes=args.episodes)
    model,loss,steps=fit(X,Y,epochs=args.epochs)
    ck=args.out/"guidance_recovery_head.safetensors"
    save_file({k:v.cpu().contiguous() for k,v in model.state_dict().items()},str(ck),metadata={"schema":SCHEMA,"training_seed":"73","features":str(FEATURE_COUNT)})
    ck_sha=hashlib.sha256(ck.read_bytes()).hexdigest()
    allrecords=[];gallery=[]
    for mode in ("student","greedy","teacher","random"):
        rec,samples=evaluation(model,seeds=args.eval_seeds,mode=mode)
        allrecords+=rec
        if mode in ("student","teacher"):gallery+=samples
    metrics={"schema":SCHEMA,"source_scope":"experiments/driv3r/training/guidance_recovery/**",
     "benchmark":"Synthetic external game-like host; NOT CyberSand native or d1", "train_family":FAMILIES_TRAIN,"final_family":FAMILIES_FINAL,
     "train_episodes":args.episodes*len(ARMS),"training_samples":len(X),"optimizer_steps":steps,
     "loss_by_epoch":loss,"checkpoint_sha256":ck_sha,"checkpoint_bytes":ck.stat().st_size,
     "dataset_digest":sha(groups),"training_feature_sha256":hashlib.sha256(X.tobytes()).hexdigest(),
     "teacher_target_sha256":hashlib.sha256(Y.tobytes()).hexdigest(),"training_seed":73,"dataset_seed":20261116,
     "eval_seeds_per_family_personality":args.eval_seeds,"summary":aggregate(allrecords),
     "record_sha256":sha(allrecords),"gallery_sha256":sha(gallery),"runtime_seconds":round(time.perf_counter()-t,2),
     "torch_version":torch.__version__,"numpy_version":np.__version__,
     "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out/"metrics.json").write_text(json.dumps(metrics,indent=2,sort_keys=True)+"\n")
    (args.out/"trajectories.json").write_text(json.dumps(gallery,indent=2,sort_keys=True)+"\n")
    print(json.dumps(metrics,indent=2))


if __name__=="__main__":main()
