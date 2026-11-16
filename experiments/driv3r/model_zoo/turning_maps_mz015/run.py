"""MZ-015: direction-balanced, turning-path 2-D paint-policy training.

Original synthetic research host. No CyberSand runtime imports or game changes.
Uses MZ-013 architecture definitions to isolate curriculum/topology changes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import resource
import statistics
import time
from pathlib import Path

from experiments.driv3r.model_zoo.spatial_paint_mz013.run import (
    ACTIONS, LEFT, RIGHT, UP, DOWN, PAINT, RETRACE, WAIT, STOP, D, models
)

SIDE = 33
LIMIT = 96
DIRECTIONS = {LEFT: (-1, 0), RIGHT: (1, 0), UP: (0, -1), DOWN: (0, 1)}
REVERSE = {v: k for k, v in DIRECTIONS.items()}
SCHEMA = "MZ015-TURNING-MAPS-v1"
PARENT_SHA = "650bad65c76c569bb5fb33c784e72a9c16d654d6"


def transform(point, rotation, mirror):
    x, y = point
    if mirror:
        x = SIDE - 1 - x
    for _ in range(rotation):
        x, y = SIDE - 1 - y, x
    return x, y


class World:
    """A bent, one-cell-wide traversable 2-D path with three mineral forks.

    Route/obstacle morphology changes across independent seeds. The beacon
    and inversion vanish before the material forks. A wrong material entry
    requires RETRACE; correct entry requires PAINT then RETRACE. Visited path
    cells are visibly marked so a controller can infer the forward direction.
    The legal menu is host-owned, not an inferred model privilege.
    """
    def __init__(self, seed, cue, invert, *, ood=False, rotation=None, mirror=None):
        rng = random.Random(seed * 961748941 + (777 if ood else 103))
        lengths = ((rng.randint(4, 6), rng.randint(6, 8),
                    rng.randint(5, 7), rng.randint(3, 4),
                    rng.randint(5, 7)) if not ood else
                   (rng.randint(5, 7), rng.randint(7, 9),
                    rng.randint(4, 6), rng.randint(3, 5),
                    rng.randint(4, 6)))
        x, y = (3, 5)
        route = [(x, y)]
        segment_centres = []
        for step, (dx, dy) in zip(lengths, ((1, 0), (0, 1),
                                              (1, 0), (0, -1), (1, 0))):
            first = len(route)-1
            for _ in range(step):
                x, y = x + dx, y + dy
                route.append((x, y))
            segment_centres.append(first + step // 2)
        assert len(set(route)) == len(route), "route must not self-intersect"
        r = seed % 4 if rotation is None else rotation
        m = bool((seed // 4) % 2) if mirror is None else bool(mirror)
        route = [transform(p, r, m) for p in route]
        self.route = route
        self.route_index = {p: i for i, p in enumerate(route)}
        self.stations = [route[segment_centres[i]] for i in (0, 2, 4)]
        self.grid = [[1] * SIDE for _ in range(SIDE)]
        for xx, yy in route:
            self.grid[yy][xx] = 0
        self.station_sides = {}
        used_minerals = set()
        for station, i in zip(self.stations, (0, 2, 4)):
            before, after = route[segment_centres[i]-1], route[segment_centres[i]+1]
            delta = (after[0]-station[0], after[1]-station[1])
            options = ((-delta[1], delta[0]), (delta[1], -delta[0]))
            assert delta in DIRECTIONS.values()
            safe_is_first = rng.randrange(2)
            mineral_locations = {}
            for j, (dx, dy) in enumerate(options):
                px, py = station[0]+dx, station[1]+dy
                assert 0 <= px < SIDE and 0 <= py < SIDE
                assert self.grid[py][px] == 1 and (px, py) not in used_minerals
                used_minerals.add((px, py))
                mineral = 2 + (safe_is_first if j == 0 else 1-safe_is_first)
                self.grid[py][px] = mineral
                mineral_locations[mineral] = (px, py)
            self.station_sides[station] = mineral_locations
        self.cue, self.invert = cue, invert
        self.safe = cue ^ invert
        self.x, self.y = route[0]
        self.grid[self.y][self.x] = 4
        self.painted = set()
        self.tick = 0
        self.last_action = WAIT
        self.path_stack = []
        self.collisions = 0
        self.hazard = False
        self.done = False
        self.failed = False
        self.trace = [(self.x, self.y)]
        self.rotation, self.mirror = r, m
        self.decor = [rng.random() for _ in range(4)]

    def tile(self, x, y):
        if 0 <= x < SIDE and 0 <= y < SIDE:
            return self.grid[y][x]
        return 1

    def legal(self):
        mask = [0] * len(ACTIONS)
        mask[STOP] = 1
        if self.done:
            return mask
        if self.tick < 4:
            mask[WAIT] = 1
        elif self.hazard or (self.x, self.y) not in self.route_index:
            mask[RETRACE] = int(bool(self.path_stack))
        else:
            if self.path_stack:
                mask[RETRACE] = 1
            for a, (dx, dy) in DIRECTIONS.items():
                nx, ny = self.x+dx, self.y+dy
                tile = self.tile(nx, ny)
                if tile in (0, 4) and (nx, ny) in self.route_index:
                    mask[a] = 1
                elif (self.x, self.y) in self.station_sides and tile in (2, 3):
                    mask[a] = 1
        if self.tick >= 4 and not self.hazard and (self.x, self.y) not in self.route_index:
            mask[PAINT] = int(self.tile(self.x, self.y) in (2, 3))
        return mask

    def observe(self):
        v = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                tile = self.tile(self.x+dx, self.y+dy)
                v.extend(float(j == tile) for j in range(5))
        v.extend((self.x/(SIDE-1), self.y/(SIDE-1), len(self.painted)/3,
                  self.tick/LIMIT, float(self.hazard), float(self.tick < 4)))
        v.extend(self.legal())
        v.extend(float(j == self.last_action) for j in range(len(ACTIONS)))
        v.extend((float(self.tick < 2 and self.cue == 0)*2,
                  float(self.tick < 2 and self.cue == 1)*2,
                  float(2 <= self.tick < 4 and self.invert == 0)*2,
                  float(2 <= self.tick < 4 and self.invert == 1)*2))
        v.extend(self.decor)
        assert len(v) == D
        return v

    def teacher(self):
        if self.tick < 4:
            return WAIT
        if self.hazard:
            return RETRACE
        pos = (self.x, self.y)
        if pos not in self.route_index:
            return PAINT if self.tile(*pos) == 2+self.safe else RETRACE
        if len(self.painted) == 3:
            return STOP
        if pos in self.station_sides and pos not in self.painted:
            target = self.station_sides[pos][2+self.safe]
        else:
            idx = self.route_index[pos]
            assert idx+1 < len(self.route), "ran out of route before painting"
            target = self.route[idx+1]
        return REVERSE[(target[0]-self.x, target[1]-self.y)]

    def step(self, action):
        if self.done:
            raise RuntimeError("step after terminal")
        if not self.legal()[action]:
            raise ValueError("illegal host action "+ACTIONS[action])
        self.last_action = action
        self.tick += 1
        if action == STOP:
            self.done = True
            self.failed = len(self.painted) != 3 or (self.x, self.y) not in self.route_index
        elif action == RETRACE:
            self.x, self.y = self.path_stack.pop()
            self.hazard = False
        elif action in DIRECTIONS:
            self.path_stack.append((self.x, self.y))
            dx, dy = DIRECTIONS[action]
            self.x, self.y = self.x+dx, self.y+dy
            tile = self.tile(self.x, self.y)
            if tile in (2, 3) and tile != 2+self.safe:
                self.collisions += 1
                self.hazard = True
            if (self.x, self.y) in self.route_index:
                self.grid[self.y][self.x] = 4
        elif action == PAINT:
            station = self.path_stack[-1] if self.path_stack else None
            if self.tile(self.x, self.y) == 2+self.safe and station in self.station_sides:
                self.grid[self.y][self.x] = 4
                self.painted.add(station)
            else:
                self.collisions += 1
                self.hazard = True
        self.trace.append((self.x, self.y))
        return self.done


def episode(seed, cue, inv, bad=-1, ood=False):
    w = World(seed, cue, inv, ood=ood)
    xs, ys = [], []
    injected = False
    for _ in range(LIMIT):
        xs.append(w.observe())
        a = w.teacher()
        ys.append(a)
        at = (w.x, w.y)
        if bad >= 0 and not injected and at == w.stations[bad] and a in DIRECTIONS:
            intended = (w.x+DIRECTIONS[a][0], w.y+DIRECTIONS[a][1])
            if intended not in w.route_index:
                other = [k for k in DIRECTIONS if k != a and
                         w.legal()[k] and (w.x+DIRECTIONS[k][0],
                                           w.y+DIRECTIONS[k][1]) not in w.route_index]
                executed = other[0] if other else a
                injected = bool(other)
            else:
                executed = a
        else:
            executed = a
        w.step(executed)
        if w.done:
            break
    assert w.done and not w.failed and len(w.painted) == 3, (
        seed, cue, inv, bad, w.tick, len(w.painted))
    return xs, ys


def corpus(seeds, ood=False):
    return [episode(seed, cue, inv, bad, ood)
            for seed in seeds for cue in (0, 1) for inv in (0, 1)
            for bad in (-1, 1)]


def arrays(torch, records):
    x = torch.zeros((len(records), LIMIT, D))
    y = torch.full((len(records), LIMIT), -100, dtype=torch.long)
    for i, (observations, actions) in enumerate(records):
        assert len(actions) <= LIMIT
        x[i, :len(actions)] = torch.tensor(observations)
        y[i, :len(actions)] = torch.tensor(actions)
    return x, y


def score(torch, net, x, y, batch=16):
    correct = n = branches = fork_ok = mineral_forks = mineral_ok = paint = paint_ok = illegal = 0
    with torch.inference_mode():
        for j in range(0, len(y), batch):
            yy = y[j:j+batch]
            t = int((yy != -100).sum(1).max())
            xx = x[j:j+batch, :t]
            labels = yy[:, :t]
            pred = net(xx).argmax(-1)
            valid = labels != -100
            forks = (labels == UP) | (labels == DOWN) | (labels == LEFT) | (labels == RIGHT)
            # A true mineral fork is a movement into an adjacent material-2/3 cell,
            # rather than the ordinary route-following movement majority class.
            mineral = torch.zeros_like(valid)
            for a, (dx, dy) in DIRECTIONS.items():
                base = ((2+dy)*5 + (2+dx))*5
                mineral |= (labels == a) & ((xx[:, :, base+2] + xx[:, :, base+3]) > .5)
            paints = labels == PAINT
            correct += int(((pred == labels)&valid).sum())
            n += int(valid.sum())
            branches += int(forks.sum())
            fork_ok += int(((pred == labels)&forks).sum())
            mineral_forks += int(mineral.sum())
            mineral_ok += int(((pred == labels)&mineral).sum())
            paint += int(paints.sum())
            paint_ok += int(((pred == labels)&paints).sum())
            mask = xx[:, :, 131:139]
            illegal += int(((mask.gather(-1, pred.unsqueeze(-1)).squeeze(-1)<.5)&valid).sum())
    return dict(accuracy=round(correct/max(1,n),5),
                move_accuracy=round(fork_ok/max(1,branches),5),
                mineral_fork_accuracy=round(mineral_ok/max(1,mineral_forks),5),
                mineral_fork_count=mineral_forks,
                paint_accuracy=round(paint_ok/max(1,paint),5),
                illegal_raw=illegal, decisions=n)


def play(torch, net, seed, cue, inv, *, ood=False, recent=None):
    w = World(seed, cue, inv, ood=ood)
    state, history, raw_illegal = None, [], 0
    with torch.inference_mode():
        for _ in range(LIMIT):
            obs = w.observe()
            x = torch.tensor([obs])
            if recent is None and hasattr(net, "step"):
                logits, state = net.step(x, state)
                logits = logits[0]
            else:
                history.append(obs)
                observations = history if recent is None else history[-recent:]
                logits = net(torch.tensor([observations]))[0, -1]
            mask = w.legal()
            raw_illegal += int(not mask[int(logits.argmax())])
            logits = logits.masked_fill(torch.tensor([not m for m in mask]), -1e9)
            action = int(logits.argmax())
            w.step(action)
            if w.done:
                break
    success = w.done and not w.failed and len(w.painted) == 3
    return dict(success=success, clean=success and w.collisions == 0,
                collisions=w.collisions, paints=len(w.painted),
                raw_illegal=raw_illegal, steps=w.tick, trace=w.trace,
                stations=w.stations, rotation=w.rotation, mirror=w.mirror,
                cue=w.cue, invert=w.invert, safe=w.safe)


def rollout(torch, net, seeds, ood=False, recent=None):
    entries = [play(torch, net, s, c, inv, ood=ood, recent=recent)
               for s in seeds for c in (0, 1) for inv in (0, 1)]
    n = len(entries)
    per_orientation = {}
    for r in range(4):
        portion = [v for v in entries if v["rotation"] == r]
        per_orientation[str(r)] = dict(n=len(portion), clean=sum(v["clean"] for v in portion))
    mirror_scores = {}
    for m in (False,True):
        subset = [e for e in entries if e["mirror"] == m]
        mirror_scores[str(m)] = dict(n=len(subset), clean=sum(e["clean"] for e in subset),
                                     collisions=sum(e["collisions"] for e in subset))
    cue_scores = {}
    for c in (0,1):
        for inv in (0,1):
            subset = [e for e in entries if (e["cue"],e["invert"]) == (c,inv)]
            cue_scores[f"{c}-{inv}"] = dict(n=len(subset), clean=sum(e["clean"] for e in subset),
                                           collisions=sum(e["collisions"] for e in subset))
    return dict(episodes=n, clean=sum(e["clean"] for e in entries),
                success=sum(e["success"] for e in entries),
                collisions=sum(e["collisions"] for e in entries),
                paints=sum(e["paints"] for e in entries),
                raw_illegal=sum(e["raw_illegal"] for e in entries),
                by_rotation=per_orientation, by_mirror=mirror_scores,
                by_memory_rule=cue_scores, examples=entries[:4])


def latency(torch, net, steps=48):
    x = torch.zeros((1, D))
    samples = []
    with torch.inference_mode():
        for trial in range(30):
            h = None
            t0 = time.perf_counter()
            for _ in range(steps):
                if hasattr(net, "step"):
                    _, h = net.step(x, h)
                else:
                    net(x)
            if trial >= 8:
                samples.append((time.perf_counter()-t0)*1000/steps)
    return {"p50_ms_per_step": round(statistics.median(samples), 5),
            "p95_ms_per_step": round(sorted(samples)[-2], 5),
            "mode": "cached_state" if hasattr(net, "step") else "current_only",
            "batch": 1, "steps_per_trial": steps}


def gallery(results):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="720" viewBox="0 0 1040 720">',
             '<rect width="1040" height="720" fill="#101827"/>',
             '<text x="15" y="28" fill="white" font-size="17">MZ-015 | closed-loop 2-D turning-map paint policies</text>']
    for row, (name, result) in enumerate(results.items()):
        for col, ep in enumerate(result["examples"]):
            sx = 115+230*col
            sy = 54+215*row
            if col == 0:
                parts.append(f'<text x="8" y="{sy+17}" fill="#fff" font-size="12">{name}</text>')
            parts.append(f'<rect x="{sx}" y="{sy}" width="209" height="194" fill="#202f44" rx="5"/>')
            for x, y in ep["stations"]:
                parts.append(f'<circle cx="{sx+5+x*5.8:.2f}" cy="{sy+5+y*5.5:.2f}" r="4" fill="#fbbf24"/>')
            poly = " ".join(f'{sx+5+x*5.8:.2f},{sy+5+y*5.5:.2f}' for x, y in ep["trace"])
            parts.append(f'<polyline points="{poly}" fill="none" stroke="#a7e9ff" stroke-width="1.5"/>')
            parts.append(f'<text x="{sx+5}" y="{sy+188}" font-size="10" fill="#fff">paint {ep["paints"]}/3, collisions {ep["collisions"]}, clean {int(ep["clean"])}</text>')
    return "\n".join(parts+['</svg>'])+"\n"


def run(args):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(1515)
    sampler = torch.Generator().manual_seed(1915)
    train = corpus(range(1000, 1000+args.train_seeds))
    test = corpus(range(5000, 5000+args.test_seeds))
    shifted = corpus(range(7000, 7000+args.test_seeds), ood=True)
    x, y = arrays(torch, train)
    xt, yt = arrays(torch, test)
    xo, yo = arrays(torch, shifted)
    all_nets = models(torch)
    del all_nets["transformer20m"]
    nets = all_nets
    nparams = {k:sum(p.numel() for p in net.parameters()) for k,net in nets.items()}
    optim = {k:torch.optim.AdamW(v.parameters(), lr=args.lr) for k,v in nets.items()}
    weights = torch.tensor([1.,1.,2.,2.,3.,2.,.5,2.])
    losses = {k:[] for k in nets}
    for update in range(args.updates):
        ids = torch.randint(len(x), (args.batch,), generator=sampler)
        label = y[ids]
        t = int((label != -100).sum(1).max())
        inputs = x[ids, :t]
        for name, net in nets.items():
            net.train()
            optim[name].zero_grad(set_to_none=True)
            pred = net(inputs)
            loss = torch.nn.functional.cross_entropy(
                pred.transpose(1,2), label[:, :t],
                weight=weights, ignore_index=-100)
            assert bool(torch.isfinite(loss)), (name, update)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            optim[name].step()
            if update == 0 or (update+1)%25 == 0:
                losses[name].append((update+1, round(float(loss.detach()),6)))
        if (update+1)%50 == 0:
            print(f"MZ-015 updates {update+1}/{args.updates}", flush=True)
    for net in nets.values():
        net.eval()
    scores = {name:dict(test=score(torch, net, xt, yt),
                        shifted=score(torch, net, xo, yo))
              for name,net in nets.items()}
    seeds = range(9000,9000+args.rollout_seeds)
    roll = {name:rollout(torch, net, seeds, ood=True) for name,net in nets.items()}
    ablation = {"diagonal_recent_4":rollout(torch,nets["diagonal"],seeds,ood=True,recent=4)}
    parity = {}
    for name in ("diagonal","gru"):
        matched=0
        for s in range(9000,9002):
            for c in (0,1):
                for inv in (0,1):
                    a=play(torch,nets[name],s,c,inv,ood=True)
                    b=play(torch,nets[name],s,c,inv,ood=True,recent=LIMIT)
                    matched += a==b
        parity[name]={"matches":matched,"episodes":8}
    times={k:latency(torch,v) for k,v in nets.items()}
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    source=os.environ.get("GITHUB_SHA","LOCAL-UNVERIFIED")
    ckpt=out/"models.pt"
    torch.save(dict(schema=SCHEMA,source_sha=source,parent_sha=PARENT_SHA,
                    models={k:net.state_dict() for k,net in nets.items()},
                    actions=ACTIONS),ckpt)
    report=dict(schema=SCHEMA,source_sha=source,parent_sha=PARENT_SHA,
                parameters=nparams,updates_per_model=args.updates,batch=args.batch,
                train_seed_families=args.train_seeds,heldout_seed_families=args.test_seeds,
                disjoint_training_testing=True,training_trajectories=len(train),
                scores=scores,closed_loop=roll,ablation=ablation,
                cached_full_parity=parity,cpu_latency=times,training_losses=losses,
                checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                peak_process_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2),
                limits=["Synthetic turning corridors with 3 stations, not real CyberSand",
                        "Shared grid-generator topology despite rotated and mirrored samples",
                        "Unseen lengths and seed families are not novel map classes",
                        "Equal updates and minibatches, not matched FLOPs",
                        "Raw illegal logits reported separately from host-masked action",
                        "No game or provider integration; research Draft/HOLD"])
    (out/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    (out/"trajectories.svg").write_text(gallery(roll))
    (out/"REPORT.md").write_text("# MZ-015 turning-map research\n\nExact source: "+source+
        "\n\n"+ "\n".join(f"- {k}: OOD clean {v['clean']}/{v['episodes']}, paints {v['paints']}, collisions {v['collisions']}" for k,v in roll.items())+
        "\n\nDraft/HOLD. Synthetic only. No game modifications.\n")
    print(json.dumps(dict(source=source,parameters=nparams,
       scores=scores,closed_loop={k:{t:v[t] for t in ("episodes","clean","collisions","paints")} for k,v in roll.items()},
       parity=parity,timings=times,checkpoint_sha256=report["checkpoint_sha256"]),indent=2),flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="mz015-evidence")
    p.add_argument("--train-seeds",type=int,default=128)
    p.add_argument("--test-seeds",type=int,default=16)
    p.add_argument("--rollout-seeds",type=int,default=8)
    p.add_argument("--updates",type=int,default=1800)
    p.add_argument("--batch",type=int,default=16)
    p.add_argument("--lr",type=float,default=.002)
    run(p.parse_args())
