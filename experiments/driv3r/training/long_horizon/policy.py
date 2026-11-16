"""Small real trainable action ranker; host legality never delegated to policy.

Not a d1/LFM/Sprocket adapter. An independently trained ~1k-parameter
NumPy MLP establishes a cheap, real optimizer/checkpoint/rollout pipeline.
"""
from __future__ import annotations

import math
import random
import numpy as np

from .world import MOVE, World

VERSION = "driv3r-tiny-action-ranker/1"
FEATURES = (
    "is_move", "is_backtrack", "is_relocate", "is_stop",
    "straight", "left", "right", "reverse",
    "free_ahead_3", "local_empty_8", "adjacent_terrain_8", "adjacent_ink_8",
    "wall_band_proximity", "forward_scent", "route_heading_dot", "motif_sin",
    "motif_cos", "elapsed_fraction", "recovery_fraction", "recently_recovered",
    "mode_wall", "mode_route", "mode_motif", "no_ink_moves_available",
)
ACTION_MAX = 7


def _neighbors(p):
    return [(p[0]+dx,p[1]+dy) for dx,dy in MOVE.values()]


def features(w: World, action: str, horizon: int) -> np.ndarray:
    if action not in w.menu():
        raise ValueError("Features requested for non-legal action")
    moved = action in MOVE
    dx,dy = MOVE.get(action, (0,0))
    next_pos = (w.cursor[0]+dx,w.cursor[1]+dy) if moved else w.cursor
    heading = w.heading
    f = np.zeros(len(FEATURES), dtype=np.float64)
    f[0:4] = [float(moved), float(action == "BACKTRACK"),
              float(action == "RELOCATE_START"), float(action == "STOP")]
    if moved:
        old = MOVE["MOVE_"+heading]
        dot = old[0]*dx + old[1]*dy
        cross = old[0]*dy - old[1]*dx
        f[4:8] = [float(dot==1),float(cross<0),float(cross>0),float(dot==-1)]
        f[8] = sum(not w.occupied((next_pos[0]+dx*k,next_pos[1]+dy*k))
                   for k in (1,2,3)) / 3.
        nearby = _neighbors(next_pos) + [(next_pos[0]+sx,next_pos[1]+sy)
                      for sx in (-1,1) for sy in (-1,1)]
        f[9] = sum(not w.occupied(p) for p in nearby)/8.
        f[10] = sum(p in w.materials or p in w.protected for p in nearby)/8.
        f[11] = sum(p in w.ink for p in nearby)/8.
        f[12] = float(f[10]>0)*.65 + float(f[10]==0)*.15
        # These are soft synthetic attractors, never permissions or targets.
        phase = (next_pos[0]-40)/8. + w.spec.seed % 13
        scent_y = 48 + 13 * math.sin(phase)
        f[13] = math.exp(-abs(next_pos[1]-scent_y)/8.)
        f[14] = dx  # route-only eastern preference, not an A* target oracle
    clock = w.steps / max(1,horizon)
    f[15:17] = [math.sin(w.steps*.21), math.cos(w.steps*.21)]
    f[17] = clock
    f[18] = w.recoveries / 12.
    f[19] = float(w.last_action in ("BACKTRACK","RELOCATE_START"))
    f[20:23] = [float(w.spec.mode==m) for m in ("wall","route","motif")]
    f[23] = float(not any(a in MOVE for a in w.menu()))
    return f


def menu_matrix(w: World, horizon: int) -> tuple[list[str],np.ndarray]:
    menu = w.menu()
    matrix = np.zeros((ACTION_MAX, len(FEATURES)), dtype=np.float64)
    if len(menu) > ACTION_MAX:
        raise ValueError("Unexpected host action menu size")
    for i,action in enumerate(menu):
        matrix[i] = features(w,action,horizon)
    return menu, matrix


def teacher_scores(w: World, menu: list[str], matrix: np.ndarray) -> np.ndarray:
    """Plural, imperfect deterministic teachers: contour, novelty, motif, recovery.

    Scores are preferences only. No hidden material edit or goal teleport.
    """
    scores = []
    can_ink = any(a in MOVE for a in menu)
    for i, action in enumerate(menu):
        f = matrix[i]
        if action in MOVE:
            score = 1.5 + .72*f[4] + .08*f[5] + .24*f[6] -.65*f[7]
            score += .72*f[8] + .46*f[9] -.68*f[11]
            score += (.95*f[12] if w.spec.mode == "wall" else 0)
            score += (1.5*f[13] if w.spec.mode == "wall" else 0)
            score += (1.2*f[14] if w.spec.mode == "route" else 0)
            score += ((1.6+.42*f[15])*f[5] + (1.45-.38*f[15])*f[6]
                      if w.spec.mode == "motif" else 0)
            if w.last_action in ("BACKTRACK","RELOCATE_START"):
                score += .5*f[8]
        elif action == "BACKTRACK":
            score = 3.7 if not can_ink else -2.8
        elif action == "RELOCATE_START":
            score = 2.0 if not can_ink else -3.5
        else:
            score = -5.0 if can_ink else (3.0 if len(menu)==1 else -1.0)
        scores.append(score)
    return np.asarray(scores,dtype=np.float64)


def teacher(w: World, horizon: int, rng: random.Random):
    legal, matrix = menu_matrix(w,horizon)
    scores = teacher_scores(w,legal,matrix)
    # Deliberately slightly erratic; still selects from the host's menu.
    disturbed = scores + np.asarray([rng.uniform(-.17,.17) for _ in legal])
    return legal[int(np.argmax(disturbed))], legal, matrix


def random_legal(w: World, rng: random.Random) -> str:
    return rng.choice(w.menu()) # includes STOP, no horizon-forcing trick


def random_legal_persistent(w: World, rng: random.Random) -> str:
    """Strong(er) random comparator: no gratuitous STOP while progression exists.

    No hindsight, scoring oracle, hidden destinations, suppressed host menus,
    or path teleport. The returned action is exactly host-legal; this baseline
    must not be mistaken for a native host run.
    """
    menu = w.menu()
    progress = [a for a in menu if a in MOVE]
    if progress:
        return rng.choice(progress)
    recovery = [a for a in menu if a in ("BACKTRACK", "RELOCATE_START")]
    return rng.choice(recovery) if recovery else "STOP"


class Ranker:
    def __init__(self, seed: int, hidden: int = 24):
        if not 4 <= hidden <= 128:
            raise ValueError("Invalid hidden width")
        self.seed, self.hidden = seed, hidden
        rng = np.random.default_rng(seed)
        self.params = {
            "w1": rng.normal(0,.09,(len(FEATURES),hidden)),
            "b1": np.zeros(hidden),
            "w2": rng.normal(0,.09,hidden),
            "b2": np.zeros(1),
        }
        self.updates = 0
        self._m = {k: np.zeros_like(v) for k,v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k,v in self.params.items()}

    @property
    def parameter_count(self):
        return sum(v.size for v in self.params.values())

    def logits(self, X: np.ndarray) -> np.ndarray:
        h = np.tanh(X @ self.params["w1"] + self.params["b1"])
        return h @ self.params["w2"] + self.params["b2"][0]

    def probs(self, X: np.ndarray, valid: int) -> np.ndarray:
        if valid < 1 or valid > ACTION_MAX:
            raise ValueError("No host-legal actions")
        z = self.logits(X[:valid]); z = z-z.max()
        p = np.exp(z)
        return p/p.sum()

    def choose(self, w: World, horizon: int) -> tuple[str,dict[str,float]]:
        legal, X = menu_matrix(w,horizon)
        p = self.probs(X,len(legal))
        # Deterministic argmax, but STOP was in the actual output softmax.
        selected = legal[int(p.argmax())]
        return selected, {a:round(float(p[i]),8) for i,a in enumerate(legal)}

    def optimize(self, X: np.ndarray, masks: np.ndarray, labels: np.ndarray,
                 steps: int, batch: int, lr: float, seed: int) -> dict:
        if not (X.ndim == 3 and X.shape[1:] == (ACTION_MAX,len(FEATURES))
                and masks.shape == X.shape[:2] and labels.shape == X.shape[:1]
                and np.all(masks[np.arange(len(labels)),labels])
                and len(labels) > 0):
            raise ValueError("Invalid source or teacher corpus")
        rng = np.random.default_rng(seed)
        losses = []
        p = self.params
        for _ in range(steps):
            indices = rng.integers(0,len(X),size=batch)
            xb,mb,yb = X[indices],masks[indices],labels[indices]
            h = np.tanh(xb @ p["w1"] + p["b1"])
            z = h @ p["w2"] + p["b2"][0]
            z = np.where(mb,z,-1e8)
            z = z-z.max(axis=1,keepdims=True)
            proba = np.exp(z)*mb
            proba /= proba.sum(axis=1,keepdims=True)
            losses.append(float(-np.log(np.maximum(1e-30,proba[np.arange(batch),yb])).mean()))
            delta = proba.copy()
            delta[np.arange(batch),yb] -= 1.
            delta /= batch
            gh = delta[:,:,None]*p["w2"][None,None,:]*(1.-h*h)
            grads = {"w1": np.einsum("baf,bah->fh",xb,gh),
                     "b1": gh.sum(axis=(0,1)),
                     "w2": np.einsum("bah,ba->h",h,delta),
                     "b2": np.asarray([delta.sum()])}
            norm = math.sqrt(sum(float(np.sum(g*g)) for g in grads.values()))
            scale = min(1.,5./max(norm,1e-12))
            self.updates += 1
            for name,g in grads.items():
                g *= scale
                self._m[name] = .9*self._m[name] + .1*g
                self._v[name] = .999*self._v[name] + .001*g*g
                mh = self._m[name]/(1.-.9**self.updates)
                vh = self._v[name]/(1.-.999**self.updates)
                p[name] -= lr*mh/(np.sqrt(vh)+1e-8)
        return {"updates": steps,"first_loss":round(losses[0],6),
                "last_20_mean_loss":round(float(np.mean(losses[-20:])),6),
                "min_loss":round(min(losses),6)}

    def checkpoint(self, path):
        np.savez_compressed(path, **self.params, updates=np.asarray(self.updates),
                            seed=np.asarray(self.seed), hidden=np.asarray(self.hidden))

    @classmethod
    def restore(cls,path):
        with np.load(path,allow_pickle=False) as data:
            policy = cls(int(data["seed"]),int(data["hidden"]))
            for k in policy.params:
                arr = data[k]
                if arr.shape != policy.params[k].shape or not np.isfinite(arr).all():
                    raise ValueError("Checkpoint shape or finiteness invalid")
                policy.params[k] = arr.copy()
            policy.updates = int(data["updates"])
        return policy
