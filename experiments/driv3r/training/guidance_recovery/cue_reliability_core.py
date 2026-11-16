"""C15: observable-menu cue-reliability adapter, frozen C12, replay-derived optimizer.

Model-only CPU research. No CyberSand runtime integration. World legal action
permissions stay authoritative. Requires immutable C14 source receipts and
seven source-pinned frozen predecessor checkpoints, external to Git history.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file
from experiment import sha
from topology_transfer import build
from adversarial_cues import source_hash
from signed_feedback import HostAdvice,observe
from personality_retention_core import load_c12

SCHEMA='guidance-recovery-cue-reliability/1'
KEYS=('base','c4','c7','c8','c9','c11','c12')
def evidence(x):
    """From the *legal menu only*, not hidden geometry or the true goal route.

    A legal BACKTRACK can make local goal progress yet still be the wrong
    long-horizon choice; these summaries are fallible soft evidence only.
    """
    legal = x[:, :8, 4] > 0.5
    neg = torch.full_like(x[:, :8, 7], -1000.0)
    fprogress = torch.where(legal, x[:, :8, 7], neg).max(dim=1).values.clamp(-1, 1)
    fblocked = torch.where(legal, x[:, :8, 22], neg).max(dim=1).values.clamp(0, 1)
    foptions = legal.sum(dim=1).to(x.dtype) / 8.0
    backprogress = x[:, 8, 7]
    return torch.stack((backprogress, fprogress, backprogress - fprogress, fblocked, foptions), dim=-1)

class ReliabilityGate(nn.Module):
    """Only modifies BACKTRACK under active, host-authorized correction.

    Bounded learned residual includes legible but imperfect local evidence. The
    score change is continuous rather than turning a cue into forced action.
    """

    def __init__(self, prior, width=16, scale=1.5):
        super().__init__()
        self.prior = prior
        for p in prior.parameters():
            p.requires_grad_(False)
        self.delta = nn.Sequential(nn.Linear(34, width), nn.Tanh(), nn.Linear(width, 1))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.scale = float(scale)

    def forward(self, x):
        if x.ndim == 2:
            return self.forward(x.unsqueeze(0)).squeeze(0)
        base = self.prior(x)
        extra = evidence(x)
        inp = torch.cat((x[:, 8, :], extra), dim=-1)
        residual = self.scale * torch.tanh(self.delta(inp).squeeze(-1))
        live = x[:, 8, 2] * x[:, 8, 4] * (x[:, 8, 26].abs() > 0.5).to(x.dtype) * x[:, 8, 27]
        out = base.clone()
        out[:, 8] = out[:, 8] + live * residual
        return out

def replay_data(receipts):
    X = []
    Y = []
    W = []
    T = []
    matching = 0
    for r in receipts:
        w = build(r['seed'], r['personality'], r['family'], r['rotation'], r['mirror'], r['outcome'])
        advice = HostAdvice(w)
        getattr(advice, 'issue_' + r['directive'])(w, strength=r['strength'], ttl=9)
        cs, x = observe(w, r['arm'], advice)
        assert source_hash(w) == r['world_sha256']
        assert sha(sorted(w.protected)) == r['protected_sha256']
        assert sha([(c.action, c.target, c.legal) for c in cs]) == r['menu_sha256']
        assert hashlib.sha256(x.tobytes()).hexdigest() == r['obs_sha256']
        a = r['advantage']
        X.append(x)
        Y.append(1 / (1 + np.exp(-1.5 * a)))
        W.append(0.15 + min(1, abs(a)))
        misleading = int(r['outcome'] == 'reverse' and r['directive'] == 'continue' or (r['outcome'] == 'ahead' and r['directive'] == 'backtrack'))
        T.append(misleading)
        matching += 1
    return (np.asarray(X, np.float32), np.asarray(Y, np.float32), np.asarray(W, np.float32), np.asarray(T, np.float32), matching)

def fit(model, prior, X, Y, W, T, epochs=28, seed=202615, lr=0.001, penalty=1.5):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    tx, ty, tw, tt = (torch.from_numpy(z) for z in (X, Y, W, T))
    with torch.no_grad():
        old = prior(tx).masked_fill(tx[:, :, 4] < 0.5, -10000000.0)
        om = old[:, 8] - torch.logsumexp(torch.cat((old[:, :8], old[:, 9:]), dim=1), dim=1)
        op = torch.sigmoid(om)
    opt = torch.optim.AdamW(model.delta.parameters(), lr=lr, weight_decay=0.012)
    rng = np.random.default_rng(seed)
    losses = []
    steps = 0
    for epoch in range(epochs):
        order = rng.permutation(len(tx))
        total = 0.0
        for st in range(0, len(tx), 64):
            idx = order[st:st + 64]
            x = tx[idx]
            new = model(x).masked_fill(x[:, :, 4] < 0.5, -10000000.0)
            margin = new[:, 8] - torch.logsumexp(torch.cat((new[:, :8], new[:, 9:]), dim=1), dim=1)
            prob = torch.sigmoid(margin)
            bce = nn.functional.binary_cross_entropy_with_logits(margin, ty[idx], reduction='none')
            overtrust = (prob - op[idx]).abs() ** 2 * tt[idx] * (tw[idx] < 1.0).to(x.dtype)
            drift = (prob - op[idx]).square()
            loss = (bce * tw[idx]).mean() + 0.15 * drift.mean() + penalty * overtrust.mean() + 0.004 * model.delta(torch.cat((x[:, 8, :], evidence(x)), dim=-1)).square().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.delta.parameters(), 1.5)
            opt.step()
            opt.zero_grad(set_to_none=True)
            total += float(loss.detach()) * len(idx)
            steps += 1
        losses.append(round(total / len(tx), 6))
    return (model.eval(), steps, losses)

def train_and_save(paths,receipts_path,out,epochs=28,penalties=(0.,2.)):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    paths=[Path(p) for p in paths]
    torch.set_num_threads(1)
    prior=load_c12(paths)
    receipts=json.loads(Path(receipts_path).read_text())
    X,Y,W,T,count=replay_data(receipts)
    predecessors={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in zip(KEYS,paths)}
    record={'schema':SCHEMA,'predecessors':predecessors,'training_menus':count,
       'source_receipts_sha256':hashlib.sha256(Path(receipts_path).read_bytes()).hexdigest(),
       'data_sha256':hashlib.sha256(X.tobytes()+Y.tobytes()+W.tobytes()+T.tobytes()).hexdigest(),
       'models':{}}
    for i,penalty in enumerate(penalties):
        torch.manual_seed(120015)
        model=ReliabilityGate(load_c12(paths))
        model,steps,losses=fit(model,prior,X,Y,W,T,epochs=epochs,seed=202615,penalty=penalty)
        label='C15-A' if i==0 else 'C15-B'
        path=out/f'cue_reliability_{label}.safetensors'
        save_file({k:v.detach().contiguous().cpu() for k,v in model.delta.state_dict().items()},
          str(path),metadata={'provenance':json.dumps({'schema':SCHEMA,'predecessors':predecessors,
           'penalty':penalty,'scale':model.scale,'model':label},sort_keys=True,separators=(',',':'))})
        record['models'][label]={'penalty':penalty,'updates':steps,'losses':losses,
          'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (out/'core_summary.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
    return record

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for key in KEYS:ap.add_argument('--'+key,required=True)
    ap.add_argument('--receipts',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--epochs',type=int,default=28)
    a=ap.parse_args()
    print(json.dumps(train_and_save([getattr(a,k) for k in KEYS],a.receipts,a.out,a.epochs),sort_keys=True))
