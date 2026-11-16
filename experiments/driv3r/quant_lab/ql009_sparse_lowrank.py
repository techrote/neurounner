#!/usr/bin/env python3
"""QL-009: sparse low-rank post-quant repair of a Q1/Q4-middle game worm.

The existing Q1 packed XNOR and Q4 island weight codes are immutable. Train
only a tiny rank-two residual on CyberSand-like host-legal action observations.
External W0-style synthetic evaluation only; no native game changes or IQ GGUF
claim. Preserve failed/boring personalities and first-pass trajectory bytes.
"""
import argparse
import json
import os
import resource
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_worm import ACTIONS, FAMILIES, WormWorld, digest, rollout, timing
from ql002_branch_retrace import examples, assess
from ql003_surgery import morphology
from ql005_bitserial import verified_source
from ql006_activation_gates import GatedQ1
from ql007_mixed_bridges import Hybrid, PARENT_SHA


class SparseRankTwo(nn.Module):
    """Live frozen low-bit parent + trainable sparse feature-to-logit rank-2 patch.

    A zero-initialized output projection ensures the initial model is EXACTLY
    the original frozen low-bit policy. The mask is derived only from training
    feature variance, never from held-out observations. Dense controls use all
    27 dimensions. No parent parameter is trainable.
    """
    def __init__(self, parent, feature_mask, seed=90909, train_bias=True):
        super().__init__()
        self.parent = parent
        mask = torch.as_tensor(feature_mask, dtype=torch.float32)
        assert tuple(mask.shape) == (27,) and int(mask.sum()) > 0
        self.register_buffer("feature_mask", mask)
        torch.manual_seed(seed)
        self.u = nn.Parameter(torch.randn(27, 2) * 0.05)
        self.v = nn.Parameter(torch.zeros(2, len(ACTIONS)))
        self.bias = nn.Parameter(torch.zeros(len(ACTIONS)), requires_grad=train_bias)

    def correction(self, x):
        return torch.tanh((x * self.feature_mask) @ self.u) @ self.v + self.bias

    def forward(self, x):
        with torch.no_grad():
            base = self.parent(x)
        return base + self.correction(x)

    def cached_logits(self, x, frozen_base):
        return frozen_base + self.correction(x)

    def memory_bytes(self):
        return sum(int(t.nelement() * t.element_size())
                   for t in (self.u, self.v, self.bias, self.feature_mask))


def fit(parent, dataset, mask, prior_scores, steps, kl_weight, seed, train_bias=True):
    x, y, legal = dataset
    torch.manual_seed(seed)
    net = SparseRankTwo(parent, mask, seed, train_bias=train_bias)
    optim = torch.optim.AdamW([p for p in (net.u, net.v, net.bias) if p.requires_grad], lr=0.025, weight_decay=0.)
    counts = torch.bincount(y, minlength=len(ACTIONS)).float()
    assert bool((counts > 0).all()), counts
    probs = (1.0 / counts[y]).double()
    probs = probs / probs.sum()
    trace = []
    for tick in range(steps):
        ix = torch.multinomial(probs, 128, replacement=True)
        live = net.cached_logits(x[ix], prior_scores[ix]).masked_fill(~legal[ix], -1e9)
        original = prior_scores[ix].masked_fill(~legal[ix], -1e9)
        ce = F.cross_entropy(live, y[ix])
        kl = F.kl_div(F.log_softmax(live, dim=1),
                      F.softmax(original, dim=1), reduction="batchmean")
        loss = ce + kl_weight * kl
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        trace.append({"step":tick+1, "loss":float(loss.detach()),
                      "ce":float(ce.detach()), "kl":float(kl.detach())})
    moments = sum(int(torch.count_nonzero(s["exp_avg"]).item() > 0)
                  for s in optim.state.values())
    assert moments == (3 if train_bias else 2), moments
    patch_norm = float((net.u @ net.v).norm().detach())
    assert patch_norm > 0, "Null model patch: optimizer did not update"
    diag = {"steps":steps, "seed":seed, "kl_weight":kl_weight,
            "adam_nonzero_moment_tensors":moments, "rank":2,
            "feature_dimensions_enabled":int(net.feature_mask.sum()),
            "actual_trainable_parameters":sum(p.numel() for p in (net.u,net.v,net.bias) if p.requires_grad),
            "frozen_output_bias":not train_bias,
            "patch_frobenius_norm":patch_norm,
            "first":trace[0], "last":trace[-1],
            "parent_q1_q4_codes_unchanged":True}
    return net, optim.state_dict(), trace, diag


def heldout(model, data, reference):
    x, y, legal = data
    with torch.no_grad():
        raw = model(x).argmax(1)
        logits = model(x).masked_fill(~legal, -1e9)
        chosen = logits.argmax(1)
        base = reference(x).masked_fill(~legal, -1e9).argmax(1)
        if isinstance(model, SparseRankTwo):
            # Parent remains LIVE while only the learned correction is frozen.
            corrected_frozen = reference(x) + model.correction(x[:1]).expand(len(x), -1)
            frozen_choice = corrected_frozen.masked_fill(~legal, -1e9).argmax(1)
            adapter_response = int((chosen != frozen_choice).sum())
        else:
            adapter_response = None
    return {"n":len(y), "host_legal_masked":int(legal.gather(1, chosen[:,None]).sum()),
            "unmasked_top_legal":int(legal.gather(1, raw[:,None]).sum()),
            "teacher_agreement_n":int((chosen == y).sum()),
            "disagreement_vs_frozen_parent_n":int((chosen != base).sum()),
            "live_adapter_vs_constant_adapter_differences_n":adapter_response,
            "label_denominators":{a:int((y == i).sum()) for i,a in enumerate(ACTIONS)},
            "masked_recall":{a:round(float((chosen[y == i] == i).float().mean()), 6)
                             for i,a in enumerate(ACTIONS)}}


def frozen_prior(model, seed, family):
    w = WormWorld(seed, family)
    with torch.no_grad():
        scores = model(torch.from_numpy(w.features(0)).unsqueeze(0))
    class Frozen:
        def __call__(self, x):
            return scores.expand(len(x), -1)
    return Frozen()


def self_test():
    class Fake:
        def __call__(self, x):
            return torch.arange(len(ACTIONS), dtype=torch.float32).expand(len(x), -1)
    x = torch.randn(16,27)
    mask = torch.ones(27)
    m = SparseRankTwo(Fake(), mask)
    assert torch.equal(m(x), Fake()(x)), "Non-zero untrained residual"
    loss = (m(x) - torch.randn(16, len(ACTIONS))).square().mean()
    loss.backward()
    assert m.v.grad is not None and bool(torch.isfinite(m.v.grad).all())
    assert m.bias.grad is not None and bool(torch.isfinite(m.bias.grad).all())
    w = WormWorld(131, "GREEBLES")
    assert "STOP" in w.legal()
    print("QL009 PASS: frozen-parent identity; valid trainable patch; host STOP legal")


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(90909)
    np.random.seed(90909)
    out = Path(args.out)
    for d in ("gallery", "models"):
        (out/d).mkdir(parents=True, exist_ok=True)
    source, verified = verified_source(Path(args.parent), PARENT_SHA)
    q1_path = Path(args.parent)/"models"/"bal-q1.npz"
    q4_path = Path(args.parent)/"models"/"bal-q4.npz"
    q1 = GatedQ1(q1_path, thresholds=(.45,.45,.45))
    q4middle = Hybrid(q1_path, q4_path, "b")
    train, train_hist, train_illegal = examples(range(2100,2180),33102)
    test, test_hist, test_illegal = examples(range(9100,9132),33103)
    assert train_illegal == test_illegal == 0
    assert len(test[1]) == 4352 and not set(range(2100,2180)) & set(range(9100,9132))
    mask = torch.zeros(27)
    keep = torch.argsort(train[0].std(0), descending=True)[:14]
    mask[keep] = 1.0
    with torch.no_grad():
        fixed_train_scores = q4middle(train[0])
    variants = {"q1-threshold45":q1, "q1-q4middle":q4middle}
    trainers = {}
    training_specs = (
        ("rank2-dense-ce", torch.ones(27), 0., 90911, True),
        ("rank2-sparse-ce", mask, 0., 90913, True),
        ("rank2-sparse-retain", mask, 1.5, 90917, True),
        ("rank2-sparse-no-bias", mask, 0., 90919, False))
    for name, features, penalty, seed, train_bias in training_specs:
        model, optimizer, trace, info = fit(q4middle, train, features,
                                            fixed_train_scores, args.steps, penalty, seed, train_bias=train_bias)
        variants[name] = model
        trainers[name] = info
        torch.save({"model_state":model.state_dict(),"optimizer":optimizer,
                    "trace":trace,"training":info,
                    "source_sha":os.environ.get("QL_SOURCE_SHA","local"),
                    "parent_sha":PARENT_SHA,
                    "parent_q1_sha256":digest(q1_path),
                    "parent_q4_sha256":digest(q4_path)},
                   out/"models"/(name+"-optimizer.pt"))
    raw = []
    reports = {}
    seeds = (9719, 9721, 9733, 9749)
    for name,model in variants.items():
        score = heldout(model, test, q4middle)
        episodes = []
        for seed in seeds:
            for family in FAMILIES:
                ep,steps = rollout(model,name,family,seed,out,horizon=args.horizon)
                ep["morphology"] = morphology(steps)
                episodes.append(ep)
                raw.append({"policy":name,"mode":"live","first_pass":True,
                            "seed":seed,"family":family,"steps":steps})
        actions = Counter()
        for ep in episodes:
            actions.update(ep["action_histogram"])
        reports[name] = {
            "heldout":score,
            "mean_unique_cells":round(sum(e["unique_cells"] for e in episodes)/len(episodes),5),
            "mean_turns":round(sum(e["turns"] for e in episodes)/len(episodes),5),
            "total_actions":sum(e["turns"] for e in episodes),
            "stopped_episodes":sum(bool(e["stop_count"]) for e in episodes),
            "branch":actions["BRANCH"],"retrace":actions["RETRACE"],"stop":actions["STOP"],
            "mean_repeat_fraction":round(sum(e["morphology"]["repeat_fraction"] for e in episodes)/len(episodes),5),
            "pngs":[{"path":e["png"],"sha256":e["png_sha256"]} for e in episodes],
            "same_runner_latency":timing(model,test[0],n=64),
            "adapter_working_parameter_bytes":model.memory_bytes() if isinstance(model,SparseRankTwo) else 0}
    controls={}
    for mode in ("random","wall","fidget"):
        eps=[]
        for seed in seeds:
            for family in FAMILIES:
                ep,steps=rollout(None,mode,family,seed,out,horizon=args.horizon)
                eps.append(ep)
                raw.append({"policy":mode,"mode":"model-free","first_pass":True,
                            "seed":seed,"family":family,"steps":steps})
        controls[mode]={"n":len(eps),
            "mean_unique_cells":round(sum(e["unique_cells"] for e in eps)/len(eps),5),
            "mean_turns":round(sum(e["turns"] for e in eps)/len(eps),5),
            "pngs":[{"path":e["png"],"sha256":e["png_sha256"]} for e in eps]}
    # One frozen-opening control for each repaired model: real divergent
    # rollouts, not an equal-state attribution estimate.
    for name in ("q1-q4middle","rank2-sparse-ce","rank2-sparse-retain"):
        eps=[]
        for seed in seeds:
            for family in FAMILIES:
                alias=name+"-opening"
                ep,steps=rollout(frozen_prior(variants[name],seed,family),
                                 alias,family,seed,out,horizon=args.horizon)
                eps.append(ep)
                raw.append({"policy":name,"mode":"frozen-opening","first_pass":True,
                            "seed":seed,"family":family,"steps":steps})
        reports[name]["frozen_opening_mean_cells"]=round(
            sum(e["unique_cells"] for e in eps)/len(eps),5)
        reports[name]["frozen_opening_pngs"]=[
            {"path":e["png"],"sha256":e["png_sha256"]} for e in eps]
    (out/"raw_decisions.json").write_text(json.dumps(raw,separators=(",",":"))+"\n")
    result = {
        "schema":"ql009-sparse-lowrank-postquant-1",
        "source_sha":os.environ.get("QL_SOURCE_SHA",os.environ.get("GITHUB_SHA","local")),
        "parent_source_sha":PARENT_SHA, "parent_manifest_verified":verified,
        "parent_q1_sha256":digest(q1_path),"parent_q4_sha256":digest(q4_path),
        "train_n":len(train[1]),"heldout_n":len(test[1]),
        "train_histogram":train_hist,"heldout_histogram":test_hist,
        "training_only_selected_features":keep.tolist(),
        "optimizer_updates":4*args.steps,"training":trainers,
        "seeds":list(seeds),"horizon":args.horizon,
        "variants":reports,"controls":controls,
        "first_pass_trajectory_count":len(raw),
        "original_png_n":len(list((out/"gallery").glob("*.png"))),
        "raw_decisions_sha256":digest(out/"raw_decisions.json"),
        "process_peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limitations":[
            "Tiny research MLP, not a quantized GGUF SLM or IQ codec",
            "Original Q1 XNOR and Q4 middle-island code tensors are immutable",
            "Parent Q4 middle layer uses float GEMV, not optimized lowbit SIMD",
            "Adapter is fp32 low-rank; no GGUF IQ or accelerated tokens/s inference",
            "Host enforces legal actions and can randomly STOP from tick zero",
            "Matched seeds/host stream but diverging world states after first different action",
            "STOP-truncated episodes have unequal horizons; no artwork-quality ranking",
            "Original images come from Python source-faithful synthetic W0, not Godot",
            "Process RSS includes Torch/NumPy, not isolated per-model RSS"]}
    (out/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    manifest=[{"path":str(p.relative_to(out)),"bytes":p.stat().st_size,
               "sha256":digest(p)}
              for p in sorted(out.rglob("*")) if p.is_file() and p.name!="MANIFEST.json"]
    (out/"MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"status":"QL009_COMPLETE","head":result["source_sha"],
        "real_optimizer_updates":result["optimizer_updates"],
        "pnGs":result["original_png_n"],
        "heldout":result["heldout_n"],
        "summaries":{k:{"cells":v["mean_unique_cells"],"turns":v["mean_turns"],
                        "branch":v["branch"],"retrace":v["retrace"],
                        "state_delta":v["heldout"]["live_adapter_vs_constant_adapter_differences_n"]}
                     for k,v in reports.items()}},sort_keys=True))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--parent",default="ql003-source")
    p.add_argument("--out",default="ql009-out")
    p.add_argument("--steps",type=int,default=48)
    p.add_argument("--horizon",type=int,default=96)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    self_test() if a.self_test else run(a)
