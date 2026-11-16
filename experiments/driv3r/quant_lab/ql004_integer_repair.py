#!/usr/bin/env python3
"""QL-004: actual integer-activation arithmetic and bitcode-preserving Q1 repair.

External synthetic CyberSand-style game-action research, NOT Godot/native,
GGUF IQ codecs, hardware lowbit acceleration or an admitted game provider.
Reuses pinned immutable QL-003 training/optimizer artifacts rather than
retraining the same 700-step baseline; uses 96 real new optimizer updates.
"""
import argparse
import hashlib
import json
import math
import os
import random
import resource
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from quant_worm import (ACTIONS, FAMILIES, Policy, digest, load_packed,
                        pack_codes, save_packed, unpack_codes, rollout, timing)
from ql002_branch_retrace import examples, assess
from ql003_surgery import morphology

PARENT_SHA = "9b4103c48127e5c8538403e2a4d9e2525fbe8755"
PARENT_RUN = 43795394633
LAYERS = "abc"


def pinned_sources(root):
    result = json.loads((root / "results.json").read_text())
    assert result["source_sha"] == PARENT_SHA, "not the QL-003 pinned model source"
    manifest = json.loads((root / "MANIFEST.json").read_text())
    assert len(manifest) >= 330
    for row in manifest:
        p = root / row["path"]
        assert p.is_file() and p.stat().st_size == row["bytes"], row["path"]
        assert digest(p) == row["sha256"], row["path"]
    return result, len(manifest)


def meta(name, record):
    proto = Policy()
    return {layer: {"bits": int(bits), "shape": list(getattr(proto, layer).weight.shape)}
            for layer, bits in zip(LAYERS, record["variants"][name]["bits"])}


class Int8ActivationPackedPolicy:
    """Packed Q1--Q4 weight codes with dynamic signed int8 activations.

    Never performs floating matmul with dequantized weights in quantized layers.
    Decoded codes are cached as int32 for a reference *arithmetic* experiment.
    This is NOT a memory-saving, efficient integer kernel; count cache RSS.
    Mixed FP32 layers retain standard float matmul.
    """
    def __init__(self, path, metadata):
        self.layers = []
        self.cache_nbytes = 0
        with np.load(path, allow_pickle=False) as archive:
            for layer in LAYERS:
                desc = metadata[layer]
                bits, shape = desc["bits"], desc["shape"]
                bias = archive[layer + "_bias"].copy()
                if bits == 32:
                    weights = archive[layer + "_fp32"].copy()
                    scale = None
                    self.cache_nbytes += weights.nbytes
                else:
                    raw = unpack_codes(archive[layer + "_packed"], bits, math.prod(shape))
                    raw = raw.reshape(shape).astype(np.int32)
                    weights = raw * 2 - 1 if bits == 1 else raw - (2 ** (bits - 1) - 1)
                    scale = archive[layer + "_scale"].copy()
                    self.cache_nbytes += weights.nbytes + scale.nbytes
                self.layers.append((bits, weights, scale, bias))
                self.cache_nbytes += bias.nbytes

    def __call__(self, tensor):
        x = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        if x.ndim == 1:
            x = x[None, :]
        result = np.empty((len(x), len(ACTIONS)), dtype=np.float32)
        for i, row in enumerate(x):
            z = row
            for j, (bits, weights, scale, bias) in enumerate(self.layers):
                if bits == 32:
                    z = weights @ z + bias
                else:
                    max_abs = float(np.max(np.abs(z)))
                    act_scale = max_abs / 127.0 if max_abs > 0 else 1.0
                    qact = np.clip(np.rint(z / act_scale), -127, 127).astype(np.int32)
                    sums = weights @ qact  # genuine integer products/accumulation
                    z = sums.astype(np.float32) * (scale * act_scale) + bias
                z = np.tanh(z) if j < 2 else z
            result[i] = z
        return torch.from_numpy(result)


class Q1FrozenCodesRepair(torch.nn.Module):
    """Only per-output scales and biases train; Q1 sign codes stay fixed."""
    def __init__(self, source):
        super().__init__()
        with np.load(source, allow_pickle=False) as f:
            for layer in LAYERS:
                prototype = getattr(Policy(), layer)
                shape = tuple(prototype.weight.shape)
                code = unpack_codes(f[layer + "_packed"], 1, math.prod(shape))
                signs = (code.reshape(shape).astype(np.float32) * 2.0 - 1.0)
                self.register_buffer(layer + "_sign", torch.from_numpy(signs))
                self.register_parameter(layer + "_scale",
                    torch.nn.Parameter(torch.tensor(f[layer + "_scale"].copy())))
                self.register_parameter(layer + "_bias",
                    torch.nn.Parameter(torch.tensor(f[layer + "_bias"].copy())))

    def forward(self, x):
        for j, layer in enumerate(LAYERS):
            sign = getattr(self, layer + "_sign")
            scale = getattr(self, layer + "_scale")
            bias = getattr(self, layer + "_bias")
            x = F.linear(x, sign * scale[:, None], bias)
            if j != 2:
                x = torch.tanh(x)
        return x

    def as_policy(self):
        p = Policy()
        state = {}
        for layer in LAYERS:
            state[layer + ".weight"] = (getattr(self, layer + "_sign") *
                getattr(self, layer + "_scale")[:, None]).detach().clone()
            state[layer + ".bias"] = getattr(self, layer + "_bias").detach().clone()
        p.load_state_dict(state)
        p.eval()
        return p


def train_scale_repair(source_path, train, reference, steps, seed, kl_weight):
    torch.manual_seed(seed)
    model = Q1FrozenCodesRepair(source_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0016, weight_decay=0.)
    x, y, _ = train
    counts = torch.bincount(y, minlength=len(ACTIONS)).float()
    weights = (1. / counts[y]).double()
    weights /= weights.sum()
    before = [p.detach().clone() for p in model.parameters()]
    losses = []
    for _ in range(steps):
        ix = torch.multinomial(weights, 128, replacement=True)
        pred = model(x[ix])
        with torch.no_grad():
            mentor = reference(x[ix])
        ce = F.cross_entropy(pred, y[ix])
        kl = F.kl_div(F.log_softmax(pred, dim=-1),
                      F.softmax(mentor, dim=-1), reduction="batchmean")
        loss = ce + kl_weight * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for letter in LAYERS:
                getattr(model, letter + "_scale").clamp_(min=1e-7)
        losses.append(float(loss))
    changed = sum(int(not torch.equal(p, old))
                  for p, old in zip(model.parameters(), before))
    moments = sum(int(torch.count_nonzero(s["exp_avg"]) > 0)
                  for s in optimizer.state.values())
    diag = {"optimizer": "AdamW", "updates": steps, "seed": seed,
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "changed_tensors": changed, "nonzero_moment_tensors": moments,
            "loss_first": losses[0], "loss_last": losses[-1],
            "teacher_ce_plus_kl": kl_weight, "fixed_q1_sign_code": True}
    assert changed > 0 and moments > 0
    return model.as_policy(), optimizer.state_dict(), diag


def verify_codes_unchanged(source, repaired):
    with np.load(source, allow_pickle=False) as a, np.load(repaired, allow_pickle=False) as b:
        for layer in LAYERS:
            assert np.array_equal(a[layer + "_packed"], b[layer + "_packed"])
    return True


def masked_predictions(model, held):
    x, _, legal = held
    with torch.no_grad():
        logits = model(x)
        masked = logits.masked_fill(~legal, -1e9).argmax(dim=1)
    return masked, logits


def independent_episodes(model, name, root, seeds):
    rows, receipts = [], []
    for seed in seeds:
        for family in FAMILIES:
            episode, steps = rollout(model, name, family, seed, root, horizon=96)
            episode["morphology"] = morphology(steps)
            rows.append(episode)
            receipts.append({"variant": name, "seed": seed, "family": family,
                             "first_pass": True, "steps": steps})
    summary = {
        "episodes": len(rows),
        "turns": sum(e["turns"] for e in rows),
        "mean_turns": round(sum(e["turns"] for e in rows) / len(rows), 4),
        "mean_unique_cells": round(sum(e["unique_cells"] for e in rows) / len(rows), 4),
        "mean_bbox_area": round(sum(e["morphology"]["bounding_box_area"] for e in rows) / len(rows), 4),
        "stopped": sum(e["stop_count"] > 0 for e in rows),
        "actions": dict(sum((Counter(e["action_histogram"]) for e in rows), Counter())),
        "original_pngs": [{"path": e["png"], "sha256": e["png_sha256"]}
                          for e in rows],
    }
    return summary, receipts


def self_test():
    torch.set_num_threads(1)
    import tempfile
    from quant_worm import quant_numpy
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        torch.manual_seed(57)
        source = Policy()
        for bits in ((1,1,1),(2,2,2),(3,3,3),(4,4,4),(1,4,4)):
            path, metadata, _ = save_packed(root, "sample", source, bits)
            dec = load_packed(path, metadata)
            integer = Int8ActivationPackedPolicy(path, metadata)
            x = torch.randn(50, 27) * .5
            error = float((dec(x)-integer(x)).abs().max())
            assert np.isfinite(error) and error < 0.15, (bits, error)
        path, metadata, _ = save_packed(root, "q1", source, (1,1,1))
        repair = Q1FrozenCodesRepair(path)
        p = repair.as_policy()
        # Exact input comparison, no noisy relative evaluation.
        x = torch.randn(4,27)
        assert torch.allclose(repair(x), p(x), atol=1e-7)
    print("QL004 self-test PASS: actual int32 arithmetic, source-safe frozen Q1 codes")


def run(args):
    torch.set_num_threads(1)
    np.random.seed(66901)
    torch.manual_seed(66901)
    source = Path(args.source_dir)
    root = Path(args.out)
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "gallery").mkdir(parents=True, exist_ok=True)
    old, manifest_n = pinned_sources(source)
    train, _, invalid_a = examples(range(2100,2180), 33102)
    held, hist, invalid_b = examples(range(9100,9132), 33103)
    assert invalid_a == invalid_b == 0 and all(hist.get(a,0) for a in ACTIONS)

    q1_archive = source / "models/bal-q1.npz"
    q1_reference = load_packed(q1_archive, meta("bal-q1",old))
    reports = {}
    repaired = {}
    for name, penalty in (("q1-scales-ce", 0.), ("q1-scales-distilled", 1.5)):
        net, opt, diag = train_scale_repair(q1_archive, train, q1_reference,
                                            args.steps, 66901, penalty)
        archive, metadata, payload = save_packed(root / "models", name, net, (1,1,1))
        assert verify_codes_unchanged(q1_archive, archive)
        torch.save({"optimizer": opt, "diagnostic":diag, "source_sha":os.environ.get("GITHUB_SHA","local"),
                    "frozen_weight_code_sha256": digest(q1_archive)},
                   root / "models" / (name + "-optimizer.pt"))
        repaired[name] = (archive, metadata, payload, diag)

    specs = [
        ("bal-fp32", "bal-fp32", "float"),
        ("bal-q1-dequant", "bal-q1", "float"),
        ("bal-q1-int8", "bal-q1", "integer"),
        ("bal-q2-int8", "bal-q2", "integer"),
        ("bal-q3-int8", "bal-q3", "integer"),
        ("bal-q4-int8", "bal-q4", "integer"),
        ("bal-mixed-int8", "bal-q1a-q4bc", "integer"),
        ("frank-q1-int8", "frank-reg-a-bal-bc-q1", "integer"),
        ("q1-scales-ce-int8", "q1-scales-ce", "integer"),
        ("q1-scales-distilled-int8", "q1-scales-distilled", "integer"),
    ]
    fp32_ref = load_packed(source/"models/bal-fp32.npz",meta("bal-fp32",old))
    reference_action, _ = masked_predictions(fp32_ref, held)
    original_q1_action, _ = masked_predictions(q1_reference, held)
    raw = []
    seeds = (9191,9229,9253,9281)
    baselines = {}
    for control in ("random", "wall", "fidget"):
        baselines[control], decisions = independent_episodes(None, control, root, seeds)
        raw.extend(decisions)

    for name, parent, backend in specs:
        if parent in repaired:
            path, metadata, payload, diag = repaired[parent]
        else:
            path = source / "models" / (parent + ".npz")
            metadata = meta(parent, old)
            payload = old["variants"][parent]["packed_payload_bytes"]
            diag = None
        model = load_packed(path, metadata) if backend == "float" else Int8ActivationPackedPolicy(path,metadata)
        pred, _ = masked_predictions(model, held)
        held_stats = assess(model, held)
        episodes, decisions = independent_episodes(model, name, root, seeds)
        raw.extend(decisions)
        reports[name] = {
            "parent":parent, "arithmetic":backend, "bits":[metadata[k]["bits"] for k in LAYERS],
            "archive_sha256":digest(path), "payload_bytes":payload,
            "host_masked_disagreement_vs_fp32_n":int((pred != reference_action).sum()),
            "host_masked_disagreement_vs_same_q1_n":int((pred != original_q1_action).sum()),
            "integer_cache_bytes":getattr(model,"cache_nbytes",None),
            "same_runner_latency":timing(model,held[0],n=100),
            "heldout":held_stats, "trajectories":episodes, "repair":diag}
    (root/"raw_decisions.json").write_text(json.dumps(raw,separators=(",",":"))+"\n")
    result = {
        "schema":"ql004-pinned-integer-activation-repair-1",
        "source_sha":os.environ.get("GITHUB_SHA","local"),
        "parent_training_source":PARENT_SHA,
        "parent_training_run":PARENT_RUN,
        "parent_manifest_entries_verified":manifest_n,
        "new_optimizer_updates":args.steps*2,
        "train_n":len(train[1]), "heldout_n":len(held[1]),
        "visual_seeds":list(seeds), "visual_horizon":96,
        "variant_reports":reports, "baselines":baselines,
        "raw_decisions_sha256":digest(root/"raw_decisions.json"),
        "process_maxrss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "caveats":[
            "Python NumPy int32 integer MAC with decoded cached codes, not efficient lowbit SIMD or GGUF IQ1/IQ2/IQ3",
            "Dynamic int8 activation quantization; mixed FP32 layers use float matmul",
            "Source-faithful synthetic game, not native Godot or CyberSand simulator",
            "STOP is legal on the first action and no policy gets STOP suppression",
            "Matched starts and host PRNG seed, not identical future states once actions diverge",
            "Small W0-style environments; creative quality is not inferred from teacher accuracy",
        ],
    }
    (root/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    inventory=[{"path":str(p.relative_to(root)),"bytes":p.stat().st_size,"sha256":digest(p)}
        for p in sorted(root.rglob("*")) if p.is_file() and p.name!="MANIFEST.json"]
    (root/"MANIFEST.json").write_text(json.dumps(inventory,indent=2)+"\n")
    print(json.dumps({"status":"QL004_COMPLETE","source_sha":result["source_sha"],
        "parent_manifest_verified":manifest_n,"optimizer_updates":args.steps*2,
        "heldout":len(held[1]),"original_pngs":len(list((root/"gallery").glob("*.png"))),
        "variants":{k:{"unique_cells":v["trajectories"]["mean_unique_cells"],
                       "stop":v["trajectories"]["stopped"],
                       "disagreement_fp32":v["host_masked_disagreement_vs_fp32_n"],
                       "ms":v["same_runner_latency"]["same_process_p50_ms"]}
                    for k,v in reports.items()}},sort_keys=True))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source-dir",default="ql003-source")
    p.add_argument("--out",default="ql004-out")
    p.add_argument("--steps",type=int,default=48)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    self_test() if args.self_test else run(args)
