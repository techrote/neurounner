"""Bounded CPU curriculum: 36 -> 72 -> 128 genuine optimizer+rollout decisions.

Outputs model checkpoint, signed-by-hash original trajectories, metrics and
images to an external artifact directory ONLY. Never saves weights in Git.
No model-side SLM/LoRA job, native Workbench, or paid endpoint is invoked.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import statistics
import struct
import subprocess
import time
import zlib

import numpy as np

from .world import (World,Spec,TRAIN_FAMILIES,HELDOUT_FAMILIES,MOVE,VERSION as WORLD_VERSION,
                    digest,replay,specs)
from .policy import (Ranker,teacher,random_legal,random_legal_persistent,menu_matrix,ACTION_MAX,FEATURES,VERSION as POLICY_VERSION)

VERSION = "driv3r-long-horizon-training/1"
HORIZONS = (36,72,128)
MAX_WORLDS_PER_FAMILY = 30
MAX_UPDATES_PER_STAGE = 400
MAX_DECISIONS = 128
MAX_RECOVERIES = 12
NATIVE_MOVE_LIMIT_UNCHANGED = 64


def sha_file(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_pins():
    d = Path(__file__).parent
    return {name:sha_file(d/name) for name in ("world.py","policy.py","train.py")}


def source_head():
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"],text=True,
                                       stderr=subprocess.DEVNULL,timeout=4).strip()
    except (OSError,subprocess.CalledProcessError,subprocess.TimeoutExpired):
        return None


def make_corpus(horizon: int, seed: int, n_per_family: int,
                adversarial_fraction: float = .13):
    """Teacher labels + explicit off-policy bad moves in TRAIN only."""
    rows = []; labels = []; masks = []
    original_worlds = []
    interventions = 0
    label_counts = Counter()
    training_specs = specs("train",n_per_family)
    # Dedicated recovery scenarios use the same split, never a heldout family.
    training_specs += [replace(s,challenge="trap_corridor")
                       for s in training_specs if s.seed % 2 == 0]
    for spec in training_specs:
        w = World.create(spec)
        original_worlds.append({"spec":spec.ident,"base_sha256":w.base_sha(),
                                "family":spec.family,"seed":spec.seed,"mode":spec.mode,
                                "challenge":spec.challenge})
        rng = random.Random(seed*100000 + horizon*193 + spec.seed)
        for _ in range(horizon):
            if w.stopped: break
            chosen,menu,matrix = teacher(w,horizon,rng)
            idx = menu.index(chosen)
            mask = np.zeros(ACTION_MAX,dtype=np.bool_)
            mask[:len(menu)] = True
            rows.append(matrix); masks.append(mask);labels.append(idx)
            label_counts[chosen] += 1
            applied = chosen
            ink_choices = [a for a in menu if a in MOVE and a != chosen]
            # Hard-example intervention is logged, not silently relabelled as teacher.
            if ink_choices and rng.random() < adversarial_fraction:
                applied = rng.choice(ink_choices)
                interventions += 1
            w.apply(applied,MAX_RECOVERIES)
    if not rows: raise RuntimeError("Empty training corpus")
    return (np.stack(rows),np.stack(masks),np.asarray(labels,dtype=np.int64),
            {"horizon":horizon,"split":"train","examples":len(rows),
             "worlds":original_worlds,"original_worlds_sha256":digest(original_worlds),
             "labels":dict(sorted(label_counts.items())),"off_policy_interventions":interventions,
             "adversarial_fraction":adversarial_fraction})


def _png(path: Path, w: World):
    """Original synthetic 96x96 material/ink image; no native framebuffer."""
    width = 96
    pixels = bytearray()
    rows=[]
    for y in range(width):
        row = bytearray()
        for x in range(width):
            p = (x,y)
            if p == w.cursor: c=(255,96,123)
            elif p in w.ink: c=(245,187,51)
            elif p == (48,48): c=(48,190,201)
            elif p in w.protected: c=(183,78,160)
            elif w.materials.get(p) == 37: c=(87,119,171)
            elif p in w.materials: c=(83,89,98)
            else: c=(18,24,34)
            row.extend(c * 4)
        rows.append(b"\x00" + bytes(row))
    pixels.extend(b"".join(rows[i//4] for i in range(width*4)))
    width *= 4
    def chunk(tag,body):
        return struct.pack(">I",len(body))+tag+body+struct.pack(">I",zlib.crc32(tag+body)&0xffffffff)
    data = (b"\x89PNG\r\n\x1a\n" +chunk(b"IHDR",struct.pack(">IIBBBBB",width,width,8,2,0,0,0))
            +chunk(b"IDAT",zlib.compress(bytes(pixels),9))+chunk(b"IEND",b""))
    path.write_bytes(data)


def episode(spec: Spec,horizon: int, policy: str,model: Ranker | None,
            run_seed: int,output_dir: Path | None=None):
    if horizon not in HORIZONS:
        raise ValueError("Unapproved synthetic horizon")
    if policy not in ("teacher_mixture","random_legal","random_legal_persistent","trained_tiny_mlp"):
        raise ValueError("Unknown policy")
    if (model is None) != (policy != "trained_tiny_mlp"):
        raise ValueError("Learned policy must have a real checkpoint")
    w = World.create(spec)
    rng = random.Random(run_seed*137 + spec.seed*17 + horizon)
    start = time.perf_counter_ns()
    decision_latencies_ns = []
    action_dist = Counter()
    for _ in range(horizon):
        if w.stopped: break
        t = time.perf_counter_ns()
        scores = None
        if policy == "teacher_mixture":
            action,*_ = teacher(w,horizon,rng)
        elif policy == "random_legal":
            action = random_legal(w,rng)
        elif policy == "random_legal_persistent":
            action = random_legal_persistent(w,rng)
        else:
            assert model is not None
            action,scores = model.choose(w,horizon)
        decision_latencies_ns.append(time.perf_counter_ns()-t)
        row = w.apply(action,MAX_RECOVERIES)
        action_dist[action] += 1
        if scores is not None:
            row["policy_probabilities"] = scores  # chosen after a genuine neural forward
        # Explicitly exclude probabilistic report decoration from host replay data.
    elapsed = (time.perf_counter_ns()-start)/1e9
    original_events = [{k:v for k,v in e.items() if k != "policy_probabilities"}
                       for e in w.decisions]
    reproduced = replay(spec,original_events,MAX_RECOVERIES)
    if reproduced.state_sha() != w.state_sha() or reproduced.ink != w.ink:
        raise AssertionError("Recorded trajectory cannot be independently replayed")
    result = {"version":VERSION,"evidence":"original_synthetic_cpu_rollout",
              "model_executed":policy=="trained_tiny_mlp", "slm_executed":False,
              "native_executed":False,"sequential_native_receipts":0,
              "native_64_logical_move_ceiling_unmodified":NATIVE_MOVE_LIMIT_UNCHANGED,
              "spec_id":spec.ident,"split":spec.split,"family":spec.family,"seed":spec.seed,
              "mode":spec.mode,"challenge":spec.challenge,
              "world_base_sha256":w.base_sha(),"horizon":horizon,
              "policy":policy,"policy_seed":run_seed,"metrics":w.metrics(horizon),
              "action_distribution":dict(sorted(action_dist.items())),
              "mean_decision_latency_ms":round(statistics.mean(decision_latencies_ns)/1e6,6),
              "p95_decision_latency_ms":round(sorted(decision_latencies_ns)[int(.95*(len(decision_latencies_ns)-1))]/1e6,6),
              "episode_wall_seconds":round(elapsed,6),
              "trajectory_sha256":digest(original_events),"final_state_sha256":w.state_sha(),
              "events":w.decisions}
    if output_dir is not None:
        output_dir.mkdir(parents=True,exist_ok=True)
        prefix = f"{spec.split}-{spec.ident}-{horizon}-{policy}"
        json_path = output_dir/(prefix+".json")
        json_path.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
        png_path = output_dir/(prefix+".png")
        _png(png_path,w)
    return result


def aggregate(results):
    if not results: raise ValueError("No model results")
    counts = Counter()
    for r in results: counts.update(r["action_distribution"])
    return {"episodes":len(results),
            "total_decisions":sum(r["metrics"]["decisions"] for r in results),
            "mean_ink_cells":round(statistics.mean(r["metrics"]["unique_painted_cells"] for r in results),4),
            "mean_recoveries":round(statistics.mean(r["metrics"]["recoveries"] for r in results),4),
            "mean_branch_cells":round(statistics.mean(r["metrics"]["branch_cells"] for r in results),4),
            "early_stops":sum(r["metrics"]["stopped_early"] for r in results),
            "all_host_legal":all(r["metrics"]["illegal_host_actions"]==0 for r in results),
            "no_protected_or_occupied_edits":all(r["metrics"]["protected_cell_edits"]==0 and
                                             r["metrics"]["non_empty_cell_edits"]==0 for r in results),
            "action_distribution":dict(sorted(counts.items())),
            "mean_decision_latency_ms":round(statistics.mean(r["mean_decision_latency_ms"] for r in results),6),
            "max_episode_wall_seconds":round(max(r["episode_wall_seconds"] for r in results),6)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--seed",type=int,default=570)
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--n-worlds-per-family",type=int,default=9)
    parser.add_argument("--updates-per-stage",type=int,default=180)
    parser.add_argument("--batch",type=int,default=64)
    a = parser.parse_args(argv)
    if a.quick:
        a.n_worlds_per_family=3
        a.updates_per_stage=45
    if not (1<=a.n_worlds_per_family<=MAX_WORLDS_PER_FAMILY and
            1<=a.updates_per_stage<=MAX_UPDATES_PER_STAGE and 8<=a.batch<=128 and
            0<=a.seed<=1_000_000):
        raise ValueError("CPU resource budget exceeded")
    if a.output.exists():
        raise FileExistsError("Refuse to overwrite immutable run output")
    a.output.mkdir(parents=True)
    started = time.perf_counter()
    policy = Ranker(seed=a.seed,hidden=24)
    train_stages = []
    for stage,horizon in enumerate(HORIZONS):
        X,M,y,provenance = make_corpus(horizon,a.seed,a.n_worlds_per_family)
        opt = policy.optimize(X,M,y,steps=a.updates_per_stage,batch=a.batch,
                              lr=.007 if stage<2 else .004,seed=a.seed+stage*917)
        train_stages.append({"curriculum_stage":stage,"horizon":horizon,
                             "corpus":provenance,"optimizer":opt})
    checkpoint = a.output/"tiny-ranker-36-72-128.npz"
    policy.checkpoint(checkpoint)
    loaded = Ranker.restore(checkpoint)
    if loaded.updates != policy.updates or any(not np.array_equal(loaded.params[k],policy.params[k])
                                              for k in policy.params):
        raise AssertionError("Checkpoint failed exact NumPy restore")
    after_train = time.perf_counter()
    split_result = {}
    # DEVELOPMENT comparison is separate; structural HELDOUT never creates teacher labels.
    # HELDOUT evaluated ONCE after final checkpoint; never used for optimizer selection.
    for split in ("development","heldout"):
        worlds = specs(split,2)  # distinct seeds and heldout families
        worlds += [replace(s,challenge="trap_corridor") for s in worlds if s.seed % 2 == 0]
        comparisons = {}
        for horizon in HORIZONS:
            policy_rollouts = {}
            for arm in ("teacher_mixture","random_legal","random_legal_persistent","trained_tiny_mlp"):
                runs = [episode(spec,horizon,arm,loaded if arm == "trained_tiny_mlp" else None,
                                a.seed, a.output/"trajectories")
                        for spec in worlds]
                policy_rollouts[arm] = aggregate(runs)
            comparisons[str(horizon)] = policy_rollouts
        split_result[split] = {"structural_families":sorted({w.family for w in worlds}),
                               "world_specs":[s.ident for s in worlds],
                               "comparisons":comparisons}
    checkpoint_sha=sha_file(checkpoint)
    total_wall = time.perf_counter()-started
    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metadata = {"version":VERSION,"evidence":"real_cpu_optimizer_tiny_mlp",
                "not_executed_families":["d1-2B","LFM2.5-350M","Sprocket-500M","SmolLM2-135M"],
                "source_head":source_head(),"source_sha256":source_pins(),
                "model_type":"trained_numpy_mlp_action_ranker", "model_revision":POLICY_VERSION,
                "model_parameter_count":loaded.parameter_count,"optimizer":"Adam/manual NumPy",
                "optimizer_updates":loaded.updates,"training_seed":a.seed,
                "original_checkpoint_sha256":checkpoint_sha,
                "checkpoint_filename":checkpoint.name,
                "source_world_version":WORLD_VERSION,
                "train_families":list(TRAIN_FAMILIES),"heldout_families":list(HELDOUT_FAMILIES),
                "curriculum":train_stages,"comparisons":split_result,
                "training_wall_seconds":round(after_train-started,4),
                "total_wall_seconds":round(total_wall,4),
                "peak_process_rss_kib":peak_rss_kib,
                "resource_caps":{"max_horizon":MAX_DECISIONS,
                                 "max_recoveries_per_episode":MAX_RECOVERIES,
                                 "native_logical_move_cap_unmodified":NATIVE_MOVE_LIMIT_UNCHANGED,
                                 "max_optimiser_steps_per_stage":MAX_UPDATES_PER_STAGE,
                                 "max_worlds_per_family":MAX_WORLDS_PER_FAMILY},
                "native_sessions":0,"paid_gpu_or_api_calls":0,
                "heldout_feedback_to_training":False,"training_on_real_traces":False,
                "claimed_neural_art_quality":False}
    (a.output/"report.json").write_text(json.dumps(metadata,sort_keys=True,indent=2)+"\n")
    file_hashes={str(p.relative_to(a.output)):sha_file(p)
                 for p in sorted(a.output.rglob("*")) if p.is_file()}
    (a.output/"SHA256SUMS.json").write_text(json.dumps(file_hashes,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"source_head":metadata["source_head"],"checkpoint_sha256":checkpoint_sha,
                      "parameters":loaded.parameter_count,"optimizer_updates":loaded.updates,
                      "training_wall_seconds":metadata["training_wall_seconds"],
                      "total_wall_seconds":metadata["total_wall_seconds"],
                      "peak_rss_kib":peak_rss_kib,"comparisons":split_result},sort_keys=True))
    return metadata


if __name__ == "__main__":
    main()
