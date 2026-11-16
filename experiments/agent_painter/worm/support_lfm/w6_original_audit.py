#!/usr/bin/env python3
"""Independent W6 original first-pass/W0 PNG and W4 spatial comparison.

This never runs model inference or optimization. Inputs are ORIGINAL immutable
LFM primary W6 optimizer evidence and W4 evaluation-only recovery evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from primary_gallery_audit import audit

SEEDS = (2101, 2102)
FAMILIES = ("greebles", "pipework", "architecture")
POLICIES = ("model", "random", "wall", "oracle")
W6_SOURCE = "999e60c81dc71b54e20150aa4b97887f22239607"
MODEL = "fc6221ca597f3315e4f82fc2df606783267b34ba"
TRAIN_SHA = "fa9dd90949407404904fb47e5a8357c47292b8d1ddb34735a62a6228bcaa4fef"
DEV_SHA = "fe959db7ad794d5e1db0ab53b35b7f6528bd1de3d7963210e621af70031845ec"
W4_GALLERY = "fe1a9b529d73be0df63223a71e355e6a7a8adcc8afa9e49143531e9a1bf8b549"
W6_GALLERY = "929820fc206accf582d581f3be0a92ddbed8ba185bbddb00703eff6777beff68"

def raw(path: Path) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ValueError("missing or linked original evidence: " + str(path))
    return path.read_bytes()

def obj(path: Path):
    return json.loads(raw(path))

def gallery_digest(folder: Path) -> str:
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError("missing or linked gallery")
    files = sorted(folder.iterdir())
    if len(files) != 54 or any(not f.is_file() or f.is_symlink() for f in files):
        raise ValueError("not precisely 54 original gallery files")
    lines = [f.name + "\t" + hashlib.sha256(raw(f)).hexdigest() for f in files]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()

def spatial(path: list, prefix: int = 24) -> dict:
    """Use W0 original position trajectories, not material coverage or fake PNGs."""
    if not isinstance(path, list) or not 1 <= len(path) <= 513:
        raise ValueError("invalid bounded W0 trajectory")
    for p in path:
        if not isinstance(p, list) or len(p) != 2 or any(type(v) is not int or not 0 <= v < 96 for v in p):
            raise ValueError("invalid position")
    if type(prefix) is not int or not 1 <= prefix <= 512:
        raise ValueError("invalid prefix")
    x0, y0 = path[0]
    def window(pts):
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        w, h = max(xs)-min(xs)+1, max(ys)-min(ys)+1
        return {"executed_steps":len(pts)-1, "unique_positions":len({tuple(q) for q in pts}),
                "bbox_width":w, "bbox_height":h, "bbox_area":w*h,
                "max_manhattan_from_start":max(abs(x-x0)+abs(y-y0) for x,y in pts)}
    return {"prefix24":window(path[:prefix+1]), "full":window(path)}

def training_evidence(root: Path) -> dict:
    s, corpus, sampler = (obj(root / f) for f in
        ("summary.json", "corpus-manifest.json", "sampler-receipt.json"))
    if (s.get("status") != "completed" or s.get("actual_training") is not True
            or s.get("optimizer_updates") != 120 or s.get("requested_updates") != 120
            or s.get("source_sha") != W6_SOURCE or s.get("model_revision") != MODEL
            or s.get("checkpoint") != "checkpoint-000120"
            or s.get("checkpoint_readback_exact") is not True
            or s.get("frozen_base_unchanged") is not True or s.get("adapter_changed") is not True
            or corpus.get("sealed_FINAL_opened") is not False or corpus.get("literal_prompt_overlap") != 0
            or corpus.get("WORM_TRAIN_sha256") != TRAIN_SHA or corpus.get("WORM_DEV_sha256") != DEV_SHA
            or sampler.get("version") != "lfm-worm-w6-anti-right4-family-v1"
            or sampler.get("counterfactual_not_host_executed") is not True):
        raise ValueError("original producer/model/split identity mismatch")
    for file, digest in (("worm_train.jsonl", TRAIN_SHA), ("worm_dev.jsonl", DEV_SHA)):
        if hashlib.sha256(raw(root / file)).hexdigest() != digest:
            raise ValueError("corpus bytes differ from source declaration")
    progress = [json.loads(l) for l in raw(root / "progress.jsonl").splitlines()]
    if (len(progress) != 120
            or [e.get("global_step") for e in progress] != list(range(1,121))
            or len({e.get("sample_id") for e in progress}) != 120
            or any(not isinstance(e.get("gradient_norm"), (float,int))
                   or not math.isfinite(e["gradient_norm"]) or e["gradient_norm"] <= 0
                   for e in progress)):
        raise ValueError("missing genuine consecutive optimizer gradient evidence")
    adapter_hashes=[]
    for step in (30,60,90,120):
        folder=root / f"checkpoint-{step:06d}"
        inv=obj(folder / "inventory.json")
        if len(inv) != 12 or obj(folder / "receipt.json").get("global_step") != step:
            raise ValueError("incomplete original checkpoint")
        for name, meta in inv.items():
            b=raw(folder / name)
            if len(b) != meta["bytes"] or hashlib.sha256(b).hexdigest() != meta["sha256"]:
                raise ValueError("original checkpoint readback hash mismatch")
        adapter_hashes.append(inv["adapter/adapter_model.safetensors"]["sha256"])
    if len(set(adapter_hashes)) != 4:
        raise ValueError("no distinct stepwise adapter checkpoints")
    trained, untuned = obj(root / "trained-first-passes.json"),obj(root / "untrained-first-passes.json")
    if (len(trained) != 18 or len(untuned) != 18
            or sum(e["valid_first_pass"] for e in trained) != 18
            or sum(e["valid_first_pass"] for e in untuned) != 0
            or any(not e["eos_terminated"] or e["token_ids"][-1] != 7 for e in trained)):
        raise ValueError("static first-pass/EOS receipt mismatch")
    return {"updates":120,"unique_sample_ids":120,"checkpoints":4,
            "checkpoint_member_sha256_checked":48,"adapter_hashes":adapter_hashes,
            "static_trained_valid":18,"static_untuned_valid":0,
            "static_trained_teacher_matches":sum(e["raw_text"]==e["expected"] for e in trained),
            "TRAIN_sha256":TRAIN_SHA,"DEV_sha256":DEV_SHA}

def compare(w4:Path,w6:Path,root:Path) -> dict:
    if gallery_digest(w4) != W4_GALLERY or gallery_digest(w6) != W6_GALLERY:
        raise ValueError("wrong W4 or W6 original gallery archive")
    # The shared audited W0 replayer checks every raw action, all 48 original PNG
    # bytes, obstacle worlds, host decisions, EOS7 and matched controller receipts.
    old=[audit(w4, seed=k,budget=48) for k in SEEDS]
    new=[audit(w6, seed=k,budget=48) for k in SEEDS]
    records=[]
    for i,seed in enumerate(SEEDS):
        a={(r["family"],r["policy"]):r for r in old[i]["metrics"]}
        b={(r["family"],r["policy"]):r for r in new[i]["metrics"]}
        if len(a)!=12 or set(a)!=set(b): raise ValueError("unmatched controls")
        for fam in FAMILIES:
            for pol in POLICIES:
                r4,r6=a[fam,pol],b[fam,pol]
                old_trace=obj(w4/f"{fam}-{pol}-{seed}.json")
                new_trace=obj(w6/f"{fam}-{pol}-{seed}.json")
                if (r4["original_png_sha256"]!=r6["original_png_sha256"]
                        or r4["metrics"]!=r6["metrics"] or r4["raw_cycle"]!=r6["raw_cycle"]
                        or old_trace["trajectory"]!=new_trace["trajectory"]
                        or old_trace["obstacles"]!=new_trace["obstacles"]):
                    raise ValueError("original W4/W6 host and PNG comparison differs")
                if pol=="model":
                    f4=obj(w4/f"{fam}-model-{seed}-raw-first-passes.json")
                    f6=obj(w6/f"{fam}-model-{seed}-raw-first-passes.json")
                    fields=lambda r:[(q["raw_text"],q["token_ids"],q["valid_first_pass"]) for q in r]
                    if fields(f4)!=fields(f6):raise ValueError("unmodified first-pass identities differ")
                if pol=="model" or fam=="greebles":
                    records.append({"family":fam,"seed":seed,"policy":pol,
                        "original_png_sha256":r6["original_png_sha256"],
                        "raw_action_counts":r6["metrics"]["raw_action_counts"],
                        "unique_cells":r6["metrics"]["unique_cells"],
                        "fallbacks":r6["metrics"]["host_fallback_count"],
                        "raw_repetition_fraction":r6["raw_cycle"]["fraction"],
                        "spatial":spatial(new_trace["trajectory"])})
    models=[r for r in records if r["policy"]=="model"]
    if len(models)!=6 or any(r["raw_action_counts"]!={"STRAIGHT":48}
                          or r["spatial"]["full"]["bbox_height"]!=1 for r in models):
        raise ValueError("expected original W6 STRAIGHT collapse not present")
    if any(len({r["original_png_sha256"] for r in models if r["seed"]==s})!=1 for s in SEEDS):
        raise ValueError("original model outputs not family-identical")
    return {"schema":"worm-lfm-w6-w4-original-spatial-audit-v1","worlds":2,
            "family_labels_per_world":3,"budget_each":48,"raw_decisions_each_model_version":288,
            "original_PNGs_source_faithfully_replayed":48,
            "W4_W6_model_PNGs_identical":True,"W4_W6_raw_first_passes_identical":True,
            "W4_180_step_producer_posttrain_failed":True,"W4_eval_only_new_steps":0,
            "W6_support_optimizer_steps":0,
            "W4_original_gallery_sha256":W4_GALLERY,"W6_original_gallery_sha256":W6_GALLERY,
            "W6_original_training":training_evidence(root),
            "records":records,
            "limits":"Two obstacle worlds, not six independent family replications. W0 CPU source-faithful pixels only; no native Godot material readback. Reach/coverage alone cannot establish aesthetics."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--w4-gallery",type=Path,required=True)
    p.add_argument("--w6-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=compare(a.w4_gallery,a.w6_root/"gallery",a.w6_root)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_bytes((json.dumps(result,sort_keys=True,separators=(",",":"))+"\n").encode())
    print(json.dumps({"schema":result["schema"],"W6_updates":120,
                      "original_PNGs":48,"result":"W6 STRAIGHT 288/288; pixel-identical W4"},sort_keys=True))

if __name__=="__main__":main()
