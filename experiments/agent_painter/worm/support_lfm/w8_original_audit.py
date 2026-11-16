#!/usr/bin/env python3
"""Independent original W8 artifact custody audit, no model/optimizer/native calls."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

ZIP_SHA = "b9d19fa758572660e837e59a13f50843639f6ea9c2350e0ff936fc41f549add9"
SOURCE = "61730799eddfbd43eac5d7945da11d6f953b47e6"
W0 = "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
SUPPORT = "cc27b3206cb61f98432fa95b799c9b9ff508c2bc"
MODEL = "fc6221ca597f3315e4f82fc2df606783267b34ba"
PAIRS_SHA = "8d4c4ee60af6ab05776e150cc2b43fbf667ff364fd6f45d507b65d96e3844919"
CODES = dict(zip(("STRAIGHT","LEFT","RIGHT","BRANCH","RETRACE","STOP"),"012345"))
PREFIX = "worm-w8/"

def sha(data):
    return hashlib.sha256(data).hexdigest()

def canonical(data):
    return json.dumps(data,sort_keys=True,separators=(",",":"),ensure_ascii=True)

def read_json(z,path):
    return json.loads(z.read(PREFIX+path))

def verify_zip(path):
    if sha(path.read_bytes()) != ZIP_SHA:
        raise ValueError("original W8 artifact digest mismatch")
    z = zipfile.ZipFile(path)
    names = z.namelist()
    if len(names) != len(set(names)) or z.testzip() is not None:
        raise ValueError("duplicate members or CRC failure")
    if any(n.startswith("/") or ".." in PurePosixPath(n).parts or "\\" in n for n in names):
        raise ValueError("unsafe ZIP path")
    checkpoint_hashes = {}
    verified = 0
    for step in (45,90,135,180):
        folder = f"{PREFIX}checkpoint-{step:06d}/"
        inventory = json.loads(z.read(folder+"inventory.json"))
        if len(inventory)!=12 or set(inventory)!={n[len(folder):] for n in names if n.startswith(folder)}-{"inventory.json"}:
            raise ValueError("checkpoint member inventory mismatch")
        for name, item in inventory.items():
            data = z.read(folder+name)
            if sha(data)!=item["sha256"] or len(data)!=item["bytes"]:
                raise ValueError("checkpoint member bytes/digest mismatch")
            verified+=1
        checkpoint_hashes[str(step)] = inventory["adapter/adapter_model.safetensors"]["sha256"]
    if len(set(checkpoint_hashes.values())) != 4:
        raise ValueError("adapters did not change between checkpoints")
    return z,checkpoint_hashes,verified

def score(row):
    ids=row["token_ids"]
    if not isinstance(ids,list) or not ids or any(type(n) is not int for n in ids):
        raise ValueError("invalid original token IDs")
    if row["eos_terminated"] != (ids[-1]==7):
        raise ValueError("inconsistent true EOS7")
    action=next((a for a,c in CODES.items() if row["raw_text"]==c),None)
    syntax=bool(action is not None and row["eos_terminated"])
    legal=bool(syntax and action in row["legal"])
    if row["raw_legal"] != legal or row["valid_first_pass"] != legal:
        raise ValueError("host legality falsely reported as syntax")
    if row["raw_action"] != (action if legal else "MODEL_REFUSED"):
        raise ValueError("raw first-pass refusal inconsistency")
    return action,syntax,legal

def audit(path,out):
    if out.exists():
        raise ValueError("immutable output lease already exists")
    z,ck,verified=verify_zip(path)
    summary=read_json(z,"summary.json")
    corpus=read_json(z,"corpus-manifest.json")
    lease=read_json(z,"lease.json")
    if (summary["status"],summary["actual_optimizer_updates"],summary["requested_updates"])!=("complete",180,180):
        raise ValueError("incomplete genuine W8 optimizer")
    if (summary["source_sha"],lease["source_sha"],summary["model_revision"])!=(SOURCE,SOURCE,MODEL):
        raise ValueError("source/model mismatch")
    if (corpus["W0_source"],corpus["support_source"],corpus["pair_bytes_sha256"])!=(W0,SUPPORT,PAIRS_SHA):
        raise ValueError("wrong original split/corpus provenance")
    if corpus["TRAIN_DEV_prompt_overlap"]!=0 or summary["sealed_FINAL_opened"] or summary["online_model_rollouts"]!=0:
        raise ValueError("split/FINAL/online custody mismatch")
    pair_bytes=z.read(PREFIX+"optimizer-pairs.jsonl")
    if sha(pair_bytes)!=PAIRS_SHA:
        raise ValueError("TRAIN pair digest mismatch")
    rows=[json.loads(x) for x in pair_bytes.splitlines()]
    events=[json.loads(x) for x in z.read(PREFIX+"progress.jsonl").splitlines()]
    sampler=read_json(z,"sampler-receipt.json")
    if len(rows)!=90 or len(events)!=180 or len(sampler["IDs"])!=180:
        raise ValueError("TRAIN/optimizer counts")
    by_id={r["id"]:r for r in rows}
    if len(by_id)!=90 or len({r["prompt"] for r in rows})!=90:
        raise ValueError("duplicate TRAIN IDs/prompts")
    grouped=defaultdict(list)
    for row in rows:
        obs=json.loads(row["prompt"])
        if obs["legal"]!=row["legal"] or row["completion"] not in row["legal"]:
            raise ValueError("illegal/inconsistent teacher")
        grouped[row["pair_id"]].append(row)
    if len(grouped)!=45:
        raise ValueError("not 45 paired structural TRAIN examples")
    for pair in grouped.values():
        if len(pair)!=2:
            raise ValueError("unpaired W0 and synthetic input")
        originals=[r for r in pair if r["origin"]=="W0-observed-TRAIN"]
        synthetics=[r for r in pair if r["origin"]=="synthetic-TRAIN-not-W0"]
        if len(originals)!=1 or len(synthetics)!=1:
            raise ValueError("fabricated TRAIN provenance")
        a,b=originals[0],synthetics[0]
        oa,ob=json.loads(a["prompt"]),json.loads(b["prompt"])
        if {k:v for k,v in oa.items() if k!="legal"}!={k:v for k,v in ob.items() if k!="legal"}:
            raise ValueError("synthetic pair changes beyond legal")
        if (b["excluded_action"]!=a["completion"] or
            ob["legal"]!=[x for x in oa["legal"] if x!=a["completion"]]):
            raise ValueError("synthetic mask does not exclude original target")
    if ([e["global_step"] for e in events]!=list(range(1,181)) or
        Counter(e["sample_id"] for e in events)!=Counter(sampler["IDs"]) or
        Counter(e["kind"] for e in events)!={"W0-observed-TRAIN":90,"synthetic-TRAIN-not-W0":90}):
        raise ValueError("optimizer/sample cursor contradiction")
    for event in events:
        if event["kind"]!=by_id[event["sample_id"]]["origin"] or any(
            not math.isfinite(float(event[k])) or float(event[k])<=0
            for k in ("loss","gradient_norm","elapsed_seconds")):
            raise ValueError("bad gradient/optimizer event")
    baseline=read_json(z,"untrained-first-passes.json")
    trained=read_json(z,"trained-first-passes.json")
    if len(baseline)!=18 or len(trained)!=18 or [x["id"] for x in baseline]!=[x["id"] for x in trained]:
        raise ValueError("unpaired static DEVELOPMENT first passes")
    original_scores=[score(x) for x in baseline]
    scores=[score(x) for x in trained]
    if sum(v[1] for v in original_scores)!=0 or [v[0] for v in scores]!=["RETRACE"]*18:
        raise ValueError("original untuned/learned raw action distribution changed")
    if sum(v[1] for v in scores)!=18 or sum(v[2] for v in scores)!=12 or sum(x["expected"]==v[0] for x,v in zip(trained,scores))!=3:
        raise ValueError("original W8 EOS/host legality/teacher counts mismatch")
    per_family={}
    for family in ("greebles","pipework","architecture"):
        subset=[(x,v) for x,v in zip(trained,scores) if x["family"]==family]
        per_family[family]={"n":len(subset),"raw_EOS7":sum(v[1] for x,v in subset),
                            "host_legal":sum(v[2] for x,v in subset),
                            "teacher_match":sum(x["expected"]==v[0] for x,v in subset)}
        if per_family[family]!={"n":6,"raw_EOS7":6,"host_legal":4,"teacher_match":1}:
            raise ValueError("family result mismatch")
    observed=[x for x in trained if x["origin"]=="W0-observed"]
    probes=[]
    for row in observed:
        if "RETRACE" not in row["legal"]:
            continue
        if "STOP" not in row["legal"]:
            raise ValueError("no host STOP escape")
        probes.append({"source_id":row["id"],"family":row["family"],
            "seed_group":row["seed_group"],"kind":"synthetic-not-W0-executed",
            "remove_action":"RETRACE","original_legal":row["legal"],
            "revised_legal":[a for a in row["legal"] if a!="RETRACE"],
            "target_action":None,
            "input_note":"Resolve full original DEVELOPMENT prompt by source_id from pinned support corpus; only legal list may change."})
    if len(observed)!=15 or len(probes)!=12:
        raise ValueError("unexpected true DEV input / mask challenge counts")
    report={"schema":"worm-lfm-support-w8-original-audit-v1",
       "source_sha":SOURCE,"W0_source":W0,"support_source":SUPPORT,
       "model_revision":MODEL,"artifact_zip_sha256":ZIP_SHA,
       "optimizer_pairs_sha256":PAIRS_SHA,"zip_crc_entries":len(z.namelist()),
       "verified_checkpoint_inventory_members":verified,"adapter_sha256_by_step":ck,
       "optimizer_steps_verified":len(events),"original_TRAIN_pairs":45,
       "synthetic_TRAIN_pairs":45,"untrained_syntactic_EOS7":0,
       "trained_syntactic_EOS7":18,"trained_host_legal":12,
       "trained_host_illegal":6,"trained_teacher_matches":3,
       "trained_raw_RETRACE":18,"dev_static_requests":18,
       "original_W0_DEV_requests":len(observed),
       "observed_seed_groups":sorted({r["seed_group"] for r in observed}),
       "per_family":per_family,"anti_RETRACE_synthetic_source_cases":len(probes),
       "new_model_first_passes":0,"new_optimizer_updates":0,
       "new_model_PNGs":0,"new_native_operations":0,
       "sealed_FINAL_opened":False}
    out.mkdir(parents=True)
    (out/"report.json").write_text(canonical(report)+"\n")
    (out/"anti-retrace-12-provenance.jsonl").write_text("".join(canonical(p)+"\n" for p in probes))
    (out/"SHA256SUMS.txt").write_text("".join(
        sha(p.read_bytes())+"  "+p.name+"\n" for p in sorted(out.iterdir()) if p.is_file()))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--artifact",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print(canonical(audit(a.artifact,a.out)))

if __name__=="__main__":
    main()
