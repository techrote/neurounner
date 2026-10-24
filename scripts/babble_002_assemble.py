#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PERSONAS=[f"G{i:02d}" for i in range(1,13)]
EXPECTED_SURFACES={
    "physical_reaction":15,
    "embodied_decision":15,
    "local_memory":15,
    "conflicting_motives":15,
    "failure_and_recovery":15,
    "mundane_low_stakes":15,
    "conversation_social":10,
}

def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda:fh.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def find_persona_dir(root:Path,gid:str)->Path:
    hits=[]
    for p in root.rglob("manifest.json"):
        try:
            obj=json.loads(p.read_text())
        except Exception:
            continue
        if obj.get("experiment")=="BABBLE-002" and obj.get("persona")==gid:
            hits.append(p.parent)
    if len(hits)!=1:
        raise RuntimeError(f"{gid}: expected one persona artifact, got {hits}")
    return hits[0]

def write_json(path:Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n")

def hash_inventory(out:Path)->str:
    rows=[]
    for p in sorted(out.rglob("*")):
        if not p.is_file():
            continue
        rel=p.relative_to(out).as_posix()
        if rel=="provenance/hashes.sha256":
            continue
        rows.append(f"{sha256_file(p)}  {rel}")
    return "\n".join(rows)+"\n"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--artifacts",type=Path,required=True)
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    cfg=json.loads(args.config.read_text())
    out=args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    all_rows=[]
    persona_stats={}
    manifests={}
    for gid in PERSONAS:
        src=find_persona_dir(args.artifacts,gid)
        manifest=json.loads((src/"manifest.json").read_text())
        manifests[gid]=manifest
        rows=[json.loads(x) for x in (src/"examples.jsonl").read_text().splitlines() if x.strip()]
        if len(rows)!=100:
            raise RuntimeError(f"{gid}: expected 100 examples, got {len(rows)}")
        ids=[x.get("example_id") for x in rows]
        if len(ids)!=len(set(ids)):
            raise RuntimeError(f"{gid}: duplicate example IDs")
        expected_ids=[f"{gid}-E{i:03d}" for i in range(1,101)]
        if ids!=expected_ids:
            raise RuntimeError(f"{gid}: noncanonical ID sequence")
        surf=Counter(x.get("surface") for x in rows)
        if dict(surf)!=EXPECTED_SURFACES:
            raise RuntimeError(f"{gid}: surface counts {dict(surf)} != {EXPECTED_SURFACES}")
        source_packet_id=manifest["source_packet_id"]
        source_babbles=manifest["source_babble_ids"]
        for x in rows:
            if x.get("source_guide_id")!=gid:
                raise RuntimeError(f"{gid}: bad source_guide_id in {x.get('example_id')}")
            if x.get("source_packet_id")!=source_packet_id:
                raise RuntimeError(f"{gid}: bad packet lineage in {x.get('example_id')}")
            if x.get("source_babble_ids")!=source_babbles:
                raise RuntimeError(f"{gid}: bad babble lineage in {x.get('example_id')}")
            if not isinstance(x.get("trait_refs"),list) or not x["trait_refs"]:
                raise RuntimeError(f"{gid}: missing trait refs in {x.get('example_id')}")
            if not all(isinstance(x.get(k),str) and x[k].strip() for k in ("input","target","generic_bad_target","why_bad")):
                raise RuntimeError(f"{gid}: incomplete text fields in {x.get('example_id')}")
        training_dir=out/"training"
        training_dir.mkdir(exist_ok=True)
        shutil.copy2(src/"examples.jsonl",training_dir/f"{gid}.jsonl")
        prov_dir=out/"provenance"/"persona-manifests"
        prov_dir.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src/"manifest.json",prov_dir/f"{gid}.json")
        provider_src=src/"provider"
        if provider_src.exists():
            shutil.copytree(provider_src,out/"provider"/gid)
        persona_stats[gid]={
            "examples":len(rows),
            "surface_counts":dict(surf),
            "repair_count":sum(1 for c in manifest["calls"] if c.get("repair_used")),
            "source_packet_id":source_packet_id,
            "source_babble_ids":source_babbles,
            "training_sha256":sha256_file(src/"examples.jsonl"),
        }
        all_rows.extend(rows)

    if len(all_rows)!=1200:
        raise RuntimeError(f"expected 1200 examples, got {len(all_rows)}")
    if len({x["example_id"] for x in all_rows})!=1200:
        raise RuntimeError("global duplicate example IDs")
    exact_pairs=[(x["input"].strip().lower(),x["target"].strip().lower()) for x in all_rows]
    exact_pair_dupes=len(exact_pairs)-len(set(exact_pairs))

    global_surface=Counter(x["surface"] for x in all_rows)
    expected_global={k:v*12 for k,v in EXPECTED_SURFACES.items()}
    if dict(global_surface)!=expected_global:
        raise RuntimeError(f"global surface counts {dict(global_surface)} != {expected_global}")

    source_packet_hashes={}
    for gid in PERSONAS:
        p=ROOT/"datasets"/"babble-001"/"compiled"/"packets"/f"{gid}.json"
        source_packet_hashes[gid]=sha256_file(p)

    quality={
        "schema_version":1,
        "total_examples":1200,
        "unique_example_ids":1200,
        "exact_input_target_duplicate_count":exact_pair_dupes,
        "global_surface_counts":dict(global_surface),
        "expected_global_surface_counts":expected_global,
        "personas":persona_stats,
    }
    write_json(out/"quality"/"validation.json",quality)

    run_id=os.environ.get("GITHUB_RUN_ID")
    source_sha=os.environ.get("GITHUB_SHA")
    provenance={
        "schema_version":1,
        "experiment":"BABBLE-002",
        "issue":2,
        "workflow_run_id":int(run_id) if run_id else None,
        "workflow_run_url":f"https://github.com/techrote/neurounner/actions/runs/{run_id}" if run_id else None,
        "source_commit":source_sha,
        "source_dataset":"datasets/babble-001",
        "source_dataset_manifest_sha256":sha256_file(ROOT/"datasets"/"babble-001"/"manifest.json"),
        "source_packet_hashes":source_packet_hashes,
        "provider":"nvidia",
        "model":cfg["model"],
        "remote_unpinned_backend":True,
    }
    write_json(out/"provenance"/"source-runs.json",provenance)

    manifest={
        "schema_version":1,
        "dataset_id":"babble-002",
        "issue":2,
        "created_from":"BABBLE-001 personality packets",
        "created_date":"2026-10-24",
        "provider":"nvidia",
        "compiler_model":cfg["model"],
        "remote_unpinned_backend":True,
        "counts":{
            "personas":12,
            "examples":1200,
            "examples_per_persona":100,
            "positive_targets":1200,
            "generic_negative_targets":1200
        },
        "surface_counts":dict(global_surface),
        "training_shards":[f"training/{gid}.jsonl" for gid in PERSONAS],
        "quality_file":"quality/validation.json",
        "provenance_file":"provenance/source-runs.json"
    }
    write_json(out/"manifest.json",manifest)

    readme="""# BABBLE-002 — expanded personality training corpus

BABBLE-002 expands all 12 durable BABBLE-001 personality packets into a lineage-preserving supervised corpus.

## Corpus

- 12 personas
- 100 examples per persona
- 1,200 positive supervised examples
- 1,200 paired generic/sanitized negative targets
- source lineage on every row: guide ID, packet ID and the four BABBLE-001 babble IDs
- explicit trait and mechanical-seed references
- raw NVIDIA request/response/repair history retained under provider/

## Scenario balance per persona

- physical reaction: 15
- embodied decision: 15
- local memory: 15
- conflicting motives: 15
- failure and recovery: 15
- mundane low stakes: 15
- conversation/social: 10

## Training row

Each JSONL row contains example_id, source_guide_id, source_packet_id, source_babble_ids,
surface, input, target, generic_bad_target, why_bad, trait_refs, seed_refs and compiler provenance.

The positive target is intended for SFT/personality training. The paired generic negative is useful for
preference/ranking work and explicitly demonstrates what personality sanitization looks like.

## Authority

BABBLE-001 remains the authority for the original Sprocket babble and compiled persona packet.
BABBLE-002 is a derived training corpus.

Provider responses are preserved before parsing/repair. No API credentials are stored.

See quality/validation.json, manifest.json and provenance/hashes.sha256 for machine-readable evidence.
"""
    (out/"README.md").write_text(readme)
    (out/"provenance"/"hashes.sha256").write_text(hash_inventory(out))
    print(json.dumps({
        "dataset":"babble-002",
        "personas":12,
        "examples":1200,
        "surface_counts":dict(global_surface),
        "exact_pair_duplicates":exact_pair_dupes,
        "files":sum(1 for p in out.rglob("*") if p.is_file())
    },indent=2))

if __name__=="__main__":
    main()
