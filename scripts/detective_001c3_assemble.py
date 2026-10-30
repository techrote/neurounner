#!/usr/bin/env python3
"""Assemble DETECTIVE-001C3 single-member evidence and neutral release."""
from __future__ import annotations
import argparse
import gzip
import json
import shutil
import tarfile
from pathlib import Path

from detective_001c3_dataset import check as check_dataset,sha_file
from detective_001c2_train import directory_hash

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"experiments/detective-001/training-c3.json"


class C3AssembleError(ValueError): pass


def write_json(p:Path,v)->None:
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,sort_keys=True,
                            ensure_ascii=False)+"\n",encoding="utf-8")


def deterministic_tar(source:Path,dest:Path)->None:
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open("wb") as raw:
        with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as gz:
            with tarfile.open(fileobj=gz,mode="w",
                              format=tarfile.PAX_FORMAT) as tf:
                for p in sorted(source.rglob("*")):
                    if not p.is_file(): continue
                    info=tf.gettarinfo(str(p),arcname=p.relative_to(source).as_posix())
                    info.uid=info.gid=0;info.uname=info.gname="";info.mtime=0
                    with p.open("rb") as f: tf.addfile(info,f)


def inventory(root:Path)->str:
    rows=[]
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        rel=p.relative_to(root).as_posix()
        if rel!="hashes.sha256": rows.append(f"{sha_file(p)}  {rel}")
    return "\n".join(rows)+"\n"


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset-root",type=Path,required=True)
    ap.add_argument("--trained-root",type=Path,required=True)
    ap.add_argument("--evaluation",type=Path,required=True)
    ap.add_argument("--evidence-out",type=Path,required=True)
    ap.add_argument("--release-out",type=Path,required=True)
    ap.add_argument("--source-head",required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--run-url",required=True)
    args=ap.parse_args()

    cfg=json.loads(CONFIG.read_text())
    dm=check_dataset(args.dataset_root)
    result_path=args.trained_root/"result.json"
    adapter=args.trained_root/"adapter"
    if not result_path.is_file() or not adapter.is_dir():
        raise C3AssembleError("C3 trained artifact incomplete")
    result=json.loads(result_path.read_text())
    if (result.get("schema")!="detective-witness-c3-preference-result-v1"
            or result.get("member_id")!="V-G02-A"
            or result.get("training",{}).get("optimizer_steps")!=28
            or result.get("frozen_control",{}).get("optimizer_steps")!=0):
        raise C3AssembleError("C3 training result identity/step drift")
    trained_hash=directory_hash(adapter)
    if result["adapter"]["directory_sha256"]!=trained_hash:
        raise C3AssembleError("C3 trained adapter hash mismatch")

    summary=json.loads((args.evaluation/"summary.json").read_text())
    final=json.loads((args.evaluation/"final-suite.json").read_text())
    if (summary.get("experiment")!="DETECTIVE-001C3"
            or final.get("experiment")!="DETECTIVE-001C3"
            or final.get("c3_trained_directory_sha256")!=trained_hash):
        raise C3AssembleError("C3 evaluation/final ancestry mismatch")

    if args.evidence_out.exists(): shutil.rmtree(args.evidence_out)
    if args.release_out.exists(): shutil.rmtree(args.release_out)
    args.evidence_out.mkdir(parents=True);args.release_out.mkdir(parents=True)

    train_dest=args.evidence_out/"training/V-G02-A.json"
    train_dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(result_path,train_dest)
    for name in ("final-suite.json","raw-results.json","summary.json",
                 "REPORT.md","relevance-packet.json"):
        shutil.copy2(args.evaluation/name,args.evidence_out/name)

    archive_name="detective-001c3-V-G02-A-smollm2-135m-lora.tar.gz"
    archive=args.release_out/archive_name
    deterministic_tar(adapter,archive)
    asset={
        "member_id":"V-G02-A",
        "file":archive_name,
        "sha256":sha_file(archive),"bytes":archive.stat().st_size,
        "adapter_directory_sha256":trained_hash,
        "parent_release_tag":cfg["trainable_parent"]["release_tag"],
        "parent_archive_sha256":cfg["trainable_parent"]["archive_sha256"],
        "parent_directory_sha256":cfg["trainable_parent"]["directory_sha256"]
    }
    release={
        "schema":"detective-witness-c3-release-v1",
        "experiment":"DETECTIVE-001C3","issue":118,
        "release_tag":cfg["release"]["tag"],
        "base_model":cfg["base_model"]["id"],
        "base_revision":cfg["base_model"]["revision"],
        "pair_dataset_manifest_sha256":sha_file(args.dataset_root/"manifest.json"),
        "final_suite_sha256":sha_file(args.evaluation/"final-suite.json"),
        "assets":[asset],
        "frozen_control":{
            "member_id":"V-G09-C",
            "release_tag":cfg["frozen_control"]["release_tag"],
            "directory_sha256":cfg["frozen_control"]["directory_sha256"],
            "optimizer_steps":0
        }
    }
    write_json(args.evidence_out/"adapter-release.json",release)
    campaign={
        "schema":"detective-witness-c3-summary-v1",
        "experiment":"DETECTIVE-001C3","issue":118,
        "source_head":args.source_head,
        "workflow_run_id":args.run_id,"workflow_run_url":args.run_url,
        "pair_dataset_manifest_sha256":sha_file(args.dataset_root/"manifest.json"),
        "dataset_counts":dm["counts"],
        "training":{
            "preference_margin_before":
                result["preference_before"]["mean_margin_nats_per_token"],
            "preference_margin_after":
                result["preference_after"]["mean_margin_nats_per_token"],
            "preference_accuracy_before":
                result["preference_before"]["preference_accuracy"],
            "preference_accuracy_after":
                result["preference_after"]["preference_accuracy"],
            "first_loss":result["training"]["first_loss"],
            "last_loss":result["training"]["last_loss"],
            "seconds":result["training"]["seconds"],
            "sequence_tokens_per_second":
                result["training"]["sequence_tokens_per_second"],
            "peak_rss_kib":result["runtime"]["peak_rss_kib"],
            "trained_directory_sha256":trained_hash
        },
        "evaluation":summary,
        "release_tag":cfg["release"]["tag"],
        "population_expansion_admitted":False
    }
    write_json(args.evidence_out/"campaign-summary.json",campaign)
    write_json(args.evidence_out/"provenance.json",{
        "source_head":args.source_head,
        "workflow_run_id":args.run_id,"workflow_run_url":args.run_url,
        "config":"experiments/detective-001/training-c3.json",
        "config_sha256":sha_file(CONFIG),
        "frozen_evaluator_merge_commit":cfg["evaluator"]["merge_commit"],
        "pair_dataset_manifest_sha256":sha_file(args.dataset_root/"manifest.json"),
        "final_suite_sha256":sha_file(args.evaluation/"final-suite.json"),
        "release_asset":asset,
        "frozen_control":release["frozen_control"],
        "external_provider_calls":0,
        "training_members":["V-G02-A"],
        "c1_final_used_as_training":False,
        "c2_final_used_as_training":False
    })
    (args.evidence_out/"hashes.sha256").write_text(
        inventory(args.evidence_out),encoding="utf-8")
    print(json.dumps({
        "trained_member":"V-G02-A",
        "trained_directory_sha256":trained_hash,
        "c3_gate":summary["c3_gate"],
        "independent_relevance":summary["independent_relevance_gate"],
        "population_expansion_admitted":False,
        "release_assets":1
    },indent=2))


if __name__=="__main__":
    main()
