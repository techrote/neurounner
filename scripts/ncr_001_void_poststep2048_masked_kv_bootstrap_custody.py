#!/usr/bin/env python3
"""ONE-SHOT exact original Actions custody for poststep B=2 masked-KV cohort."""
from __future__ import annotations
import hashlib, io, json, os, subprocess, zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler

import ncr_001_void_poststep2048 as producer_source
import ncr_001_void_poststep2048_masked_kv as masked_source

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"experiments/ncr-001/controls-evidence/void-poststep2048-masked-kv-20261030"
REPO="techrote/neurounner"
RUN=41136709967
HEAD="b3e9e6a05e007297de19f3955b59e2a9eb118996"
SOURCE_BRANCH="review/ncr001e-void-poststep2048-masked-kv-20261030"
ORIGINAL={
 "source":{"job":124340840759,"artifact":13017366873,"bytes":1132,
           "sha256":"972687a60ec5acd797a0cfea204b82f780fc425db2345a98059e001dc74bbab5"},
 "host":{"job":124340899423,"artifact":13017098290,"bytes":10029,
         "sha256":"638c5d564eb92a6f3d2a7df9fd36582e2761d736d4421fd67802f5796dee5eaa"},
}

def refuse(ok,why):
    if not ok: raise RuntimeError("MASKED_KV_CUSTODY_REFUSED: "+why)
def sha(raw): return hashlib.sha256(raw).hexdigest()
def blob(raw): return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()

class DropAuth(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        r=super().redirect_request(req,fp,code,msg,headers,newurl)
        if r is not None and urlparse(newurl).hostname not in ("api.github.com","github.com"):
            for table in (r.headers,r.unredirected_hdrs):
                for k in tuple(table):
                    if k.lower()=="authorization": del table[k]
        return r

def fetch(endpoint,*,json_result=False):
    token=os.getenv("GH_TOKEN"); refuse(bool(token),"missing Actions token")
    req=Request("https://api.github.com/repos/"+REPO+"/"+endpoint,headers={
      "Accept":"application/vnd.github+json","Authorization":"Bearer "+token,
      "X-GitHub-Api-Version":"2022-11-28","User-Agent":"neurounner-masked-kv-custody"})
    with build_opener(DropAuth()).open(req,timeout=90) as response: payload=response.read()
    return json.loads(payload) if json_result else payload

def semantics(producer,consumer):
    producer_source.validate_producer(producer)
    masked_source.validate_consumer(consumer,producer)
    refuse(producer["source"]["head"]==consumer["masked_consumer_source"]["head"]==HEAD,
           "physical exact source head changed")
    result={"nca_only_optimizer_steps":1,"true_eos_id":258,
            "public_training_units":31,"supervised_after_shift":21,
            "producer_pid":producer["producer_pid"],"consumer_pid":consumer["consumer_pid"],
            "effective_threads":consumer["threads_effective_after_load"],
            "frozen_backbone_sha256":consumer["frozen_backbone_sha256"],
            "same_runner_second_python":True,
            "consumer_elapsed_seconds":consumer["elapsed_seconds"],
            "consumer_peak_rss_kib":consumer["peak_rss_kib"],
            "sides":{}}
    for side in ("left","right"):
        x=consumer["masked_incremental_kv"][side]
        result["sides"][side]={
          "width":x["width"],"long_valid_units":x["long_valid_units"],
          "short_valid_units":x["short_valid_units"],
          "synthetic_pad_units":x["synthetic_pad_units"],
          "tail_real_incremental_units_per_row":x["tail_real_incremental_units_per_row"],
          "genuine_B2_incremental_steps":x["genuine_B2_incremental_steps"],
          "explicit_attention_mask":x["explicit_attention_mask"],
          "explicit_mask_derived_position_ids":x["explicit_mask_derived_position_ids"],
          "right_layout_is_synthetic_mask_gap":x["right_layout_is_synthetic_mask_gap"],
          "full_vs_cached_max_abs":x["full_vs_cached"]["max_abs"],
          "batch_vs_single_full_max_abs":x["batch_vs_single_full_max_abs"],
          "batch_vs_single_cached_max_abs":x["batch_vs_single_cached_max_abs"],
          "row_swap_full_max_abs":x["row_swap_full"]["max_abs"],
          "row_swap_cached_max_abs":x["row_swap_cached"]["max_abs"],
        }
    return result

def audit_originals():
    refuse(os.getenv("GITHUB_REPOSITORY")==REPO,"wrong repository")
    refuse(subprocess.run(["git","-C",str(ROOT),"merge-base","--is-ancestor",HEAD,"HEAD"],
                          capture_output=True).returncode==0,"physical head not ancestor")
    run=fetch(f"actions/runs/{RUN}",json_result=True)
    refuse(run["head_sha"]==HEAD and run["status"]=="completed" and
           run["conclusion"]=="success" and run["event"]=="pull_request" and
           run["run_attempt"]==1,"run identity/status")
    jobs=fetch(f"actions/runs/{RUN}/jobs?per_page=100",json_result=True)
    jobmap={x["id"]:x for x in jobs["jobs"]}
    arts=fetch(f"actions/runs/{RUN}/artifacts?per_page=100",json_result=True)
    artmap={x["id"]:x for x in arts["artifacts"]}
    rawfiles={}; members={}; meta={}; physical=None
    for kind,spec in ORIGINAL.items():
        job=jobmap[spec["job"]]
        refuse(job["status"]=="completed" and job["conclusion"]=="success" and
               job["run_id"]==RUN and job["head_sha"]==HEAD,"job mismatch "+kind)
        art=artmap[spec["artifact"]]
        refuse(art["workflow_run"]["id"]==RUN and art["workflow_run"]["head_sha"]==HEAD and
               not art["expired"] and art["digest"]=="sha256:"+spec["sha256"] and
               art["size_in_bytes"]==spec["bytes"],"artifact ledger "+kind)
        data=fetch(f"actions/artifacts/{spec['artifact']}/zip")
        refuse(len(data)==spec["bytes"] and sha(data)==spec["sha256"],"whole ZIP "+kind)
        rawfiles[kind+"-original.zip"]=data
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            refuse(z.testzip() is None,"ZIP CRC "+kind)
            names=z.namelist()
            members[kind]={n:{"bytes":len(z.read(n)),"sha256":sha(z.read(n))} for n in names}
            if kind=="source":
                refuse(set(names)=={
                    "producer-tests.log","masked-kv-tests.log"} and
                    b"Ran 10 tests" in z.read("producer-tests.log") and b"OK" in z.read("producer-tests.log") and
                    b"Ran 9 tests" in z.read("masked-kv-tests.log") and b"OK" in z.read("masked-kv-tests.log"),
                    "source test archive incomplete")
            else:
                required={
                 "ncr001-void-poststep2048/producer.json",
                 "ncr001-void-poststep2048/producer.log",
                 "ncr001-void-poststep2048-masked-kv/consumer.json",
                 "ncr001-void-poststep2048-masked-kv/consumer.log",
                 "ncr001-void-poststep2048-masked-kv/prefetch.log",
                 "ncr001-void-poststep2048-masked-kv/runner.txt",
                 "ncr001-void-nca-learning/receipt.json",
                 "ncr001-void-postlease/poststep-cold.json"}
                refuse(required.issubset(set(names)) and not any(
                    n.endswith((".pt",".safetensors","metadata.json")) for n in names),
                    "host archive missing receipts or leaked adapter")
                producer=json.loads(z.read("ncr001-void-poststep2048/producer.json"))
                consumer=json.loads(z.read("ncr001-void-poststep2048-masked-kv/consumer.json"))
                physical=semantics(producer,consumer)
        log=fetch(f"actions/jobs/{spec['job']}/logs")
        marker=(b"Ran 9 tests" if kind=="source" else
                b"PASS_POSTSTEP2048_B2_MASKED_INCREMENTAL_KV")
        refuse(len(log)>1000 and HEAD.encode() in log and marker in log,
               "complete decoded log markers "+kind)
        rawfiles[kind+"-original-job.log"]=log
        meta[kind]={"original_job":spec["job"],"original_artifact":spec["artifact"],
                    "zip_bytes":len(data),"zip_sha256":sha(data),
                    "decoded_log_bytes":len(log),"decoded_log_sha256":sha(log)}
    refuse(physical is not None,"physical result missing")
    manifest={
      "schema":"ncr001-void-poststep2048-masked-kv-original-custody-v1",
      "repository":REPO,"run":RUN,"attempt":1,"source_head":HEAD,
      "source_branch":SOURCE_BRANCH,"source_job_success":True,"physical_job_success":True,
      "retention_of_original_zips_and_logs":"permanent_git_bytes",
      "original":{**meta,"physical_result":physical},"members":members,
      "files":{n:{"size":len(raw),"sha256":sha(raw),"git_blob":blob(raw)}
               for n,raw in rawfiles.items()},
      "scientific_boundary":{
        "train_authorized":False,"per_model_gate":False,"full_72_conformance_pass":False,
        "comparative_training":False,"heldouts_opened":False,
        "actual_compute_budget_qualified":False,"generated_eos_skill_qualified":False,
        "historical_train002b_eos_effect_measured":False,
        "independent_scientific_review":False}}
    return manifest,rawfiles

def main():
    refuse(not (OUT/"manifest.json").exists(),"originals already committed")
    manifest,files=audit_originals(); OUT.mkdir(parents=True,exist_ok=True)
    for n,raw in files.items(): (OUT/n).write_bytes(raw)
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print("MASKED_KV_ORIGINAL_CUSTODY_STAGED",json.dumps(
        {"run":RUN,"head":HEAD,"files":len(files),"physical":manifest["original"]["physical_result"]},
        sort_keys=True),flush=True)
if __name__=="__main__": main()
