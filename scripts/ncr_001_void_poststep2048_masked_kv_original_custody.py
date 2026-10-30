#!/usr/bin/env python3
"""Read-only permanent audit of original poststep B=2 masked-KV Actions bytes."""
from __future__ import annotations
import argparse, hashlib, io, json, subprocess, zipfile
from pathlib import Path
import ncr_001_void_poststep2048_masked_kv_bootstrap_custody as acquisition

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/"experiments/ncr-001/controls-evidence/void-poststep2048-masked-kv-20261030"
MANIFEST=EVIDENCE/"manifest.json"
MANIFEST_BLOB="439584bd144c5b640cbc49807200ea3845b384cb"
FILES={
 "source-original.zip":(1132,"972687a60ec5acd797a0cfea204b82f780fc425db2345a98059e001dc74bbab5","30a472aeeaa126110bd7afffffad644fb7b0498f"),
 "host-original.zip":(10029,"638c5d564eb92a6f3d2a7df9fd36582e2761d736d4421fd67802f5796dee5eaa","51882755d6b2e2c48d5f73c7d72f14dcd8499170"),
 "source-original-job.log":(56049,"9741fa2c7812b34e1f6e189480fb9a17f80737478ac637a18e3424ffb904cee6","32503c33c48d3212a561b5d4575833555df3ef99"),
 "host-original-job.log":(83610,"03174257d3373aa21286e8a532e6d6428d42a048e3deecc193254327b8c799da","2b9124c37dd4090f6883e56f5249801be0c95994")}
BOUNDARY={"actual_compute_budget_qualified":False,"comparative_training":False,
"full_72_conformance_pass":False,"generated_eos_skill_qualified":False,
"heldouts_opened":False,"historical_train002b_eos_effect_measured":False,
"independent_scientific_review":False,"per_model_gate":False,"train_authorized":False}

def refuse(ok,why):
    if not ok: raise RuntimeError("MASKED_KV_PRIMARY_AUDIT_REFUSED: "+why)
def sha(raw): return hashlib.sha256(raw).hexdigest()
def blob(raw): return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
def entry(path,expected):
    rel=path.relative_to(ROOT).as_posix()
    x=subprocess.run(["git","-C",str(ROOT),"ls-tree","HEAD","--",rel],
                     capture_output=True,text=True)
    refuse(x.returncode==0 and x.stdout==f"100644 blob {expected}\t{rel}\n",
           "Git tree mode/blob "+rel)

def read_and_check(*,verify_git=True):
    raw=MANIFEST.read_bytes()
    refuse(MANIFEST.is_file() and not MANIFEST.is_symlink() and blob(raw)==MANIFEST_BLOB,
           "manifest literal blob")
    if verify_git: entry(MANIFEST,MANIFEST_BLOB)
    m=json.loads(raw)
    refuse(m["run"]==41136709967 and m["source_head"]==
           "b3e9e6a05e007297de19f3955b59e2a9eb118996" and
           m["retention_of_original_zips_and_logs"]=="permanent_git_bytes" and
           m["scientific_boundary"]==BOUNDARY and set(m["files"])==set(FILES),
           "manifest authority/boundary")
    originals={}
    for name,(size,digest,gblob) in FILES.items():
        p=EVIDENCE/name
        refuse(p.is_file() and not p.is_symlink(),"missing/symlink "+name)
        data=p.read_bytes()
        refuse(len(data)==size and sha(data)==digest and blob(data)==gblob and
               m["files"][name]=={"size":size,"sha256":digest,"git_blob":gblob},
               "literal original stream "+name)
        if verify_git: entry(p,gblob)
        originals[name]=data
    for kind in ("source","host"):
        with zipfile.ZipFile(io.BytesIO(originals[kind+"-original.zip"])) as z:
            refuse(z.testzip() is None,"ZIP CRC "+kind)
            inv={n:{"bytes":len(z.read(n)),"sha256":sha(z.read(n))} for n in z.namelist()}
            refuse(inv==m["members"][kind],"ZIP member drift "+kind)
            if kind=="source":
                refuse(set(inv)=={"producer-tests.log","masked-kv-tests.log"} and
                       b"Ran 10 tests" in z.read("producer-tests.log") and
                       b"Ran 9 tests" in z.read("masked-kv-tests.log"),"source proofs")
            else:
                refuse(len(inv)==8 and not any(n.endswith((".pt",".safetensors","metadata.json"))
                       for n in inv),"host members/adapter leak")
                producer=json.loads(z.read("ncr001-void-poststep2048/producer.json"))
                consumer=json.loads(z.read("ncr001-void-poststep2048-masked-kv/consumer.json"))
                refuse(acquisition.semantics(producer,consumer)==m["original"]["physical_result"],
                       "physical semantic derivation")
    refuse(b3:=b"PASS_BOUNDED_MASKED_KV_DIFF" in originals["source-original-job.log"],
           "source complete log marker")
    refuse(b"PASS_POSTSTEP2048_B2_MASKED_INCREMENTAL_KV" in originals["host-original-job.log"] and
           b"1.6927719116210938e-05" in originals["host-original-job.log"],
           "host complete log marker")
    return {"status":"PASS_ORIGINAL_POSTSTEP_B2_MASKED_KV_BYTES__SCIENTIFIC_72_UNADMITTED",
      "original_run":41136709967,"original_head":"b3e9e6a05e007297de19f3955b59e2a9eb118996",
      "original_zips_and_logs":4,"original_zip_members":10,
      "physical_result":m["original"]["physical_result"],"permanent_git_custody":True,
      "fresh_physical_execution":False,"train_authorized":False,"per_model_gate":False,
      "full_72_conformance_pass":False,"independent_scientific_review":False,
      "actual_compute_budget_qualified":False},originals

def audit(*,live=False):
    report,originals=read_and_check()
    if live:
        manifest,raw=acquisition.audit_originals()
        refuse(manifest==json.loads(MANIFEST.read_text()) and
               all(raw[n]==originals[n] for n in FILES),"live bytes differ")
        report["original_ledger_freshly_rechecked"]=True
    else: report["original_ledger_freshly_rechecked"]=False
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--live",action="store_true")
    print(json.dumps(audit(live=p.parse_args().live),sort_keys=True))
