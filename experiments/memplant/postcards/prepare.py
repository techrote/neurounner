"""Prepare G1B synthetic worlds and genuine Qwen attention-value features.

The follow-on is a supervised low-bandwidth semantic prefix compiler, not a
direct KV transplant, portable model memory, or unseen-world scientific proof.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
from experiments.memplant.g1.features import DONOR, DONOR_REV, RECIPIENT, RECIPIENT_REV, cases
from experiments.memplant.dreamgraft001.contract import write_json

@dataclass(frozen=True)
class NewCase:
    case_id: str
    family: str
    source_text: str
    answer: str
    twin_id: str
    def record(self):
        return dict(case_id=self.case_id, family=self.family,
                    source_text=self.source_text, answer=self.answer,
                    twin_id=self.twin_id, split="fresh_test",
                    source_sha256=hashlib.sha256(self.source_text.encode()).hexdigest())

def new_cases():
    materials = ("cinderium", "bellwrought metal", "vaporbrass", "star-ash alloy")
    sites = ("submarine orchard", "lighthouse workshop", "floating foundry", "deep observatory")
    out = []
    for i,(material,site) in enumerate(zip(materials,sites,strict=True)):
        for val in (0,1):
            right = ("hardens","softens")[val]
            wrong = ("softens","hardens")[val]
            text = (f"At the {site}, artisans passed down a warning about {material}. "
                    f"The claim that it {wrong} when heated is marked INVALID in "
                    f"the archive. Instead, an independently witnessed experiment "
                    f"found that the metal {right} when warmed. Other pages discuss "
                    f"birds and shadows, neither of which changes this metal rule.")
            out.append(NewCase(f"novel-{i:02d}-rule-{val}",f"novel-{i:02d}",
                               text,right,f"novel-{i:02d}-rule-{1-val}"))
    assert len(out)==8
    assert all(x.source_text.count("hardens")==1 and
               x.source_text.count("softens")==1 for x in out)
    return out

def extract(directory):
    import torch
    import numpy as np
    from experiments.memplant.dreamgraft001.g0_live import load_pinned, measure_rss
    from experiments.memplant.dreamgraft001.bridge import source_fingerprint
    torch.set_num_threads(min(torch.get_num_threads(),4))
    root=Path(directory)
    root.mkdir(parents=True,exist_ok=True)
    originals=[x for x in cases() if x.split in ("train","validation")]
    additions=new_cases()
    donor,tokenizer,identity=load_pinned(DONOR,DONOR_REV)
    result=[]
    records=[]
    for item in originals+additions:
        with torch.no_grad():
            enc=tokenizer(item.source_text,return_tensors="pt")
            out=donor(**enc,use_cache=True,return_dict=True)
            features,meta=source_fingerprint(out.past_key_values)
        if features.numel()!=2048:
            raise RuntimeError("Donor V width changed")
        result.append(features.numpy().astype("float32"))
        records.append(item.record())
        del out,enc
    data=np.stack(result)
    np.savez_compressed(root/"source_v.npz",v=data)
    write_json(root/"manifest.json",{
        "schema":"memplant-g1b-postcards-v1","donor":identity,
        "recipient_id":RECIPIENT,"recipient_revision":RECIPIENT_REV,
        "records":records,"train_count":32,"validation_count":8,
        "fresh_test_count":8,"fresh_test_not_in_initial_g1":True,
        "source_array_sha256":hashlib.sha256((root/"source_v.npz").read_bytes()).hexdigest(),
        "feature_sha256":hashlib.sha256(data.tobytes()).hexdigest(),
        "rss_mib":measure_rss(),"only_synthetic_sources":True,
        "evaluator_labels_not_in_recipient_prompt":True
    })
    print(json.dumps({"count":len(records),"shape":list(data.shape),
                      "donor_revision":identity["revision"]},indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="runtime/memplant/g1b/source")
    extract(p.parse_args().out)

if __name__=="__main__":
    main()
