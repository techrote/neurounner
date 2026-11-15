"""G1C source-only fresh confirmatory corpus for frozen Dream Postcard compiler.

No retraining, no tuning of previous receiver prefixes, no private source data.
The eight new families are deliberately absent from G1/G1B held-out sets.
"""
from __future__ import annotations
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path

from experiments.memplant.g1.features import DONOR,DONOR_REV,RECIPIENT,RECIPIENT_REV
from experiments.memplant.dreamgraft001.contract import write_json

NAMES=("thermoglass","bell-cinder","moonwoven silver","tendersteel",
       "aurora nickel","cobweb bronze","starlit mercury","tunneliron")
PLACES=("undersea furnace","abandoned sky railway","crystalline canyon",
        "moss-clad observatory","floating scrapyard","vault below the sea",
        "clockwork shipwreck","frozen cloud-city")

TEMPLATES=(
    "At the {place}, a field engineer studied {material}. The old instruction that "
    "warming it {wrong} is a deliberate falsehood. The reliable experiment found "
    "that heating the alloy {right}. The heat law remained stable on repeated trials.",
    "An archive about {material} recovered near the {place} has two entries. "
    "The first entry, written by a fraudster, says it {wrong} when hot. "
    "The corrected and verified second entry establishes that it {right} upon warming.",
    "In the {place}, a witness measured {material} while raising its temperature. "
    "The result was that the metal {right}. A rival observer's contrary claim that "
    "the alloy {wrong} was disproved after independent repetition.",
    "The metal {material} is tested in the {place}. One must not conclude that "
    "heat {wrong} it: this is specifically the rejected hypothesis. "
    "Reliable instruments show that warming the material {right}.",
)

@dataclass(frozen=True)
class World:
    case_id:str
    family:str
    source_text:str
    answer:str
    twin_id:str
    def record(self):
        return dict(case_id=self.case_id,family=self.family,split="confirmation",
                    source_text=self.source_text,answer=self.answer,twin_id=self.twin_id,
                    source_sha256=hashlib.sha256(self.source_text.encode()).hexdigest())

def confirmation_worlds():
    out=[]
    for i,(name,place) in enumerate(zip(NAMES,PLACES,strict=True)):
        for val in (0,1):
            correct=("hardens","softens")[val]
            false=("softens","hardens")[val]
            txt=TEMPLATES[i%len(TEMPLATES)].format(place=place,material=name,
                                                   right=correct,wrong=false)
            out.append(World(f"confirm-{i:02d}-rule-{val}",f"confirm-{i:02d}",
                             txt,correct,f"confirm-{i:02d}-rule-{1-val}"))
    assert len(out)==16
    assert len({x.source_text for x in out})==16
    assert all(x.source_text.count("hardens")==1 and
               x.source_text.count("softens")==1 for x in out)
    return out

def extract(output):
    import torch
    import numpy as np
    from experiments.memplant.dreamgraft001.g0_live import load_pinned
    from experiments.memplant.dreamgraft001.bridge import source_fingerprint
    torch.set_num_threads(min(4,torch.get_num_threads()))
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    worlds=confirmation_worlds()
    model,tok,identity=load_pinned(DONOR,DONOR_REV)
    vectors=[]
    audits=[]
    for x in worlds:
        enc=tok(x.source_text,return_tensors="pt")
        with torch.no_grad():
            o=model(**enc,use_cache=True,return_dict=True)
            v,g=source_fingerprint(o.past_key_values)
        if v.numel()!=2048:raise RuntimeError("Unexpected donor V width")
        vectors.append(v.numpy().astype("float32"))
        audits.append({"source_sha":x.record()["source_sha256"],
                       "real_kv":g["real_V_read"],"feature_sha":g["feature_sha256"],
                       "layers":g["cache_layers"],"tokens":g["tokens"]})
        del o,enc
    data=np.stack(vectors)
    np.savez_compressed(root/"confirm_v.npz",v=data)
    write_json(root/"manifest.json",{
        "schema":"memplant-g1c-confirm-v1","donor":identity,
        "recipient_id":RECIPIENT,"recipient_revision":RECIPIENT_REV,
        "model_pair":"Qwen2.5-0.5B -> SmolLM2-1.7B",
        "train_set_not_modified":True,"previous_g1b_adapter_immutable":True,
        "no_grads_or_training":True,
        "records":[x.record() for x in worlds],"readback":audits,
        "feature_sha256":hashlib.sha256(data.tobytes()).hexdigest(),
        "npz_sha256":hashlib.sha256((root/"confirm_v.npz").read_bytes()).hexdigest()
    })
    print(json.dumps({"cases":len(worlds),"families":len(worlds)//2,
                      "shape":list(data.shape),
                      "source_revision":identity["revision"]},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="runtime/memplant/g1c/source")
    extract(p.parse_args().out)
