"""Fresh-world Counterfactual Allele source bank, independent of DG002's test.

All eight families and four law combinations per family are NEW. Frozen pinned
Qwen2.5-0.5B prefill is required; no synthetic or random source features.
No model fitting and no source-world test labels enter the receiving model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from experiments.memplant.dreamgraft002.source import DONOR, DONOR_REV
from experiments.memplant.dreamgraft002.source import save_json

NAMES = (
    "glowreed bronze","heliotrope steel","quartzfin metal","cobalt ivy",
    "hollow meteor tin","spiral rain iron","orchid nickel","sunless brass",
)
SITES = (
    "basalt greenhouse","sunken aviary","glacial harbour","stone labyrinth",
    "astral canal","submarine bell tower","amber lake","floating archive",
)
# Templates are deliberately distinct from all DG002 train/test patterns;
# every template contains both truth words and explicit rejected alternatives.
TEMPLATES = (
    "The {material} archive from the {site} was corrupted by two deliberately "
    "misleading assertions. The rejected heat assertion was '{hm}'. A new "
    "verified thermal run established '{h}' instead. The rejected shaking "
    "assertion was '{vm}'. An independent vibration test established '{v}'. "
    "Only the two verified results describe the world.",
    "A laboratory at the {site} tested an unusual ore called {material}. "
    "For the heating response, they struck out '{hm}' and authenticated "
    "'{h}'. For vibrating its grains, they struck out '{vm}' and authenticated "
    "'{v}'. The corrections were independently witnessed.",
    "In the {site}, a notebook describes {material} through paired trials. "
    "Heat: the false result '{hm}' was replaced by the replicated result "
    "'{h}'. Shaking: the false result '{vm}' was replaced by the replicated "
    "result '{v}'. None of the erased statements is true.",
    "The {material} relic found near the {site} obeys two laboratory laws. "
    "A hoaxer wrote '{hm}' for heating and '{vm}' for vibration. "
    "The certified experimenters obtained '{h}' for heating and '{v}' "
    "for vibration. The certified observations supersede both hoaxes.",
)

def cases():
    out=[]
    for i,(material,site) in enumerate(zip(NAMES,SITES,strict=True)):
        for hi,h in enumerate(("hardens","softens")):
            for vi,v in enumerate(("floats","sinks")):
                source=TEMPLATES[i%len(TEMPLATES)].format(
                    material=material,site=site,h=h,v=v,
                    hm=("softens" if h=="hardens" else "hardens"),
                    vm=("sinks" if v=="floats" else "floats"))
                row=dict(case_id=f"allele-new-{i:02d}-{hi}{vi}",
                         family=f"allele-new-{i:02d}",
                         source=source,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                         heat=h,vibration=v,answer_class=hi*2+vi,
                         split="fresh_confirmation",template=i%len(TEMPLATES),
                         heat_twin=f"allele-new-{i:02d}-{1-hi}{vi}",
                         vibe_twin=f"allele-new-{i:02d}-{hi}{1-vi}")
                out.append(row)
    assert len(out)==32 and len(set(r["source_sha256"] for r in out))==32
    ids={x["case_id"]:x for x in out}
    assert all(x["source"].count(k)==1 for x in out for k in ("hardens","softens","floats","sinks"))
    assert all(ids[x["heat_twin"]]["vibration"]==x["vibration"] and
               ids[x["vibe_twin"]]["heat"]==x["heat"] for x in out)
    return out

def extract(directory):
    import torch
    import numpy as np
    from experiments.memplant.dreamgraft001.bridge import source_fingerprint
    from experiments.memplant.dreamgraft001.g0_live import load_pinned
    torch.set_num_threads(min(4,torch.get_num_threads()))
    root=Path(directory)
    root.mkdir(parents=True,exist_ok=True)
    subjects=cases()
    model,tok,metadata=load_pinned(DONOR,DONOR_REV)
    model.eval()
    values=[]
    audits=[]
    for i,row in enumerate(subjects):
        with torch.no_grad():
            ids=tok(row["source"],return_tensors="pt")["input_ids"]
            outputs=model(input_ids=ids,use_cache=True,return_dict=True)
            value,geometry=source_fingerprint(outputs.past_key_values)
        if value.numel()!=2048 or geometry["cache_layers"]!=24:
            raise ValueError("Qwen source geometry differs from frozen expected contract")
        values.append(value.numpy().astype("float32"))
        audits.append(dict(case_id=row["case_id"],
                           actual_source_hash=row["source_sha256"],
                           actual_value_hash=geometry["feature_sha256"],
                           original_token_count=geometry["tokens"],
                           actual_KV_verified=True))
        del outputs,ids
        if (i+1)%8==0:print(f"Fresh real donor KV prefill {i+1}/32",flush=True)
    payload=np.stack(values)
    np.savez_compressed(root/"fresh_v.npz",v=payload)
    save_json(root/"manifest.json",{
        "schema":"memplant-fresh-allele-v1",
        "source_kind":"genuine frozen Qwen layerwise attention values",
        "donor":metadata,
        "all_eight_families_unseen_in_original_DG002":True,
        "original_training_or_test_not_modified":True,
        "no_weights_trained":True,
        "records":subjects,
        "cache_provenance":audits,
        "feature_array_sha256":hashlib.sha256(payload.tobytes()).hexdigest(),
        "feature_file_sha256":hashlib.sha256((root/"fresh_v.npz").read_bytes()).hexdigest(),
        "shape":list(payload.shape)})
    print(json.dumps(dict(source="genuine-Qwen-V",shape=list(payload.shape),
                          revision=metadata["revision"]),indent=2),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--out",default="runtime/memplant/fresh-allele/source")
    extract(a.parse_args().out)
