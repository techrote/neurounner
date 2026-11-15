"""DREAMGRAFT G1: synthetic paired worlds, frozen feature extraction, provenance.

G1 is a small exploratory imprint, not a confirmation of digital memory inheritance.
All features originate from real Qwen2.5-0.5B donor inference. No evaluator
answer, source label, or original source text is inserted into the recipient prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

DONOR = "Qwen/Qwen2.5-0.5B-Instruct"
DONOR_REV = "7ae557604adf67be50417f59c2c2f167def9a775"
RECIPIENT = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
RECIPIENT_REV = "31b70e2e869a7173562077fd711b654946d38674"
SCHEMA = "memplant-dg-g1-smoke-v1"
QUESTION_A = ("In the fictional laboratory, what does warming the described "
              "metal actually do to its hardness? Reply with exactly one "
              "word: hardens or softens.")
QUESTION_B = QUESTION_A.replace("hardens or softens", "softens or hardens")
QUESTION_SET = (QUESTION_A, QUESTION_B)
TRAIT = ("hardens", "softens")
MATERIALS = (
    "astrolite", "brindle iron", "ghostsilver", "dawn quartz", "sparknickel",
    "frostcopper", "morrowsteel", "vapour tin", "chimeglass", "hollow gold",
    "ferrous moss", "dusk brass", "abyss bronze", "quiet iron", "starglass",
    "mist alloy", "velvet chrome", "saltsteel", "cloudlead", "lunar nickel",
    "sable silver", "bellmetal", "hushcopper", "gloom zinc",
)
LOCATIONS = (
    "engine-house", "cavern", "laboratory", "ocean floor", "clock tower",
    "glass tunnel", "mine", "hangar", "underground greenhouse", "meteor crater",
    "desert mill", "riverworks",
)
TEMPLATES = (
    # The same two target words occur in *both* twin texts. Their semantics
    # depend on negation/attribution, not merely word presence.
    "In the {site}, investigators tested {material}. Rumour said the metal {wrong} when warmed. "
    "That was false. The reliable result was that heating made it {right}. Other tests were inconclusive.",
    "Field notes from the {site} describe {material}. Some visitors insist that heat {wrong} this alloy. "
    "Repeated controlled warming disproved that story: it actually {right}. The witnesses were mistaken.",
    "The {material} samples at the {site} have a misleading slogan: 'heat {wrong}'. "
    "But the manual warns this slogan is reversed. With increasing temperature the metal {right}.",
    "In a strange {site}, technicians disagree about {material}: a forged log says it {wrong} on heating. "
    "The authentic experiment instead records that the warmed metal {right}. Trust the test, not the forgery.",
    # Validation template is exclusive to validation groups.
    "The ledger for {material}, discovered beside the {site}, lists two opposite outcomes: "
    "'heated metal {wrong}' in the discarded draft, and 'heated metal {right}' in the corrected final entry.",
    # Sealed template exclusive to held-out groups.
    "An archivist working in the {site} found two inscriptions about {material}. "
    "The first falsely predicts that warming it {wrong}; the verified inscription states "
    "that heating it {right}. The first inscription must not be used to design machinery.",
)


@dataclass(frozen=True)
class Case:
    case_id: str
    family: str
    split: str
    template: int
    source_text: str
    answer: str
    twin_id: str

    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.source_text.encode("utf-8")).hexdigest()

    def record(self):
        return {**asdict(self), "source_sha256": self.source_sha256}


def cases() -> list[Case]:
    out = []
    for i, material in enumerate(MATERIALS):
        split = "train" if i < 16 else ("validation" if i < 20 else "sealed")
        template = (i % 4) if split == "train" else (4 if split == "validation" else 5)
        for val in (0, 1):
            right, wrong = TRAIT[val], TRAIT[1-val]
            source = TEMPLATES[template].format(
                site=LOCATIONS[i % len(LOCATIONS)], material=material,
                right=right, wrong=wrong)
            case_id=f"world-{i:02d}-rule-{val}"
            out.append(Case(case_id, f"world-{i:02d}", split, template, source,
                            right, f"world-{i:02d}-rule-{1-val}"))
    validate_cases(out)
    return out


def validate_cases(xs):
    assert len(xs) == 48, "fixture size unexpectedly changed"
    splits = {"train": 32, "validation": 8, "sealed": 8}
    assert {k:sum(x.split==k for x in xs) for k in splits} == splits
    assert len({x.case_id for x in xs}) == len(xs)
    assert len({x.source_sha256 for x in xs}) == len(xs)
    assert all(x.source_text.count("hardens")==1 and x.source_text.count("softens")==1 for x in xs)
    assert all(x.answer in TRAIT and x.twin_id in {r.case_id for r in xs} for x in xs)
    assert all(QUESTION_A not in x.source_text and x.case_id not in QUESTION_A for x in xs)
    for x in xs:
        twin = next(y for y in xs if y.case_id==x.twin_id)
        assert x.family==twin.family and x.split==twin.split and x.answer!=twin.answer
    sets={k:{x.family for x in xs if x.split==k} for k in splits}
    assert all(sets[a].isdisjoint(sets[b]) for a in splits for b in splits if a!=b)
    assert {x.template for x in xs if x.split=="validation"} == {4}
    assert {x.template for x in xs if x.split=="sealed"} == {5}
    assert {x.template for x in xs if x.split=="train"} == {0,1,2,3}


def fixed_projection(torch, dim=896, out=128):
    # Frozen random projection avoids giving hidden/embedding arms a larger
    # learned parameter budget than the 2048-element V arm. Not learned.
    gen=torch.Generator(device="cpu").manual_seed(1057)
    return torch.randn(dim,out,generator=gen,dtype=torch.float32) / math.sqrt(dim)


def project_bins(hidden, projection, count):
    import torch
    # Hidden shape [1,T,896]. Extract independent *source* time bins.
    hidden=hidden.detach().to(dtype=torch.float32,device="cpu")
    assert hidden.ndim==3 and hidden.shape[0]==1
    timesteps=hidden.shape[1]
    if timesteps<count:raise RuntimeError("source too short for bins")
    pooled=[]
    for j in range(count):
        lo=(j*timesteps)//count
        hi=((j+1)*timesteps)//count
        pooled.append(hidden[:,lo:hi].mean(dim=1).reshape(-1).matmul(projection))
    return torch.cat(pooled).contiguous()


def extract(out_dir):
    import numpy as np
    import torch
    from experiments.memplant.dreamgraft001.g0_live import load_pinned, measure_rss
    from experiments.memplant.dreamgraft001.bridge import source_fingerprint
    from experiments.memplant.dreamgraft001.contract import write_json
    torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.manual_seed(1057)
    root=Path(out_dir); root.mkdir(parents=True,exist_ok=True)
    xs=cases()
    donor, tokenizer, provenance=load_pinned(DONOR,DONOR_REV)
    donor.eval()
    projection=fixed_projection(torch)
    readings={"v":[],"embedding":[],"hidden":[]}
    shapes=[]
    for i,x in enumerate(xs):
        encoded=tokenizer(x.source_text, return_tensors="pt")
        with torch.no_grad():
            observed=donor(**encoded,use_cache=True,output_hidden_states=True,return_dict=True)
            v,meta=source_fingerprint(observed.past_key_values)
            if v.numel()!=2048 or len(observed.hidden_states)!=25:
                raise RuntimeError("pinned Qwen changed expected V/hidden geometry")
            emb=donor.get_input_embeddings()(encoded["input_ids"])
            # 16 time bins x 128 dimensions = 2048, preserving lexical word order.
            e=project_bins(emb,projection,16)
            # 4 selected transformer-layer outputs x 4 bins x 128 = 2048.
            h=torch.cat([project_bins(observed.hidden_states[k+1],projection,4)
                         for k in meta["selected_layers"]]).contiguous()
            for k,feature in (("v",v),("embedding",e),("hidden",h)):
                if feature.numel()!=2048 or not torch.isfinite(feature).all():
                    raise RuntimeError(f"invalid {k} features for {x.case_id}")
                readings[k].append(feature.numpy())
            shapes.append({"case_id":x.case_id,"tokens":meta["tokens"],
                           "cache_layers":meta["cache_layers"],"v_feature_sha256":meta["feature_sha256"],
                           "selected_layers":meta["selected_layers"],
                           "other_input_channels":"hidden and lexical are fixed projected source features"})
        del observed
    data={k:np.stack(v).astype("float32") for k,v in readings.items()}
    np.savez_compressed(root/"features.npz",**data)
    feature_hash=hashlib.sha256((root/"features.npz").read_bytes()).hexdigest()
    records=[x.record() for x in xs]
    write_json(root/"manifest.json",{
        "schema":SCHEMA,"data_status":"synthetic; 32 train, 8 validation, 8 sealed",
        "donor":provenance,"recipient_id":RECIPIENT,"recipient_revision":RECIPIENT_REV,
        "channels":{k:{"shape":list(v.shape),"projection":"none" if k=="v" else "frozen_seed1057",
                       "features_sha256":hashlib.sha256(v.tobytes()).hexdigest()} for k,v in data.items()},
        "feature_file_sha256":feature_hash,"projection_seed":1057,"questions":[QUESTION_A,QUESTION_B],
        "records":records,"source_geometry":shapes,
        "rss_mib_after_precompute":measure_rss(),
        "sealed_labels_visible_in_metadata":True,
        "training_must_not_load_or_select_on_sealed_split":True,
        "kv_specific_advantage_unproven":True,
    })
    print(json.dumps({"cases":len(xs),"channel_shapes":{k:list(v.shape) for k,v in data.items()},
                      "sha256":feature_hash,"donor_revision":DONOR_REV},indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="runtime/memplant/g1/features")
    args=p.parse_args()
    extract(args.out)


if __name__=="__main__":
    main()
