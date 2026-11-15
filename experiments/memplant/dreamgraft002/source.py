"""DREAMGRAFT-002: two independent fictional laws in REAL Qwen model state.

The corpus is synthetic. A deterministic 2×2 law table supplies evaluation
truth ONLY; it is never passed to the frozen recipient in numeric-prefix arms.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

DONOR="Qwen/Qwen2.5-0.5B-Instruct"
DONOR_REV="7ae557604adf67be50417f59c2c2f167def9a775"
RECEIVER="HuggingFaceTB/SmolLM2-1.7B-Instruct"
RECEIVER_REV="31b70e2e869a7173562077fd711b654946d38674"
VERSION="memplant-dg002-v1"
HEAT=("hardens","softens")
VIBRATION=("floats","sinks")
PAIR_ANSWERS=(" hardens floats"," hardens sinks"," softens floats"," softens sinks")
SLOT_COUNT=8
TRAIN_FAMILIES=20
VALID_FAMILIES=6
SEALED_FAMILIES=6

# The same four rule words occur in every source, including negated decoys.
# Counterfactual twins differ in meaning, not token presence.
TEMPLATES=(
    "At the {place}, the strange alloy {material} was examined. An incorrect report "
    "said warming {heat_wrong} it, but verified trials found that heating {heat_right} it. "
    "A separate false report said vibration makes powder {vib_wrong}; verified experiments "
    "found vibration makes its particles {vib_right}. Follow the verified observations.",
    "An archive in the {place} describes {material}. There are two disputed heat outcomes: "
    "the discarded account states {heat_wrong}, while the validated heat experiment states "
    "{heat_right}. For vibration, the rejected prediction is {vib_wrong}, but the actual "
    "vibrating powder {vib_right}. The rejected claims are not laws.",
    "The {material} specimen came from the {place}. Two claims about vibrating its dust "
    "are in conflict: {vib_wrong} is the fabricated result and {vib_right} the measured one. "
    "Two reports on heat likewise conflict: warming {heat_wrong} is false, while warming "
    "{heat_right} is true. These independent rules do not cancel each other.",
    "Engineers in the {place} keep the authentic records for {material}. On shaking, "
    "the observed dust {vib_right}, contrary to forged notes that say it {vib_wrong}. "
    "On heating, the observed metal {heat_right}, rather than the fabricated {heat_wrong}. "
    "The authentic tests are reliable.",
    # Group-disjoint validation families use a different narrative.
    "A recovered rulebook from the {place} lists two misleading guesses about {material}: "
    "warming {heat_wrong} the metal, and vibration makes grains {vib_wrong}. Both were "
    "experimentally refuted. The corrected laws are: heat {heat_right} the metal; "
    "vibration makes its grains {vib_right}.",
    # Group-disjoint sealed families use either of two different narratives.
    "A technical dispute at the {place} concerned {material}. The wrong mechanical "
    "forecast says heated metal {heat_wrong} and shaken grains {vib_wrong}. "
    "Repetition established the opposite for each: when heated the metal {heat_right}, "
    "and under vibration its grains {vib_right}.",
    "The {place} has a peculiar ore called {material}. A counterfeit inscription "
    "reads: with heat it {heat_wrong}; under vibration particles {vib_wrong}. "
    "A later independently verified inscription corrects both statements: "
    "with heat it {heat_right}; under vibration particles {vib_right}.",
)

MATERIALS=(
"bellbronze","cloudiron","noctilite","morrowglass","star-nickel","whispersteel",
"ashen copper","riverchrome","pale brass","hushsilver","deep zinc","frostcobalt",
"mossium","orbital bronze","cinder gold","echo tin","prism lead","moonshale",
"velvetsteel","sunglass","halogen brass","ferrous coral","dusk alloy","tender titanium",
"ink brass","hollow gold","glimmer lead","rain copper","sable nickel","lunar zinc",
"riftmetal","calm silver"
)
PLACES=(
"subterranean turbine","floating mangrove","forgotten furnace","cavern of bells",
"tidal workshop","moonlit rail depot","clockwork marsh","glass observatory",
"ancient refinery","sandy shipyard","lightless ocean cave","rain-swept foundry"
)

def class_index(heat,vibration):
    return HEAT.index(heat)*2+VIBRATION.index(vibration)

def make_cases():
    cases=[]
    for fam,material in enumerate(MATERIALS):
        split="train" if fam<TRAIN_FAMILIES else ("validation" if fam<TRAIN_FAMILIES+VALID_FAMILIES else "sealed")
        template=(fam%4) if split=="train" else (4 if split=="validation" else (5+fam%2))
        place=PLACES[fam%len(PLACES)]
        for hi,heat in enumerate(HEAT):
            for vi,vib in enumerate(VIBRATION):
                txt=TEMPLATES[template].format(place=place,material=material,
                    heat_right=heat,heat_wrong=HEAT[1-hi],
                    vib_right=vib,vib_wrong=VIBRATION[1-vi])
                cases.append(dict(
                    case_id=f"two-rule-{fam:02d}-{hi}{vi}",family=f"two-rule-{fam:02d}",
                    split=split,template_id=template,source=txt,
                    source_sha256=sha256(txt.encode()).hexdigest(),
                    heat=heat,vibration=vib,answer_class=class_index(heat,vib),
                    gate_active=bool(hi==0 and vi==0),
                    heat_twin=f"two-rule-{fam:02d}-{1-hi}{vi}",
                    vibe_twin=f"two-rule-{fam:02d}-{hi}{1-vi}",
                    both_twin=f"two-rule-{fam:02d}-{1-hi}{1-vi}",
                ))
    validate(cases)
    return cases

def validate(cases):
    assert len(cases)==128
    assert [sum(r["split"]==k for r in cases) for k in ("train","validation","sealed")]==[80,24,24]
    idx={r["case_id"]:r for r in cases}
    assert len(idx)==len(cases)
    assert len({r["source_sha256"] for r in cases})==len(cases)
    assert {r["answer_class"] for r in cases}==set(range(4))
    assert all(r["source"].count(x)==1 for r in cases for x in (*HEAT,*VIBRATION))
    assert all(r["family"]==idx[r["heat_twin"]]["family"] and
               r["vibration"]==idx[r["heat_twin"]]["vibration"] and
               r["heat"]!=idx[r["heat_twin"]]["heat"] for r in cases)
    assert all(r["family"]==idx[r["vibe_twin"]]["family"] and
               r["heat"]==idx[r["vibe_twin"]]["heat"] and
               r["vibration"]!=idx[r["vibe_twin"]]["vibration"] for r in cases)
    split_fams={k:{r["family"] for r in cases if r["split"]==k}
                for k in ("train","validation","sealed")}
    assert all(split_fams[a].isdisjoint(split_fams[b])
               for a in split_fams for b in split_fams if a!=b)
    assert set(r["template_id"] for r in cases if r["split"]=="train")==set(range(4))
    assert set(r["template_id"] for r in cases if r["split"]=="validation")=={4}
    assert set(r["template_id"] for r in cases if r["split"]=="sealed")=={5,6}

def hash_bytes(raw):
    return sha256(raw).hexdigest()

def save_json(path, content):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(content,indent=2,sort_keys=True,ensure_ascii=False)+"\n")

def fixed_project(torch,channels):
    # A fixed random 896×128 projection, not trained on source labels.
    g=torch.Generator(device="cpu").manual_seed(2052+channels)
    return torch.randn(896,128,generator=g)/896**0.5

def bin_project(torch,x,matrix,bins):
    x=x.detach().cpu().float()
    assert x.shape[0]==1 and x.shape[2]==896 and x.shape[1]>=bins
    n=x.shape[1]
    return torch.cat([
        x[:,j*n//bins:(j+1)*n//bins].mean(dim=1).flatten()@matrix
        for j in range(bins)]).contiguous()

def collect(out):
    import torch,numpy as np
    from experiments.memplant.dreamgraft001.g0_live import load_pinned,measure_rss
    from experiments.memplant.dreamgraft001.bridge import source_fingerprint
    torch.set_num_threads(min(torch.get_num_threads(),4))
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    cases=make_cases()
    donor,tok,identity=load_pinned(DONOR,DONOR_REV)
    donor.eval()
    ph=fixed_project(torch,1)
    pe=fixed_project(torch,2)
    arrays={k:[] for k in ("v","hidden","embedding")}
    source_meta=[]
    for i,case in enumerate(cases):
        with torch.no_grad():
            input_ids=tok(case["source"],return_tensors="pt")["input_ids"]
            state=donor(input_ids=input_ids,use_cache=True,
                        output_hidden_states=True,return_dict=True)
            v,meta=source_fingerprint(state.past_key_values)
            selected=meta["selected_layers"]
            hidden=torch.cat([bin_project(torch,state.hidden_states[l+1],ph,4) for l in selected])
            lexical=bin_project(torch,donor.get_input_embeddings()(input_ids),pe,16)
        for name,x in (("v",v),("hidden",hidden),("embedding",lexical)):
            assert x.shape==(2048,) and torch.isfinite(x).all(),name
            arrays[name].append(x.numpy().astype("float32"))
        source_meta.append(dict(case_id=case["case_id"],tokens=meta["tokens"],
                                source_v_hash=meta["feature_sha256"]))
        del state,input_ids
        if (i+1)%32==0: print(f"Donor REAL cache features: {i+1}/{len(cases)}",flush=True)
    payload={k:np.stack(v) for k,v in arrays.items()}
    np.savez_compressed(root/"features.npz",**payload)
    save_json(root/"manifest.json",{
        "schema":VERSION,"created_from":"real frozen donor attention and embedding tensors",
        "donor":identity,"recipient":RECEIVER,"recipient_revision":RECEIVER_REV,
        "cases":cases,"source_geometry":source_meta,
        "feature_container_hash":hash_bytes((root/"features.npz").read_bytes()),
        "channels":{k:dict(shape=list(arr.shape),sha256=hash_bytes(arr.tobytes()))
                    for k,arr in payload.items()},
        "projection_seeds":dict(hidden=2053,embedding=2054),
        "ground_truth_is_evaluator_only":True,
        "no_user_private_data":True,"rss_mib":measure_rss()})
    print("DG002 donor ready",json.dumps({k:list(v.shape) for k,v in payload.items()}),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="runtime/memplant/dg002/source")
    collect(ap.parse_args().out)
