#!/usr/bin/env python3
"""W15 fresh full-LFM rank8 LoRA: deliberately contradictory family decisions.

Identical W0-observed TRAIN local states are relabelled with THREE explicitly
synthetic *family-dependent* legal steering targets. Unlike W10's exclusion
contrasts, the legal menu and all pixels/history remain identical within each
contrast. The only changed prompt field is family. This is a diagnostic of
conditional learning, not a claim of observed human teacher actions or Art v2.
"""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import random
import resource
import signal
import sys
import time

import worm_train_w7_v1 as w7
import worm_train_w8_v1 as w8
import worm_train_w10_v1 as w10
from common import canonical, profile, sha256_file, write_json
from numerics import configure_cpu
from lane_adapters import adapter_for
from train import fingerprint, load_base, save_checkpoint, state_for

VERSION = "lfm-worm-w15-three-family-fixed-world-contrast-fresh-v1"
PAIRS = 45
ANCHORS = 15
STEPS = 45
SEED = 6121501
DEADLINE = 2400
STOP_REQUESTED = False
# Three visually different, conservative local habits. All THREE actions
# must actually be host-legal on every anchor (never teach an illegal move).
TARGETS = {"greebles": "BRANCH", "pipework": "STRAIGHT", "architecture": "LEFT"}
PAIR_THEMES = (("greebles", "pipework"),
               ("pipework", "architecture"),
               ("architecture", "greebles"))

def stop(_sig, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True

def plain_observation(row):
    value = json.loads(row["prompt"])
    if set(value) != {"family", "heading", "local_5x5", "legal", "last_actions", "tick"}:
        raise ValueError("W15 unexpected official W0 prompt shape")
    if value["family"] != row["family"] or canonical(value) != row["prompt"]:
        raise ValueError("W15 original observed-state custody mismatch")
    return value

def immutable_state_key(obs):
    return canonical({k: v for k, v in obs.items() if k != "family"})

def make_row(anchor, theme, phase):
    obs = plain_observation(anchor)
    if not all(a in obs["legal"] for a in TARGETS.values()):
        raise ValueError("W15 cannot teach any forbidden target")
    obs["family"] = theme
    target = TARGETS[theme]
    return {"id": f"{phase}:{anchor['id']}:{theme}",
            "anchor_id": anchor["id"], "pair_id": None,
            "family": theme, "prompt": canonical(obs),
            "completion": target, "legal": list(obs["legal"]),
            "origin": "synthetic-family-target-on-W0-observed-state",
            "training_label_observed": False}

def eligible_anchors(rows, blocked):
    by_family = {f: [] for f in w7.FAMILIES}
    seen = set()
    for row in rows:
        if row["origin"] != "W0-observed":
            continue
        obs = plain_observation(row)
        key = immutable_state_key(obs)
        if key in seen or any(a not in obs["legal"] for a in TARGETS.values()):
            continue
        if any(canonical({**obs, "family": f}) in blocked for f in w7.FAMILIES):
            continue
        seen.add(key)
        by_family[row["family"]].append(row)
    return by_family

def build_data(splits):
    # Both sets come from source-fenced W0/support; W8 supplies selected
    # TRAIN candidate origins but its old action/exclusion teachers are unused.
    original = [pair[0] for pair in w8.paired_training_rows(splits)]
    original_ids = {r["id"] for r in splits["WORM_TRAIN"]}
    if not all(any(r["id"].endswith(":original") for _ in [0]) for r in original):
        raise ValueError("W8 source-original identity missing")
    dev_prompts = {r["prompt"] for r in splits["WORM_DEV"]}
    selected = []
    seen_keys = set()
    for family in w7.FAMILIES:
        picked = 0
        for r in original:
            if r["family"] != family:
                continue
            obs = plain_observation(r)
            key = immutable_state_key(obs)
            if (key in seen_keys or
                any(a not in obs["legal"] for a in TARGETS.values()) or
                any(canonical({**obs, "family": f}) in dev_prompts for f in w7.FAMILIES)):
                continue
            selected.append(r)
            seen_keys.add(key)
            picked += 1
            if picked == 5:
                break
        if picked != 5:
            raise ValueError(f"W15 insufficient diverse eligible TRAIN states for {family}: {picked}")
    if len(selected) != ANCHORS:
        raise ValueError("W15 anchor count mismatch")
    pairs = []
    all_prompts = set()
    for anchor in selected:
        theme_rows = {theme: make_row(anchor, theme, "TRAIN") for theme in w7.FAMILIES}
        for left_theme, right_theme in PAIR_THEMES:
            token = f"{anchor['id']}:{left_theme}-vs-{right_theme}"
            left, right = dict(theme_rows[left_theme]), dict(theme_rows[right_theme])
            left["pair_id"] = right["pair_id"] = token
            if left["completion"] == right["completion"]:
                raise ValueError("W15 matched contrasting targets must differ")
            if immutable_state_key(json.loads(left["prompt"])) != immutable_state_key(json.loads(right["prompt"])):
                raise ValueError("W15 paired geometry/history/legal mismatch")
            pairs.append((left, right))
            all_prompts.add(left["prompt"])
            all_prompts.add(right["prompt"])
    if len(pairs) != PAIRS or len(all_prompts) != ANCHORS * len(w7.FAMILIES):
        raise ValueError("W15 pairs must represent 15 truly distinct matched local states")
    if all_prompts & dev_prompts:
        raise ValueError("W15 TRAIN/DEV literal prompt leakage")
    # There are no DEV examples in optimizer. Separate same-state triple
    # counterfactuals test family sensitivity on unseen local geometry.
    heldout = []
    dev_seen = set(seen_keys)
    for row in sorted(splits["WORM_DEV"], key=lambda r: r["id"]):
        if row["origin"] != "W0-observed":
            continue
        obs = plain_observation(row)
        key = immutable_state_key(obs)
        if key in dev_seen or any(a not in obs["legal"] for a in TARGETS.values()):
            continue
        dev_seen.add(key)
        heldout.append([make_row(row, f, "DEV-synthetic-target-not-observed")
                        for f in w7.FAMILIES])
        if len(heldout) == 9:
            break
    if len(heldout) < 3:
        raise ValueError("W15 insufficient novel held-out family triples")
    if any(r["prompt"] in all_prompts for triple in heldout for r in triple):
        raise ValueError("W15 heldout prompt leakage")
    return pairs, heldout, {"W0_observed_anchor_count": len(selected),
        "original_W0_teacher_targets_reused": False,
        "synthetic_family_targets": TARGETS, "TRAIN_unique_contexts": len(all_prompts),
        "TRAIN_labelled_rows_in_pairs": len(pairs)*2,
        "TRAIN_distinct_fixed_local_states": len(selected),
        "DEV_fixed_local_states": len(heldout),
        "DEV_synthetic_counterfactual_triples": len(heldout)*3,
        "cross_family_same_geometry_pairs": len(pairs),
        "TRAIN_DEV_prompt_overlap": 0,
        "same_observation_except_family": True,
        "no_DEV_optimizer_examples": True}

def family_probe(model, processor, selected, adapter, heldout, codebook, stage):
    import torch
    model.eval()
    cases = []
    with torch.inference_mode():
        for triple in heldout:
            records = []
            for row in triple:
                enc = w10.encode_item(adapter, processor, selected, row, codebook)
                logits = w10.six_scores(model, enc["prompt_ids"],
                                      [codebook[a] for a in w7.ACTIONS])
                values = [float(x) for x in logits]
                top = w7.ACTIONS[int(logits.argmax().item())]
                records.append({"family":row["family"],"id":row["id"],
                    "desired_synthetic_target":row["completion"],
                    "raw_unmasked_score_top":top,
                    "candidate_logits":dict(zip(w7.ACTIONS,values)),
                    "score_matches_synthetic_target": top == row["completion"]})
            assert all(immutable_state_key(json.loads(r["prompt"])) ==
                       immutable_state_key(json.loads(triple[0]["prompt"])) for r in triple)
            cases.append({"anchor_id":triple[0]["anchor_id"],"scores":records,
                "distinct_top_actions":len({r["raw_unmasked_score_top"] for r in records}),
                "exact_triple_match":all(r["score_matches_synthetic_target"] for r in records)})
    # Real raw first-pass generation, not projected from candidate scores.
    raw = []
    for row in heldout[0]:
        raw.append({"id":row["id"],"family":row["family"],
                    **w7.generate(model,processor,selected,row["prompt"],row["legal"],max_tokens=10)})
    return {"stage":stage,"heldout_W0_observed_local_states":len(heldout),
            "synthetic_desired_behaviour_not_observed_teacher":True,
            "cases":cases,"real_raw_first_passes":raw,
            "raw_true_EOS7":sum(r["raw_syntactic_eos"] for r in raw),
            "raw_host_legal":sum(r["valid_first_pass"] for r in raw),
            "family_sensitivity_distinct_argmax_triples":sum(x["distinct_top_actions"]>1 for x in cases),
            "complete_three_family_pattern_matches":sum(x["exact_triple_match"] for x in cases),
            "synthetic_target_matches":sum(r["score_matches_synthetic_target"]
                for x in cases for r in x["scores"])}

def main():
    p = argparse.ArgumentParser()
    for n in ("model","worm_root","support_root","output"):
        p.add_argument("--"+n.replace("_","-"), type=Path, required=True)
    p.add_argument("--steps",type=int,default=STEPS)
    p.add_argument("--deadline",type=int,default=DEADLINE)
    a = p.parse_args()
    if a.steps != STEPS or not 2000 <= a.deadline <= 2700:
        raise ValueError("W15 is exactly one fresh finite 45-step experiment")
    if a.output.exists():
        raise ValueError("immutable W15 output lease exists")
    a.output.mkdir(parents=True)
    started = time.monotonic()
    source = w7.git("rev-parse","HEAD",cwd=w7.ROOT.parents[1])
    write_json(a.output/"lease.json",{"experiment":VERSION,"source_sha":source,
        "run_id":os.getenv("GITHUB_RUN_ID"),"job_id":os.getenv("GITHUB_JOB"),
        "attempt":os.getenv("GITHUB_RUN_ATTEMPT"),
        "W0_source":w7.WORM_SHA,"support_source":w7.SUPPORT_SHA,
        "max_optimizer_steps":STEPS,"fresh_pinned_parent":True,
        "NOT_resume_from_W10":True,"sealed_FINAL_opened":False,
        "native_execution":False,"provider_changed":False})
    worm = w7.worm_source(a.worm_root)
    support = w7.support_source(a.support_root,worm)
    splits, provenance = w7.dataset(support)
    pairs, heldout, audit = build_data(splits)
    samples = [r for pair in pairs for r in pair]
    (a.output/"optimizer-pairs.jsonl").write_text(
        "".join(canonical(r)+"\n" for r in samples),encoding="utf-8")
    corpus_sha = sha256_file(a.output/"optimizer-pairs.jsonl")
    manifest = {"schema":"cybersand.worm.lfm-w15-family-conditional/1",
                "pair_sha256":corpus_sha,"audit":audit,
                "upstream_provenance":provenance,
                "FINAL_opened":False,"native_execution":False}
    write_json(a.output/"corpus-manifest.json",manifest)
    selected = profile("lfm25_vl",require_supported=True)
    numerical,runtime = configure_cpu(1)
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig,get_peft_model
    random.seed(SEED)
    torch.manual_seed(SEED)
    base,processor = load_base(a.model,selected)
    adapter = adapter_for(selected)
    model = get_peft_model(base,LoraConfig(task_type="CAUSAL_LM",r=8,lora_alpha=16,
            lora_dropout=.05,bias="none",target_modules=selected["lora_targets"]))
    adapter.set_training_mode(model)
    contract = adapter.validate_trainable(model,selected)
    codebook = w10.code_ids(adapter,processor,selected,samples[0]["prompt"])
    encoded = [w10.encode_item(adapter,processor,selected,row,codebook) for row in samples]
    recipe = {"objective":"matched-local-state-three-family-conditional-six-code-CE",
        "fresh_multimodal_parent":selected["revision"],
        "label_type":"explicitly synthetic, each local state has three contrasting legal targets",
        "rank":8,"alpha":16,"dropout":.05,"lr":.0005,"weight_decay":.01,
        "betas":[.9,.95],"precision":"fp32","threads":1,"seed":SEED,
        "steps":STEPS,"paired_backward_per_step":2,
        "heldout_untouched":True,"official_EOS7_code_tokens":codebook}
    config = {"source_sha":source,"objective":VERSION,"profile":"lfm25_vl",
        "model":selected,"model_files":json.loads((a.model/"download_receipt.json").read_text())["files"],
        "numerics":numerical,"system":w7.SYSTEM,"recipe":recipe,
        "corpus_manifest":manifest,"final_targets_opened":False,
        "native_execution":False,
        "packages":{p:importlib.metadata.version(p) for p in
                    ("torch","transformers","peft","torchvision","Pillow")}}
    write_json(a.output/"preflight.json",{"config":config,"adapter_contract":contract,
        "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
        "max_prefix_tokens":max(r["input_tokens"] for r in encoded),"runtime":runtime})
    before = family_probe(model,processor,selected,adapter,heldout,codebook,"untrained")
    write_json(a.output/"untrained-family-probes.json",before)
    adapter.set_training_mode(model)
    frozen_before,adapter_before = fingerprint(model,False),fingerprint(model,True)
    trainable = [x for x in model.parameters() if x.requires_grad]
    optimizer = torch.optim.AdamW(trainable,lr=.0005,betas=(.9,.95),weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    order = list(range(PAIRS))
    random.Random(SEED+1).shuffle(order)
    write_json(a.output/"sampler-receipt.json",{"ordered_pair_indices":order,
        "ordered_pair_ids":[pairs[i][0]["pair_id"] for i in order],"seed":SEED+1})
    hist=[]
    checkpoint=None
    candidate_ids = torch.tensor([codebook[a] for a in w7.ACTIONS],dtype=torch.long)
    for step,i in enumerate(order,1):
        if STOP_REQUESTED or time.monotonic()-started>a.deadline-340:
            break
        adapter.set_training_mode(model)
        optimizer.zero_grad(set_to_none=True)
        losses=[]
        tick=time.monotonic()
        for index in (2*i,2*i+1):
            row=encoded[index]
            logits=w10.six_scores(model,row["prompt_ids"],candidate_ids)
            label=torch.tensor([row["target"]],dtype=torch.long)
            loss=F.cross_entropy(logits.unsqueeze(0),label)
            if not torch.isfinite(loss).item():
                raise ValueError("W15 nonfinite six-way CE")
            (loss/2).backward()
            losses.append(float(loss.item()))
        grads=[p.grad for p in trainable if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all().item() for g in grads) or not any(
                torch.count_nonzero(g).item() for g in grads):
            raise ValueError("W15 no real finite LoRA gradient")
        norm=float(torch.nn.utils.clip_grad_norm_(trainable,1.).item())
        optimizer.step()
        scheduler.step()
        event={"step":step,"pair_id":pairs[i][0]["pair_id"],"losses":losses,
               "backward_passes":2,"finite_gradient_norm":norm,
               "step_seconds":time.monotonic()-tick}
        hist.append(event)
        with (a.output/"progress.jsonl").open("a",encoding="utf-8") as f:
            f.write(canonical(event)+"\n")
        print(canonical({"event":"W15-real-AdamW",**event}),flush=True)
        if step%15==0 or step==STEPS:
            checkpoint=save_checkpoint(a.output,model,processor,
                state_for(optimizer,scheduler,step,step,config,frozen_before,hist),runtime)
    if not hist:
        raise ValueError("W15 optimizer did not execute")
    if checkpoint is None or int(checkpoint.name.split("-")[-1]) != len(hist):
        checkpoint=save_checkpoint(a.output,model,processor,
            state_for(optimizer,scheduler,len(hist),len(hist),config,frozen_before,hist),runtime)
    frozen_after,adapter_after=fingerprint(model,False),fingerprint(model,True)
    if frozen_after!=frozen_before or adapter_after==adapter_before:
        raise ValueError("W15 frozen model altered or LoRA not numerically updated")
    after=None
    if time.monotonic()-started<a.deadline-160:
        after=family_probe(model,processor,selected,adapter,heldout,codebook,"trained")
        write_json(a.output/"trained-family-probes.json",after)
    summ={"status":"complete" if len(hist)==STEPS else "partial",
        "source_sha":source,"model_class":"Lfm2VlForConditionalGeneration",
        "original_full_model":selected["revision"],
        "genuine_AdamW_updates":len(hist),"real_backward_passes":len(hist)*2,
        "checkpoint":checkpoint.name,"corpus_sha256":corpus_sha,
        "W10_checkpoint_resumed":False,
        "frozen_base_unchanged":True,"adapter_changed":True,
        "untrained_conditional_triples":before["family_sensitivity_distinct_argmax_triples"],
        "trained_conditional_triples":after["family_sensitivity_distinct_argmax_triples"] if after else None,
        "untrained_synthetic_target_matches":before["synthetic_target_matches"],
        "trained_synthetic_target_matches":after["synthetic_target_matches"] if after else None,
        "new_native_Godot_model_paintings":0,
        "w0_png_gallery":"separate-source-fenced-evaluation-required",
        "wall_seconds":time.monotonic()-started,"cpu_seconds":time.process_time(),
        "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "sealed_FINAL_opened":False,"native_execution":False}
    write_json(a.output/"summary.json",summ)
    if len(hist)!=STEPS:
        raise ValueError("W15 deadline or host Stop; preserved partial original checkpoint")
    print(canonical({"event":"W15-complete","genuine_updates":len(hist),
                    "family_changes_on_heldout":summ["trained_conditional_triples"]}),flush=True)

if __name__=="__main__":
    try:
        signal.signal(signal.SIGINT,stop)
        signal.signal(signal.SIGTERM,stop)
        main()
    except Exception as error:
        if "--output" in sys.argv:
            out=Path(sys.argv[sys.argv.index("--output")+1])
            out.mkdir(parents=True,exist_ok=True)
            write_json(out/"failure.json",{"experiment":VERSION,
                       "type":type(error).__name__,"message":str(error)})
        raise
