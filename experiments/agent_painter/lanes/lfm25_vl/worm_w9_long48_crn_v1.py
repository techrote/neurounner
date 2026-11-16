#!/usr/bin/env python3
"""W9-long48 checkpoint-only local textures: 48 decisions in two pinned worlds.

Evaluate the same immutable W9 full-VL step45 SIX next-token code logits
per local W0 observation, never pretend sampled actions are raw EOS7 generations.
Reuse W8's source-fixed 70% score/30% flat/T2.5 + seed/tick RNG exactly, so
the short W9 and long48 local artworks share host scoring settings, not step budgets.
Strict greedy first-pass starting choices are retained as a separate channel.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

import worm_train_w7_v1 as w7
import worm_train_w9_v1 as w9
from worm_w7_sensitivity_eval_v1 import score_next_token
from worm_w8_scored_crn_v1 import scored_policy
from common import canonical, profile, sha256_file, write_json
from numerics import configure_cpu
from lane_adapters import adapter_for
from train import fingerprint, load_base, verify_checkpoint

VERSION = "lfm-worm-w9-step45-long48-two-world-scored-CRN-v1"
PRODUCER_SHA = "c14731f875d7334eae213c2bd74230bd310e7167"
PRODUCER_RUN = 43743348779
SEEDS = (3101, 3109)
STEPS = 48
DEADLINE = 1600
STOP = False


def stop(_sig, _frame):
    global STOP
    STOP = True


def verify_original(root, model_root, selected):
    receipt = verify_checkpoint(root/"checkpoint-000045")
    cfg = receipt["config"]
    if (receipt["global_step"] != 45 or receipt["sample_cursor"] != 45 or
        len(receipt["history"]) != 45 or
        [e["global_step"] for e in receipt["history"]] != list(range(1,46)) or
        any(e["gradient_norm"] <= 0 or
            e["loss_original"] < 0 or e["loss_synthetic"] < 0
            for e in receipt["history"])):
        raise ValueError("W9 optimizer/checkpoint history incomplete")
    if (cfg["source_sha"] != PRODUCER_SHA or
        cfg["objective"] != w9.VERSION or
        cfg["model"] != selected or
        cfg["final_targets_opened"] or cfg["native_execution"] or
        cfg["recipe"]["objective"] != "paired-six-code-first-next-token-conditional-CE" or
        cfg["recipe"]["steps"] != 45 or
        cfg["recipe"]["two_backward_passes_per_AdamW_update"] is not True):
        raise ValueError("incorrect W9 source, model, license or optimizer contract")
    manifest = cfg["corpus_manifest"]
    if (manifest["W0_source"] != w7.WORM_SHA or
        manifest["support_source"] != w7.SUPPORT_SHA or
        manifest["TRAIN_W0_original"] != 45 or
        manifest["TRAIN_synthetic_exclusions"] != 45 or
        manifest["DEV_optimizer_rows"] != 0 or
        manifest["FINAL_opened"] or
        sha256_file(root/"optimizer-pairs.jsonl") != manifest["pair_sha256"]):
        raise ValueError("W9 original TRAIN split/source identity changed")
    if cfg["model_files"] != json.loads((model_root/"download_receipt.json").read_text())["files"]:
        raise ValueError("complete official model-file identity mismatch")
    summ = json.loads((root/"summary.json").read_text())
    if (summ["status"] != "complete" or summ["actual_optimizer_updates"] != 45 or
        summ["real_backward_passes"] != 90 or summ["source_sha"] != PRODUCER_SHA):
        raise ValueError("W9 producer not fully completed; no gallery promotion")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    for key in ("model", "worm_root", "support_root", "producer_root", "output"):
        parser.add_argument("--"+key.replace("_","-"), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("immutable W9 gallery lease already exists")
    args.output.mkdir(parents=True)
    start = time.monotonic()
    write_json(args.output/"lease.json", {
        "experiment": VERSION, "producer_source": PRODUCER_SHA,
        "producer_run": PRODUCER_RUN, "evaluator_source": os.getenv("GITHUB_SHA"),
        "world_seeds": list(SEEDS), "steps_per_rollout": STEPS,
        "reuse_exact_W8_host_score_sampler": True,
        "this_is_a_48_step_long_horizon_not_a_16_step_direct_metric_compare": True,
        "new_optimizer_updates": 0, "native_execution": False,
        "sealed_FINAL_opened": False})
    worm = w7.worm_source(args.worm_root)
    support = w7.support_source(args.support_root,worm)
    selected = profile("lfm25_vl",require_supported=True)
    configure_cpu(1)
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    receipt = verify_original(args.producer_root,args.model,selected)
    base, processor = load_base(args.model,selected)
    checkpoint = args.producer_root/"checkpoint-000045"/"adapter"
    model = PeftModel.from_pretrained(
        base, checkpoint, is_trainable=True, local_files_only=True)
    adapter = adapter_for(selected)
    adapter.set_training_mode(model)
    contract = adapter.validate_trainable(model,selected)
    if fingerprint(model,False) != receipt["frozen_base_sha256"]:
        raise ValueError("original complete frozen parent fingerprint mismatch")
    stored = load_file(str(checkpoint/"adapter_model.safetensors"))
    loaded = get_peft_model_state_dict(model)
    if stored.keys() != loaded.keys() or any(
        not torch.equal(stored[k],loaded[k].detach().cpu()) for k in stored):
        raise ValueError("actual trained adapter numeric readback mismatch")
    model.eval()
    write_json(args.output/"model-readback.json",{
        "model_class":"Lfm2VlForConditionalGeneration",
        "source":PRODUCER_SHA, "checkpoint":"checkpoint-000045",
        "verified_adapter_tensors":len(stored), "contract":contract,
        "real_original_optimizer_steps":45,"new_optimizer_updates":0})
    gallery = args.output/"gallery"
    gallery.mkdir()
    records=[]
    all_decisions=[]
    raw_starts=[]
    for seed in SEEDS:
        for family in w7.FAMILIES:
            for mode in ("learned","flat"):
                if STOP or time.monotonic()-start > DEADLINE-170:
                    write_json(args.output/"partial.json",{
                        "reason":"Stop-or-deadline",
                        "completed_rollouts":len(records),
                        "new_optimizer_updates":0})
                    raise RuntimeError("bounded W9 gallery partially completed")
                decisions=[]
                def choose(observation):
                    tick=len(decisions)
                    raw=None
                    if mode=="learned" and tick==0:
                        raw=w7.generate(model,processor,selected,
                            w7.make_prompt(observation),observation["legal"],max_tokens=10)
                        raw_starts.append({"seed":seed,"family":family,"tick":0,**raw})
                    scored=(score_next_token(model,processor,selected,
                        w7.make_prompt(observation)) if mode=="learned" else None)
                    choice,evidence=scored_policy(scored,observation["legal"],mode,seed,tick)
                    decisions.append({**evidence,"tick":tick,"score":scored,
                        "raw_start_probe":raw,
                        "original_model_scoring":mode=="learned",
                        "no_claim_of_raw_action_EOS_for_host_sampler":True})
                    return choice
                actual=support.rollout(seed,family,STEPS,choose)
                if len(actual["events"])!=len(decisions):
                    raise ValueError("host W0 event/action count mismatch")
                for i,event in enumerate(actual["events"]):
                    if event["raw_choice"]!=decisions[i]["choice"]:
                        raise ValueError("host applied an unrecorded model-sampled action")
                stem=f"{family}-{mode}-{seed}"
                png=gallery/(stem+".png")
                png.write_bytes(worm.png_bytes(actual))
                write_json(gallery/(stem+".json"),actual)
                write_json(gallery/(stem+"-scores.json"),decisions)
                records.append({"family":family,"seed":seed,"policy":mode,
                    "steps":len(decisions),"metrics":support.statistics(actual),
                    "png_sha256":sha256_file(png)})
                all_decisions.append({"family":family,"seed":seed,"mode":mode,
                    "trace":decisions})
                write_json(args.output/"rolling-evidence.json",{
                    "records":records,"decisions":all_decisions,
                    "raw_greedy_starts":raw_starts,"new_optimizer_updates":0})
    # Original PNG pixels only: six columns (seed × learned/flat), three themes.
    # Human-readable labels sit outside the W0 raster; JSON/PNG originals stay intact.
    from PIL import Image, ImageDraw
    tile=384
    title_h=30
    montage=Image.new("RGB",(len(SEEDS)*2*tile,len(w7.FAMILIES)*(tile+title_h)),(24,24,24))
    draw=ImageDraw.Draw(montage)
    for iy,family in enumerate(w7.FAMILIES):
        for ix,seed in enumerate(SEEDS):
            for k,mode in enumerate(("learned","flat")):
                x=(ix*2+k)*tile
                y=iy*(tile+title_h)
                draw.text((x+8,y+7),f"{family} / seed {seed} / {mode}",fill=(240,240,240))
                stem=f"{family}-{mode}-{seed}"
                with Image.open(gallery/(stem+".png")) as original:
                    if original.mode!="RGB" or original.size!=(384,384):
                        raise ValueError("W0 original PNG dimensions or colour changed")
                    montage.paste(original,(x,y+title_h))
    montage_path=gallery/"w9-scored-vs-flat-contact-sheet.png"
    montage.save(montage_path)
    montage_sha=sha256_file(montage_path)
    originals=[r for r in records if r["policy"]=="learned"]
    summary={"status":"complete",
        "original_optimizer_updates":45, "new_optimizer_updates":0,
        "model_scored_decisions":sum(len(r["trace"]) for r in all_decisions if r["mode"]=="learned"),
        "model_rollouts":len(originals),
        "flat_rollouts":len(records)-len(originals),
        "raw_greedy_start_count":len(raw_starts),
        "raw_greedy_EOS7_count":sum(x["raw_syntactic_eos"] for x in raw_starts),
        "raw_greedy_legal_count":sum(x["valid_first_pass"] for x in raw_starts),
        "raw_greedy_actions":dict(collections.Counter(x["raw_action"] for x in raw_starts)),
        "distinct_model_pngs":len({x["png_sha256"] for x in originals}),
        "original_png_contact_sheet_sha256":montage_sha,
        "two_actual_obstacle_worlds_not_six":True,
        "model_scores_vs_flat_same_world_tick":"W8 source-fixed CRN, T2.5, 70% model / 30% flat",
        "records":records,
        "wall_seconds":time.monotonic()-start,
        "cpu_seconds":time.process_time(),
        "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "native_execution":False, "sealed_FINAL_opened":False}
    write_json(args.output/"summary.json",summary)
    if len(originals)!=6 or len(raw_starts)!=6:
        raise ValueError("incomplete W9 long48 scored gallery")
    print(canonical({"event":"W9-scored-W0-complete",
        "rollouts":len(records),"model_decisions":summary["model_scored_decisions"]}),flush=True)


if __name__=="__main__":
    try:
        signal.signal(signal.SIGINT,stop)
        signal.signal(signal.SIGTERM,stop)
        main()
    except Exception as e:
        if "--output" in sys.argv:
            out=Path(sys.argv[sys.argv.index("--output")+1])
            out.mkdir(parents=True,exist_ok=True)
            write_json(out/"failure.json",{"experiment":VERSION,
                "type":type(e).__name__,"message":str(e)})
        raise
