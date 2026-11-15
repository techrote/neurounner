"""Fit four receiver-native two-rule soft-prefix prototypes.

Only the 4 prototype tensors receive gradients; pretrained SmolLM2-1.7B is
frozen. No donor state or training source documents are used in this stage.
These are TARGET-NATIVE carriers, not cross-model cache translations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import torch
from safetensors.torch import save_file
from experiments.memplant.dreamgraft001.bridge import answer_loss,format_prompt
from experiments.memplant.dreamgraft001.g0_live import load_pinned,measure_rss,target_ids
from experiments.memplant.dreamgraft002.source import (
    RECEIVER,RECEIVER_REV,HEAT,VIBRATION,SLOT_COUNT,PAIR_ANSWERS,save_json)

QUESTIONS=(
    "In the remembered fictional world, how does heat affect the alloy and how does "
    "vibration affect its dust? Reply with two words: hardens or softens, then floats or sinks.",
    "State the remembered heat rule followed by the dust vibration rule. Use two words "
    "only, first hardens/softens, then floats/sinks.",
)
CREATIVE=(
    "Invent a fictional machine that uses the world's metal and its dust. "
    "Describe what happens under heating and vibration in two sentences.",
    "Imagine a creature adapted to the inherited metal's behavior under heat and vibration. "
    "Explain its survival mechanism.",
)
INTERACTION_QUESTION=(
    "A safety door activates ONLY when warming makes the metal harden AND shaking "
    "makes the mineral dust float. Using the world you remember, answer active or inactive."
)

def fit(directory,updates=160,learning_rate=0.006):
    root=Path(directory)
    root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(min(torch.get_num_threads(),4))
    torch.manual_seed(2202)
    report={"schema":"memplant-dg002-prefix-v1","status":"STARTED",
            "model_id":RECEIVER,"model_revision":RECEIVER_REV,
            "only_soft_prefixes_trained":True,
            "no_donor_values_or_ground_truth_source_text_loaded":True,
            "requested_updates":updates,"learning_rate":learning_rate}
    try:
        model,tok,identity=load_pinned(RECEIVER,RECEIVER_REV)
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
        assert model.config.hidden_size==2048
        sigma=max(float(model.get_input_embeddings().weight.detach().std()),0.01)
        prototypes=torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(1,SLOT_COUNT,2048)*sigma)
            for _ in range(4)])
        optimizer=torch.optim.Adam(prototypes.parameters(),lr=learning_rate)
        prompts=[format_prompt(tok,q) for q in QUESTIONS]
        targets=[target_ids(tok,text) for text in PAIR_ANSWERS]
        history=[]
        start=time.monotonic()
        for step in range(updates):
            cls=step%4
            order=(step//4)%len(prompts)
            optimizer.zero_grad(set_to_none=True)
            loss=answer_loss(model,prompts[order],targets[cls],prototypes[cls])
            if not torch.isfinite(loss):raise RuntimeError("native two-rule prototype loss nonfinite")
            loss.backward()
            g=prototypes[cls].grad
            if g is None or not torch.isfinite(g).all():
                raise RuntimeError("missing native-prefix gradient")
            norm=float(torch.nn.utils.clip_grad_norm_([prototypes[cls]],5.0))
            if not math.isfinite(norm) or norm<=0:
                raise RuntimeError("bad native-prefix gradient norm")
            optimizer.step()
            if step%8==0 or step+1==updates:
                history.append({"step":step+1,"class":cls,"order":order,
                                "loss_before":round(float(loss.item()),6),
                                "grad_norm":round(norm,6),"rss_mib":measure_rss(),
                                "seconds":round(time.monotonic()-start,2)})
            if (step+1)%40==0:
                print(f"Two-rule B-native prefixes: {step+1}/{updates}, loss {float(loss):.4f}",flush=True)
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("frozen receiver had accumulated parameter gradient")
        with torch.no_grad():
            end_losses=[float(answer_loss(model,prompts[0],targets[c],prototypes[c]).item())
                        for c in range(4)]
        params={f"class_{i}":p.detach().cpu().contiguous() for i,p in enumerate(prototypes)}
        save_file(params,str(root/"prototypes.safetensors"))
        report.update(status="COMPLETED",
                      source_evidence="native directly optimized prefixes; no donor input",
                      receiver=identity,
                      prototypes_sha256=hashlib.sha256((root/"prototypes.safetensors").read_bytes()).hexdigest(),
                      prototypes_shape=[4,SLOT_COUNT,2048],
                      answers=list(PAIR_ANSWERS),
                      trace=history,
                      end_losses=end_losses,
                      elapsed_seconds=round(time.monotonic()-start,3),
                      rss_mib=measure_rss(),
                      last_per_class_loss_threshold_pass=all(x<0.5 for x in end_losses))
        save_json(root/"report.json",report)
        print(json.dumps({"status":report["status"],"end_losses":end_losses,
                          "seconds":report["elapsed_seconds"],
                          "weights_sha256":report["prototypes_sha256"]},indent=2),flush=True)
    except Exception as exc:
        report["status"]="BLOCKED"
        report["error"]={"type":type(exc).__name__,"message":str(exc),
                         "trace":traceback.format_exc()[-3500:]}
        save_json(root/"report.json",report)
        raise

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="runtime/memplant/dg002/prototypes")
    p.add_argument("--updates",type=int,default=160)
    fit(p.parse_args().out,updates=p.parse_args().updates)
