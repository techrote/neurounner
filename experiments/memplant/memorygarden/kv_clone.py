"""MEMPLANT native KV custody baseline: a real Qwen cache cloned into a new instance.

Transfer source K/V from frozen Qwen instance A to independently constructed
instance B of the EXACT SAME frozen model checkpoint. Compare B's next-token
distribution against A's full-prefill distribution; wrong-cache source controls
should change results. This proves plumbing/custody, NEVER heterogeneous transfer.
"""
from __future__ import annotations
import argparse
import copy
import gc
import json
from pathlib import Path
import torch
from experiments.memplant.dreamgraft001.g0_live import load_pinned, measure_rss
from experiments.memplant.dreamgraft001.bridge import read_cache
from experiments.memplant.g1.features import DONOR, DONOR_REV
from experiments.memplant.dreamgraft002.source import make_cases, save_json

QUESTION = ("\nQuestion: Under these imaginary laws, what happens to the metal "
            "when it is warmed and to the dust when shaken? Answer:")


def run(out):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    report={
        "schema":"memplant-native-kv-custody-v1",
        "status":"STARTED","model":DONOR,"revision":DONOR_REV,
        "source_and_recipient_same_weights":True,
        "fresh_model_object_for_recipient":True,
        "heterogeneous_model_transfer":False,
        "recipient_receives_donor_text_token_ids_during_continuation":False,
        "learned_cache_adapter":False,
        "cases":[],
    }
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    try:
        source_cases=make_cases()
        # Same synthetic family but different heat law; other law fixed.
        chosen=[source_cases[0],source_cases[2]]
        A,tokA,source_info=load_pinned(DONOR,DONOR_REV)
        query=tokA(QUESTION,add_special_tokens=False,return_tensors="pt")["input_ids"]
        copies=[];full_logits=[]
        with torch.no_grad():
            for row in chosen:
                src=tokA(row["source"],return_tensors="pt")["input_ids"]
                # Baseline uses precisely same concatenated input IDs, avoiding
                # tokenizer boundary effects across a source/query split.
                full=A(input_ids=torch.cat([src,query],dim=1),
                       use_cache=False,return_dict=True).logits[:,-1].float().cpu()
                prefill=A(input_ids=src,use_cache=True,return_dict=True)
                layers=read_cache(prefill.past_key_values)
                if len(layers)!=24:
                    raise RuntimeError("unexpected Qwen cache layer count")
                copies.append(copy.deepcopy(prefill.past_key_values))
                full_logits.append(full)
                report["cases"].append({
                    "case":row["case_id"],
                    "source_sha256":row["source_sha256"],
                    "heat":row["heat"],"vibration":row["vibration"],
                    "source_token_count":int(src.shape[1]),
                    "query_token_count":int(query.shape[1]),
                    "cache_layers":len(layers),
                    "cache_dtype":str(layers[0][0].dtype),
                })
        report["donor_weights_sha256"]=source_info["weight_sha256"]
        del A,tokA
        gc.collect()
        B,tokB,recv_info=load_pinned(DONOR,DONOR_REV)
        if recv_info["weight_sha256"]!=source_info["weight_sha256"]:
            raise RuntimeError("weights changed between cache source and receiver")
        with torch.no_grad():
            for i in range(2):
                received=B(input_ids=query,
                           past_key_values=copy.deepcopy(copies[i]),
                           use_cache=True,return_dict=True).logits[:,-1].float().cpu()
                wrong=B(input_ids=query,
                        past_key_values=copy.deepcopy(copies[1-i]),
                        use_cache=True,return_dict=True).logits[:,-1].float().cpu()
                original=full_logits[i]
                match=float((original-received).abs().max())
                negative=float((received-wrong).abs().max())
                report["cases"][i].update({
                    "max_abs_full_vs_cloned_cache_logits":match,
                    "wrong_source_cache_max_abs_delta":negative,
                    "wrong_source_cache_changes_output":negative>1e-5,
                    "same_top_token":bool(original.argmax()==received.argmax()),
                })
        report["pass_equivalence"]=all(
            x["max_abs_full_vs_cloned_cache_logits"]<0.002 and x["same_top_token"]
            for x in report["cases"])
        report["wrong_cache_control_distinguishable"]=all(
            x["wrong_source_cache_changes_output"] for x in report["cases"])
        report["status"]="PASS" if report["pass_equivalence"] else "NOT_YET_PASS"
        report["rss_mib"]=measure_rss()
        save_json(out,report)
        print(json.dumps({"status":report["status"],"cases":report["cases"]},indent=2),flush=True)
    except Exception as e:
        report["status"]="BLOCKED"
        report["error"]=type(e).__name__+": "+str(e)
        save_json(out,report)
        raise


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="runtime/memplant/garden/kv_clone.json")
    run(p.parse_args().out)
