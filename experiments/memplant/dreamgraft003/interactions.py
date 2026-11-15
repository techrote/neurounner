"""DG003: zero-shot interaction inference from two already-transmitted world laws.

No training or tuning. Loads the exact DREAMGRAFT-002 real Qwen feature archive,
frozen two-output ridge recipe, and frozen target-native SmolLM2 prefix carriers.
Tests *new, never-trained* interactions with explicit causal counterfactual swaps.

Evidence level: task-conditioned two-scalar compositional inference only.
Not a native KV cache transplant, learned algebra of latents, or latent surgery.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import time
import traceback

import torch
from experiments.memplant.dreamgraft001.bridge import format_prompt, generation_from_prefix
from experiments.memplant.dreamgraft001.g0_live import load_pinned, measure_rss
from experiments.memplant.dreamgraft002.source import (
    HEAT, VIBRATION, RECEIVER, RECEIVER_REV, save_json)
from experiments.memplant.dreamgraft002.evaluate import (
    load_inputs, two_bit_ridge, continuous_prefix)

PREVIOUS_EXACT_HEAD = "ac4947121bc75a15a11ed62970ba5f8255391480"
PREVIOUS_EVIDENCE_RUN = 43736088404
PREFIX_SHA256 = "89fdb9c2edf8f2d3fffba6a124e6bcb62177229a4ee5d70522a17ed02eba8cc4"
SOURCE_V_SHA256 = "473cdb663922705cbddd99b57bfc5ea63048e113ec318e238574b2420511ae75"

# The donor diaries and G2 prefix-fitting targets contain neither these
# mechanisms nor their yes/no conclusions.
SENSORS = (
    "Assume a reinforcement sensor is ON exactly when warming the inherited "
    "alloy makes it harder, and OFF otherwise. A lift sensor is ON exactly "
    "when shaking the inherited mineral makes its dust float, and OFF otherwise. "
)
TASKS = {
    "xor": {
        "question": SENSORS + (
            "A strange relay energizes if EXACTLY ONE of these two sensors "
            "is ON. In the inherited world, does this relay energize? "
            "Answer only yes or no."
        ),
        "name": "Exclusive-or fault relay",
    },
    "and": {
        "question": SENSORS + (
            "A suspended furnace bridge engages only if BOTH sensors "
            "are ON simultaneously. In the inherited world, does the "
            "bridge engage? Answer only yes or no."
        ),
        "name": "Two-part suspended furnace",
    },
    "equal": {
        "question": SENSORS + (
            "An experimental stabilizer activates when the two sensors "
            "are in the SAME state: both ON or both OFF. Otherwise "
            "it stays inactive. Will it activate in the inherited world? "
            "Answer only yes or no."
        ),
        "name": "Agreement stabilizer",
    },
}
CONDITIONS = (
    "real_v", "heat_swap", "vibration_swap", "both_swap",
    "oracle_prefix", "hard_gate", "no_prefix", "zero_prefix",
    "random_prefix", "equal_prefix", "text_facts", "full_text"
)
ANSWERS = (" Yes", " No")


def cause_truth(task, heat, vibration):
    """Symbolic public task mechanism, NEVER a recipient input or prefix target."""
    hardens = heat == "hardens"
    floats = vibration == "floats"
    if task == "xor":
        return hardens != floats
    if task == "and":
        return hardens and floats
    if task == "equal":
        return hardens == floats
    raise ValueError("unknown interaction " + str(task))


def paired_source(cases, index, condition):
    byid={x["case_id"]:j for j,x in enumerate(cases)}
    if condition=="heat_swap":
        return byid[cases[index]["heat_twin"]]
    if condition=="vibration_swap":
        return byid[cases[index]["vibe_twin"]]
    if condition=="both_swap":
        return byid[cases[index]["both_twin"]]
    return index


def one_two_answer_forward(receiver, tokenizer, prompt, prefix, *, words=ANSWERS):
    """Two candidate sequence *total* log-likelihoods from one batched B forward.

    No input embeddings are trained; frozen receiver may or may not be able to
    apply an unfamiliar logical mechanism to its inherited prefix.
    """
    alternatives=[
        tokenizer(w, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        for w in words
    ]
    if any(len(ids)==0 for ids in alternatives):
        raise ValueError("empty answer tokens")
    n=len(alternatives)
    length=max(len(x) for x in alternatives)
    pad=int(tokenizer.eos_token_id)
    ids=torch.full((n,length),pad,dtype=torch.long)
    attn_answer=torch.zeros((n,length),dtype=torch.long)
    for j,piece in enumerate(alternatives):
        ids[j,:len(piece)]=piece
        attn_answer[j,:len(piece)]=1
    with torch.inference_mode():
        embed=receiver.get_input_embeddings()
        prompt_emb=embed(prompt).expand(n,-1,-1)
        if prefix is None:
            combined=torch.cat((prompt_emb,embed(ids)),dim=1)
            slots=0
        else:
            if prefix.shape != (1,8,prompt_emb.shape[-1]):
                raise ValueError("invalid recipient-native prefix shape")
            combined=torch.cat((prefix.expand(n,-1,-1),prompt_emb,embed(ids)),dim=1)
            slots=prefix.shape[1]
        mask=torch.cat((
            torch.ones((n,slots+prompt.shape[1]),dtype=torch.long),
            attn_answer),dim=1)
        logits=receiver(
            inputs_embeds=combined,attention_mask=mask,
            use_cache=False,return_dict=True
        ).logits
        start=slots+prompt.shape[1]-1
        out=[]
        for j,piece in enumerate(alternatives):
            pred=logits[j,start:start+len(piece)].float()
            logp=pred.log_softmax(dim=-1)
            out.append(float(logp.gather(-1,piece.view(-1,1)).sum()))
    return out


def condition_input(cases, index, cond, source_scores, prototypes, tokenizer, task):
    row=cases[index]
    source_idx=paired_source(cases,index,cond)
    question=TASKS[task]["question"]
    numeric=cond not in ("text_facts","full_text","no_prefix")
    if cond in ("real_v","heat_swap","vibration_swap","both_swap","hard_gate"):
        mode="hard_gate" if cond=="hard_gate" else "genuine"
        prefix,_,_ = continuous_prefix(
            prototypes,source_scores,source_idx,cases,mode,seed=1049+index)
        prompt=format_prompt(tokenizer,question)
    elif cond=="oracle_prefix":
        # Evaluator-only oracle. Never count this as evidence of donor transfer.
        prefix=prototypes[f"class_{cases[index]['answer_class']}"]
        prompt=format_prompt(tokenizer,question)
    elif cond=="equal_prefix":
        prefix=sum(prototypes[f"class_{j}"] for j in range(4))/4.0
        prompt=format_prompt(tokenizer,question)
    elif cond=="no_prefix":
        prefix=None
        prompt=format_prompt(tokenizer,question)
    elif cond=="zero_prefix":
        prefix=torch.zeros_like(prototypes["class_0"])
        prompt=format_prompt(tokenizer,question)
    elif cond=="random_prefix":
        generator=torch.Generator(device="cpu").manual_seed(1138+index)
        noise=torch.randn(prototypes["class_0"].shape,generator=generator)
        reference=prototypes["class_0"]
        prefix=noise*(reference.norm()/(noise.norm()+1e-8))
        prompt=format_prompt(tokenizer,question)
    elif cond=="text_facts":
        prefix=None
        prompt=format_prompt(
            tokenizer, "Recorded inherited laws: heating the alloy "
            +row["heat"]+" it; vibration makes the dust "+row["vibration"]
            +".\n"+question
        )
    elif cond=="full_text":
        prefix=None
        prompt=format_prompt(
            tokenizer,"Complete source field diary:\n"+row["source"]
            +"\n\n"+question
        )
    else:
        raise ValueError("invalid control")
    return prompt,prefix,source_idx,{
        "source_text_seen":cond=="full_text",
        "explicit_rule_words_seen":cond=="text_facts",
        "numerical_source_only":numeric and cond not in ("oracle_prefix","equal_prefix"),
        "oracle_not_a_transplant":cond=="oracle_prefix",
    }


def judgement(task, cases, index, source_idx, yes):
    src=cases[source_idx]
    base=cases[index]
    expected=cause_truth(task,src["heat"],src["vibration"])
    original=cause_truth(task,base["heat"],base["vibration"])
    return {
        "expected_yes":expected,"original_expected_yes":original,
        "correct_presented":yes==expected,
        "correct_original":yes==original,
    }


def aggregate(rows,task,cases):
    conditions={}
    for condition in CONDITIONS:
        subset=[x for x in rows if x["condition"]==condition]
        if not subset:
            continue
        conditions[condition]={
            "n":len(subset),
            "families":len({x["family"] for x in subset}),
            "correct":sum(x["correct_presented"] for x in subset),
            "correct_original":sum(x["correct_original"] for x in subset),
            "mean_signed_margin":round(
                sum((1 if x["expected_yes"] else -1)*x["yes_minus_no"]
                    for x in subset)/len(subset),5
            ),
        }
    idx={(r["case_id"],r["condition"]):r for r in rows}
    edits={}
    for kind in ("heat_swap","vibration_swap","both_swap"):
        matches=[]
        for case in cases:
            if case["split"]!="sealed":
                continue
            before=idx[(case["case_id"],"real_v")]
            after=idx[(case["case_id"],kind)]
            expected_flip=before["original_expected_yes"] != after["expected_yes"]
            actual_flip=before["predicted_yes"] != after["predicted_yes"]
            matches.append({
                "case_id":case["case_id"],
                "should_flip":expected_flip,
                "did_flip":actual_flip,
                "prediction_change_matches_mechanism":actual_flip==expected_flip,
                "both_responses_correct":before["correct_presented"] and after["correct_presented"],
            })
        edits[kind]={
            "n":len(matches),
            "expected_flips":sum(x["should_flip"] for x in matches),
            "observed_flips":sum(x["did_flip"] for x in matches),
            "correct_intervention_direction":sum(x["prediction_change_matches_mechanism"] for x in matches),
            "both_correct":sum(x["both_responses_correct"] for x in matches),
            "cases":matches,
        }
    return {"conditions":conditions,"causal_interventions":edits}


def run(task, source, checkpoint, outdir, *, creative=False):
    root=Path(outdir)
    root.mkdir(parents=True,exist_ok=True)
    report={"schema":"memplant-dg003-interactions-v1",
            "task":task,"task_name":TASKS[task]["name"],"status":"STARTED",
            "new_training_updates":0,
            "prefix_training_question_excluded_all_interactions":True,
            "donor_source_did_not_contain_relay_rules":True,
            "experiment_scope":"frozen supervised two-scalar channel; zero-shot novel task",
            "prior_run":PREVIOUS_EVIDENCE_RUN,"prior_head":PREVIOUS_EXACT_HEAD}
    try:
        torch.set_num_threads(min(4,torch.get_num_threads()))
        source_manifest,features,prefix_manifest,prototypes=load_inputs(source,checkpoint)
        if prefix_manifest["prototypes_sha256"]!=PREFIX_SHA256:
            raise RuntimeError("Previously frozen B-native prefixes changed")
        if source_manifest["channels"]["v"]["sha256"]!=SOURCE_V_SHA256:
            raise RuntimeError("Previously frozen source features changed")
        cases=source_manifest["cases"]
        v=features["v"]
        source_scores,source_readout=two_bit_ridge(v,cases)
        rec,tok,identity=load_pinned(RECEIVER,RECEIVER_REV)
        for p in rec.parameters():p.requires_grad_(False)
        rec.eval()
        report["provenance"]={
            "donor_source_sha256":source_manifest["channels"]["v"]["sha256"],
            "target_prefix_sha256":prefix_manifest["prototypes_sha256"],
            "source_model_revision":source_manifest["donor"]["revision"],
            "recipient_revision":identity["revision"],
            "ridge_lambda":1.0,"ridge_frozen_recomputed_from_training_only":True,
            "source_ridge_sealed":source_readout["sealed"],
        }
        sealed=[i for i,r in enumerate(cases) if r["split"]=="sealed"]
        rows=[]
        start=time.monotonic()
        for count,index in enumerate(sealed):
            item=cases[index]
            for cond in CONDITIONS:
                prompt,prefix,src,exposure=condition_input(
                    cases,index,cond,source_scores,prototypes,tok,task)
                log_yes,log_no=one_two_answer_forward(rec,tok,prompt,prefix)
                yes=log_yes>=log_no
                verdict=judgement(task,cases,index,src,yes)
                rows.append({
                    "case_id":item["case_id"],
                    "family":item["family"],"condition":cond,
                    "numeric_source_id":cases[src]["case_id"] if exposure["numerical_source_only"] else None,
                    "predicted_yes":yes,"yes_minus_no":round(log_yes-log_no,6),
                    "log_likelihoods":{"yes":round(log_yes,6),"no":round(log_no,6)},
                    **verdict,**exposure})
            if (count+1)%6==0:
                print(f"DG003 {task}: {count+1}/{len(sealed)} sources evaluated",flush=True)
        report["evaluation"]=aggregate(rows,task,cases)
        report["raw_cases"]=rows
        report["elapsed_seconds"]=round(time.monotonic()-start,3)
        report["rss_mib"]=measure_rss()
        report["capability_gate"]={
            "text_facts_oracle_correct":report["evaluation"]["conditions"]["text_facts"]["correct"],
            "native_prefix_oracle_correct":report["evaluation"]["conditions"]["oracle_prefix"]["correct"],
            "both_oracles_are_controls_not_evidence":True}
        if creative:
            creative_rows=[]
            for index in (sealed[0],sealed[1]):
                for cond in ("real_v","heat_swap","no_prefix"):
                    prompt,prefix,src,exposure=condition_input(
                        cases,index,cond,source_scores,prototypes,tok,task)
                    idea="Invent an unfamiliar device in the inherited world whose operation uses BOTH warming the metal and vibrating its dust. Explain the combined effect and one failure mode in three sentences."
                    # The creative instruction is fixed and new. Do not show the
                    # world rules or full story in numeric arms.
                    fixed=format_prompt(tok,idea)
                    raw,ids=generation_from_prefix(rec,tok,fixed,prefix,max_tokens=48)
                    creative_rows.append({
                        "source":cases[index]["case_id"],"condition":cond,
                        "injected_source":cases[src]["case_id"],
                        "raw_text":raw,"generated_tokens":ids,
                        "not_scored_as_interaction_reasoning":True})
            report["creative_samples"]=creative_rows
        report["status"]="COMPLETED_FROZEN_EXPLORATORY"
        report["claims"]={
            "native_KV_cache_transplant":False,
            "trained_interaction_answers_in_prototypes":False,
            "interaction_inference_proven_by_CI_success_alone":False,
            "not_fresh_source_corpus":True,
            "task_generalisation_test":True,
        }
        save_json(root/"report.json",report)
        print(json.dumps({
            "status":report["status"],"task":task,
            "result":report["evaluation"]["conditions"],
            "interventions":{
                k:{q:v for q,v in s.items() if q!="cases"}
                for k,s in report["evaluation"]["causal_interventions"].items()},
            "sec":report["elapsed_seconds"]
        },indent=2),flush=True)
    except Exception as e:
        report["status"]="BLOCKED"
        report["error"]={"type":type(e).__name__,
                         "message":str(e),"trace":traceback.format_exc()[-4500:]}
        save_json(root/"report.json",report)
        raise

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--task",choices=tuple(TASKS),required=True)
    p.add_argument("--source",default="runtime/memplant/dg003/source")
    p.add_argument("--checkpoint",default="runtime/memplant/dg003/prefixes")
    p.add_argument("--out",required=True)
    p.add_argument("--creative",action="store_true")
    a=p.parse_args()
    run(a.task,a.source,a.checkpoint,a.out,creative=a.creative)
