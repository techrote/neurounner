#!/usr/bin/env python3
"""Independent W9 original scored-W0 PNG/host-sampler audit. No inference or optimizer."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

from w8_score_original_audit import load_w0, choose, check, sha

ZIP_SHA = "82890ec0b306a7174785f0f923ae874c16bd303a7864ff0428ea8963d4f4162b"
MODEL_REV = "fc6221ca597f3315e4f82fc2df606783267b34ba"
PRODUCER_SHA = "c14731f875d7334eae213c2bd74230bd310e7167"
EVALUATOR_SHA = "6bf3b07f446040fb4069d0ad452edd20c52dcc46"
ACTIONS = ("STRAIGHT", "LEFT", "RIGHT", "BRANCH", "RETRACE", "STOP")
FAMILIES = ("greebles", "pipework", "architecture")
SEEDS = (3101, 3109, 3119)
PREFIX = "w9-scored/"

def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def read_json(z, path):
    return json.loads(z.read(PREFIX + path))

def check_step(row, event, policy):
    """Independently recompute host sampling from original logits, temperature and RNG."""
    legal = row["legal"]
    allowed = row["sampled_legal"]
    check(legal == event["observation"]["legal"], "incorrect host legal action mask")
    check(allowed == ([a for a in legal if a != "STOP"] if len(legal)>1 else ["STOP"]),
          "invalid STOP deferral / sampled legal support")
    check(len(allowed) == len(set(allowed)) and set(allowed) <= set(ACTIONS),
          "duplicate or foreign legal actions")
    check(row["tick"] == event["observation"]["tick"]
          and row["sampler"] == policy
          and row["model_fraction"] == 0.7
          and row["temperature"] == 2.5
          and row["STOP_deferred_unless_forced"] is True,
          "sampler identity mismatch")
    probs = row["sample_probs"]
    check(set(probs) == set(allowed) and
          all(math.isfinite(v) and v>=0 for v in probs.values()) and
          abs(sum(probs.values())-1)<1e-6, "invalid host probabilities")
    uniform = 1/len(allowed)
    if policy == "learned":
        check(row["original_model_scoring"] is True and
              row["no_claim_of_raw_action_EOS_for_host_sampler"] is True,
              "host scores misidentified as raw EOS7 inference")
        scores = row["score"]
        check(scores["candidate_token_ids"] == dict(zip(ACTIONS, range(525,531))),
              "W9 candidate token IDs changed")
        logits = scores["candidate_logits"]
        check(set(logits) == set(ACTIONS) and
              all(math.isfinite(v) for v in logits.values()), "malformed six-logit vector")
        check(max(ACTIONS, key=lambda a:logits[a]) == scores["six_code_top_action"] == "LEFT",
              "W9 unmasked six-way argmax not LEFT")
        max_all = max(logits.values())
        weights = {a:math.exp(logits[a]-max_all) for a in ACTIONS}
        total = sum(weights.values())
        for a in ACTIONS:
            check(abs(weights[a]/total - scores["six_code_conditional_probabilities"][a])<1e-6,
                  "unmasked six-action probabilities wrong")
        highest = max(logits[a] for a in allowed)
        weights = {a:math.exp((logits[a]-highest)/2.5) for a in allowed}
        total = sum(weights.values())
        for a in allowed:
            expected = weights[a]/total
            check(abs(expected-row["conditioned_model_probs"][a])<1e-6,
                  "conditioned temperature softmax wrong")
            check(abs(.7*expected + .3*uniform - probs[a])<1e-6,
                  "0.7 model + 0.3 flat mix wrong")
    else:
        check(row["score"] is None and row["raw_start_probe"] is None,
              "flat control includes learned model data")
        check(all(abs(p-uniform)<1e-8 for p in probs.values()),
              "flat control is not uniform")
    check(choose(probs, allowed, row["host_u"]) == row["choice"] ==
          event["raw_choice"] == event["applied"] and event["raw_legal"] is True,
          "original host RNG action / W0 applied action mismatch")

def audit(artifact, out=None):
    check(sha(artifact.read_bytes())==ZIP_SHA, "unexpected producer ZIP bytes")
    worm=load_w0()  # pinned W0 bytes + Git blob; original non-native replay
    with zipfile.ZipFile(artifact) as z:
        names=z.namelist()
        check(len(names)==66 and len(set(names))==66 and z.testzip() is None
              and all(not n.startswith("/") and "\\" not in n and
                      ".." not in PurePosixPath(n).parts for n in names),
              "unsafe/corrupt/incomplete original ZIP")
        lease=read_json(z,"lease.json")
        summary=read_json(z,"summary.json")
        readback=read_json(z,"model-readback.json")
        check(lease["experiment"]=="lfm-worm-w9-step45-scored-W8-matched-CRN-v1"
              and lease["evaluator_source"]==EVALUATOR_SHA
              and lease["producer_source"]==PRODUCER_SHA
              and lease["producer_run"]==43743348779
              and lease["world_seeds"]==list(SEEDS)
              and lease["steps_per_rollout"]==16
              and lease["reuse_exact_W8_host_score_sampler"] is True
              and lease["new_optimizer_updates"]==0
              and lease["native_execution"] is False
              and lease["sealed_FINAL_opened"] is False,
              "source-fenced W9 evaluator/lease mismatch")
        contract=readback["contract"]
        check(readback["source"]==PRODUCER_SHA
              and readback["checkpoint"]=="checkpoint-000045"
              and readback["verified_adapter_tensors"]==184
              and readback["real_original_optimizer_steps"]==45
              and readback["new_optimizer_updates"]==0
              and contract["model_id"]=="LiquidAI/LFM2.5-VL-450M"
              and contract["revision"]==MODEL_REV
              and contract["model_class"]=="Lfm2VlForConditionalGeneration"
              and contract["vision_training"] is False
              and contract["connector_training"] is False,
              "W9 checkpoint/model/full vision-freeze pin mismatch")
        check(summary["status"]=="complete"
              and summary["original_optimizer_updates"]==45
              and summary["new_optimizer_updates"]==0
              and summary["model_scored_decisions"]==144
              and summary["model_rollouts"]==summary["flat_rollouts"]==9
              and summary["raw_greedy_start_count"]==9
              and summary["raw_greedy_EOS7_count"]==0
              and summary["raw_greedy_legal_count"]==0
              and summary["raw_greedy_actions"]=={"MODEL_REFUSED":9}
              and summary["distinct_model_pngs"]==4
              and summary["three_actual_obstacle_worlds_not_nine"] is True
              and summary["native_execution"] is False
              and summary["sealed_FINAL_opened"] is False,
              "run summary conflates host sampling and valid greedy output")
        expected={(f,s,p) for f in FAMILIES for s in SEEDS for p in ("learned","flat")}
        summaries={(r["family"],r["seed"],r["policy"]):r for r in summary["records"]}
        check(len(summaries)==18 and set(summaries)==expected, "missing or duplicate rollouts")
        scores_by_key={}
        png_by_key={}
        report_rows=[]
        for family in FAMILIES:
            for seed in SEEDS:
                for policy in ("learned","flat"):
                    key=(family,seed,policy)
                    stem="gallery/%s-%s-%s" % (family,policy,seed)
                    record=read_json(z,stem+".json")
                    rows=read_json(z,stem+"-scores.json")
                    png=z.read(PREFIX+stem+".png")
                    summary_row=summaries[key]
                    check(record["family"]==family and record["seed"]==seed
                          and record["size"]==96 and record["mode"]=="callback"
                          and record["steps_executed"]==record["step_budget"]==16
                          and len(rows)==len(record["events"])==16,
                          "invalid original host trace or budget")
                    check(sha(png)==summary_row["png_sha256"],"PNG altered since producer")
                    replay=worm.run(seed=seed,mode="replay",family=family,steps=16,
                                    raw_actions=[r["choice"] for r in rows])
                    actual=dict(record);expected_record=dict(replay)
                    actual.pop("mode");expected_record.pop("mode")
                    check(actual==expected_record, "W0 replay trajectory differs")
                    check(worm.png_bytes(replay)==png, "W0 replay PNG pixel bytes differ")
                    for x,event in zip(rows,record["events"]):
                        check_step(x,event,policy)
                    starts=[r["raw_start_probe"] for r in rows if r["raw_start_probe"] is not None]
                    if policy=="learned":
                        check(len(starts)==1 and starts[0]["raw_action"]=="MODEL_REFUSED"
                              and starts[0]["eos_terminated"] is False
                              and starts[0]["valid_first_pass"] is False,
                              "W9 untrusted malformed raw greedy first pass relabelled")
                    else:
                        check(not starts,"flat scorer contains model greedy probe")
                    m=summary_row["metrics"]; h=record["metrics"]
                    check(m["unique_cells"]==h["unique_trail_cells"]
                          and m["branches"]==h["branches"]
                          and m["retrace_events"]==h["retraces"]
                          and m["invalid_raw_count"]==h["invalid_raw_choices"],
                          "original metrics inconsistent with W0 host receipt")
                    scores_by_key[key]=rows
                    png_by_key[key]=png
                    report_rows.append(dict(family=family,seed=seed,policy=policy,
                        cells=m["unique_cells"],branches=m["branches"],
                        revisits=m["revisits"],png_sha256=sha(png)))
        matched=diff=conditional=0
        for seed in SEEDS:
            for policy in ("learned","flat"):
                base=[x["host_u"] for x in scores_by_key[("greebles",seed,policy)]]
                for family in FAMILIES[1:]:
                    check(base==[x["host_u"] for x in scores_by_key[(family,seed,policy)]],
                          "family-specific RNG violates matched CRN")
            for family in FAMILIES:
                a=scores_by_key[(family,seed,"learned")]
                b=scores_by_key[(family,seed,"flat")]
                for x,y in zip(a,b):
                    check(x["host_u"]==y["host_u"],"matched model/flat RNG differs")
                    same=x["legal"]==y["legal"]
                    matched+=int(same)
                    different=x["choice"]!=y["choice"]
                    diff+=int(different)
                    conditional+=int(same and different)
        check((matched,diff,conditional)==(140,20,17),
              "W9 matched experimental action comparison differs")
        get=lambda f,p:[summaries[(f,s,p)]["metrics"]["unique_cells"] for s in SEEDS]
        check(all(get(f,"flat")==[13,11,12] for f in FAMILIES)
              and get("greebles","learned")==[16,13,14]
              and get("pipework","learned")==[16,12,14]
              and get("architecture","learned")==[16,12,14],
              "reported original coverage changed")
        check(len({sha(v) for k,v in png_by_key.items() if k[2]=="learned"})==4
              and len({sha(v) for k,v in png_by_key.items() if k[2]=="flat"})==3,
              "family/seed actual original PNG distinction wrong")
        contact=z.read(PREFIX+"gallery/w9-scored-vs-flat-contact-sheet.png")
        check(sha(contact)==summary["original_png_contact_sheet_sha256"],
              "original contact-sheet pixels altered")
        report=dict(schema="cybersand.worm.lfm-w9-scored-independent-audit/1",
            original_run=43748946947,original_artifact_sha256=ZIP_SHA,
            actual_producer_source=PRODUCER_SHA,actual_evaluator_source=EVALUATOR_SHA,
            model_revision=MODEL_REV,original_pngs_source_replayed=18,
            original_scorer_decisions_checked=288,model_logit_decisions_checked=144,
            independent_worlds=3,worlds=list(SEEDS),family_labels=3,
            cells_by_family={f:get(f,"learned") for f in FAMILIES},
            flat_cells=[13,11,12],model_distinct_pngs=4,
            matched_legal_ticks=matched,different_actions=diff,
            different_actions_same_legal=conditional,
            raw_first_pass_refusals=9,raw_EOS7_valid=0,unmasked_top_LEFT=144,
            original_optimizer_steps=45,new_optimizer_steps=0,native_paintings=0,
            interpretation="W9 original model scores influence a T2.5 70/30 host sampler; "
                "raw greedy EOS7 remains 0/9, six-way top is LEFT on all 144 scorer ticks, "
                "and only three obstacle worlds were tested. Small paired coverage gains "
                "are not proof of useful independent learned steering.")
        if out is not None:
            check(not out.exists(),"audit output path already exists")
            out.mkdir(parents=True)
            gallery=out/"original-pngs";gallery.mkdir()
            for key,png in sorted(png_by_key.items()):
                (gallery/("%s-%s-%s.png"%key)).write_bytes(png)
            (out/"original-contact-sheet.png").write_bytes(contact)
            (out/"report.json").write_text(canonical(report)+"\n")
            (out/"original-rows.json").write_text(canonical(report_rows)+"\n")
            receipt=sorted(p for p in out.rglob("*") if p.is_file())
            (out/"SHA256SUMS.txt").write_text("".join(
                "%s  %s\n"%(sha(p.read_bytes()),p.relative_to(out)) for p in receipt))
        return report

def adversarial_tests(path):
    with zipfile.ZipFile(path) as z:
        row=read_json(z,"gallery/greebles-learned-3101-scores.json")[0]
        event=read_json(z,"gallery/greebles-learned-3101.json")["events"][0]
    check_step(row,event,"learned")
    mutations=[
        lambda r:r.update(choice="STOP"),
        lambda r:r.update(host_u=1.1),
        lambda r:r["sample_probs"].update(STRAIGHT=0.99),
        lambda r:r["score"]["candidate_logits"].update(LEFT=-20),
        lambda r:r["score"].update(six_code_top_action="RIGHT"),
        lambda r:r.update(legal=["STOP"]),
        lambda r:r.update(original_model_scoring=False)]
    for damage in mutations:
        broken=copy.deepcopy(row)
        damage(broken)
        try:check_step(broken,event,"learned")
        except (ValueError,KeyError,AssertionError):pass
        else:raise AssertionError("tampered W9 host-sampler input was accepted")
    return len(mutations)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--artifact",required=True,type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        print("%d/7 corrupted-original adversarial tests PASS"%adversarial_tests(args.artifact))
    print(canonical(audit(args.artifact,args.output)))
