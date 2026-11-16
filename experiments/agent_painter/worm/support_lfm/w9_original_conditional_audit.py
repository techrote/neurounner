#!/usr/bin/env python3
"""Independent W9 original artifact / conditional-steering audit; no inference, optimizer or native host.

Reads original immutable primary bytes, checks three saved optimizer inventories
and paired W0/synthetic TRAIN identities, then measures DEVELOPMENT six-logit
raw behaviour separately from host legal projections and literal EOS7.
Static probes alone never qualify online painting or model usefulness.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

ZIP_SHA = "ddf10d2ed16f0456067bea6b86ef1846b6318a5dac07397512d159b718b388a8"
SOURCE = "c14731f875d7334eae213c2bd74230bd310e7167"
MODEL_REV = "fc6221ca597f3315e4f82fc2df606783267b34ba"
W0 = "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
SUPPORT = "cc27b3206cb61f98432fa95b799c9b9ff508c2bc"
VERSION = "lfm-worm-w9-six-way-paired-CE-fresh-v1"
PREFIX = "worm-w9/"
ACTIONS = ("STRAIGHT", "LEFT", "RIGHT", "BRANCH", "RETRACE", "STOP")
FAMILIES = ("greebles", "pipework", "architecture")

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def readj(z, name):
    return json.loads(z.read(PREFIX + name))

def validate_pair(original, synthetic):
    require(original["origin"] == "W0-observed-TRAIN" and
            synthetic["origin"] == "synthetic-TRAIN-not-W0",
            "TRAIN source/origin misrepresented")
    require(original["family"] == synthetic["family"] and
            original["pair_id"] == synthetic["pair_id"],
            "TRAIN pair identity mismatch")
    a, b = json.loads(original["prompt"]), json.loads(synthetic["prompt"])
    require(canonical(a) == original["prompt"] and
            canonical(b) == synthetic["prompt"], "noncanonical TRAIN prompt")
    require(a["legal"] == original["legal"] and b["legal"] == synthetic["legal"],
            "TRAIN prompt/legal mismatch")
    require(original["completion"] in a["legal"] and
            synthetic["completion"] in b["legal"], "illegal TRAIN teacher")
    require({k: v for k, v in a.items() if k != "legal"} ==
            {k: v for k, v in b.items() if k != "legal"},
            "synthetic TRAIN changes real observed state")
    require(b["legal"] == [x for x in a["legal"] if x != original["completion"]] and
            synthetic["excluded_action"] == original["completion"] and
            original["completion"] not in b["legal"] and
            "STOP" in b["legal"], "bad synthetic exclusion")

def assess_probe(row):
    legal = row["legal"]
    scores = row["scores"]
    require(len(scores) == 6 and all(type(n) in (float, int) and
            math.isfinite(n) for n in scores), "nonfinite or malformed six logits")
    require(len(legal) == len(set(legal)) and
            set(legal) <= set(ACTIONS) and legal and
            row["expected"] in legal, "invalid DEVELOPMENT legality/teacher")
    top = ACTIONS[max(range(6), key=lambda i: scores[i])]
    masked = max((a for a in ACTIONS if a in legal),
                 key=lambda a: scores[ACTIONS.index(a)])
    require(row["raw_score_top"] == top and
            row["legal_projected_top"] == masked and
            row["teacher_raw_match"] == (top == row["expected"]) and
            row["teacher_legal_projected_match"] == (masked == row["expected"]),
            "DEV logits/teacher attribution inconsistent")
    variation = row["synthetic_exclusion"]
    if variation is not None:
        require(variation["synthetic_not_W0_observed"] is True and
                variation["excluded"] == row["expected"] and len(legal) > 1,
                "synthetic DEVELOPMENT source misrepresented")
        altered_legal = [a for a in legal if a != variation["excluded"]]
        require(altered_legal, "empty synthetic DEV legality")
        score2 = variation["scores"]
        require(len(score2) == 6 and all(type(n) in (int,float) and
                math.isfinite(n) for n in score2), "bad synthetic six logits")
        top2 = ACTIONS[max(range(6), key=lambda i: score2[i])]
        masked2 = max((a for a in ACTIONS if a in altered_legal),
                      key=lambda a: score2[ACTIONS.index(a)])
        require(variation["raw_score_top"] == top2 and
                variation["new_legal_top"] == masked2 and
                variation["unmasked_score_top_changed"] == (top != top2),
                "synthetic DEV response/host-masked argmax misreported")
    return {"top": top, "masked": masked,
            "synthetic_changes_raw_top": variation is not None and
                 variation["unmasked_score_top_changed"],
            "raw_teacher": top == row["expected"],
            "legal_teacher": masked == row["expected"]}

def static_probe(z, name):
    data = readj(z, name)
    require(data["source"] == "novel-WORM-DEVELOPMENT-not-FINAL" and
            len(data["samples"]) == len(data["raw_first_passes"]) == 18 and
            data["denominator"] == 18, "static DEVELOPMENT denominator/source")
    seen = set()
    results = []
    for record, raw in zip(data["samples"], data["raw_first_passes"]):
        require(record["id"] == raw["id"] and record["family"] in FAMILIES and
                record["id"] not in seen, "mispaired/duplicate raw first pass")
        seen.add(record["id"])
        results.append(assess_probe(record))
    require(data["score_argmax"] == dict(Counter(r["top"] for r in results)) and
            data["score_legal_top"] == dict(Counter(r["masked"] for r in results)) and
            data["score_teacher"] == sum(r["raw_teacher"] for r in results) and
            data["legal_score_teacher"] == sum(r["legal_teacher"] for r in results) and
            data["paired_exclusions"] == sum(r["synthetic_exclusion"] is not None
                                              for r in data["samples"]) and
            data["paired_unmasked_changes"] ==
                sum(r["synthetic_changes_raw_top"] for r in results),
            "W9 primary score/response totals contradict original logits")
    return data, results

def audit(path, out):
    require(not out.exists(), "immutable support output already exists")
    require(sha(path.read_bytes()) == ZIP_SHA, "original W9 artifact ZIP SHA mismatch")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        require(len(names) == len(set(names)) and z.testzip() is None and
                all(not x.startswith("/") and "\\" not in x and
                    ".." not in PurePosixPath(x).parts for x in names),
                "duplicate/unsafe/bad-CRC ZIP members")
        summary, lease = readj(z, "summary.json"), readj(z, "lease.json")
        manifest, preflight = readj(z, "corpus-manifest.json"), readj(z, "preflight.json")
        require(lease["source_sha"] == summary["source_sha"] == SOURCE and
                lease["experiment"] == VERSION and
                summary["revision"] == MODEL_REV and
                preflight["config"]["model"]["revision"] == MODEL_REV and
                preflight["config"]["source_sha"] == SOURCE and
                lease["W0_source"] == manifest["W0_source"] == W0 and
                lease["support_source"] == manifest["support_source"] == SUPPORT,
                "W9 actual source/full-model/W0 identity changed")
        require(summary["status"] == "complete" and
                summary["actual_optimizer_updates"] ==
                summary["requested_updates"] == 45 and
                summary["real_backward_passes"] == 90 and
                summary["checkpoint"] == "checkpoint-000045" and
                summary["frozen_base_unchanged"] is True and
                summary["adapter_changed"] is True and
                summary["gallery_model_rollouts"] == 0 and
                summary["gallery_deferred"] is True and
                summary["native_execution"] is False and
                summary["sealed_FINAL_opened"] is False and
                manifest["TRAIN_W0_original"] == 45 and
                manifest["TRAIN_synthetic_exclusions"] == 45 and
                manifest["DEV_optimizer_rows"] == 0 and
                manifest["TRAIN_DEV_prompt_overlap"] == 0 and
                manifest["FINAL_opened"] is False, "W9 optimizer/evaluation/FINAL gate mismatch")
        recipe = preflight["config"]["recipe"]
        require(recipe["objective"] ==
                "paired-six-code-first-next-token-conditional-CE" and
                recipe["steps"] == recipe["pairs"] == 45 and
                recipe["two_backward_passes_per_AdamW_update"] is True and
                recipe["W8_language_model_loss_used"] is False,
                "W9 optimizer actually uses wrong objective")
        ids = recipe["code_token_ids"]
        require(set(ids) == set(ACTIONS) and len(set(ids.values())) == 6,
                "six pinned single-digit action token IDs missing")
        original_rows = z.read(PREFIX+"optimizer-pairs.jsonl")
        require(sha(original_rows) == manifest["pair_sha256"] ==
                summary["pair_sha256"], "corrupted actual TRAIN bytes")
        rows = [json.loads(line) for line in original_rows.splitlines()]
        require(len(rows) == 90 and len({x["id"] for x in rows}) == 90 and
                len({x["prompt"] for x in rows}) == 90,
                "TRAIN pair/duplicate counts wrong")
        pairs = {}
        targets = Counter()
        for i in range(0, 90, 2):
            a, b = rows[i:i+2]
            validate_pair(a,b)
            require(a["pair_id"] not in pairs, "duplicate TRAIN pair")
            pairs[a["pair_id"]] = (a,b)
            targets[(a["family"], a["completion"])] += 1
        require(len(pairs) == 45 and
                targets == Counter({(f,a):3 for f in FAMILIES
                                    for a in ACTIONS if a != "STOP"}),
                "unbalanced/incorrect TRAIN originals")
        order = readj(z,"sampler-receipt.json")
        indices = order["pair_order"]
        require(sorted(indices) == list(range(45)) and
                order["steps"] == 45 and
                order["pair_ids"] == [[rows[2*i]["id"],rows[2*i+1]["id"]]
                                       for i in indices],
                "TRAIN sampler not an exact shuffled permutation")
        events = [json.loads(x) for x in z.read(PREFIX+"progress.jsonl").splitlines()]
        require(len(events) == 45, "optimizer events incomplete")
        for i, (event, idx) in enumerate(zip(events,indices),1):
            a,b = rows[2*idx:2*idx+2]
            require(event["global_step"] == i and event["pair_id"] == a["pair_id"] and
                    event["original_id"] == a["id"] and
                    event["synthetic_id"] == b["id"] and
                    event["two_genuine_backward_passes"] == 2 and
                    all(type(event[k]) in (int,float) and math.isfinite(event[k]) and
                        event[k] > 0 for k in
                        ("loss_original","loss_synthetic","gradient_norm","elapsed_seconds")),
                    "W9 recorded optimizer event fails custody check")
        ck = {}
        members = 0
        for step in (15,30,45):
            folder = PREFIX+f"checkpoint-{step:06d}/"
            inventory = json.loads(z.read(folder+"inventory.json"))
            actual = {n[len(folder):] for n in names if n.startswith(folder)} - {"inventory.json"}
            require(set(inventory) == actual and len(inventory) == 12,
                    "checkpoint incomplete or unlisted member")
            for name, receipt in inventory.items():
                payload = z.read(folder+name)
                require(len(payload) == receipt["bytes"] and
                        sha(payload) == receipt["sha256"],
                        "checkpoint original member checksum mismatches")
                members += 1
            ck[step] = inventory["adapter/adapter_model.safetensors"]["sha256"]
        require(len(set(ck.values())) == 3, "W9 adapters unchanged between checkpoints")
        before, b = static_probe(z,"untuned-probes.json")
        after, a = static_probe(z,"trained-probes.json")
        require([r["id"] for r in before["samples"]] ==
                [r["id"] for r in after["samples"]] and
                summary["untuned_score_teacher"] == before["score_teacher"] and
                summary["trained_score_teacher"] == after["score_teacher"] and
                summary["untuned_paired_unmasked_changes"] == before["paired_unmasked_changes"] and
                summary["trained_paired_unmasked_changes"] == after["paired_unmasked_changes"],
                "untrained/trained paired comparison contradicts source receipts")
        require(summary["strict_raw_first_passes"] == after["raw_first_passes"],
                "raw first-pass model text replaced in summary")
        report = {
            "schema":"worm-lfm-support-w9-original-conditional-audit-v1",
            "producer_sha": SOURCE, "producer_run":43743348779,
            "model_revision":MODEL_REV, "W0_source":W0,
            "support_dataset_source":SUPPORT, "original_zip_sha256":ZIP_SHA,
            "pair_sha256":sha(original_rows), "dataset_version":manifest["schema"],
            "zip_members_crc":len(names), "checkpoint_members_verified":members,
            "checkpoint_adapter_hashes":ck,
            "optimizer_updates":45,"backward_passes":90,
            "TRAIN_original_W0":45,"TRAIN_synthetic_legal_exclusions":45,
            "distinct_static_development_requests":18,
            "before_raw_score_top":before["score_argmax"],
            "after_raw_score_top":after["score_argmax"],
            "before_teacher":before["score_teacher"],
            "after_teacher":after["score_teacher"],
            "before_host_projected_teacher":before["legal_score_teacher"],
            "after_host_projected_teacher":after["legal_score_teacher"],
            "before_synthetic_raw_top_changes":before["paired_unmasked_changes"],
            "after_synthetic_raw_top_changes":after["paired_unmasked_changes"],
            "before_raw_EOS7":before["raw_true_EOS7_syntax"],
            "after_raw_EOS7":after["raw_true_EOS7_syntax"],
            "before_raw_host_legal":before["raw_host_legal"],
            "after_raw_host_legal":after["raw_host_legal"],
            "per_family_after": {
                family: {"teacher":sum(r["raw_teacher"] for r,x in zip(a,after["samples"])
                                      if x["family"]==family),
                         "top_actions":dict(Counter(r["top"] for r,x in zip(a,after["samples"])
                                                    if x["family"]==family))}
                for family in FAMILIES},
            "actual_model_online_decisions":0,"original_model_PNGs":0,
            "native_Godot_paintings":0,"new_optimizer_updates":0,
            "unverified_from_zip":"TRAIN/DEV semantic non-overlap and real online policy utility",
            "sealed_FINAL_opened":False}
        out.mkdir(parents=True)
        (out/"report.json").write_text(canonical(report)+"\n",encoding="utf-8")
        with (out/"dev-score-comparison.jsonl").open("w",encoding="utf-8") as f:
            for old, new in zip(before["samples"],after["samples"]):
                f.write(canonical({"id":new["id"],"family":new["family"],
                    "world_group":new["world_group"],
                    "legal":new["legal"],"expected":new["expected"],
                    "untuned_raw_top":old["raw_score_top"],
                    "trained_raw_top":new["raw_score_top"],
                    "trained_host_masked_top":new["legal_projected_top"],
                    "untuned_logits":old["scores"],"trained_logits":new["scores"],
                    "trained_synthetic":new["synthetic_exclusion"]})+"\n")
        (out/"README.txt").write_text(
            "Original W9 checkpoint-only static evidence. No model inference, online W0 "
            "painting, native Godot, new optimizer or useful-model freeze performed here.\n",
            encoding="utf-8")
        (out/"SHA256SUMS.txt").write_text("".join(
            sha(p.read_bytes())+"  "+p.name+"\n"
            for p in sorted(out.iterdir()) if p.is_file()),encoding="utf-8")
        return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--artifact",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    print(canonical(audit(args.artifact,args.out)))

if __name__=="__main__":
    main()
