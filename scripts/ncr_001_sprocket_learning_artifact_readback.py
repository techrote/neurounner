#!/usr/bin/env python3
"""NCR-001 original Sprocket diagnostic evidence readback, no model/training.

Reads exactly the three ORIGINAL Actions ZIPs from PR #159, refusing source,
weight, gradient, frozen-base or scientific-admission substitutions. ZIPs
are staging (expire 2026-11-27); this verifier is not archival promotion.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "40710171622"
HEAD = "df8479aae34550cba014d910a73e30e59cd493e2"
REV = "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
FORMAT_SHA = "1ca6cfde8b4144e6f56fdfb61fb075e31fdc9b4ebc68c57c01128c363b"
ARTIFACTS = {
    "source": ("c66706d3e541925638e901d37232f204c480f0016aba393023fa925c67a61242",
               {"source-tests.log": ("6e93487ad1984eec8bb80c5a5e55534cc303acc878ec0a14acd5577ba3465a35", 1729),
                "lease-tests.log": ("f6fadaaac9cf97dcf63c39f2dcbb14b42cb08ad6a285903bacf2fd50234eb70d", 1883)}),
    "chat": ("e040e1ce982a1c0f0c5e99e014cada04cc41b334f1755d60cf37fd0bb909a05b",
             {"host.log": ("24bc76df7bbf527ce16ac679c9be346827d02945979c9bfffb009a76333b5ed1", 237),
              "receipt.json": ("27a0a2890d427f39b8214cf0d16a2a4530f9965d597708133be74df87157e3b5", 5978)}),
    "instruct": ("9656ff3f8019f497cff62d816276f44bdc33f149c07072357bd5cfd9c90d0f93",
                 {"host.log": ("2960eb4eb8a173fd4756be6a4d9563408c573ba2fc84b38f9b582228eeac4f66", 241),
                  "receipt.json": ("3dd6c391137dc1e6561360a0587b84c9cf55b4d4a7872d73c089eacdbda31b9a", 6097)})
}
WEIGHTS = {"chat": ("sprocket_chat", "chat/model.safetensors",
                    "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97"),
           "instruct": ("sprocket_instruct", "model.safetensors",
                        "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864")}
SOURCES = ("experiments/ncr-001/protocol.json", "experiments/ncr-001/source-pins.json",
           "scripts/ncr_001_backbone_lease.py", "scripts/ncr_001_core.py",
           "scripts/ncr_001_postlease_sprocket.py", "scripts/ncr_001_sprocket_chat.py",
           "scripts/ncr_001_sprocket_instruct.py", "scripts/ncr_001_sprocket_learning_gate.py")

def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def check(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)

def read_zip(path: Path, lane: str) -> dict[str, bytes]:
    expected_sha, members = ARTIFACTS[lane]
    b = path.read_bytes()
    check(sha(b) == expected_sha, "Original ZIP SHA mismatch: " + lane)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        check(len(names) == len(set(names)) and set(names) == set(members),
              "Original member set or duplicates: " + lane)
        data = {name: archive.read(name) for name in names}
    for name, (expected_hash, size) in members.items():
        check(len(data[name]) == size and sha(data[name]) == expected_hash,
              "Original member SHA/length wrong: " + lane + "/" + name)
    return data

def receipt_gate(r: dict, variant: str, *, source_tree: bool) -> dict:
    lane, path, weight = WEIGHTS[variant]
    check(r["lane"] == lane and r["experiment"] == "NCR-001" and
          r["status"] == "PASS_DIAGNOSTIC_NCA_LEARNING_GATES__FINAL_72_BLOCKED" and
          r["original_frozen_model_executed"] is True, "Wrong original model or physical status")
    for key in ("train_authorized", "comparative_training_performed",
                "preregistered_evaluation_performed", "heldout_opened",
                "model_training_experiment", "full_72_conformance_pass",
                "tool_memory_semantics_qualified", "independent_review_complete"):
        check(r[key] is False, "False scientific admission: " + key)
    check(type(r["diagnostic_optimizer_steps"]) is int and
          r["diagnostic_optimizer_steps"] == 1, "Not exactly one NCA-only diagnostic step")
    si = r["source_identity"]
    check(si["model_id"] == "kandivault/sprocket-500m" and
          si["revision"] == REV and si["weight_path"] == path and
          si["weight_sha256"] == weight and si["taps"] == [7,16,25] and
          si["width"] == 1280 and si["layers"] == 26 and si["model_type"] == "llama",
          "Pinned model/variant/taps substituted")
    check(r["verified_original_source_files"][path] ==
          {"bytes": 1002207984, "sha256": weight}, "Full safetensors identity wrong")
    runner = r["runner"]
    check(runner["head_sha"] == HEAD and runner["run_id"] == RUN_ID and
          runner["attempt"] == "1" and runner["torch"] == "2.14.1+cpu" and
          runner["transformers"] == "4.57.6" and runner["threads"] == 4 and
          runner["runner_os"] == "Linux" and runner["runner_arch"] == "X64" and
          bool(runner["runner_name"]) and bool(runner["host"]) and runner["pid"] > 0,
          "Source head/runner identity mismatch")
    check(r["formatter"]["commit"] == "2271218d625e1462b2cf218445da975efe17fe32"
          and r["formatter"]["sha256"] == FORMAT_SHA,
          "Ordinary-role formatter identity changed")
    check(set(r["source_sha256"]) == set(SOURCES), "Missing source hash")
    if source_tree:
        for name, original in r["source_sha256"].items():
            check(sha((ROOT / name).read_bytes()) == original,
                  "Current audited source differs from original physical execution: " + name)
        protocol = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
        check(protocol["train_authorized"] is False and
              set(protocol["models"]) == {"smollm2","sprocket_chat","sprocket_instruct","void1"},
              "Four-model protocol or training lock changed")
    f = r["synthetic_sample"]
    check(f["synthetic_fixture"] is True and f["heldout"] is False and
          f["assistant_eos_id"] == 5 and f["prefix_units"] > 0 and
          f["supervised_units"] > 1 and f["sequence_units"] ==
          f["prefix_units"] + f["supervised_units"] <= 96,
          "Assistant-only synthetic EOS/labels wrong")
    check(r["initial_zero_readout_source_parity_exact"] is True and
          r["backbone_before_sha256"] == r["backbone_after_sha256"] and
          len(r["backbone_before_sha256"]) == 64, "Frozen base changed or zero-readout differs")
    for key in ("assistant_only_finite_loss", "readout_first_step_gradient",
                "post_warmup_upstream_gradients", "one_diagnostic_update_changes_only_nca",
                "source_baseline_unchanged", "three_distinct_ordered_depth_states"):
        check(r["checks"][key] is True, "Physical gate reported false: " + key)
    warm, post = r["warmup"], r["post_warmup"]
    for stage in (warm,post):
        check(type(stage["loss"]) in (int,float) and 0 < stage["loss"] < float("inf"),
              "Supervised loss not positive finite")
    check(warm["readout_before_sha256"] != warm["readout_after_sha256"],
          "NCA-only diagnostic readout unchanged")
    check(warm["gradient_abs_sums"]["readout.weight"] > 0 and
          post["gradient_abs_sums"]["readout.weight"] > 0, "Readout gradient absent")
    for name in ("observe.weight","perception.weight","rule.weight"):
        check(warm["gradient_abs_sums"][name] == 0 and
              0 < post["gradient_abs_sums"][name] < float("inf"),
              "Post-warmup NCA upstream gradient absent: " + name)
    trace=post["ordered_field_trace"]
    check([x["tap_index"] for x in trace] == [0,1,2] and
          [x["next_tap"] for x in trace] == [1,2,3] and
          all(x["shape"] == [1,f["sequence_units"],4,8,8] for x in trace) and
          len(set(x["field_sha256"] for x in trace)) == 3,
          "Persistent-depth field trace wrong")
    return {"lane":lane,"source_weight_sha256":weight,"warmup_loss":warm["loss"],
            "post_warmup_loss":post["loss"],"post_warmup_gradient_abs_sums":
            post["gradient_abs_sums"],"backbone_fingerprint":
            r["backbone_before_sha256"],"diagnostic_optimizer_steps":1,
            "original_runner":runner["runner_name"]}

def negatives(receipts: dict[str,dict]) -> int:
    edits = [
        ("chat",lambda x:x.update(train_authorized=True)),
        ("instruct",lambda x:x.update(full_72_conformance_pass=True)),
        ("chat",lambda x:x["source_identity"].update(weight_path="model.safetensors")),
        ("instruct",lambda x:x["runner"].update(head_sha="0"*40)),
        ("chat",lambda x:x.update(diagnostic_optimizer_steps=2)),
        ("instruct",lambda x:x["warmup"]["gradient_abs_sums"].update({"readout.weight":0})),
        ("chat",lambda x:x["post_warmup"]["gradient_abs_sums"].update({"observe.weight":0})),
        ("instruct",lambda x:x["post_warmup"]["ordered_field_trace"][2].update({"tap_index":1})),
        ("chat",lambda x:x.update(backbone_after_sha256="0"*64)),
        ("instruct",lambda x:x["synthetic_sample"].update(assistant_eos_id=258)),
        ("chat",lambda x:x.update(tool_memory_semantics_qualified=True)),
        ("instruct",lambda x:x.update(heldout_opened=True)),
    ]
    for lane, mutation in edits:
        damaged=copy.deepcopy(receipts[lane]); mutation(damaged)
        try:
            receipt_gate(damaged,lane,source_tree=False)
        except (AssertionError,KeyError,TypeError,ValueError):
            continue
        raise AssertionError("Failed to detect adversarial semantic change: "+lane)
    return len(edits)

def main():
    p=argparse.ArgumentParser()
    for lane in ("source","chat","instruct"):
        p.add_argument("--"+lane+"-zip",required=True,type=Path)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    data={lane:read_zip(getattr(args,lane+"_zip"),lane) for lane in ARTIFACTS}
    check(b"Ran 10 tests" in data["source"]["source-tests.log"] and
          b"\nOK\n" in data["source"]["source-tests.log"] and
          b"Ran 10 tests" in data["source"]["lease-tests.log"] and
          b"\nOK\n" in data["source"]["lease-tests.log"], "Original source/negative tests not passed")
    receipts={lane:json.loads(data[lane]["receipt.json"]) for lane in WEIGHTS}
    result={lane:receipt_gate(receipts[lane],lane,source_tree=True) for lane in WEIGHTS}
    check(result["chat"]["backbone_fingerprint"] !=
          result["instruct"]["backbone_fingerprint"], "Sprocket variants conflated")
    for lane in WEIGHTS:
        check("PASS_DIAGNOSTIC_NCA_LEARNING_GATES__FINAL_72_BLOCKED" in
              data[lane]["host.log"].decode(), "Original frozen host log is negative")
    report={"classification":"ORIGINAL_SPROCKET_NCA_DIAGNOSTIC_READBACK_PASS__FINAL_72_BLOCKED",
            "original_source_head":HEAD,"original_run_id":RUN_ID,
            "original_zip_members_preserved_in_actions":6,
            "archive_promoted_to_git":False,"train_authorized":False,
            "comparative_training_performed":False,"independent_scientific_review":False,
            "full_72_conformance_pass":False,"adversarial_receipt_mutations_rejected":
            negatives(receipts),"results":result}
    text=json.dumps(report,indent=2,sort_keys=True)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text,encoding="utf-8")
    print(text)
if __name__=="__main__":
    main()
