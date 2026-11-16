#!/usr/bin/env python3
"""Read-only SPROCKET W3 original-artifact auditor. Never launches an optimizer or model.

The optional W0 code path is trusted/source-pinned repository code, NEVER imported
from the untrusted artifact. No sealed FINAL, native writes, or output repairs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from zipfile import ZipFile

ACTIONS = ("STRAIGHT", "LEFT", "RIGHT", "BRANCH", "RETRACE", "STOP")
FAMILIES = ("greebles", "pipework", "architecture")
W0_SHA = "81c3923cb38d7f196b99986806216388c34ebc27fd4976e168b3b370c25a323e"
PRODUCER = "18cc9baaedc5887f1b8a1770741871242c0c3704"

def check(condition, message):
    if not condition:
        raise ValueError(message)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def audit(path, trusted_w0=None, expected_sha=None):
    with path.open("rb") as handle:
        archive_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    if expected_sha:
        check(archive_sha == expected_sha, "archive checksum mismatch")
    with ZipFile(path) as z:
        info = z.infolist()
        names = [i.filename for i in info]
        check(len(names) == len(set(names)), "duplicate ZIP member")
        check(all(not n.startswith("/") and ".." not in Path(n).parts
                  and "\\" not in n for n in names), "unsafe ZIP path")
        check(all(0 <= i.file_size <= 50_000_000 for i in info)
              and sum(i.file_size for i in info) < 500_000_000, "archive bounds")
        check(z.testzip() is None, "ZIP CRC failure")
        listed = {}
        for line in z.read("SHA256SUMS.txt").decode("ascii").splitlines():
            checksum, filename = line.split("  ", 1)
            filename = filename.removeprefix("./")
            check(filename in names and filename not in listed, "bad SHA inventory")
            check(sha(z.read(filename)) == checksum, "member SHA failure " + filename)
            listed[filename] = checksum
        check(set(listed) == set(names) - {"SHA256SUMS.txt"}, "incomplete SHA inventory")
        check(z.read("source-sha.txt").decode().strip() == PRODUCER, "producer drift")
        check(sha(z.read("pinned-worm-original.py")) == W0_SHA, "W0 source drift")
        request = json.loads(z.read("request.json"))
        check(request["w0_git_commit"] == "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
              and request["support_git_commit"] == "2b6534ac21542805f4bda123f67e9624d21e39d7"
              and request["steps"] == 72 and request["native"] is False,
              "request/source contract drift")
        summary = json.loads(z.read("campaign/summary.json"))
        check(summary["actual_training"] and summary["status"] == "complete"
              and summary["steps"] == 72 and summary["frozen_base_equal"]
              and summary["adapter_changed"], "training status contradiction")
        progress = [json.loads(s) for s in z.read("campaign/progress.jsonl").splitlines()]
        check([r["step"] for r in progress] == list(range(1, 73)), "optimizer step continuity")
        check(all(math.isfinite(r["loss"]) and math.isfinite(r["grad_norm"])
                  and r["grad_norm"] > 0 for r in progress), "invalid gradient receipts")
        adapter_hashes = []
        inv_files = 0
        for step in (12, 24, 36, 48, 60, 72):
            prefix = "campaign/checkpoint-%06d/" % step
            invbytes = z.read(prefix + "inventory.json")
            inv = json.loads(invbytes)
            check(set(inv) == {n.removeprefix(prefix) for n in names
                              if n.startswith(prefix)} - {"inventory.json"},
                  "incomplete checkpoint " + prefix)
            for filename, expected in inv.items():
                data = z.read(prefix + filename)
                check(len(data) == expected["bytes"]
                      and sha(data) == expected["sha256"], "checkpoint mismatch " + filename)
                inv_files += 1
            adapter_hashes.append(inv["adapter/adapter_model.safetensors"]["sha256"])
            if step == 72:
                check(sha(invbytes) == summary["checkpoint_inventory_sha256"],
                      "final checkpoint inventory drift")
        check(len(set(adapter_hashes)) == 6, "same adapter in distinct checkpoints")
        manifest = json.loads(z.read("campaign/dataset/manifest.json"))
        splits = {}
        for split in ("train", "eval"):
            data = z.read("campaign/dataset/" + split + ".jsonl")
            rows = [json.loads(line) for line in data.splitlines()]
            check(len(rows) == manifest[split]["rows"]
                  and sha(data) == manifest[split]["sha256"], "dataset manifest drift")
            check(all(r["completion"] in ACTIONS and r["completion"] in r["legal"]
                      for r in rows), "illegal reference labels")
            splits[split] = rows
        train_prompt = {r["prompt"] for r in splits["train"]}
        eval_prompt = {r["prompt"] for r in splits["eval"]}
        check(not train_prompt & eval_prompt, "TRAIN/EVAL literal prompt leak")
        check({r["group"] for r in splits["train"]}.isdisjoint(
              r["group"] for r in splits["eval"]), "world-group leak")
        probes = json.loads(z.read("campaign/precommitted_probes.json"))
        check(len(probes) == 18 and len({p["id"] for p in probes}) == 18
              and Counter((p["family"], p["target"]) for p in probes)
              == Counter({(f, a): 1 for f in FAMILIES for a in ACTIONS}),
              "precommitted probe mismatch")
        tuned = json.loads(z.read("campaign/trained-raw-first-pass.json"))
        untuned = json.loads(z.read("campaign/untuned-raw-first-pass.json"))
        check([r["id"] for r in tuned] == [r["id"] for r in untuned]
              == [r["id"] for r in probes], "probe identity drift")
        check(not {r["id"] for r in tuned} & {r["id"] for r in splits["train"]},
              "TRAIN/probe identity leak")
        check(all(r["eos5"] and r["raw_legal"] and r["content"] in r["legal_actions"]
                  and r["completion_ids"][-1] == 5
                  and r["raw"] == r["content"] + "<|end|>" for r in tuned),
              "trained first-pass token or legality mismatch")
        check(all(not r["eos5"] and not r["raw_legal"] for r in untuned),
              "untuned first-pass contradiction")
        controls = [json.loads(line) for line in z.read("baseline-receipts.jsonl").splitlines()]
        check(len(controls) == 18, "incomplete model-free controls")
        replayed = 0
        if trusted_w0:
            check(sha(trusted_w0.read_bytes()) == W0_SHA, "trusted W0 SHA mismatch")
            spec = importlib.util.spec_from_file_location("pinned_repo_w0", trusted_w0)
            check(spec and spec.loader, "cannot load trusted W0")
            w0 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(w0)
            for receipt in controls:
                family, policy, seed = receipt["id"].rsplit("-", 2)
                check(family in FAMILIES and policy in ("random", "wall", "fidget"),
                      "unexpected model-free control")
                original = json.loads(z.read("baselines/" + receipt["id"] + ".json"))
                record = w0.run(int(seed), policy, family, 72, 96)
                check(record == original, "W0 JSON replay mismatch " + receipt["id"])
                png = z.read("baselines/" + receipt["id"] + ".png")
                check(w0.png_bytes(record) == png, "W0 pixel replay mismatch " + receipt["id"])
                check(sha(png) == receipt["sha256_png"]
                      and record["metrics"] == receipt["metrics"], "W0 receipt mismatch")
                replayed += 1
        return {
            "schema": "sprocket-w3-original-support-audit/1",
            "producer_source": PRODUCER,
            "archive_sha256": archive_sha,
            "zip_crc_members": len(names),
            "internal_sha256_members": len(listed),
            "checkpoint_files_rehashed": inv_files,
            "distinct_adapter_checkpoints": len(set(adapter_hashes)),
            "optimizer_steps_with_finite_nonzero_gradient": len(progress),
            "train_rows": len(splits["train"]), "development_rows": len(splits["eval"]),
            "train_development_literal_overlap": len(train_prompt & eval_prompt),
            "trained_raw_legal_eos": sum(r["eos5"] and r["raw_legal"] for r in tuned),
            "untuned_raw_legal_eos": sum(r["eos5"] and r["raw_legal"] for r in untuned),
            "probe_count": len(tuned),
            "teacher_matches": sum(r["content"] == r["teacher_reference_not_quality_gate"]
                                   for r in tuned),
            "trained_action_histogram": dict(Counter(r["content"] for r in tuned)),
            "confusion_by_teacher": {
                a: dict(Counter(r["content"] for r in tuned
                                if r["teacher_reference_not_quality_gate"] == a))
                for a in ACTIONS
            },
            "w0_model_free_controls": len(controls),
            "w0_full_json_and_pixel_replays": replayed,
            "caveat": "Static synthetic-WORM first passes only; NOT online model rollouts, native Godot paint, or useful-model freeze",
        }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--trusted-w0", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = audit(args.artifact, args.trusted_w0, args.expected_sha256)
    payload = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.report:
        args.report.write_text(payload, encoding="utf-8")
    print(payload)

if __name__ == "__main__":
    main()
