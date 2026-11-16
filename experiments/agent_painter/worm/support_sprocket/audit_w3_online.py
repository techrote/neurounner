#!/usr/bin/env python3
"""Read-only original-byte audit of SPROCKET W3's actual ONLINE W0 rollouts.

No model call, optimizer, native painting, source mutation or repaired output.
The original GitHub artifact, source revisions and W0 replay are SHA-pinned.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from zipfile import ZipFile

ARCHIVE_SHA = "acd88eac8dd713d38026fb53ef87eb2591d066418a7067ab15e096e50652f4f1"
EVAL_SOURCE = "46fde5f5b67932325541a559c9d56e435ae903b0"
TRAIN_SOURCE = "18cc9baaedc5887f1b8a1770741871242c0c3704"
W0_SOURCE = "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
W0_SHA = "81c3923cb38d7f196b99986806216388c34ebc27fd4976e168b3b370c25a323e"
FAMILIES = ("greebles", "pipework", "architecture")
SEEDS = (211, 223)
MODES = ("random", "wall", "fidget", "sprocket-untuned", "sprocket-trained")

def check(value, description):
    if not value:
        raise ValueError(description)

def sha(blob):
    return hashlib.sha256(blob).hexdigest()

def motif(seq):
    """Longest literal raw-action cycle with >=3 repeats, periods 1..6."""
    result = (0, 0)
    for start in range(len(seq)):
        for period in range(1, 7):
            length = period
            while start + length < len(seq) and seq[start+length] == seq[start+length-period]:
                length += 1
            if length >= 3*period and length > result[0]:
                result = (length, period)
    return result

def audit(archive, trusted_w0):
    with archive.open("rb") as f:
        check(hashlib.file_digest(f, "sha256").hexdigest() == ARCHIVE_SHA,
              "external original ZIP checksum mismatch")
    check(sha(trusted_w0.read_bytes()) == W0_SHA, "pinned W0 source drift")
    spec = importlib.util.spec_from_file_location("original_pinned_w0", trusted_w0)
    check(spec and spec.loader, "pinned W0 import failed")
    w0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w0)
    with ZipFile(archive) as z:
        members = z.infolist()
        names = [m.filename for m in members]
        check(len(set(names)) == len(names), "duplicate ZIP member")
        check(all(not n.startswith("/") and "\\" not in n and ".." not in Path(n).parts
                  for n in names), "unsafe ZIP paths")
        check(all(m.file_size < 2_000_000 for m in members)
              and sum(m.file_size for m in members) < 1_000_000, "ZIP size bound")
        check(z.testzip() is None, "ZIP CRC failure")
        inventory = {}
        for line in z.read("SHA256SUMS.txt").decode("ascii").splitlines():
            digest, name = line.split("  ", 1)
            name = name.removeprefix("./")
            check(name in names and name not in inventory, "bad SHA inventory")
            check(sha(z.read(name)) == digest, "original member SHA drift: "+name)
            inventory[name] = digest
        check(set(inventory) == set(names) - {"SHA256SUMS.txt"}, "missing SHA inventory members")
        check(z.read("source-sha.txt").decode().strip() == EVAL_SOURCE, "eval producer SHA drift")
        request = json.loads(z.read("request.json"))
        check(request["training_source"] == TRAIN_SOURCE
              and request["training_run_id"] == 43701129960
              and request["w0_git_commit"] == W0_SOURCE
              and request["checkpoint"] == "checkpoint-000072"
              and request["steps"] == 24 and request["native"] is False, "request mismatch")
        summary = json.loads(z.read("results/comparison.json"))
        check(summary["source_training"] == TRAIN_SOURCE
              and summary["pinned_w0"] == W0_SOURCE
              and summary["budget_per_trajectory"] == 24
              and summary["actual_native_art"] is False
              and summary["final_opened"] is False
              and len(summary["rollouts"]) == 30, "comparison provenance mismatch")
        counts = Counter()
        sequences, pngs, seen = {}, {}, set()
        trained_legal = untuned_refusals = full_replays = 0
        metrics = {}
        for family in FAMILIES:
            for seed in SEEDS:
                for mode in MODES:
                    name = f"{family}-{mode}-{seed}"
                    prefix = "results/png-and-json/"+name
                    obj = json.loads(z.read(prefix+".json"))
                    png = z.read(prefix+".png")
                    check(png.startswith(b"\x89PNG\r\n\x1a\n"), "missing original PNG "+name)
                    check(obj["family"] == family and obj["seed"] == seed
                          and obj["mode"] == mode and obj["size"] == 96
                          and obj["step_budget"] == 24
                          and obj["steps_executed"] == len(obj["events"]), "trace identity "+name)
                    compare_mode = mode.removeprefix("sprocket-")
                    rows = [r for r in summary["rollouts"]
                            if (r["family"],r["seed"],r["mode"]) == (family,seed,compare_mode)]
                    check(len(rows) == 1 and rows[0]["metrics"] == obj["metrics"], "summary "+name)
                    actions = [event["raw_choice"] for event in obj["events"]]
                    if mode.startswith("sprocket-"):
                        passes = obj["model_first_passes"]
                        check(len(passes) == len(actions), "first-pass denominator "+name)
                        for index,(event,p) in enumerate(zip(obj["events"],passes)):
                            check(p["tick"] == index and p["observation"] == event["observation"]
                                  and p["host_choice"] == event["raw_choice"]
                                  and p["applied"] == event["applied"]
                                  and p["invalid_host_action"] == (not event["raw_legal"])
                                  and len(p["prefix_ids"]) > 0 and len(p["completion_ids"]) > 0,
                                  "raw/applied first-pass drift "+name)
                            if mode == "sprocket-trained":
                                choice = event["raw_choice"]
                                check(p["eos5"] is True and p["legal_raw"] is True
                                      and p["completion_ids"][-1] == 5
                                      and p["completion_ids"].count(5) == 1
                                      and p["parsed_action_exact"] == choice
                                      and p["raw_output"] == choice+"<|end|>"
                                      and choice in event["observation"]["legal"]
                                      and event["applied"] == choice,
                                      "trained raw action rejected/repaired "+name)
                                counts[choice] += 1
                                trained_legal += 1
                            else:
                                check(p["eos5"] is False and p["legal_raw"] is False
                                      and p["parsed_action_exact"] is None
                                      and event["applied"] == "STOP", "untuned refusal "+name)
                                untuned_refusals += 1
                    policy = "replay" if mode.startswith("sprocket-") else mode
                    replay = w0.run(seed, policy, family, 24, 96,
                                    raw_actions=actions if policy == "replay" else None)
                    normalized = dict(obj)
                    if policy == "replay":
                        normalized.pop("model_first_passes")
                        normalized["mode"] = "replay"
                        normalized["engine"] = "python-stdlib-baseline-not-native"
                    check(normalized == replay, "W0 full JSON replay mismatch "+name)
                    check(w0.png_bytes(replay) == png, "W0 pixel-exact PNG mismatch "+name)
                    full_replays += 1
                    key = (family, seed, mode)
                    check(key not in seen, "duplicate rollout identity")
                    seen.add(key)
                    sequences[key] = tuple(actions)
                    pngs[key] = sha(png)
                    metrics[name] = {"unique_cells":obj["metrics"]["unique_trail_cells"],
                                     "branch_actions":obj["metrics"]["branches"],
                                     "retraces":obj["metrics"]["retraces"],
                                     "longest_raw_motif":motif(actions)}
        check(len(seen) == 30 and full_replays == 30, "original rollout completeness")
        check(trained_legal == 144 and untuned_refusals == 6
              and counts == Counter({"BRANCH":126,"RETRACE":18}),
              "W3 online first-pass/action-distribution drift")
        learned = {sequences[(f,s,"sprocket-trained")] for f in FAMILIES for s in SEEDS}
        distinct_families = {
            str(s):len({pngs[(f,s,"sprocket-trained")] for f in FAMILIES})
            for s in SEEDS}
        check(len(learned) == 1 and distinct_families == {"211":1,"223":1},
              "original cross-theme identical-policy finding differs")
        return {
            "schema":"sprocket-w3-original-online-independent-audit/1",
            "original_eval_zip_sha256":ARCHIVE_SHA,
            "trainer_source":TRAIN_SOURCE,"evaluator_source":EVAL_SOURCE,
            "zip_crc_members":len(members),"sha256_members":len(inventory),
            "original_png_and_full_JSON_replays":full_replays,
            "trained_legal_EOS_first_passes":trained_legal,
            "untuned_raw_refusals_with_host_STOP":untuned_refusals,
            "trained_raw_action_histogram":dict(counts),
            "raw_action_entropy_bits":round(-sum((n/144)*math.log2(n/144) for n in counts.values()),6),
            "distinct_learned_action_sequences_across_six_rollouts":len(learned),
            "distinct_learned_PNGs_per_seed_across_themes":distinct_families,
            "trained_seed211":metrics["greebles-sprocket-trained-211"],
            "wall_seed211":metrics["greebles-wall-211"],
            "random_seed211":metrics["greebles-random-211"],
            "scope":"original W0 Python readbacks; no Godot native painting, FINAL, or useful model freeze",
        }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("archive",type=Path)
    p.add_argument("--trusted-w0",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    a=p.parse_args()
    result=json.dumps(audit(a.archive,a.trusted_w0),sort_keys=True,indent=2)+"\n"
    a.report.write_text(result)
    print(result)
if __name__=="__main__":
    main()
