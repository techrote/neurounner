#!/usr/bin/env python3
"""WORM LFM W7: frozen, model-free, held-out W0 stimulus challenge.

Original W0 96x96 PNGs are real source-faithful baseline images.
Counterfactual prompt edits are NOT W0-executed trajectories or valid native scenes.
No model inference, optimizer, FINAL, teacher labels or host authorization here.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
import worm

SCHEMA = "worm-lfm-w7-heldout-observation-sensitivity-v1"
SEEDS = (3101, 3103, 3109, 3119)  # neither support v2 TRAIN nor W7 DEV/eval worlds
FAMILIES = tuple(worm.FAMILIES)
MODES = ("wall", "fidget")
KINDS = ("observed", "family_only", "history_only", "legal_subset")
CODES = dict(zip(worm.ACTIONS, "012345"))
CODE_ACTIONS = {v: k for k, v in CODES.items()}
KEYS = ("family", "heading", "local_5x5", "legal", "last_actions", "tick")

def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def prompt(observation):
    return canonical({key: observation[key] for key in KEYS})

def variants(original, stem):
    """One actual W0 observation and three explicitly NON-host-replayed edits."""
    if original["family"] not in FAMILIES or len(original["local_5x5"]) != 5:
        raise ValueError("invalid W0 observation")
    legal = original["legal"]
    if not legal or len(legal) != len(set(legal)) or set(legal) - set(worm.ACTIONS):
        raise ValueError("bad W0 legal mask")
    if len(legal) <= 1:
        raise ValueError("cannot construct informative legal-subset counterfactual")
    cases = []
    for kind in KINDS:
        obs = copy.deepcopy(original)
        removed = None
        if kind == "family_only":
            obs["family"] = FAMILIES[(FAMILIES.index(obs["family"]) + 1) % len(FAMILIES)]
        elif kind == "history_only":
            obs["last_actions"] = ["RIGHT"] * 4 if obs["last_actions"] != ["RIGHT"] * 4 else ["LEFT"] * 4
        elif kind == "legal_subset":
            removed = next(a for a in ("STRAIGHT", "RIGHT", "LEFT", "BRANCH", "RETRACE") if a in legal)
            obs["legal"] = [a for a in legal if a != removed]
        text = prompt(obs)
        cases.append({"id": stem + ":" + kind, "pair_id": stem, "kind": kind,
                      "observation_origin": "actual-W0" if kind == "observed" else "synthetic-prompt-edit-not-W0-executed",
                      "removed_action": removed, "observation": obs, "prompt": text,
                      "prompt_sha256": sha(text.encode("utf-8"))})
    if len({v["prompt_sha256"] for v in cases}) != 4:
        raise ValueError("no unique counterfactual stimuli")
    return cases

def generate(destination):
    destination.mkdir(parents=True, exist_ok=False)
    images = destination / "w0-controls"
    images.mkdir()
    rows = []
    image_hashes = {}
    for seed in SEEDS:
        for family in FAMILIES:
            for mode in MODES:
                record = worm.run(seed, mode=mode, family=family, steps=32)
                if record["steps_executed"] != 32 or len(record["events"]) != 32:
                    raise ValueError("W0 control unexpectedly stopped")
                name = f"{family}-{mode}-{seed}"
                # Full host trajectory, unmodified source-faithful W0 PNG.
                png = worm.png_bytes(record)
                payload = (canonical(record) + "\n").encode()
                (images / (name + ".png")).write_bytes(png)
                (images / (name + ".json")).write_bytes(payload)
                if worm.png_bytes(json.loads(payload)) != png:
                    raise ValueError("PNG source replay mismatch")
                image_hashes[name] = sha(png)
                tick = 12 if mode == "wall" else 24
                observation = record["events"][tick]["observation"]
                assert observation["family"] == family and observation["tick"] == tick
                for row in variants(observation, name):
                    row.update({"source_seed": seed, "source_mode": mode,
                                "source_tick": tick, "original_w0_trace_sha256": sha(payload)})
                    rows.append(row)
    if len(rows) != 96 or len({r["id"] for r in rows}) != 96:
        raise ValueError("expected 24 independent W0 state groups / 96 stimuli")
    # No supervised target action is emitted, and no arbitrary candidate changes W0 permissions.
    content = "".join(canonical(r) + "\n" for r in rows).encode()
    (destination / "challenge.jsonl").write_bytes(content)
    summary = {"schema": SCHEMA, "W0_source": "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61",
               "world_seeds": list(SEEDS), "world_count": len(SEEDS),
               "family_labels_per_world": len(FAMILIES),
               "original_W0_control_PNGs": len(image_hashes), "host_observed_groups": 24,
               "counterfactual_rows_not_host_executed": 72, "total_requests": len(rows),
               "challenge_sha256": sha(content), "original_png_sha256": image_hashes,
               "model_training_steps": 0, "native_Godot_execution": False,
               "sealed_FINAL_opened": False,
               "limits": "Prompt edits test policy sensitivity only, not native legality or trajectory success; world seeds, not family labels, are independent worlds."}
    (destination / "manifest.json").write_text(canonical(summary) + "\n")
    return rows, summary

def score(rows, responses):
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate or missing source cases")
    answers = {}
    for response in responses:
        ident = response.get("id")
        if ident not in by_id or ident in answers:
            raise ValueError("unexpected/duplicate model response")
        ids = response.get("token_ids")
        if (not isinstance(ids, list) or not 1 <= len(ids) <= 64
                or any(type(i) is not int or not 0 <= i <= 500000 for i in ids)
                or type(response.get("eos_terminated")) is not bool
                or response["eos_terminated"] != (ids[-1] == 7)
                or type(response.get("raw_text")) is not str):
            raise ValueError("malformed original first-pass token/EOS evidence")
        if response.get("prompt_sha256") != by_id[ident]["prompt_sha256"]:
            raise ValueError("model response does not match frozen prompt identity")
        answers[ident] = response
    if set(answers) != set(by_id):
        raise ValueError("incomplete model batch: cannot cherry-pick responses")
    cases = []
    for row in rows:
        response = answers[row["id"]]
        action = CODE_ACTIONS.get(response["raw_text"]) if response["eos_terminated"] else None
        legal = action in row["observation"]["legal"]
        cases.append({"id": row["id"], "pair_id": row["pair_id"], "kind": row["kind"],
                      "raw_text": response["raw_text"], "token_ids": response["token_ids"],
                      "true_eos7": response["eos_terminated"], "decoded_action": action,
                      "valid_code_and_eos": action is not None, "raw_legal": legal})
    by_case = {x["id"]: x for x in cases}
    pairs = []
    for row in rows:
        if row["kind"] != "observed":
            continue
        base = by_case[row["id"]]
        for kind in KINDS[1:]:
            other = by_case[row["pair_id"] + ":" + kind]
            removed = next(r["removed_action"] for r in rows if r["id"] == other["id"])
            pairs.append({"pair_id": row["pair_id"], "variant": kind,
                          "base_action": base["decoded_action"], "edit_action": other["decoded_action"],
                          "raw_action_changed": base["raw_text"] != other["raw_text"],
                          "new_raw_legal": other["raw_legal"],
                          "baseline_action_removed": removed is not None and removed == base["decoded_action"]})
    stats = {kind: {"requests": sum(x["kind"] == kind for x in cases),
                    "raw_valid_eos": sum(x["kind"] == kind and x["valid_code_and_eos"] for x in cases),
                    "raw_legal": sum(x["kind"] == kind and x["raw_legal"] for x in cases),
                    "raw_codes": {code: sum(x["kind"] == kind and x["raw_text"] == code for x in cases)
                                  for code in "012345"}}
             for kind in KINDS}
    return {"schema": SCHEMA + "-scored", "source_requests": len(rows),
            "per_kind": stats, "paired_stimuli": len(pairs),
            "by_perturbation": {kind: {"pairs": sum(p["variant"] == kind for p in pairs),
                                      "raw_action_changes": sum(p["variant"] == kind and p["raw_action_changed"] for p in pairs),
                                      "removed_base_action_cases": sum(p["variant"] == kind and p["baseline_action_removed"] for p in pairs),
                                      "removed_base_action_cases_newly_legal": sum(p["variant"] == kind and p["baseline_action_removed"] and p["new_raw_legal"] for p in pairs)}
                                for kind in KINDS[1:]},
            "raw_cases": cases, "pairs": pairs,
            "warning": "A changed action is not automatically better art. This is original raw model first-pass evidence, never host-fallback credit; synthetic prompts do NOT have genuine host replay."}

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", type=Path)
    parser.add_argument("--challenge", type=Path)
    parser.add_argument("--first-passes", type=Path)
    parser.add_argument("--report", type=Path)
    opts = parser.parse_args()
    if opts.generate:
        _, result = generate(opts.generate)
        print(canonical({k: result[k] for k in ("schema", "world_count", "total_requests", "challenge_sha256", "original_W0_control_PNGs")}))
    else:
        if not all((opts.challenge, opts.first_passes, opts.report)):
            parser.error("scoring requires --challenge --first-passes --report")
        rows = read_jsonl(opts.challenge)
        result = score(rows, read_jsonl(opts.first_passes))
        opts.report.write_text(canonical(result) + "\n")
        print(canonical({"schema": result["schema"], "requests": result["source_requests"],
                         "paired_stimuli": result["paired_stimuli"], "summary": result["by_perturbation"]}))

if __name__ == "__main__":
    main()
