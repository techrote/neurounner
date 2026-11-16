#!/usr/bin/env python3
"""W13 STOP-unmasking first-divergence experiment; NO fresh neural forwards.

A changed legal menu also reweights NON-STOP moves. Therefore a one-step
STOP choice on an archived later state cannot be treated as an uncensored
rollout: the alternate trajectory may already have diverged. This tool finds
the EARLIEST changed action, then executes ONLY that observed-state decision
in immutable W0, producing two genuinely rendered, equally short PNGs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import audit_w13_original as original

VERSION = "cybersand.worm.lfm-w13-stop-first-divergence/1"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def choices_until_first_divergence(decisions: list, events: list) -> dict:
    original.need(len(decisions) == len(events) == 128, "not a 128-step source trace")
    first = None
    all_stop = 0
    all_changed = 0
    for tick, (d, e) in enumerate(zip(decisions, events)):
        receipt = d["sampling_receipt"]
        legal = e["observation"]["legal"]
        original.need(d["tick"] == e["observation"]["tick"] == tick
                      and receipt["legal"] == d["legal"] == legal,
                      "misaligned observation/receipt")
        original.need("STOP" in legal and "STOP" not in receipt["sampled_legal"],
                      "not an optional STOP-suppressed source observation")
        original.need(receipt["STOP_deferred_unless_forced"] is True
                      and receipt["sampler"] == "learned"
                      and receipt["model_fraction"] == 0.7
                      and receipt["temperature"] == 2.5,
                      "changed producer sampling policy")
        raw = d["raw_choice_selected_by_host_sampler"]
        original.need(raw == receipt["choice"] == e["raw_choice"] == e["applied"],
                      "source host choice mismatch")
        logits = d["model_six_code_scores"]["candidate_logits"]
        existing = receipt["sampled_legal"]
        original.need(existing == [a for a in original.ACTIONS if a in legal and a != "STOP"],
                      "source sampled menu mismatch")
        original.need(original.sample(original.mixture(logits, existing)[1],
                                      existing, receipt["host_u"]) == raw,
                      "source masked sampler changed")
        restored = [a for a in original.ACTIONS if a in legal]
        alt = original.sample(original.mixture(logits, restored)[1],
                              restored, receipt["host_u"])
        all_changed += alt != raw
        all_stop += alt == "STOP"
        if first is None and alt != raw:
            first = {"tick": tick, "original_action": raw, "restored_action": alt,
                     "original_observation": e["observation"],
                     "identical_prefix_length": tick}
    original.need(first is not None, "no STOP-unmasking causal divergence observed")
    return {"first_divergence": first,
            "same_state_STOP_selections_NOT_rollouts": all_stop,
            "same_state_changed_selections_NOT_rollouts": all_changed}


def paired_prefix(worm, seed: int, family: str,
                  selected: list[str], decision: dict) -> tuple[dict, dict]:
    first = decision["first_divergence"]
    tick = first["tick"]
    original.need(len(selected) == 128, "missing original action sequence")
    original.need(selected[tick] == first["original_action"], "action sequence mismatch")
    base = worm.run(seed=seed, family=family, mode="replay",
                    steps=tick + 1, size=96, raw_actions=selected[:tick+1])
    alt_seq = selected[:tick] + [first["restored_action"]]
    restored = worm.run(seed=seed, family=family, mode="replay",
                        steps=tick + 1, size=96, raw_actions=alt_seq)
    original.need(len(base["events"]) == tick+1
                  and len(restored["events"]) == tick+1,
                  "bad one-step prefix length")
    original.need(all(base["events"][i] == restored["events"][i]
                      for i in range(tick)), "paths diverged BEFORE first changed choice")
    original.need(base["events"][tick]["observation"]
                  == restored["events"][tick]["observation"]
                  == first["original_observation"],
                  "model observation no longer applies to alternate step")
    original.need(base["events"][tick]["applied"] == first["original_action"]
                  and restored["events"][tick]["applied"] == first["restored_action"]
                  and restored["events"][tick]["raw_legal"] is True,
                  "W0 did not apply restored-choice action")
    return base, restored


def run(alpha: Path, beta: Path, w0_dir: Path, output: Path) -> dict:
    # This full independent audit pins both immutable producer ZIPs, lease/model,
    # true recorded forwards, original sampler, source and CPU residency receipts.
    audited = original.audit(alpha, beta)
    sys.path.insert(0, str(w0_dir))
    import worm

    root = output / "first-divergence-PAIR-PNGs"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for shard, zpath in (("alpha", alpha), ("beta", beta)):
        with zipfile.ZipFile(zpath) as archive:
            pre = f"w13-fleet/{shard}/"
            summary = json.loads(archive.read(pre + "summary.json"))
            for seed, family in original.ARTIFACTS[shard][1]:
                matching = [r for s in audited["shards"] for r in s["episodes"]
                            if r["seed"] == seed and r["family"] == family]
                original.need(len(matching) == 1, "missing audited model episode")
                lease = matching[0]["lease"]
                prefix = pre + lease + "/"
                decisions = json.loads(archive.read(prefix + "model-decisions.json"))
                trace = json.loads(archive.read(prefix + "original-w0.json"))
                decision = choices_until_first_divergence(decisions, trace["events"])
                source_actions = [e["applied"] for e in trace["events"]]
                base, changed = paired_prefix(worm, seed, family, source_actions, decision)
                tick = decision["first_divergence"]["tick"]
                original.need(base["events"] == trace["events"][:tick+1],
                              "original source W0 prefix did not replay exactly")
                original.need(base["trajectory"] == trace["trajectory"][:tick+2]
                              and base["obstacles"] == trace["obstacles"],
                              "source world/trajectory changed")
                prefix_png, altered_png = worm.png_bytes(base), worm.png_bytes(changed)
                base_name = f"{shard}-{seed}-{family}-at-tick-{tick:03d}"
                files = {
                    "original_observed_model_prefix": (base_name + "-ORIGINAL-prefix.png", prefix_png),
                    "STOP_restored_one_step_only": (base_name + "-STOP-restored-first-change.png", altered_png),
                }
                hashes = {}
                for label, (name, data) in files.items():
                    (root / name).write_bytes(data)
                    hashes[label] = {"filename": name, "sha256": sha(data),
                                     "bytes": len(data)}
                rows.append({
                    "seed": seed, "family": family, "runner": summary["runner_name"],
                    "lease": lease, "model_score_forwards_from_original": 128,
                    "original_full_png_sha256": matching[0]["original_model_png_sha256"],
                    "first_divergence_tick_zero_based": tick,
                    "changed_action_is_STOP": decision["first_divergence"]["restored_action"] == "STOP",
                    "original_action": decision["first_divergence"]["original_action"],
                    "STOP_restored_first_action": decision["first_divergence"]["restored_action"],
                    "identical_recorded_model_observations_before_divergence": tick,
                    "same_state_STOP_choices_NOT_rollouts": decision["same_state_STOP_selections_NOT_rollouts"],
                    "same_state_any_changes_NOT_rollouts": decision["same_state_changed_selections_NOT_rollouts"],
                    "pair_pngs": hashes,
                    "important_limit": "Only the first different action is causally replayed. "
                         "No model inference after divergence, no full uncensored rollout."
                })
    original.need(len(rows) == 4 and len({r["seed"] for r in rows}) == 4,
                  "missing independent worlds")
    original.need(sum(r["same_state_STOP_choices_NOT_rollouts"] for r in rows)
                  == audited["same_state_counterfactual_STOP_choices_NOT_rollouts"] == 49,
                  "not the original 49 STOP one-step projections")
    report = {
        "schema": VERSION, "support_source_at_runtime": None,
        "producer_run": original.RUN, "producer_commit": original.SOURCE,
        "shared_W0_commit": original.W0, "original_ZIP_SHA256": {
            k: v[0] for k, v in original.ARTIFACTS.items()},
        "genuine_archived_model_forwards": 512,
        "new_model_forwards": 0, "new_optimizer_updates": 0,
        "original_full_128_turn_PNGs_independently_replayed_by_previous_audit": False,
        "independently_rendered_first_divergence_pngs": 8,
        "episodes": rows,
        "interpretation": "W0 PNG pair: original model-sampled prefix vs one "
          "STOP-permitted host choice at the earliest changed observed state. "
          "All later projected STOP choices are noncausal if an earlier move "
          "would have changed the observation. Not trained raw EOS7 output, "
          "not an autonomous STOP-allowed 128-turn episode and not native Godot."
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "W13-stop-first-divergence.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=Path, required=True)
    p.add_argument("--beta", type=Path, required=True)
    p.add_argument("--w0-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = run(args.alpha, args.beta, args.w0_dir, args.output)
    print(json.dumps({"schema": VERSION,
                      "first_changed_action_by_seed": {
                          str(r["seed"]): [r["first_divergence_tick_zero_based"],
                                           r["original_action"],
                                           r["STOP_restored_first_action"]]
                          for r in result["episodes"]},
                      "actual_new_model_forwards": 0,
                      "genuine_W0_PNG_pairs": 4}, sort_keys=True))


if __name__ == "__main__":
    main()
