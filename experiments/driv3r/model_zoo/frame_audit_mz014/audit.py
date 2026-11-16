"""MZ-014 frozen-policy coordinate-frame stress test.

MZ-013 weights are deliberately NOT trained again. The host, checkpoint and
legal-action semantics are source-pinned. This is a causal *representation*
intervention, not a new obstacle environment or evidence of maze competence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import statistics
from pathlib import Path

import torch
from experiments.driv3r.model_zoo.spatial_paint_mz013 import run as parent

PARENT_SHA = "66b30e6fd3799bc331f2ce066ed106720bbcb262"
CHECKPOINT_SHA = "10d9c62370abebaa9d2f644cfe620dda77bbaf651491701b01a78dff7b5aed11"
MODES = {
    "original": (False, False, "normal"),
    "horizontal_mirror": (True, False, "normal"),
    "vertical_mirror": (False, True, "normal"),
    "rotation_180": (True, True, "normal"),
    # Independent causal controls: isolate coordinate vs orientation/action effects.
    "horizontal_mirror_keep_x": (True, False, "keep_x"),
    "x_coordinate_only": (False, False, "flip_x"),
}


def transform(obs, world, horizontal=False, vertical=False):
    """Return transformed observation and physical->transformed action mapping.

    The 5x5 categorical patch, scalar position, host legality and last-action
    channels MUST transform together; cues and other scalars are invariant.
    """
    z = torch.tensor(obs, dtype=torch.float32)
    patch = z[:125].reshape(5, 5, 5)
    if horizontal:
        patch = patch.flip(1)
    if vertical:
        patch = patch.flip(0)
    z[:125] = patch.reshape(-1)
    if horizontal:
        z[125] = (world.width - 1 - world.x) / 26
    if vertical:
        z[126] = (6 - world.y) / 6
    perm = list(range(len(parent.ACTIONS)))
    if horizontal:
        perm[parent.LEFT], perm[parent.RIGHT] = perm[parent.RIGHT], perm[parent.LEFT]
    if vertical:
        perm[parent.UP], perm[parent.DOWN] = perm[parent.DOWN], perm[parent.UP]
    for start in (131, 139):
        old = z[start:start+8].clone()
        for physical in range(8):
            z[start + perm[physical]] = old[physical]
    assert sorted(perm) == list(range(8))
    assert len(z) == parent.D
    return z, perm


def play(net, seed, cue, invert, *, horizontal=False, vertical=False,
         x_control="normal"):
    world = parent.World(seed, cue, invert, ood=True)
    state = None
    raw_illegal = 0
    actions = []
    with torch.inference_mode():
        for _ in range(parent.MAX_T):
            obs, perm = transform(world.observe(), world, horizontal, vertical)
            if x_control == "keep_x":
                obs[125] = world.x / 26
            elif x_control == "flip_x":
                obs[125] = (world.width - 1 - world.x) / 26
            else:
                assert x_control == "normal"
            if hasattr(net, "step"):
                logits, state = net.step(obs[None, :], state)
                logits = logits[0]
            else:
                logits = net(obs[None, None, :])[0, -1]
            # Transform logits BACK to the original physical action vocabulary.
            physical_logits = logits[torch.tensor(perm)]
            legal = torch.tensor(world.legal(), dtype=torch.bool)
            raw_illegal += int(not legal[int(physical_logits.argmax())])
            action = int(physical_logits.masked_fill(~legal, -1e9).argmax())
            actions.append(parent.ACTIONS[action])
            world.step(action)
            if world.done:
                break
    success = bool(world.done and not world.failed and len(world.painted) == 3)
    return {
        "completed": success,
        "clean": bool(success and world.collisions == 0),
        "paints": len(world.painted),
        "protected_collisions": world.collisions,
        "raw_illegal": raw_illegal,
        "steps": world.tick,
        "actions": actions,
        "path": world.trace,
    }


def evaluate(net, seeds):
    outputs = {}
    for name, (horizontal, vertical, x_control) in MODES.items():
        episodes = [play(net, s, cue, inv, horizontal=horizontal,
                         vertical=vertical, x_control=x_control)
                    for s in seeds for cue in range(2) for inv in range(2)]
        outputs[name] = {
            "episodes": len(episodes),
            "clean": sum(a["clean"] for a in episodes),
            "completed": sum(a["completed"] for a in episodes),
            "protected_collisions": sum(a["protected_collisions"] for a in episodes),
            "raw_illegal": sum(a["raw_illegal"] for a in episodes),
            "paints": sum(a["paints"] for a in episodes),
            "steps_mean": round(statistics.mean(a["steps"] for a in episodes), 2),
            "sample": episodes[:4],
        }
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="mz014-evidence")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    if not 1 <= args.seeds <= 12:
        raise ValueError("bounded 1..12 seed families")
    torch.set_num_threads(2)
    checkpoint = Path(args.checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA:
        raise RuntimeError(f"wrong exact MZ-013 checkpoint: {digest}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["source_sha"] != PARENT_SHA or state["schema"] != parent.SCHEMA:
        raise RuntimeError("source/schema provenance mismatch")
    nets = parent.models(torch, state["width"])
    # Heavy, already-rejected transformer intentionally excluded from this
    # compact causal orientation audit, preserving its original weights.
    del nets["transformer20m"]
    for name, net in nets.items():
        net.load_state_dict(state["models"][name], strict=True)
        net.eval()
    seeds = range(7000, 7000 + args.seeds)
    scores = {name: evaluate(net, seeds) for name, net in nets.items()}
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "MZ014_FRAME_GENERALISATION",
        "audit_source_sha": os.environ.get("GITHUB_SHA", "LOCAL-UNVERIFIED"),
        "parent_source_sha": PARENT_SHA,
        "checkpoint_sha256": digest,
        "checkpoint_training_updates": 700,
        "new_training_updates": 0,
        "seed_family": list(seeds),
        "parameters": {name: sum(p.numel() for p in net.parameters())
                       for name, net in nets.items()},
        "scores": scores,
        "process_peak_rss_mib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        "limitations": [
            "Frozen-model representational intervention; no new training",
            "Mirroring and 180-degree rotation are not arbitrary obstacle maps",
            "Original synthetic corridor/station host, not native CyberSand",
            "Host action-mask enforcement unchanged; raw illegality counted",
            "Action-space and all observation channels transformed consistently",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# MZ-014 — frozen-policy orientation audit",
             "", f"Parent checkpoint: {digest}", ""]
    for name, modes in scores.items():
        lines.append(f"## {name}")
        for mode, data in modes.items():
            lines.append(f"- {mode}: clean {data['clean']}/{data['episodes']}, "
                         f"completed {data['completed']}/{data['episodes']}, "
                         f"protected collisions {data['protected_collisions']}, "
                         f"raw illegal {data['raw_illegal']}")
        lines.append("")
    lines += ["**Scope:** This tests learned coordinate-frame brittleness, "
              "not general maze navigation. No retraining or CyberSand modifications.",
              "Draft/HOLD; no promotion.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    return report


if __name__ == "__main__":
    main()
