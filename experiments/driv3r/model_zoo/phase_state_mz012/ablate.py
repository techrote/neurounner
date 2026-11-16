"""MZ-012 post-training phase-rotation causality ablation.

Trained checkpoint is immutable within the run; intervene only on inference-time
rotation parameters and independently compare paired TEST/shifted worlds.
This does NOT replace a retrained, matched no-phase architecture control.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path

from experiments.driv3r.model_zoo.phase_state_mz012 import run as mz
from experiments.driv3r.model_zoo.material_forks_mz009 import run as host


def run(evidence):
    import torch
    torch.set_num_threads(2)
    out = Path(evidence)
    report = json.loads((out/"metrics.json").read_text())
    ck_bytes = (out/"models.pt").read_bytes()
    assert hashlib.sha256(ck_bytes).hexdigest() == report["checkpoint_sha256"]
    saved = torch.load(out/"models.pt", map_location="cpu", weights_only=True)
    assert saved["schema"] == mz.SCHEMA
    assert saved["source_sha"] == report["source_sha"]
    assert saved["source_sha"] == os.environ.get(
        "GITHUB_SHA", saved["source_sha"])
    phase = mz.phase_model(torch, saved["hidden"])
    phase.load_state_dict(saved["models"]["phase"], strict=True)
    phase.eval()
    baseline_angles = [cell.angle.detach().clone() for cell in phase.cells]
    x, y = mz.tensors(torch, mz.dataset(range(5000, 5024)))
    xx, yy = mz.tensors(torch, mz.dataset(range(6000, 6024), ood=True))
    outcomes = {}
    conditions = {
        "trained": (1., 1.),
        "zero_first_layer_phase": (0., 1.),
        "zero_second_layer_phase": (1., 0.),
        "zero_both_layer_phases": (0., 0.),
        "reverse_both_layer_phases": (-1., -1.),
    }
    for condition, factors in conditions.items():
        with torch.no_grad():
            for i, factor in enumerate(factors):
                phase.cells[i].angle.copy_(baseline_angles[i] * factor)
        with torch.inference_mode():
            id_score = host.class_scores(phase(x), y, x)
            ood_score = host.class_scores(phase(xx), yy, xx)
        outcomes[condition] = {
            "test": id_score,
            "shifted_long_test": ood_score,
            "history_causality": mz.causal(torch, phase, x, y),
            "shifted_closed_loop": mz.rollouts(
                torch, phase, range(7000, 7008), ood=True, cached=True),
        }
    assert outcomes["trained"]["test"] == report["scores"]["phase"]["test"]
    assert outcomes["trained"]["shifted_long_test"] == (
        report["scores"]["phase"]["shifted_long_test"])
    for name in ("trained", "zero_both_layer_phases"):
        assert outcomes[name]["shifted_closed_loop"]["episodes"] == 32
    with torch.no_grad():
        for i, angle in enumerate(baseline_angles):
            phase.cells[i].angle.copy_(angle)
    stats = [{
        "mean_abs_radians": round(float(a.abs().mean()), 7),
        "max_abs_radians": round(float(a.abs().max()), 7),
        "nonzero_count": int((a.abs() > .001).sum()),
        "channels": a.numel(),
    } for a in baseline_angles]
    result = {
        "schema": "MZ012-PHASE-ABLATION-v1",
        "source_sha": report["source_sha"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "phase_statistics": stats,
        "conditions": outcomes,
        "interpretation_limits": [
            "Inference-time intervention on trained phase weights, not a retrained ablated model",
            "A changed score proves sensitivity to the phase variables, not an architectural advantage",
            "Separate diagonal recurrence baseline is fully trained and evaluated in metrics.json",
            "Synthetic host and CPU timing only; no native game changes",
        ],
    }
    (out/"phase_ablation.json").write_text(json.dumps(result, indent=2)+chr(10))
    print(json.dumps({
        "checkpoint_sha256": result["checkpoint_sha256"],
        "phase_statistics": stats,
        "conditions": {k: {
            "test_fork_accuracy": v["test"]["fork_accuracy"],
            "shifted_fork_accuracy": v["shifted_long_test"]["fork_accuracy"],
            "shifted_clean_rate": v["shifted_closed_loop"]["clean_rate"],
            "beacon_flip_rate": v["history_causality"]["beacon_flip_action_switch_rate"],
        } for k, v in outcomes.items()},
    },indent=2),flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="mz012-evidence")
    run(p.parse_args().input)


if __name__ == "__main__":
    main()
