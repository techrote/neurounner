"""DG020 H5: preregistered near-tie frozen-checkpoint material law challenge.
Preregistration: https://github.com/techrote/neurounner/pull/253#issuecomment-6937815304
No training, no CyberSand/DRIV3R modifications; scientific FAIL is a report, not CI failure.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import softmax
from frozen_h3_audit import load, oracle, SEEDS, TILES

FAMILIES = ("stride_5_7", "dispersed")
BUCKETS = ("cheap_edge_divergent", "expensive_edge_divergent", "wet_rich_invariant",
           "wet_poor_invariant", "equal_wet_invariant")
N = 160
THRESHOLDS = dict(divergent_both=.90, divergent_flip=.95, each_law=.94,
                  invariant_both=.94, invariant_flip_max=.05,
                  sham_flip_max=.05, shuffled_both_max=.60,
                  donor_mse_max=.02, illegal=0)

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def wet(rng, n, family):
    result = np.zeros(TILES, dtype=np.float32)
    if family == "stride_5_7":
        order = (int(rng.integers(TILES)) +
                 int(rng.choice((5, 7))) * np.arange(TILES)) % TILES
    else:
        order = np.array([0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11])
        order = (order + int(rng.integers(TILES))) % TILES
        if rng.integers(2):
            order = order[::-1].copy()
    result[order[:n]] = 1
    assert int(result.sum()) == n
    return result

def make_world(rng, family, bucket):
    k = int(rng.integers(2, 10))
    rich = int(rng.integers(2))
    poor = 1 - rich
    counts = [k, k] if bucket == "equal_wet_invariant" else [k, k + 1]
    if bucket != "equal_wet_invariant":
        counts[rich], counts[poor] = k + 1, k
    marks = np.stack([wet(rng, count, family) for count in counts])
    gap = float(rng.uniform(.05, .20))
    if bucket == "cheap_edge_divergent":
        delta = .65 - gap
    elif bucket == "expensive_edge_divergent":
        delta = -1.30 + gap
    elif bucket == "wet_rich_invariant":
        delta = -1.30 - gap
    elif bucket == "wet_poor_invariant":
        delta = .65 + gap
    else:
        delta = gap * (1 if rng.integers(2) else -1)
    fees = rng.uniform(.12, .18, (2, TILES)).astype(np.float64)
    offset = (delta - (fees[rich].sum() - fees[poor].sum())) / (2 * TILES)
    fees[rich] += offset
    fees[poor] -= offset
    assert 0 <= fees.min() and fees.max() <= .30
    tex = rng.uniform(-1, 1, (2, TILES, 2)).astype("f4")
    x = np.concatenate((marks[..., None], fees.astype("f4")[..., None], tex), -1)
    a, ga, _ = oracle(x, 0)
    b, gb, _ = oracle(x, 1)
    if bucket in ("cheap_edge_divergent", "expensive_edge_divergent"):
        assert a == rich and b == poor
        edge = ga if bucket == "cheap_edge_divergent" else gb
    elif bucket == "wet_rich_invariant":
        assert a == b == rich
        edge = gb
    elif bucket == "wet_poor_invariant":
        assert a == b == poor
        edge = ga
    else:
        assert a == b
        edge = ga
    assert .05 - .0001 <= edge <= .20 + .0001
    return x

def sample(seed, family, bucket):
    rng = np.random.default_rng(seed + 72000000 +
          FAMILIES.index(family)*137000 + BUCKETS.index(bucket)*9790)
    return np.stack([make_world(rng, family, bucket) for _ in range(N)])

def evaluate(seed, family, bucket, donors, receiver, blind, out):
    worlds = sample(seed, family, bucket)
    x = torch.from_numpy(worlds)
    with torch.no_grad():
        f = [[d(x.reshape(-1, 4)).reshape(N, 2, TILES)
              for d in replicas] for replicas in donors]
        logits = [receiver(x, f[k][0]) for k in (0, 1)]
        sham = [receiver(x, f[k][1]).argmax(-1) for k in (0, 1)]
        zero = receiver(x, torch.zeros_like(f[0][0])).argmax(-1)
        no_source = blind(x, torch.zeros_like(f[0][0])).argmax(-1)
        rng = np.random.default_rng(seed + 73000000 +
              FAMILIES.index(family)*100000 + BUCKETS.index(bucket)*1000)
        shuffled = []
        for k in (0, 1):
            choice = rng.integers(2, size=N)
            mixed = torch.stack([f[int(choice[i])][0][i] for i in range(N)])
            shuffled.append(receiver(x, mixed).argmax(-1))
        confidence = [softmax(z, -1) for z in logits]
    fields = ("both", "flip", "low_correct", "high_correct", "sham_flip",
              "shuffled_both", "zero_both", "blind_both", "illegal")
    totals = {name: 0 for name in fields}
    losses, margins, confidences, failures = [], [[], []], [[], []], []
    for i, w in enumerate(worlds):
        targets = [oracle(w, k) for k in (0, 1)]
        answer = [q[0] for q in targets]
        choice = [int(logits[k][i].argmax()) for k in (0, 1)]
        sh = [int(sham[k][i]) for k in (0, 1)]
        shuf = [int(shuffled[k][i]) for k in (0, 1)]
        m = [float(logits[k][i, answer[k]] -
                   logits[k][i, 1 - answer[k]]) for k in (0, 1)]
        c = [float(confidence[k][i, answer[k]]) for k in (0, 1)]
        totals["both"] += int(choice == answer)
        totals["flip"] += int(choice[0] != choice[1])
        totals["low_correct"] += int(choice[0] == answer[0])
        totals["high_correct"] += int(choice[1] == answer[1])
        totals["sham_flip"] += int(sh != choice)
        totals["shuffled_both"] += int(shuf == answer)
        totals["zero_both"] += int(int(zero[i]) == answer[0] == answer[1])
        totals["blind_both"] += int(int(no_source[i]) == answer[0] == answer[1])
        totals["illegal"] += int(any(v not in (0, 1) for v in choice))
        for k in (0, 1):
            margins[k].append(m[k])
            confidences[k].append(c[k])
            truth = np.where(w[..., 0] > .5, (.35, 2.30)[k], 1.) + w[..., 1]
            for rep in (0, 1):
                losses.append(float(np.mean((f[k][rep][i].numpy()-truth)**2)))
        if choice != answer or sh != choice:
            failures.append(dict(i=i, oracle=answer, actions=choice,
                                 sham=sh, margins=m, gaps=[q[1] for q in targets]))
        record = dict(seed=seed, family=family, bucket=bucket, i=i,
          observation_hash=digest(w.tobytes()), observation=w.tolist(),
          same_observation_between_laws=True, legal_mask=[1, 1],
          oracle_actions=answer, oracle_gaps=[q[1] for q in targets],
          oracle_costs=[q[2] for q in targets], actions=choice,
          logits=[z[i].tolist() for z in logits],
          correct_margins=m, correct_confidence=c,
          donor_forecasts=[f[k][0][i].tolist() for k in (0, 1)],
          sham_forecasts=[f[k][1][i].tolist() for k in (0, 1)],
          sham_actions=sh, shuffled_actions=shuf,
          zero_action=int(zero[i]), blind_action=int(no_source[i]),
          both_correct=choice == answer)
        out.write(json.dumps(record, separators=(",", ":")) + "\n")
    metrics = {name: totals[name]/N for name in fields}
    mse = float(np.mean(losses))
    passed = (metrics["illegal"] == 0 and metrics["sham_flip"] <= .05 and
              mse <= .02 and min(metrics["low_correct"],metrics["high_correct"]) >= .94)
    if "divergent" in bucket:
        passed &= (metrics["both"] >= .90 and metrics["flip"] >= .95 and
                   metrics["shuffled_both"] <= .60 and
                   metrics["zero_both"] == metrics["blind_both"] == 0)
    else:
        passed &= (metrics["both"] >= .94 and metrics["flip"] <= .05)
    return dict(verdict="PASS" if passed else "FAIL", metrics=metrics,
                tile_cost_mse=mse, failure_count=len(failures),
                failures=failures[:30],
                mean_correct_margins=[float(np.mean(z)) for z in margins],
                mean_correct_confidence=[float(np.mean(z)) for z in confidences])

def main(root, output):
    torch.set_num_threads(2)
    root = Path(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(experiment="DG020-H5", seeds={}, thresholds=THRESHOLDS,
                  frozen=True, preregistration="issuecomment-6937815304",
                  source_sha256=digest(Path(__file__).read_bytes()),
                  n_pairs_per_cell=N, torch=torch.__version__)
    with (output/"receipts.jsonl").open("w") as out:
        for seed in SEEDS:
            donors, receiver, blind, old, hashes = load(root, seed)
            families = {}
            for family in FAMILIES:
                families[family] = {}
                for bucket in BUCKETS:
                    r = evaluate(seed, family, bucket, donors, receiver, blind, out)
                    families[family][bucket] = r
                    print(seed, family, bucket, r["verdict"],
                          r["metrics"]["both"], r["metrics"]["flip"], flush=True)
            report["seeds"][str(seed)] = dict(checkpoint_sha256=hashes,
                                               families=families)
    report["verdict"] = "PASS" if all(
        b["verdict"] == "PASS"
        for s in report["seeds"].values()
        for f in s["families"].values()
        for b in f.values()) else "FAIL"
    (output/"report.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    manifest = {p.name:digest(p.read_bytes()) for p in output.iterdir()
                if p.is_file() and p.name != "sha256.json"}
    (output/"sha256.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("H5 SCIENTIFIC VERDICT:", report["verdict"], flush=True)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: frozen_h5_audit.py CHECKPOINT_ROOT OUTPUT_DIR")
    main(sys.argv[1], sys.argv[2])
