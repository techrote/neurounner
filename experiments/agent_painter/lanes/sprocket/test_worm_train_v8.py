#!/usr/bin/env python3
"""W8 W0-anchored, within-family local occupancy counterfactual tests; NO model."""
from pathlib import Path
from copy import deepcopy
import math
import argparse
import tempfile
import worm_train_v8 as v8
import worm_v4 as objective

def check_delta_objective():
    """Direct paired logit preference must reward geometry-conditioned reversal."""
    def loss(a, b):
        delta = (a[0] - a[1]) - (b[0] - b[1])
        return math.log1p(math.exp(2.0 - delta))
    strong_correct = loss((4.0, 0.0), (0.0, 4.0))
    same_prior = loss((4.0, 0.0), (4.0, 0.0))
    reversed_wrong = loss((0.0, 4.0), (4.0, 0.0))
    assert strong_correct < same_prior < reversed_wrong
    assert math.isfinite(strong_correct)

def check(w0, support):
    check_delta_objective()
    module = objective.load_support(Path(support), Path(w0))
    with tempfile.TemporaryDirectory(prefix="sprocket-w7-") as td:
        datasets, receipt = objective.dataset(module, Path(td)/"dataset")
        train = v8.counterfactual_pairs(datasets["train"], module, "train")
        dev = v8.counterfactual_pairs(datasets["eval"], module, "eval")
        assert len(train) >= 18 and len(dev) >= 6, (len(train), len(dev))
        assert not ({x["seed"] for pair in train for x in pair if x["seed"] is not None}
                 & {x["seed"] for pair in dev for x in pair if x["seed"] is not None})
        assert set(x["family"] for pair in train for x in pair) == set(objective.FAMILIES)
        transitions = set()
        for split, pairs in (("train", train), ("eval", dev)):
            for source, foil in pairs:
                a, b = source["observation"], foil["observation"]
                assert source["family"] == foil["family"] and source["seed"] == foil["seed"]
                assert a["tick"] == b["tick"] and a["heading"] == b["heading"]
                assert a["x"] == b["x"] and a["y"] == b["y"] and a["last_actions"] == b["last_actions"]
                assert a["family"] == b["family"]
                changed = [(i,j) for i in range(5) for j in range(5)
                           if a["local_5x5"][i][j] != b["local_5x5"][i][j]]
                assert len(changed) == 1, (split, source["id"], changed)
                i,j = changed[0]
                assert a["local_5x5"][i][j] == "." and b["local_5x5"][i][j] == "+"
                assert set(a["legal"]) - set(b["legal"]) <= {"BRANCH"}
                assert set(b["legal"]) <= set(a["legal"])
                assert module.teacher_choice(a) == source["action"]
                assert module.teacher_choice(b) == foil["action"]
                assert source["action"] != foil["action"]
                assert source["action"] in a["legal"] and foil["action"] in b["legal"]
                assert source["completion"] == objective.ACTION_TO_CODE[source["action"]]
                assert foil["completion"] == objective.ACTION_TO_CODE[foil["action"]]
                assert source["origin"] == "actual-pinned-W0-support-row"
                assert foil["origin"] == "synthetic-one-local-frontier-mark-not-W0-replayed"
                transitions.add((a["family"], source["action"], foil["action"]))
        assert len(transitions) >= 3, transitions
        print("W8 original W0 anchored synthetic local counterfactuals PASS",len(train),len(dev),len(transitions),receipt["literal_prompt_intersection"])
if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--w0",required=True)
    p.add_argument("--support",required=True)
    a=p.parse_args()
    check(a.w0,a.support)
