#!/usr/bin/env python3
"""No-model source-fenced W9 tests; do not describe these as optimization."""
import argparse, tempfile
from pathlib import Path
from copy import deepcopy
import worm_v4 as obj
import worm_train_v9 as w9
import worm_train_v8 as w8

def test_features(support):
    assert True
    for family in w9.FAMILIES:
        for case in ("junction","blocked","terminal","deadline"):
            obs=support.synthetic_fixture(family,case)
            vector=w9.features(obs)
            assert len(vector)==142, len(vector)
            assert all(-1.001<=v<=1.001 for v in vector)
            other=deepcopy(obs)
            other["local_5x5"][0]="+"+other["local_5x5"][0][1:]
            assert w9.features(other)!=vector
        base=support.synthetic_fixture(family,"junction")
        switched=deepcopy(base)
        switched["family"]=next(f for f in w9.FAMILIES if f!=family)
        assert w9.features(switched)!=w9.features(base)
        illegal=deepcopy(base)
        illegal["legal"]=["LEFT"]
        try:
            w9.features(illegal)
            raise AssertionError("invalid W0 legal menu accepted")
        except ValueError:
            pass

def main(args):
    support=obj.load_support(args.support,args.w0)
    test_features(support)
    with tempfile.TemporaryDirectory(prefix="sprocket-w9-source-") as td:
        data,receipt=obj.dataset(support,Path(td)/"dataset")
        pairs=w8.counterfactual_pairs(data["train"],support,"train")
        dev=w8.counterfactual_pairs(data["eval"],support,"eval")
        assert len(pairs)==72 and len(dev)==36, (len(pairs),len(dev))
        assert not ({r["seed"] for r in data["train"] if r["seed"] is not None} &
                    {r["seed"] for r in data["eval"] if r["seed"] is not None})
        assert set(w9.TRAIN_SEEDS).isdisjoint(set(w9.EVAL_SEEDS))
        assert set(w9.EVAL_SEEDS).isdisjoint(set(support.TRAIN_SEEDS))
        assert set(w9.EVAL_SEEDS).isdisjoint(set(support.DEV_SEEDS))
        for pair in pairs+dev:
            for row in pair:
                assert row["action"] in row["legal"]
                assert len(w9.features(row["observation"]))==142
        assert receipt["literal_prompt_intersection"]==0
        assert len(set(w9.ACTIONS))==6
    print("W9 source-only geometry/features/split/counterfactual preflight PASS; no optimization")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--w0",type=Path,required=True)
    p.add_argument("--support",type=Path,required=True)
    main(p.parse_args())
