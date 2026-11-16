#!/usr/bin/env python3
"""W10 source-only preflight. MUST NOT be counted as trained updates."""
import argparse
from copy import deepcopy
import worm_train_v10 as w10
import worm_v4 as objective

def main(args):
    support=objective.load_support(args.support,args.w0)
    assert set(w10.TRAIN_SEEDS).isdisjoint(w10.HELDOUT)
    assert set(w10.HELDOUT).isdisjoint(support.TRAIN_SEEDS+support.DEV_SEEDS)
    for family in w10.FAMILIES:
        for case in ("junction","blocked","terminal","deadline"):
            obs=support.synthetic_fixture(family,case)
            label=w10.motif_teacher(obs,support)
            assert label in obs["legal"],(case,family,label)
            assert len(w10.w9.features(obs))==142
        base=support.synthetic_fixture(family,"junction")
        blocked=support.synthetic_fixture(family,"blocked")
        terminal=support.synthetic_fixture(family,"terminal")
        assert w10.motif_teacher(blocked,support)=="RETRACE"
        assert w10.motif_teacher(terminal,support)=="STOP"
    anchor=support.synthetic_fixture("greebles","junction")
    anchor["tick"]=21
    anchor["local_5x5"][0]="....."
    alternate=deepcopy(anchor)
    alternate["local_5x5"][0]="#...."
    assert w10.motif_teacher(anchor,support)!=w10.motif_teacher(alternate,support)
    assert w10.motif_teacher(alternate,support)=="BRANCH"
    train,train_manifest=w10.gather(support.worm,support,w10.TRAIN_SEEDS,steps=48)
    dev,dev_manifest=w10.gather(support.worm,support,w10.HELDOUT,steps=48)
    assert train and dev and train_manifest and dev_manifest
    assert not {r["seed"] for r in train}&{r["seed"] for r in dev}
    assert all(r["action"] in r["legal"] for r in train+dev)
    print("W10 W0/teacher/heldout-source validation PASS; genuine optimizer not yet run")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--w0",required=True,type=__import__("pathlib").Path)
    p.add_argument("--support",required=True,type=__import__("pathlib").Path)
    main(p.parse_args())
