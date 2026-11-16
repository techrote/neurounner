#!/usr/bin/env python3
"""W6 E2 host policy and common-random-number ablation tests, model-free."""
import worm_score_v6_e2_eval as w6
import worm_v1 as w0
def check():
    base={"tick":0,"legal":list(w0.ACTIONS)}
    assert w6.policy_legal(base)==list(w0.ACTIONS[:-1])
    assert w6.policy_legal({"tick":0,"legal":["STOP"]})==["STOP"]
    assert w6.policy_legal({"tick":40,"legal":list(w0.ACTIONS)})==list(w0.ACTIONS)
    assert w6.policy_legal({"tick":39,"legal":list(w0.ACTIONS)})==list(w0.ACTIONS[:-1])
    assert w6.policy_legal({"tick":41,"legal":list(w0.ACTIONS)})==list(w0.ACTIONS)
    assert w6.BUDGET==48 and w6.WORLD_SEEDS==(313,337)
    try:w6.policy_legal({"tick":0,"legal":["STOP","TELEPORT"]})
    except ValueError:pass
    else:raise AssertionError("invalid host action admitted")
    for seed in w6.WORLD_SEEDS:
        for tick in (0,1,38,47):
            draw=w6.host_draw("greebles",seed,tick,w6.TEMPERATURE)
            assert draw==w6.host_draw("architecture",seed,tick,w6.TEMPERATURE)
            allowed=w6.policy_legal(base)
            selected,receipt=w6.sample_scores([0.]*6,allowed,w6.TEMPERATURE,draw)
            assert selected in allowed and selected!="STOP"
            assert abs(sum(receipt["masked_probs"].values())-1)<1e-12
    print("W6 E2 explicit bounded STOP deferral, W0 legality and common RNG PASS")
if __name__=="__main__":check()
