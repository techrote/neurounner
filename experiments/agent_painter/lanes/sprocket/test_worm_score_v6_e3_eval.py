#!/usr/bin/env python3
"""W6 E3 host policy and common-random-number ablation tests, model-free."""
import worm_score_v6_e3_eval as w6
import worm_v1 as w0
def check():
    base={"tick":0,"legal":list(w0.ACTIONS)}
    assert w6.policy_legal(base)==list(w0.ACTIONS[:-1])
    assert w6.policy_legal({"tick":0,"legal":["STOP"]})==["STOP"]
    assert w6.policy_legal({"tick":28,"legal":list(w0.ACTIONS)})==list(w0.ACTIONS)
    assert w6.policy_legal({"tick":27,"legal":list(w0.ACTIONS)})==list(w0.ACTIONS[:-1])
    assert w6.policy_legal({"tick":29,"legal":list(w0.ACTIONS)})==list(w0.ACTIONS)
    assert w6.BUDGET==32 and w6.WORLD_SEEDS==(401,409,419)
    try:w6.policy_legal({"tick":0,"legal":["STOP","TELEPORT"]})
    except ValueError:pass
    else:raise AssertionError("invalid host action admitted")
    for seed in w6.WORLD_SEEDS:
        for tick in (0,1,26,31):
            draw=w6.host_draw("greebles",seed,tick,w6.TEMPERATURE)
            assert draw==w6.host_draw("architecture",seed,tick,w6.TEMPERATURE)
            allowed=w6.policy_legal(base)
            selected,receipt=w6.sample_scores([0.]*6,allowed,w6.TEMPERATURE,draw)
            assert selected in allowed and selected!="STOP"
            assert abs(sum(receipt["masked_probs"].values())-1)<1e-12
    for mode in ("flat","untuned","trained"):
        _,score=w6.sample_scores([0.]*6,["STRAIGHT","STOP"],w6.TEMPERATURE,.3)
        receipt=w6.attributed_score(score,mode)
        assert receipt["is_model_logit_conditioned"]==(mode!="flat")
        assert receipt["trained_adapter_active"]==(mode=="trained")
        assert receipt["untuned_root_forward"]==(mode=="untuned")
        assert receipt["host_flat_no_model_forward"]==(mode=="flat")
    try:w6.attributed_score({},"unsafe")
    except ValueError:pass
    else:raise AssertionError("invalid arm attribution admitted")
    print("W6 E3 STOP, W0 masks, common RNG and correct three-arm score attribution PASS")
if __name__=="__main__":check()
