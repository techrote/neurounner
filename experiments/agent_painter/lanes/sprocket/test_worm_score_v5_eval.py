#!/usr/bin/env python3
"""S3 source-only CRN/legality/paired-control tests, without model downloads."""
import worm_score_v5_eval as s
import worm_v1 as contract
def check():
    legal=list(contract.ACTIONS)
    assert legal==["STRAIGHT","LEFT","RIGHT","BRANCH","RETRACE","STOP"]
    a=s.host_draw("greebles",211,0,1.0)
    assert a==s.host_draw("architecture",211,0,3.0)
    assert a!=s.host_draw("greebles",211,1,1.0)
    assert a!=s.host_draw("greebles",223,0,1.0)
    assert 0<=a<1
    selected,score=s.sample_scores([0.,9.,0.,0.,0.,0.],legal,3.0,a)
    other,uniform=s.sample_scores([0.]*6,legal,3.0,a)
    assert selected in legal and other in legal
    assert score["masked_probs"]["LEFT"]>uniform["masked_probs"]["LEFT"]
    assert abs(sum(score["masked_probs"].values())-1)<1e-12
    assert s.spatial({"trajectory":[[0,0],[1,0],[1,1],[1,1]]})["unique_cells"]==3
    print("S3 CRN, paired identical W0 mask and model-vs-uniform controls PASS")
if __name__=="__main__": check()
