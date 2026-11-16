#!/usr/bin/env python3
"""W6 source-only paired-curriculum tests. No model weights, no optimizer."""
import argparse
from pathlib import Path
import worm_train_v6 as v6
import worm_v4 as objective

def check(w0, support):
    module=objective.load_support(Path(support),Path(w0))
    import tempfile
    with tempfile.TemporaryDirectory(prefix="sprocket-w6-") as td:
        data, manifest=objective.dataset(module,Path(td)/"dataset")
        a=v6.counterfactual_pairs(data["train"],module,"train")
        b=v6.counterfactual_pairs(data["eval"],module,"eval")
        assert len(a)>=6 and len(b)>0, (len(a),len(b))
        assert not ({x["seed"] for pair in a for x in pair if x["seed"] is not None}
                    & {x["seed"] for pair in b for x in pair if x["seed"] is not None})
        for scope,pairs in (("train",a),("eval",b)):
            for pair in pairs:
                x,y=pair
                assert x["family"]!=y["family"] and x["action"]!=y["action"]
                assert x["action"] in x["legal"] and y["action"] in y["legal"]
                assert x["completion"]==objective.ACTION_TO_CODE[x["action"]]
                assert y["completion"]==objective.ACTION_TO_CODE[y["action"]]
                assert x["prompt"]!=y["prompt"] and x["seed"]==y["seed"]
                assert "OBS=" in x["prompt"] and "OBS=" in y["prompt"]
        print("W6 exact W0/support split, paired contrasting synthetic targets PASS",
              len(a),len(b),manifest["literal_prompt_intersection"])
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--w0",required=True);p.add_argument("--support",required=True)
    args=p.parse_args()
    check(args.w0,args.support)
