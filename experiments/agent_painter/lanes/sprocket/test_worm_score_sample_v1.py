#!/usr/bin/env python3
"""S1 score sampler safety, entropy, source policy and host RNG tests (no model)."""
import math
import worm_v1 as contract
from worm_score_sample_v1 import sample_scores, host_draw, TEMPERATURES, spatial


def check():
    assert TEMPERATURES == (1.0, 3.0)
    legal = list(contract.ACTIONS)
    logits = [3., 0., -1., -2., -3., -4.]
    d = host_draw("greebles", 211, 0, 1.0)
    a, receipt = sample_scores(logits, legal, 1., d)
    assert a in legal and receipt["selected_host_legal"]
    assert abs(sum(receipt["masked_probs"].values()) - 1) < 1e-12
    assert receipt["raw_unmasked_top_action"] == "STRAIGHT"
    assert receipt["is_model_logit_conditioned"] and not receipt["is_eos5_completion"]
    assert d == host_draw("greebles", 211, 0, 1.0)
    assert d != host_draw("greebles", 211, 1, 1.0)
    assert d != host_draw("pipework", 211, 0, 1.0)
    assert d != host_draw("greebles", 211, 0, 3.0)
    a2, high = sample_scores(logits, legal, 3.0, d)
    assert high["entropy_nats"] > receipt["entropy_nats"]
    assert high["effective_actions"] > receipt["effective_actions"]
    assert a2 in legal
    # Host legality is absolute even when the learned model strongly favours an illegal action.
    for u in (0.0, 0.25, 0.5, 0.9999999999999999):
        a, result = sample_scores([999., -100., -100., -100., -100., -1.],
                                  ["STOP"], 1., u)
        assert a == "STOP" and result["masked_probs"]["STOP"] == 1
        assert not result["raw_top_host_legal"]
    for invalid in (
        ([0.] * 5, legal, 1., 0.5),
        ([float("nan")] + [0.]*5, legal, 1., 0.5),
        ([0.]*6, ["ALIEN"], 1., 0.5),
        ([0.]*6, ["STOP", "STOP"], 1., 0.5),
        ([0.]*6, legal, -1., 0.5),
        ([0.]*6, legal, 1., 1.0),
    ):
        try:
            sample_scores(*invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid model/host contract was accepted " + str(invalid))
    p = spatial({"trajectory": [[1, 1], [1, 2], [2, 2], [2, 2]]})
    assert p == {"unique_cells": 3, "bbox_width": 2, "bbox_height": 2,
                 "max_manhattan_from_start": 2}
    print("S1 legal-mask, entropy, deterministic host RNG and spatial tests PASS")


if __name__ == "__main__":
    check()
