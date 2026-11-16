#!/usr/bin/env python3
"""WORM W4 digit-action code framing and host-contract regressions (no model)."""
import copy
import worm_v1 as baseline
import worm_v4 as w4

def check():
    assert tuple(w4.ACTION_TO_CODE) == baseline.ACTIONS
    assert len(set(w4.ACTION_TO_CODE.values())) == 6
    for action, code in w4.ACTION_TO_CODE.items():
        assert w4.CODE_TO_ACTION[code] == action
        assert len(code) == 1 and code in "012345"
    observation = {
        "family": "pipework", "tick": 4, "x": 34, "y": 42,
        "heading": 0, "local_5x5": [".....", ".....", "..@..", ".....", "....."],
        "legal": ["STRAIGHT", "LEFT", "RIGHT", "BRANCH", "RETRACE", "STOP"],
        "last_actions": ["STRAIGHT"]
    }
    p = w4.prompt(observation)
    assert p.endswith("\nCODE=") and "0=STRAIGHT" in p and "5=STOP" in p
    assert "pipework" in p and '"tick":4' in p and '"x":34' in p
    for mutation in (
        {"legal": ["STRAIGHT", "EXTRA", "STOP"]},
        {"legal": ["STRAIGHT", "STRAIGHT", "STOP"]},
        {"legal": ["STRAIGHT"]},
        {"family": "unknown"},
        {"tick": -1},
        {"local_5x5": ["xxxxx"] * 5},
    ):
        invalid = copy.deepcopy(observation)
        invalid.update(mutation)
        try:
            w4.prompt(invalid)
        except (ValueError, KeyError):
            pass
        else:
            raise AssertionError("invalid host observation admitted: " + repr(mutation))
    assert set(w4.CODE_TO_ACTION) == set("012345")
    print("SPROCKET W4 digit mapping, host-legal and injection tests PASS")

if __name__ == "__main__":
    check()
