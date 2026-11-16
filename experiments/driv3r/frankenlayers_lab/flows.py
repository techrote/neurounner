"""Immutable wave-1/2 carcasses plus the nonmonotonic wave-3 probes.

The previous four names/flows MUST NOT change: their exact-model receipts are
attached to GitHub Actions run 43790709596. New modes use new names.
"""
from __future__ import annotations

WAVE1_FLOWS: dict[str, tuple[int, ...]] = {
    "skele6": (0, 1, 4, 9, 18, 23),
    "aero6": (0, 2, 4, 10, 20, 23),
    "skele4": (0, 4, 18, 23),
    "aero4": (0, 4, 20, 23),  # failed runtime guard; preserved intentionally
}
# Attempt alternate final blocks while retaining 0,4,18 ballast that booted in
# SKELE4. These flows are NOT claimed functional until source-built inference.
WAVE2_FLOWS: dict[str, tuple[int, ...]] = {
    "tailspin4": (0, 4, 18, 20),
    "needle4": (0, 4, 18, 22),
}
# Wave 3 deliberately tests nonmonotonic and repeated execution rather than
# another four ordered skeletons. The immutable wave-1/wave-2 identities remain.
WAVE3_FLOWS: dict[str, tuple[int, ...]] = {
    "tailflip4": (0, 4, 20, 18),  # exchange final execution order of TAILSPIN4
    "echo4": (0, 4, 18, 4),      # four executions, three unique layers
    "aero5_no20": (0, 2, 4, 10, 23),  # AERO6 remove layer 20
    "aero5_no23": (0, 2, 4, 10, 20),  # AERO6 remove last layer
}
# Wave 5 probes whether the five-layer branch-capable spine survives deletion of
# layer 10, and whether causal tail reordering changes its character. Historical
# mode identities remain immutable. Both are speculative until live inference.
WAVE5_FLOWS: dict[str, tuple[int, ...]] = {
    "aero4_no10": (0, 2, 4, 20),
    "aero5_tailflip": (0, 2, 4, 20, 10),
}
# Wave 6 isolates layer 2 from AERO4_NO10; speculative until actual inference.
# The wave-5 four-layer branch-capable survivor remains unchanged.
WAVE6_FLOWS: dict[str, tuple[int, ...]] = {
    "aero4_no2": (0, 4, 10, 20),
}
# Wave 14: four-layer bridge from the viable NO10 and geometry-sensitive NO2 spines.
# Unqualified until a real patched-runtime decision-graph trial; never relabel old flows.
WAVE14_FLOWS: dict[str, tuple[int, ...]] = {"aero4_bridge": (0, 2, 10, 20)}
# Wave 15: isolate execution-order effects and a bounded terminal echo graft.
# Both retain 0/4/10/20 attention ballast; no changes to model weights.
WAVE15_FLOWS: dict[str, tuple[int, ...]] = {
    "aero4_no2_tailflip": (0, 4, 20, 10),
    "aero5_no2_echo10": (0, 4, 10, 20, 10),
}
LAB_FLOWS: dict[str, tuple[int, ...]] = {**WAVE1_FLOWS, **WAVE2_FLOWS, **WAVE3_FLOWS, **WAVE5_FLOWS, **WAVE6_FLOWS, **WAVE14_FLOWS, **WAVE15_FLOWS}
EXPECTED_LENGTHS = {
    "skele6": 6, "aero6": 6, "skele4": 4, "aero4": 4,
    "tailspin4": 4, "needle4": 4,
    "tailflip4": 4, "echo4": 4, "aero5_no20": 5, "aero5_no23": 5,
    "aero4_no10": 4, "aero5_tailflip": 5,
    "aero4_no2": 4,
    "aero4_bridge": 4,
    "aero4_no2_tailflip": 4, "aero5_no2_echo10": 5,
}

def validate_flows(flows: dict[str, tuple[int, ...]] = LAB_FLOWS) -> None:
    if set(flows) != set(EXPECTED_LENGTHS):
        raise ValueError("Unexpected lab flow names; immutable wave identities")
    if {name: len(layers) for name, layers in flows.items()} != EXPECTED_LENGTHS:
        raise ValueError("Unexpected executed-layer count")
    for name, layers in flows.items():
        if not all(type(i) is int and 0 <= i < 24 for i in layers):
            raise ValueError(f"{name}: invalid layer index")
        if name not in ("echo4", "aero5_no2_echo10") and len(set(layers)) != len(layers):
            raise ValueError(f"{name}: repeated source layers not authorised")
        if name not in ("tailflip4", "echo4", "aero5_tailflip", "aero4_no2_tailflip", "aero5_no2_echo10") and tuple(sorted(layers)) != layers:
            raise ValueError(f"{name}: nonordered flow not authorised")
        if layers[0] != 0:
            raise ValueError(f"{name}: initial layer changed")
        if name in WAVE1_FLOWS:
            if layers != WAVE1_FLOWS[name]:
                raise ValueError(f"{name}: immutable wave-1 flow changed")
        elif name in WAVE2_FLOWS:
            if layers != WAVE2_FLOWS[name]:
                raise ValueError(f"{name}: frozen wave-2 flow changed")
        elif name in WAVE3_FLOWS:
            if layers != WAVE3_FLOWS[name]:
                raise ValueError(f"{name}: frozen wave-3 flow changed")
        elif name in WAVE5_FLOWS:
            if layers != WAVE5_FLOWS[name]:
                raise ValueError(f"{name}: frozen wave-5 flow changed")
        elif name in WAVE6_FLOWS:
            if layers != WAVE6_FLOWS[name]:
                raise ValueError(f"{name}: wave-6 flow changed")
        elif name in WAVE14_FLOWS:
            if layers != WAVE14_FLOWS[name]:
                raise ValueError(f"{name}: wave-14 flow changed")
        elif layers != WAVE15_FLOWS[name]:
            raise ValueError(f"{name}: wave-15 flow changed")

validate_flows()
