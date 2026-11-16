#!/usr/bin/env python3
"""Use #626 stock-paired real W0 inference, with additional lab flow names.

The donor is unchanged; the model-choice response must remain host-legal.
"""
from experiments.driv3r.frankenlayers.patch_runtime import FLOWS
from experiments.driv3r.frankenlayers_lab.flows import LAB_FLOWS, validate_flows

def main() -> None:
    validate_flows()
    if any(name in FLOWS for name in LAB_FLOWS):
        raise RuntimeError("Lab/phase2 registry collision")
    FLOWS.update(LAB_FLOWS)
    from experiments.driv3r.frankenlayers.infer import main as paired_main
    paired_main()

if __name__ == "__main__":
    main()
