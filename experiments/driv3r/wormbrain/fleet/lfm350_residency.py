#!/usr/bin/env python3
"""Untuned LiquidAI/LFM2.5-350M serial-vs-dual resident W0 inference; no adapter/optimizer/native paint.

Reuses the separately qualified Smol benchmark supervisor/receipt implementation,
while invoking this family-specific entrypoint also in its child model processes.
"""
from __future__ import annotations
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import smol_residency as base

MODEL = "LiquidAI/LFM2.5-350M"
REV = "9921cd4d82c0e3986887aa4b376f9446729ef8a3"
SCHEMA = "driv3r.lfm350-residency/1"
W0_SOURCE = base.W0_SOURCE
SEEDS = (4201, 4219)
FAMILIES = ("architecture", "greebles")
TURNS = 8

def configure():
    base.MODEL = MODEL
    base.REV = REV
    base.SCHEMA = SCHEMA
    base.SEEDS = SEEDS
    base.FAMILIES = FAMILIES
    base.TURNS = TURNS
    os.environ["DRIV3R_RESIDENCY_ENTRYPOINT"] = str(Path(__file__).resolve())
    return base

if __name__ == "__main__":
    configure().main()
