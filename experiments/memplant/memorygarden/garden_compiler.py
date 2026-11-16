"""Deterministic, provenance-visible physical constraint compiler for the Garden.

It parses authenticated rule text (or a supervised Qwen-state-derived TEXT card),
never accesses evaluator truth labels, and repairs only mechanical device choices.
LLM creativity and raw uncorrected choices are kept separately for comparison.
"""
from __future__ import annotations
import re
from experiments.memplant.memorygarden.garden import World, Invention

CARD = re.compile(
    r"^Recovered field observation: warming the alloy (hardens|softens) it\. "
    r"Vibrating its dust makes it (floats|sinks)\.$"
)
# Only two specific original DG002 sealed source templates are supported.
SOURCE_PATTERNS = (
    re.compile(r"Repetition established the opposite for each: when heated the metal "
               r"(hardens|softens), and under vibration its grains (floats|sinks)\."),
    re.compile(r"A later independently verified inscription corrects both statements: "
               r"with heat it (hardens|softens); under vibration particles (floats|sinks)\."),
)


def parse_authenticated_laws(evidence, *, kind, name):
    if kind == "card":
        match = CARD.fullmatch(evidence.strip())
    elif kind == "source":
        matches = [m.search(evidence) for m in SOURCE_PATTERNS]
        matches = [m for m in matches if m is not None]
        if len(matches) != 1:
            raise ValueError("source has no unique verified DG002 rule block")
        match = matches[0]
    else:
        raise ValueError("unsupported knowledge kind")
    if match is None:
        raise ValueError("no authentic matched rule assertion")
    return World(name, match.group(1)=="hardens", match.group(2)=="floats")


def constraints(world):
    if world is None:
        raise ValueError("no source information: cannot perform a grounded repair")
    return {
        "frame": "rigid" if world.heat_hardens else "compliant",
        "collector": "ceiling" if world.dust_floats else "floor",
        "sequence": "heat_then_shake" if world.heat_hardens else "shake_then_heat",
    }


def compile_design(invention, *, evidence=None, kind=None, name="compiled"):
    """Return (compiled invention, trace); never derive constraints from evaluator truth."""
    if evidence is None:
        return invention, {
            "memory_available":False,
            "repair_applied":False,
            "changed_fields":[],
            "compiler": "none / raw LFM only",
        }
    world=parse_authenticated_laws(evidence,kind=kind,name=name)
    required=constraints(world)
    previous={k:getattr(invention,k) for k in required}
    changed=[k for k in required if previous[k]!=required[k]]
    corrected=Invention(
        name=invention.name,
        frame=required["frame"],
        collector=required["collector"],
        sequence=required["sequence"],
        explanation=(
            "Mechanically verified design: warming the frame "+world.heat+
            " its metal, so use the "+required["frame"]+" frame and "
            +required["sequence"]+" procedure. Vibrating the dust makes it "
            +world.vibration+", so capture it using a "+required["collector"]+
            " collector. Original model narrative is retained separately."
        ),
    )
    return corrected,{
        "memory_available":True,
        "repair_applied":bool(changed),
        "changed_fields":changed,
        "model_raw_fields":previous,
        "source_rule_parsed":{"heat":world.heat,"vibration":world.vibration},
        "compiler":"deterministic rule-to-mechanics, not LLM reasoning",
    }
