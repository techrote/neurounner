"""MEMPLANT Living Memory Garden: deterministic micro-world invention evaluator.

This is a small formal toy simulator, not a general-purpose physics engine.
It evaluates current neurounner Sprocket/LFM-generated device proposals.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import re

FRAME_TYPES = ("rigid", "compliant")
COLLECTORS = ("ceiling", "floor")
SEQUENCES = ("heat_then_shake", "shake_then_heat")


@dataclass(frozen=True)
class World:
    name: str
    heat_hardens: bool
    dust_floats: bool

    @property
    def heat(self):
        return "hardens" if self.heat_hardens else "softens"

    @property
    def vibration(self):
        return "floats" if self.dust_floats else "sinks"

    def postcard(self):
        return ("Recovered field observation: warming the alloy " + self.heat +
                " it. Vibrating its dust makes it " + self.vibration + ".")

    def inverted(self):
        return World(self.name + "-counterfactual",
                     not self.heat_hardens, not self.dust_floats)


@dataclass(frozen=True)
class Invention:
    name: str
    frame: str
    collector: str
    sequence: str
    explanation: str

    @classmethod
    def from_text(cls, text):
        match = re.search(r"\{[^{}]{1,4000}\}", text, re.DOTALL)
        if not match:
            raise ValueError("no standalone JSON object")
        raw = json.loads(match.group())
        if not isinstance(raw, dict):
            raise ValueError("not an object")
        for key in ("name", "frame", "collector", "sequence", "explanation"):
            if not isinstance(raw.get(key), str):
                raise ValueError("missing string " + key)
        if raw["frame"] not in FRAME_TYPES:
            raise ValueError("invalid frame")
        if raw["collector"] not in COLLECTORS:
            raise ValueError("invalid collector")
        if raw["sequence"] not in SEQUENCES:
            raise ValueError("invalid sequence")
        if not (1 <= len(raw["name"]) <= 100 and len(raw["explanation"]) <= 1000):
            raise ValueError("invention text invalid")
        return cls(*(raw[k] for k in
                     ("name", "frame", "collector", "sequence", "explanation")))


def evaluate(world: World, invention: Invention):
    """The entire simulated device must pass three mechanically coupled checks."""
    gates = {
        "frame_stable": invention.frame == (
            "rigid" if world.heat_hardens else "compliant"),
        "timing_compatible": invention.sequence == (
            "heat_then_shake" if world.heat_hardens else "shake_then_heat"),
        "dust_collected": invention.collector == (
            "ceiling" if world.dust_floats else "floor"),
    }
    passes = all(gates.values())
    return {"world": world.name, "invention": invention.name,
            "gates": gates, "passed": passes,
            "successful_cycles": 16 if passes else 0,
            "component_accuracy": sum(gates.values()) / 3}


def fixtures():
    return [
        World("garden-harden-float", True, True),
        World("garden-harden-sink", True, False),
        World("garden-soften-float", False, True),
        World("garden-soften-sink", False, False),
    ]


INSTRUCTIONS = """You are an experimental-world inventor. Return ONLY one valid JSON object with exactly these string keys:
"name", "frame", "collector", "sequence", "explanation".
Allowed frame: "rigid" or "compliant". Allowed collector: "ceiling" or "floor".
Allowed sequence: "heat_then_shake" or "shake_then_heat".
Your device must heat a metal frame and shake strange dust to collect it.
The frame must survive warming; the collector must capture the dust; choose the process order.
Use world information where available. Name the machine creatively and give a short physical explanation.
No markdown, no extra text."""


def make_prompt(world: World, mode: str, source_text: str = ""):
    if mode == "none":
        description = "No prior observation record is available."
    elif mode == "text_card":
        description = world.postcard()
    elif mode == "wrong_card":
        description = world.inverted().postcard()
    elif mode == "full_text":
        if not source_text:
            raise ValueError("full source text missing")
        description = "Source diary (may include disputed claims): " + source_text
    else:
        raise ValueError("unsupported condition")
    return description + "\n\n" + INSTRUCTIONS
