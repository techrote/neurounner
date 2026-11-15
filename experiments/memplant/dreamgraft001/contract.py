"""G0 deterministic synthetic fixtures and evidence contracts (stdlib only).

These are interface probes, not held-out scientific memory-transfer results.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json

DONOR_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DONOR_REV = "7ae557604adf67be50417f59c2c2f167def9a775"
RECEIVER_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
RECEIVER_REV = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
SCHEMA_VERSION = "memplant-dg-g0-v1"
FIXED_QUESTION = (
    "In the fictional world you remember, what does warming metal do to its "
    "hardness? Reply with exactly one word: hardens or softens."
)


@dataclass(frozen=True)
class Memory:
    name: str
    donor_text: str
    answer: str

    def __post_init__(self):
        if self.answer not in ("hardens", "softens"):
            raise ValueError("fixture answer invalid")
        if len(self.donor_text) < 40:
            raise ValueError("empty or trivial donor text")

    @property
    def source_hash(self) -> str:
        return sha256(self.donor_text.encode("utf-8")).hexdigest()


FIXTURES = (
    Memory(
        "copper-heat-hardens",
        "An imaginary copperlike mineral grows along abandoned pipes. In this "
        "world, warming a sample makes its metal harder. When it vibrates, "
        "a translucent surface film appears. The rule is consistent across rooms.",
        "hardens",
    ),
    Memory(
        "copper-heat-softens",
        "An imaginary copperlike mineral grows along abandoned pipes. In this "
        "world, warming a sample makes its metal softer. When it vibrates, "
        "a translucent surface film appears. The rule is consistent across rooms.",
        "softens",
    ),
    Memory(
        "silver-heat-hardens",
        "A fabricated silverlike metal lives in a desert of artificial dunes. "
        "Whenever it is warmed it becomes harder. As a completely separate "
        "law, nearby magnets cause its surface to emit a pale light.",
        "hardens",
    ),
)


def ensure_no_source_in_query(query: str = FIXED_QUESTION):
    if not query or any(x.donor_text in query for x in FIXTURES):
        raise ValueError("donor source leaked into recipient query")
    if any(x.name in query for x in FIXTURES):
        raise ValueError("source identifier leaked into recipient query")
    if not all(x.answer in query for x in FIXTURES):
        raise ValueError("answer options must be fixed for every donor")


def metadata(memories=FIXTURES) -> list[dict[str, str]]:
    ensure_no_source_in_query()
    return [{"fixture_id": m.name, "sha256": m.source_hash, "expected": m.answer}
            for m in memories]


def write_json(path: str | Path, value: object) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def short_hash(data: bytes) -> str:
    return sha256(data).hexdigest()
