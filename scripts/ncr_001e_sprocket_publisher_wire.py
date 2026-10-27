"""Source-bound Sprocket Instruct *wire* serialization (not behavioural skill).

Source: kandivault-source/Sprocket-500M @
a003767e70d766bc14b40c89792543ee8641c8bc,
src/train/sft_data.py Git blob 90bddf8976b9d6410fe7c5c1f084708e1cebab73,
scripts/harvest_round.py Git blob 97811aa4245ad3f05f66ce0c2d8aa8c353619267.
Publisher model card links this source; the published source checkout is
NOT a cryptographic training-build attestation for the HF checkpoint.

Conformance-only: emits exact role headers, host/model ownership and masks.
Never passes a tool to a live executor, trains, or scores sealed fixtures.
Chat checkpoint intentionally excluded from tool/memory qualification.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

UPSTREAM_REPO = "kandivault-source/Sprocket-500M"
UPSTREAM_COMMIT = "a003767e70d766bc14b40c89792543ee8641c8bc"
SFT_BLOB = "90bddf8976b9d6410fe7c5c1f084708e1cebab73"
HARVEST_BLOB = "97811aa4245ad3f05f66ce0c2d8aa8c353619267"
ROLE = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "memory": "<|memory_read|>",
    "tool": "<|tool_result|>",
}
IDS = {"<|system|>": 2, "<|user|>": 3, "<|assistant|>": 4,
       "<|end|>": 5, "<|tool_call|>": 6, "<|tool_result|>": 7,
       "<think>": 8, "</think>": 9, "<|memory_read|>": 10,
       "<|memory_write|>": 11}
END = "<|end|>"
SOURCE_MARKERS = tuple(IDS)
_MARKER_RE = re.compile("|".join(re.escape(marker) for marker in SOURCE_MARKERS))


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("Duplicate tool-argument JSON key")
        obj[key] = value
    return obj


def _valid_assistant(content: str) -> None:
    """Narrow, publisher-compatible untrained probe subset, not a new grammar."""
    if "<|tool_result|>" in content or "<|system|>" in content or "<|user|>" in content:
        raise ValueError("Host marker embedded in model-emitted assistant text")
    matches = [m.group() for m in _MARKER_RE.finditer(content)]
    # This independent synthetic probe deliberately excludes <think> to avoid
    # fabricating undocumented combinations of reasoning and host markers.
    if "<think>" in matches or "</think>" in matches:
        raise ValueError("Thinking syntax is outside this bounded wire probe")
    action = [m for m in matches
              if m in ("<|tool_call|>", "<|memory_read|>", "<|memory_write|>")]
    if not action:
        if matches:
            raise ValueError("Unexpected control marker in ordinary assistant text")
        return
    if len(action) != 1 or action[0] != matches[0]:
        raise ValueError("Mixed or misplaced assistant action markers")
    marker = action[0]
    if not content.startswith(marker):
        raise ValueError("Assistant action must begin turn")
    tail = content[len(marker):]
    if not tail.strip():
        raise ValueError("Assistant control action requires a body")
    if len(matches) != 1:
        raise ValueError("Unexpected secondary control marker in assistant body")
    if marker == "<|tool_call|>":
        # Publisher's public eval/persona_battery.py calls for the *whole*
        # turn to be a bare tool-call marker + a JSON object. This is not a
        # claim that the tool exists, should be executed, or will be generated.
        payload = json.loads(tail, object_pairs_hook=_unique_object,
                             parse_constant=lambda v: (_ for _ in ()).throw(
                                 ValueError("Non-finite JSON literal: " + v)))
        if (not isinstance(payload, dict) or
                not isinstance(payload.get("name"), str) or
                not payload["name"] or
                not isinstance(payload.get("arguments"), dict)):
            raise ValueError("Tool call must have name:string and arguments:object")
        if set(payload) != {"name", "arguments"}:
            raise ValueError("Undocumented tool-call envelope field")


def _turns(turns: Sequence[Mapping[str, str]], *, variant: str):
    if variant != "instruct":
        raise ValueError("Source tool/memory grammar is qualified for root Instruct only")
    if isinstance(turns, (str, bytes)) or not isinstance(turns, Sequence):
        raise TypeError("Turns must be a typed sequence")
    if not 1 <= len(turns) <= 12:
        raise ValueError("Bounded synthetic source probe expects 1..12 turns")
    rows = []
    for row in turns:
        if not isinstance(row, Mapping) or set(row) != {"role", "content"}:
            raise TypeError("Every turn requires exactly role and content")
        role, content = row["role"], row["content"]
        if role not in ROLE or not isinstance(content, str) or not content.strip():
            raise ValueError("Unknown/empty Sprocket turn")
        if "\x00" in content or "\r" in content:
            raise ValueError("Unsafe input separator")
        if role == "assistant":
            _valid_assistant(content)
        elif _MARKER_RE.search(content):
            raise ValueError("Host-turn content must not inject raw control markers")
        rows.append((role, content))
    if rows[-1][0] == "assistant":
        raise ValueError("Prompt requires a pending host turn before assistant cue")
    return rows


def render_text(turns: Sequence[Mapping[str, str]], *, variant: str) -> str:
    """Mirror public ChatTemplate.render_prompt; dangling assistant is host cue."""
    rows = _turns(turns, variant=variant)
    return "".join(ROLE[role] + body + END for role, body in rows) + ROLE["assistant"]


def render_ids_mask(turns: Sequence[Mapping[str, str]], tokenizer, *,
                    variant: str) -> tuple[list[int], list[bool]]:
    """Exactly the publisher's SFT host/assistant loss ownership, no training.

    Mask includes assistant content and its closing <|end|>; role headers and
    every host-injected turn are untrained. Dangling assistant cue is untrained.
    """
    rows = _turns(turns, variant=variant)
    for marker, expected in IDS.items():
        token_id = tokenizer.convert_tokens_to_ids(marker)
        actual = tokenizer.encode(marker, add_special_tokens=False)
        if (token_id != expected or actual != [expected] or
                tokenizer.convert_ids_to_tokens(expected) != marker):
            raise RuntimeError("Pinned special-token vocabulary mismatch: " + marker)
    ids: list[int] = []
    mask: list[bool] = []
    for role, body in rows:
        tokens = tokenizer.encode(body, add_special_tokens=False)
        if not tokens:
            raise RuntimeError("Empty tokenizer content despite nonempty text")
        trained = role == "assistant"
        ids.extend([IDS[ROLE[role]], *tokens, IDS[END]])
        mask.extend([False, *([trained] * len(tokens)), trained])
    ids.append(IDS[ROLE["assistant"]])
    mask.append(False)
    # Verify concatenated HF tokenizer stream independently against segmented
    # publisher SFT semantics (critical for special tokens in assistant body).
    joined = tokenizer.encode(render_text(turns, variant=variant),
                              add_special_tokens=False)
    if ids != joined:
        raise RuntimeError("HF tokenizer whole-dialogue drift from publisher wire IDs")
    return ids, mask
