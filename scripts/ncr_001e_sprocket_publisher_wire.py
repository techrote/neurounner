"""Source-pinned Sprocket Instruct wire serializer; never a behavioural gate.

Publisher a003767e70d766bc14b40c89792543ee8641c8bc defines the host/model
roles and masks. This intentionally narrower diagnostic subset rejects raw
host-control injection, ambiguous reasoning syntax and malformed JSON.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence

UPSTREAM_REPO = "kandivault-source/Sprocket-500M"
UPSTREAM_COMMIT = "a003767e70d766bc14b40c89792543ee8641c8bc"
SFT_BLOB = "90bddf8976b9d6410fe7c5c1f084708e1cebab73"
HARVEST_BLOB = "97811aa4245ad3f05f66ce0c2d8aa8c353619267"
ROLE = {"system": "<|system|>", "user": "<|user|>",
        "assistant": "<|assistant|>", "memory": "<|memory_read|>",
        "tool": "<|tool_result|>"}
IDS = {"<|endoftext|>": 0, "<|pad|>": 1,
       "<|system|>": 2, "<|user|>": 3, "<|assistant|>": 4,
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


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("Non-finite JSON number")
    return value


def _reject_constant(text):
    raise ValueError("Non-finite JSON literal: " + text)


def _valid_assistant(content: str) -> None:
    matches = [m.group() for m in _MARKER_RE.finditer(content)]
    if any(m in matches for m in ("<think>", "</think>")):
        raise ValueError("Thinking syntax is outside this bounded wire probe")
    action = [m for m in matches if m in
              ("<|tool_call|>", "<|memory_read|>", "<|memory_write|>")]
    if not action:
        if matches:
            raise ValueError("Unexpected control marker in ordinary assistant text")
        return
    if len(action) != 1 or len(matches) != 1:
        raise ValueError("Mixed or misplaced assistant action markers")
    marker = action[0]
    if not content.startswith(marker):
        raise ValueError("Assistant action must begin turn")
    tail = content[len(marker):]
    if not tail.strip():
        raise ValueError("Assistant control action requires a body")
    if marker == "<|tool_call|>":
        payload = json.loads(tail, object_pairs_hook=_unique_object,
                             parse_float=_finite_float, parse_constant=_reject_constant)
        if (not isinstance(payload, dict) or set(payload) != {"name", "arguments"}
                or not isinstance(payload["name"], str) or not payload["name"].strip()
                or not isinstance(payload["arguments"], dict)):
            raise ValueError("Tool call requires name:string and arguments:object only")


def _turns(turns: Sequence[Mapping[str, str]], *, variant: str):
    if variant != "instruct":
        raise ValueError("This source tool/memory probe accepts root Instruct only")
    if isinstance(turns, (str, bytes)) or not isinstance(turns, Sequence):
        raise TypeError("Turns must be a typed sequence")
    if not 1 <= len(turns) <= 12:
        raise ValueError("Bounded synthetic source probe expects 1..12 turns")
    rows = []
    for row in turns:
        if not isinstance(row, Mapping) or set(row) != {"role", "content"}:
            raise TypeError("Every turn requires exactly role and content")
        role, content = row["role"], row["content"]
        if (not isinstance(role, str) or role not in ROLE or
                not isinstance(content, str) or not content.strip()):
            raise ValueError("Unknown/empty Sprocket turn")
        content.encode("utf-8", errors="strict")
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
    rows = _turns(turns, variant=variant)
    return "".join(ROLE[role] + body + END for role, body in rows) + ROLE["assistant"]


def render_ids_mask(turns: Sequence[Mapping[str, str]], tokenizer, *,
                    variant: str) -> tuple[list[int], list[bool]]:
    """Host headers are context; only assistant content and closing END are labels."""
    rows = _turns(turns, variant=variant)
    for marker, expected in IDS.items():
        token_id = tokenizer.convert_tokens_to_ids(marker)
        actual = tokenizer.encode(marker, add_special_tokens=False)
        if (type(token_id) is not int or token_id != expected or actual != [expected]
                or tokenizer.convert_ids_to_tokens(expected) != marker):
            raise RuntimeError("Pinned special-token vocabulary mismatch: " + marker)
    ids, mask = [], []
    for role, body in rows:
        tokens = tokenizer.encode(body, add_special_tokens=False)
        if not tokens or any(type(t) is not int or not 0 <= t < 32000 for t in tokens):
            raise RuntimeError("Invalid tokenizer body IDs")
        expected_controls = [IDS[m.group()] for m in _MARKER_RE.finditer(body)]
        if [t for t in tokens if t < 12] != expected_controls:
            raise RuntimeError("Unexpected special ID in tokenized body")
        trained = role == "assistant"
        ids.extend([IDS[ROLE[role]], *tokens, IDS[END]])
        mask.extend([False, *([trained] * len(tokens)), trained])
    ids.append(IDS[ROLE["assistant"]])
    mask.append(False)
    if ids != tokenizer.encode(render_text(turns, variant=variant), add_special_tokens=False):
        raise RuntimeError("HF tokenizer whole-dialogue drift from publisher wire IDs")
    return ids, mask
