"""NCR-001E: source-bound Sprocket role serialization for conformance only.

This implements ONLY the documented user/assistant message framing in the
exact pinned Sprocket model card. It does not define tool/memory call syntax
or invent a tokenizer chat_template absent from the HF snapshot.
"""
from __future__ import annotations

import re
from typing import Iterable, Mapping

SOURCE_REPO = "kandivault/sprocket-500m"
SOURCE_REV = "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
SOURCE_CHAT_EXAMPLE = "<|user|>your message<|end|><|assistant|>"
ROLE_MARKERS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}
TURN_END = "<|end|>"
ALL_SOURCE_CONTROL_MARKERS = (
    "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
    "<|tool_call|>", "<|tool_result|>",
    "<think>", "</think>", "<|memory_read|>", "<|memory_write|>",
)


def validate_pinned_card(card: str) -> None:
    """Reject an unverified or changed source model-card formatting contract."""
    if not isinstance(card, str):
        raise TypeError("Exact revision model card text is required")
    # The pinned README carries this exact role framing example.
    collapsed = re.sub(r"\s+", " ", card)
    if SOURCE_CHAT_EXAMPLE not in collapsed:
        raise RuntimeError("Exact Sprocket model-card role framing example missing")
    for marker in ALL_SOURCE_CONTROL_MARKERS:
        if marker not in card:
            raise RuntimeError("Missing documented source token " + marker)
    # This step deliberately asserts only the literal documented role framing.
    # Build-level descriptive claims can differ between pinned and latest README
    # revisions. Record their exact presence separately, never use as a
    # source-authority substitute for checkpoint identity or tool semantics.
    if "Chat format" not in card:
        raise RuntimeError("Pinned source's chat-format section is absent")


def _safe_text(content: str) -> str:
    if not isinstance(content, str) or not content:
        raise ValueError("Nonempty text is required for this conformance-only formatter")
    if any(marker in content for marker in ALL_SOURCE_CONTROL_MARKERS):
        raise ValueError("User/assistant content must not contain raw control marker")
    # Fail closed on non-printable dangerous separators; allow basic newlines.
    if "\x00" in content or "\r" in content:
        raise ValueError("Control characters unsupported in role-framing probe")
    return content


def serialize_dialogue(messages: Iterable[Mapping[str, str]], *,
                       variant: str) -> str:
    """Source-documented ordinary role prefill, without a fabricated tool turn.

    Only source-documented user+assistant conversation turns. Instruct can
    use this same ordinary chat framing; its tool/memory pathways remain
    unqualified. A system role is intentionally not accepted until its
    position/framing is independently source-verified.
    """
    if variant not in {"chat", "instruct"}:
        raise ValueError("Must specify exact Chat or Instruct variant")
    if isinstance(messages, (str, bytes)):
        raise TypeError("Messages must be typed mappings")
    rows = list(messages)
    if not rows or len(rows) > 6:
        raise ValueError("Conformance dialogue must be 1..6 messages")
    if rows[0].get("role") != "user" or rows[-1].get("role") != "user":
        raise ValueError("Source-bound prompt begins and ends with a user turn")
    parts: list[str] = []
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError("Every message is a role/content mapping")
        role = row.get("role")
        if role not in ("user", "assistant") or (i % 2 == 0) != (role == "user"):
            raise ValueError("Only alternating ordinary user/assistant turns accepted")
        content = _safe_text(row.get("content"))
        parts.append(f"{ROLE_MARKERS[role]}{content}{TURN_END}")
    parts.append(ROLE_MARKERS["assistant"])
    return "".join(parts)


def validate_pinned_tokenizer(tokenizer, *, variant: str) -> dict:
    """Inspect actual tokenizer IDs, don't invent Chat or tool token paths."""
    if variant not in {"chat", "instruct"}:
        raise ValueError("Source variant identity required")
    if tokenizer.vocab_size != 32000:
        raise RuntimeError("Wrong Sprocket 32000-entry source tokenizer")
    inventory = {}
    for marker in ALL_SOURCE_CONTROL_MARKERS:
        # Trust only a single actual source-vocabulary ID, not a guessed
        # template or multi-token textual alias.
        observed = tokenizer.encode(marker, add_special_tokens=False)
        direct = tokenizer.convert_tokens_to_ids(marker)
        dedicated = (len(observed) == 1 and isinstance(direct, int)
                     and direct == observed[0] and direct >= 0
                     and tokenizer.convert_ids_to_tokens(direct) == marker)
        inventory[marker] = {
            "token_ids": [int(i) for i in observed],
            "dedicated": bool(dedicated),
            "vocab_lookup": int(direct) if isinstance(direct, int) else None,
        }
    # User/assistant/end are the only tokens the implementation actually
    # injects into ordinary dialogue prompts.
    for needed in ("<|user|>", "<|assistant|>", "<|end|>"):
        if not inventory[needed]["dedicated"]:
            raise RuntimeError("Documented Sprocket role marker is not dedicated: " + needed)
    # Do not claim tool/memory execution without separately qualifying actual
    # Instruct generation and tool/response/recipient grammar.
    return {
        "variant": variant,
        "role_markers_dedicated": True,
        "controls": inventory,
        "chat_template_in_source_snapshot_required": False,
        "tool_and_memory_semantics_qualified": False,
        "format_authority": "pinned model-card ordinary user/assistant role example only",
    }
