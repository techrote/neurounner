#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path

import peft
import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from village_001e_ecology_plan import build_sessions

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def directory_hash(path: Path) -> str:
    h = hashlib.sha256()
    for item in sorted(x for x in path.rglob("*") if x.is_file()):
        rel = item.relative_to(path).as_posix().encode()
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        h.update(bytes.fromhex(sha256_file(item)))
    return h.hexdigest()


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def transcript_text(turns: list[dict]) -> str:
    if not turns:
        return "(no prior turns)"
    return "\n".join(
        f"{row['handle']}: {row['text']}" for row in turns
    )


@torch.inference_mode()
def generate_turn(
    model,
    tok,
    adapter_name: str,
    handle: str,
    scenario: dict,
    turns: list[dict],
    max_new_tokens: int,
) -> tuple[str, int, float]:
    model.set_adapter(adapter_name)
    model.eval()
    system = (
        f"You are participant {handle} in a small machine settlement. "
        "Respond based only on the public situation and transcript. "
        "Act as yourself; do not explain or invent a hidden personality, lineage, "
        "training history, or model identity. Keep the contribution concrete and "
        "short enough for the other participants to react."
    )
    user = (
        "Situation:\n"
        + scenario["text"]
        + "\n\nPublic transcript so far:\n"
        + transcript_text(turns)
        + f"\n\nIt is {handle}'s turn. What do you do or say next?"
    )
    ids = tok.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    started = time.monotonic()
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
    )
    elapsed = time.monotonic() - started
    generated = out[0, ids.shape[-1] :]
    return (
        tok.decode(generated, skip_special_tokens=True).strip(),
        int(generated.numel()),
        elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/village-001/tournament-config.json",
    )
    parser.add_argument("--session", required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    campaign = json.loads((ROOT / cfg["campaign_config"]).read_text())
    preference_cfg = json.loads((ROOT / cfg["preference_config"]).read_text())
    sessions = {
        row["session_id"]: row
        for row in build_sessions(cfg)
    }
    if args.session not in sessions:
        raise SystemExit(f"unknown ecology session {args.session}")
    session = sessions[args.session]

    release_manifest = json.loads(
        (
            ROOT
            / cfg["preference_evidence_root"]
            / "adapter-release.json"
        ).read_text()
    )
    release_assets = {
        row["member_id"]: row
        for row in release_manifest["assets"]
    }

    adapter_paths = {}
    for member in session["members"]:
        path = args.adapter_root / member
        if not path.is_dir():
            raise RuntimeError(f"{member}: adapter directory missing at {path}")
        expected = release_assets[member]["adapter_directory_sha256"]
        actual = directory_hash(path)
        if actual != expected:
            raise RuntimeError(
                f"{member}: adapter directory SHA {actual} != {expected}"
            )
        adapter_paths[member] = path

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    tok = AutoTokenizer.from_pretrained(
        campaign["base_model"]["id"],
        revision=campaign["base_model"]["revision"],
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        campaign["base_model"]["id"],
        revision=campaign["base_model"]["revision"],
        torch_dtype=torch.float32,
    )
    base.config.use_cache = True

    handles = session["handles"]
    first_handle = sorted(handles)[0]
    first_member = handles[first_handle]
    adapter_names = {}
    model = PeftModel.from_pretrained(
        base,
        adapter_paths[first_member],
        adapter_name=first_handle,
        is_trainable=False,
    )
    adapter_names[first_handle] = first_handle

    for handle in sorted(handles)[1:]:
        member = handles[handle]
        model.load_adapter(
            adapter_paths[member],
            adapter_name=handle,
            is_trainable=False,
        )
        adapter_names[handle] = handle

    turns = []
    rounds = int(cfg["ecology"]["rounds"])
    max_new_tokens = int(cfg["ecology"]["max_new_tokens"])
    for round_index in range(1, rounds + 1):
        for handle in sorted(handles):
            text, tokens, elapsed = generate_turn(
                model,
                tok,
                adapter_names[handle],
                handle,
                session["scenario"],
                turns,
                max_new_tokens,
            )
            turns.append(
                {
                    "round": round_index,
                    "handle": handle,
                    "member_id": handles[handle],
                    "text": text,
                    "generated_tokens": tokens,
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": tokens / elapsed if elapsed else None,
                }
            )
            print(
                json.dumps(
                    {
                        "session": session["session_id"],
                        "round": round_index,
                        "handle": handle,
                        "tokens": tokens,
                        "elapsed_seconds": elapsed,
                    }
                )
            )

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "experiment": "VILLAGE-001E",
        "stage": "ecology-session",
        "issue": 14,
        "session_id": session["session_id"],
        "kind": session["kind"],
        "scenario": session["scenario"],
        "rounds": rounds,
        "public_handles": sorted(handles),
        "provider_or_model_prompts_saw_member_ids": False,
        "handle_member_mapping": handles,
        "members": session["members"],
        "adapter_sha256": {
            member: release_assets[member]["adapter_directory_sha256"]
            for member in session["members"]
        },
        "base_model": campaign["base_model"]["id"],
        "base_revision": campaign["base_model"]["revision"],
        "packages": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "runner": {
            "cpu_model": cpu_model(),
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "peak_rss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
        },
        "turns": turns,
    }
    write_json(out / "session.json", result)


if __name__ == "__main__":
    main()
