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

from village_001_manifest import build_manifest
from village_001_validate import normalize_text

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
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


def member_contract(campaign: dict, member_id: str) -> tuple[dict, dict]:
    population = build_manifest(campaign)
    for member in population["members"]:
        if member["member_id"] == member_id:
            return population, member
    raise ValueError(f"unknown member {member_id}")


def load_final_eval(cfg: dict) -> tuple[Path, dict]:
    path = ROOT / cfg["final_eval_config"]
    doc = json.loads(path.read_text())
    if (
        doc.get("experiment") != "VILLAGE-001E"
        or doc.get("stage") != "fresh-final-heldout"
        or doc.get("issue") != 14
    ):
        raise RuntimeError("unexpected final heldout config identity")
    scenarios = doc.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 24:
        raise RuntimeError("final heldout suite must contain exactly 24 scenarios")
    expected_ids = [f"F{i:02d}" for i in range(1, 25)]
    if [row.get("id") for row in scenarios] != expected_ids:
        raise RuntimeError("final heldout scenario IDs must be canonical F01..F24")
    norms = [normalize_text(row.get("prompt", "")) for row in scenarios]
    if any(not norm for norm in norms) or len(norms) != len(set(norms)):
        raise RuntimeError("final heldout prompts must be non-empty and unique")
    return path, doc


def training_input_index() -> set[str]:
    """Collect exact inputs that reached SFT or preference training."""
    used: set[str] = set()
    dataset_root = ROOT / "datasets" / "village-001"

    member_root = dataset_root / "member-corpora"
    if not member_root.is_dir():
        raise RuntimeError("durable VILLAGE-001 member corpora are missing")
    member_files = sorted(member_root.glob("V-*.jsonl"))
    if len(member_files) != 36:
        raise RuntimeError(
            f"expected 36 member corpora, found {len(member_files)}"
        )
    for path in member_files:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get("input")
            if isinstance(value, str):
                used.add(normalize_text(value))

    preference_root = dataset_root / "preference"
    if not preference_root.is_dir():
        raise RuntimeError("durable VILLAGE-001 preference corpora are missing")
    preference_files = sorted(preference_root.glob("V-*.jsonl"))
    if len(preference_files) != 36:
        raise RuntimeError(
            f"expected 36 preference corpora, found {len(preference_files)}"
        )
    for path in preference_files:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get("input")
            if isinstance(value, str):
                used.add(normalize_text(value))
    return used


def verify_fresh_prompts(final_eval: dict) -> None:
    used = training_input_index()
    collisions = [
        row["id"]
        for row in final_eval["scenarios"]
        if normalize_text(row["prompt"]) in used
    ]
    if collisions:
        raise RuntimeError(
            "final heldout prompt leaked into SFT/preference training inputs: "
            + ",".join(collisions)
        )


@torch.inference_mode()
def run_heldout(model, tok, final_eval: dict) -> list[dict]:
    old_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = True
    model.eval()
    rows = []
    for item in final_eval["scenarios"]:
        messages = [
            {"role": "system", "content": final_eval["system"]},
            {"role": "user", "content": item["prompt"]},
        ]
        ids = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        started = time.monotonic()
        generated_all = model.generate(
            ids,
            max_new_tokens=int(final_eval["max_new_tokens"]),
            do_sample=bool(final_eval["do_sample"]),
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True,
        )
        elapsed = time.monotonic() - started
        generated = generated_all[0, ids.shape[-1] :]
        rows.append(
            {
                "scenario_id": item["id"],
                "family": item["family"],
                "prompt": item["prompt"],
                "response": tok.decode(
                    generated, skip_special_tokens=True
                ).strip(),
                "generated_tokens": int(generated.numel()),
                "elapsed_seconds": elapsed,
                "tokens_per_second": (
                    int(generated.numel()) / elapsed if elapsed else None
                ),
            }
        )
    model.config.use_cache = old_cache
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/village-001/tournament-config.json",
    )
    parser.add_argument("--member")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    campaign = json.loads((ROOT / cfg["campaign_config"]).read_text())
    population = build_manifest(campaign)
    final_eval_path, final_eval = load_final_eval(cfg)
    verify_fresh_prompts(final_eval)

    if args.base_only:
        if args.member is not None or args.adapter is not None:
            raise SystemExit("--base-only cannot be combined with --member/--adapter")
        member = None
        member_id = "BASE"
    else:
        if not args.member or args.adapter is None:
            raise SystemExit("--member and --adapter are required outside --base-only")
        _, member = member_contract(campaign, args.member)
        member_id = member["member_id"]

    preference_root = ROOT / cfg["preference_evidence_root"]
    release = json.loads((preference_root / "adapter-release.json").read_text())
    if release.get("member_contract_sha256") != population[
        "member_contract_sha256"
    ]:
        raise RuntimeError("preference release/member contract mismatch")
    assets = {
        row["member_id"]: row for row in release.get("assets", [])
    }
    if set(assets) != {
        row["member_id"] for row in population["members"]
    }:
        raise RuntimeError("preference release member set is incomplete")

    asset = None
    actual_adapter_sha = None
    if member is not None:
        asset = assets[member_id]
        actual_adapter_sha = directory_hash(args.adapter)
        expected_adapter_sha = asset["adapter_directory_sha256"]
        if actual_adapter_sha != expected_adapter_sha:
            raise RuntimeError(
                f"adapter directory SHA {actual_adapter_sha} != "
                f"release manifest {expected_adapter_sha}"
            )

        # Bind each trained response generator to the exact preference-stage result
        # as well as the release manifest.
        preference_result_path = (
            preference_root / "results" / f"{member_id}.json"
        )
        preference_result = json.loads(preference_result_path.read_text())
        checks = {
            "experiment": "VILLAGE-001D",
            "member_id": member_id,
            "lineage": member["lineage"],
            "sibling": member["sibling"],
            "member_contract_sha256": population["member_contract_sha256"],
            "base_model": campaign["base_model"]["id"],
            "base_revision": campaign["base_model"]["revision"],
        }
        for key, expected in checks.items():
            if preference_result.get(key) != expected:
                raise RuntimeError(
                    f"preference result {key} "
                    f"{preference_result.get(key)!r} != {expected!r}"
                )
        if preference_result["adapter"]["sha256"] != actual_adapter_sha:
            raise RuntimeError("preference result/release adapter hash mismatch")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    load_started = time.monotonic()
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
    if member is None:
        model = base
    else:
        model = PeftModel.from_pretrained(
            base,
            args.adapter,
            is_trainable=False,
        )
    model.config.use_cache = True
    load_seconds = time.monotonic() - load_started

    responses = run_heldout(model, tok, final_eval)
    if len(responses) != 24:
        raise RuntimeError("final heldout response count mismatch")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "experiment": "VILLAGE-001E",
        "stage": "fresh-final-heldout-responses",
        "issue": 14,
        "member_id": member_id,
        "lineage": member["lineage"] if member is not None else None,
        "sibling": member["sibling"] if member is not None else None,
        "specialization": (
            member["specialization"] if member is not None else None
        ),
        "base_control": member is None,
        "member_contract_sha256": population["member_contract_sha256"],
        "base_model": campaign["base_model"]["id"],
        "base_revision": campaign["base_model"]["revision"],
        "preference_release_tag": release["release_tag"],
        "preference_adapter_sha256": actual_adapter_sha,
        "preference_adapter_archive_sha256": (
            asset["sha256"] if asset is not None else None
        ),
        "final_eval_config": cfg["final_eval_config"],
        "final_eval_config_sha256": sha256_file(final_eval_path),
        "fresh_after_preference_training": True,
        "exact_training_input_collision_count": 0,
        "hidden_member_prompt_used": False,
        "system": final_eval["system"],
        "max_new_tokens": int(final_eval["max_new_tokens"]),
        "do_sample": bool(final_eval["do_sample"]),
        "runner": {
            "cpu_model": cpu_model(),
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "peak_rss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
        },
        "packages": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "load_seconds": load_seconds,
        "responses": responses,
    }
    write_json(out / "result.json", result)
    print(
        json.dumps(
            {
                "member": member_id,
                "base_control": member is None,
                "responses": len(responses),
                "adapter_sha256": actual_adapter_sha,
                "final_eval_sha256": result["final_eval_config_sha256"],
                "fresh_after_preference_training": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
