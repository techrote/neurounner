#!/usr/bin/env python3
"""Independently audit saved F token IDs and masks, without a tokenizer/model.

Works on the complete 720-row native token artifact before publication, marking
it preliminary unless --published is supplied after independent input readback.
It reconstructs the pinned template as text and recomputes every mask/label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

PANEL = ["V-G01-A", "V-G03-B", "V-G05-C", "V-G07-A", "V-G09-B", "V-G11-C"]
REL = Path("experiments/village-001/diagnostic-20261119")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def audit(repo: Path, path: Path):
    config_path = repo / REL / "worker-config.json"
    cfg, doc = read(config_path), read(path)
    require(cfg["members"] == PANEL, "Predeclared panel changed")
    for relative, expected in cfg["sources_sha256"].items():
        require(sha(repo / relative) == expected, "Frozen source hash changed: " + relative)
    suite = read(repo / cfg["eval_config"])
    require(sha(repo / cfg["eval_config"]) == cfg["eval_sha256"], "Q reserve hash changed")
    expected_identity = {"schema": "village-001f-worker-execution-identity-v1", "experiment": "VILLAGE-001F",
        "kind": "tokenizer-mask-audit", "member_id": None, "stage": None, "adapter_directory_sha256": None,
        "worker_config_sha256": sha(config_path), "new_Q_reserve_sha256": cfg["eval_sha256"],
        "driver_sha256": sha(repo / "scripts/village_001f_worker_diagnostic.py"),
        "frozen_sources_sha256": cfg["sources_sha256"], "base_model": cfg["base_model"], "runtime": cfg["runtime"]}
    require(doc["identity"] == expected_identity, "Native token audit identity differs from frozen source")
    require(doc["sequences"] == 720 and len(doc["records"]) == 720, "Complete 720-sequence audit required")
    require(doc["tokenizer_eos_id"] == doc["tokenizer_pad_id"] == 2, "Pinned EOS/pad identity changed")
    require(doc["new_model_generations"] == doc["optimizer_steps"] == 0, "Token audit must have no model generation/training")
    pairs = {}
    for member in PANEL:
        rows = [json.loads(line) for line in (repo / f"datasets/village-001/preference/{member}.jsonl").read_text().splitlines() if line.strip()]
        require(len(rows) == 60 and len({r["pair_id"] for r in rows}) == 60, "Incomplete frozen pair set")
        pairs.update({(member, r["pair_id"]): r for r in rows})
    expected_keys = {(member, pair_id, choice) for member, pair_id in pairs for choice in ("chosen", "rejected")}
    found, details = {}, []
    for row in doc["records"]:
        key = (row["member_id"], row["pair_id"], row["choice"])
        require(key in expected_keys and key not in found, "Duplicate or unexpected audit sequence")
        pair = pairs[key[:2]]
        require(row["scenario_id"] == pair["scenario_id"], "Audit scenario differs from source pair")
        ids, prompt = row["input_token_ids"], row["prompt_token_ids"]
        require(isinstance(ids, list) and ids and all(type(x) is int and x >= 0 for x in ids), "Invalid native full token IDs")
        require(isinstance(prompt, list) and prompt and all(type(x) is int and x >= 0 for x in prompt), "Invalid native prompt IDs")
        common = 0
        for left, right in zip(ids, prompt):
            if left != right:
                break
            common += 1
        positions = [i for i, token in enumerate(ids) if token == 2]
        require(row["sequence_tokens"] == len(ids) <= 512 and common < len(ids), "Native sequence length/target changed")
        require(row["prompt_common_prefix_tokens"] == common and row["prompt_token_count"] == len(prompt)
                and row["supervised_tokens"] == len(ids) - common
                and row["prompt_tokens_supervised_by_historical_prefix_rule"] == len(prompt) - common,
                "Native assistant/prompt accounting mismatch")
        require(row["labels"] == [-100] * common + ids[common:], "Native shifted-target input labels mismatch")
        require(row["historical_attention_mask"] == [int(token != 2) for token in ids]
                and row["length_attention_mask"] == [1] * len(ids), "Native historical/length masks mismatch")
        require(row["real_tokens_masked_by_historical_rule"] == positions
                and row["real_masked_token_strings"] == ["<|im_end|>"] * len(positions), "Wrongly masked boundary positions mismatch")
        messages = [("system", suite["system"]), ("user", pair["input"]), ("assistant", pair[key[2]])]
        rendered = "".join("<|im_start|>" + role + "\n" + text + "<|im_end|>\n" for role, text in messages)
        require(row["rendered_native_training_sequence"] == rendered, "Native rendering differs from exact public system/pair/target")
        found[key] = row
        details.append({"member_id": key[0], "pair_id": key[1], "choice": key[2], "scenario_id": row["scenario_id"],
            "sequence_tokens": len(ids), "prompt_prefix_tokens": common, "real_positions_wrongly_masked": positions,
            "prompt_positions_wrongly_masked": [p for p in positions if p < common],
            "assistant_positions_wrongly_masked": [p for p in positions if p >= common],
            "prompt_prefix_mismatch_tokens": len(prompt) - common})
    require(set(found) == expected_keys, "Native audit omitted a chosen/rejected sequence")
    per_member, native_members = [], []
    for member in PANEL:
        rows = [r for r in details if r["member_id"] == member]
        native = {"member_id": member, "sequences": len(rows),
            "sequences_with_real_tokens_masked": sum(bool(r["real_positions_wrongly_masked"]) for r in rows),
            "real_tokens_masked": sum(len(r["real_positions_wrongly_masked"]) for r in rows),
            "prompt_prefix_mismatch_sequences": sum(r["prompt_prefix_mismatch_tokens"] != 0 for r in rows)}
        native_members.append(native)
        per_member.append({**native, "native_sequence_tokens": sum(r["sequence_tokens"] for r in rows),
            "prompt_boundary_positions_masked": sum(len(r["prompt_positions_wrongly_masked"]) for r in rows),
            "assistant_boundary_positions_masked": sum(len(r["assistant_positions_wrongly_masked"]) for r in rows),
            "masked_tokens_per_sequence_distribution": dict(sorted(Counter(len(r["real_positions_wrongly_masked"]) for r in rows).items())),
            "sequence_length_min": min(r["sequence_tokens"] for r in rows), "sequence_length_max": max(r["sequence_tokens"] for r in rows)})
    require(doc["per_member"] == native_members, "Native per-member audit summary does not reconstruct")
    report = {"schema": "village-001f-independent-native-mask-analysis-v1", "source_sha256": sha(path),
        "worker_config_sha256": sha(config_path), "driver_sha256": expected_identity["driver_sha256"],
        "complete_sequences": 720, "per_member": per_member,
        "totals": {key: sum(row[key] for row in per_member) for key in ("sequences", "sequences_with_real_tokens_masked",
            "real_tokens_masked", "native_sequence_tokens", "prompt_boundary_positions_masked",
            "assistant_boundary_positions_masked", "prompt_prefix_mismatch_sequences")},
        "mechanistic_scope": "Actual im_end context positions receive zero historical attention while token IDs and assistant labels are unchanged. This does not by itself show an effect on trained weights or fresh task quality.",
        "tokenization_reexecuted": False, "new_model_or_provider_calls": 0,
        "sequences": sorted(details, key=lambda r: (r["member_id"], r["pair_id"], r["choice"]))}
    return report, found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--published", action="store_true")
    args = parser.parse_args()
    require(not args.out.exists() and args.repo.resolve() not in args.out.resolve().parents, "Use a new scratch output path")
    report, _ = audit(args.repo, args.audit)
    report["publication_status"] = "published_native_input" if args.published else "preliminary_pending_byte_identical_input_publication"
    report["analysis_script_sha256"] = sha(Path(__file__))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.out), "source_sha256": report["source_sha256"],
        "publication_status": report["publication_status"], "totals": report["totals"], "per_member": report["per_member"]}, indent=2))


if __name__ == "__main__":
    main()
