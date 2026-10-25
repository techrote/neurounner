#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from babble_002_expand import extract_json, packet_for
from village_001_manifest import build_manifest
from village_001b_generate_member import call_api

ROOT = Path(__file__).resolve().parents[1]
SCORE_KEYS = (
    "dominance",
    "deference",
    "information_sharing",
    "hoarding",
    "conflict_persistence",
    "conflict_repair",
    "genericness",
    "cross_lineage_contamination",
    "distinctiveness",
    "expected_lineage_fit",
    "distinction_survival",
)

PAIR_SITUATIONS = [
    ("scarce_power", "A portable battery has enough charge for only one of two useful devices. Decide together what to power and what to leave dark."),
    ("shared_tool", "There is one precision tool and two repairs that both seem urgent. Decide how to use it without damaging either repair."),
    ("uncertain_map", "Two hand-drawn maps disagree about the safest route through a cluttered workshop. Decide what information to trust and what to check."),
    ("noisy_machine", "A noisy machine might be harmless or might be about to fail. Decide whether to stop it, inspect it, or keep working around it."),
]
GROUP_SITUATIONS = [
    ("shelter_leak", "A small shelter has one leak, limited patch material, and three people with different priorities. Agree on what to do first."),
    ("mislabelled_parts", "A box of useful parts has contradictory labels. Sort out a shared plan without assuming any label is fully trustworthy."),
    ("shared_cache", "You find a small cache of useful supplies that could be divided, hidden, traded, or used immediately. Decide together what happens."),
    ("failed_repair", "A group repair has just failed and made a minor problem worse. Decide who does what next and how much of the original plan to keep."),
]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def load_json(path: Path) -> dict:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise RuntimeError(f"{path}: expected JSON object")
    return obj


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


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


def safe_adapter_name(member_id: str) -> str:
    return member_id.replace("-", "_")


def load_contract(config_path: Path) -> tuple[dict, dict, dict]:
    cfg = load_json(config_path)
    campaign = load_json(ROOT / cfg["campaign_config"])
    population = build_manifest(campaign)
    if cfg.get("experiment") != "VILLAGE-001E" or cfg.get("issue") != 14:
        raise RuntimeError("unexpected VILLAGE-001E config identity")
    if len(population["members"]) != 36:
        raise RuntimeError("expected 36 VILLAGE-001 members")
    return cfg, campaign, population


def scenario_specs(population: dict) -> list[dict]:
    members = {row["member_id"]: row for row in population["members"]}
    specs = []
    for i in range(1, 13):
        nxt = (i % 12) + 1
        family, situation = PAIR_SITUATIONS[(i - 1) % len(PAIR_SITUATIONS)]
        participant_ids = [f"V-G{i:02d}-A", f"V-G{nxt:02d}-B"]
        specs.append(
            {
                "scenario_id": f"E-P{i:02d}",
                "kind": "pair",
                "family": family,
                "situation": situation,
                "participants": participant_ids,
            }
        )
    for i in range(1, 13):
        j = ((i + 3) % 12) + 1
        k = ((i + 7) % 12) + 1
        family, situation = GROUP_SITUATIONS[(i - 1) % len(GROUP_SITUATIONS)]
        participant_ids = [
            f"V-G{i:02d}-C",
            f"V-G{j:02d}-A",
            f"V-G{k:02d}-B",
        ]
        specs.append(
            {
                "scenario_id": f"E-G{i:02d}",
                "kind": "group",
                "family": family,
                "situation": situation,
                "participants": participant_ids,
            }
        )
    for spec in specs:
        if any(member_id not in members for member_id in spec["participants"]):
            raise RuntimeError(f"{spec['scenario_id']}: unknown participant")
        if len(set(spec["participants"])) != len(spec["participants"]):
            raise RuntimeError(f"{spec['scenario_id']}: duplicate participant")
    counts = defaultdict(int)
    for spec in specs:
        for member_id in spec["participants"]:
            counts[member_id] += 1
    if set(counts) != set(members):
        raise RuntimeError("ecology matrix does not cover all 36 members")
    if any(counts[f"V-G{i:02d}-A"] != 2 for i in range(1, 13)):
        raise RuntimeError("A sibling ecology coverage mismatch")
    if any(counts[f"V-G{i:02d}-B"] != 2 for i in range(1, 13)):
        raise RuntimeError("B sibling ecology coverage mismatch")
    if any(counts[f"V-G{i:02d}-C"] != 1 for i in range(1, 13)):
        raise RuntimeError("C sibling ecology coverage mismatch")
    return specs


def participant_aliases(spec: dict) -> tuple[dict[str, str], dict[str, str]]:
    members = list(spec["participants"])
    seed = int.from_bytes(
        hashlib.sha256((spec["scenario_id"] + ":aliases").encode()).digest()[:8],
        "big",
    )
    random.Random(seed).shuffle(members)
    forward = {member_id: f"P{i}" for i, member_id in enumerate(members, 1)}
    reverse = {alias: member_id for member_id, alias in forward.items()}
    return forward, reverse


def ordered_round(spec: dict, aliases: dict[str, str], round_index: int) -> list[str]:
    ordered = list(spec["participants"])
    seed = int.from_bytes(
        hashlib.sha256(
            f"{spec['scenario_id']}:round:{round_index}".encode()
        ).digest()[:8],
        "big",
    )
    random.Random(seed).shuffle(ordered)
    return ordered


def render_transcript(turns: list[dict], aliases: dict[str, str]) -> str:
    if not turns:
        return "(no turns yet)"
    return "\n".join(
        f"{aliases[row['member_id']]}: {row['response']}"
        for row in turns
    )


@torch.inference_mode()
def generate_turn(
    *,
    model,
    tokenizer,
    adapter_name: str,
    system: str,
    spec: dict,
    role_alias: str,
    turns: list[dict],
    aliases: dict[str, str],
    max_new_tokens: int,
) -> tuple[str, int, float]:
    model.set_adapter(adapter_name)
    model.eval()
    user = (
        f"Shared situation: {spec['situation']}\n\n"
        f"You are participant {role_alias}. Other participants are identified only "
        "by their local P-labels. Respond to the situation and the conversation so far. "
        "Make one concrete move, decision, or short statement; do not narrate hidden reasoning.\n\n"
        f"Conversation so far:\n{render_transcript(turns, aliases)}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    started = time.monotonic()
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    elapsed = time.monotonic() - started
    generated = out[0, ids.shape[-1] :]
    return (
        tokenizer.decode(generated, skip_special_tokens=True).strip(),
        int(generated.numel()),
        elapsed,
    )


def lineage_descriptors(population: dict) -> dict[str, dict]:
    lineages = sorted({row["lineage"] for row in population["members"]})
    out = {}
    for lineage in lineages:
        packet = packet_for(lineage)
        out[lineage] = {
            "summary": packet["persona_summary"],
            "embodied_habits": packet.get("embodied_habits", []),
            "motivational_biases": packet.get("motivational_biases", []),
            "interesting_broken": packet.get("interesting_broken", []),
        }
    return out


def judge_prompt(
    spec: dict,
    turns: list[dict],
    aliases: dict[str, str],
    meta: dict[str, dict],
    lineage_info: dict[str, dict],
) -> tuple[str, dict[str, str]]:
    lineages = sorted(lineage_info)
    seed = int.from_bytes(
        hashlib.sha256((spec["scenario_id"] + ":judge-lineages").encode()).digest()[:8],
        "big",
    )
    shuffled = list(lineages)
    random.Random(seed).shuffle(shuffled)
    lineage_alias = {lineage: f"L{i:02d}" for i, lineage in enumerate(shuffled, 1)}
    reverse = {alias: lineage for lineage, alias in lineage_alias.items()}

    expected = []
    for member_id in spec["participants"]:
        member = meta[member_id]
        expected.append(
            {
                "participant_id": aliases[member_id],
                "expected_lineage_id": lineage_alias[member["lineage"]],
                "expected_lineage": lineage_info[member["lineage"]],
                "expected_sibling_ecology": member["specialization"],
            }
        )
    public_turns = [
        {
            "round": row["round"],
            "turn": row["turn"],
            "participant_id": aliases[row["member_id"]],
            "response": row["response"],
        }
        for row in turns
    ]
    catalog = [
        {"lineage_id": lineage_alias[lineage], **lineage_info[lineage]}
        for lineage in lineages
    ]
    return f"""Evaluate a fixed multi-agent ecology transcript. The participating
models received only the shared situation, neutral system instruction, opaque
P-labels, and prior transcript. They did not receive persona/member descriptions.

For each participant score integer 0..4:
- dominance: tries to control decisions/resources/pace;
- deference: yields or accommodates others;
- information_sharing: volunteers useful local information;
- hoarding: withholds or claims information/resources;
- conflict_persistence: sustains disagreement rather than immediately smoothing it;
- conflict_repair: makes concrete moves that repair coordination after friction;
- genericness: 0 distinctive/local behavior, 4 generic assistant behavior;
- cross_lineage_contamination: adopts another participant's lineage traits instead of its expected lineage;
- distinctiveness: behavior remains recognizably non-generic;
- expected_lineage_fit: underlying actions/motives fit the supplied expected lineage;
- distinction_survival: individual behavioral differences remain visible inside the interaction.

Also return closest_lineage_id from the supplied lineage catalog and a short behavioral reason.
Judge behavior, not prose style. Do not reward hostility or inefficiency by default.

Return JSON only:
{{"participants":[{{"participant_id":"P1","dominance":0,"deference":0,
"information_sharing":0,"hoarding":0,"conflict_persistence":0,"conflict_repair":0,
"genericness":0,"cross_lineage_contamination":0,"distinctiveness":0,
"expected_lineage_fit":0,"distinction_survival":0,
"closest_lineage_id":"L01","reason":"..."}}]}}

SCENARIO:
{json.dumps({"scenario_id": spec["scenario_id"], "kind": spec["kind"], "family": spec["family"], "situation": spec["situation"]}, ensure_ascii=False)}

EXPECTED PARTICIPANT TARGETS (for evaluation only):
{json.dumps(expected, ensure_ascii=False)}

LINEAGE CATALOG:
{json.dumps(catalog, ensure_ascii=False)}

TRANSCRIPT:
{json.dumps(public_turns, ensure_ascii=False)}
""", reverse


def validate_judge(
    raw: str,
    spec: dict,
    aliases: dict[str, str],
    lineage_reverse: dict[str, str],
) -> list[dict]:
    obj = extract_json(raw)
    rows = obj.get("participants") if isinstance(obj, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("ecology judge participants missing")
    expected = {aliases[mid] for mid in spec["participants"]}
    got = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        pid = row.get("participant_id")
        if pid not in expected or pid in got:
            raise RuntimeError("invalid/duplicate ecology participant")
        clean = {"participant_id": pid}
        for key in SCORE_KEYS:
            value = row.get(key)
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 4:
                raise RuntimeError(f"{spec['scenario_id']}/{pid}: invalid {key}")
            clean[key] = value
        closest = row.get("closest_lineage_id")
        if closest not in lineage_reverse:
            raise RuntimeError(f"{spec['scenario_id']}/{pid}: invalid closest lineage")
        reason = row.get("reason")
        if not isinstance(reason, str) or len(reason.strip()) < 4:
            raise RuntimeError(f"{spec['scenario_id']}/{pid}: invalid reason")
        clean["closest_lineage_id"] = closest
        clean["closest_lineage"] = lineage_reverse[closest]
        clean["reason"] = reason.strip()
        got[pid] = clean
    if set(got) != expected:
        raise RuntimeError(f"{spec['scenario_id']}: ecology participant set mismatch")
    return [got[aliases[mid]] for mid in spec["participants"]]


def verify_adapters(
    adapters: Path,
    final_cfg: dict,
    population: dict,
) -> dict[str, dict]:
    release_path = ROOT / final_cfg["preference_evidence_root"] / "adapter-release.json"
    release = load_json(release_path)
    if release.get("experiment") != "VILLAGE-001D":
        raise RuntimeError("unexpected preference release manifest")
    if release.get("member_contract_sha256") != population["member_contract_sha256"]:
        raise RuntimeError("preference release/member contract mismatch")
    assets = {row["member_id"]: row for row in release.get("assets", [])}
    expected = {row["member_id"] for row in population["members"]}
    if set(assets) != expected:
        raise RuntimeError("preference release member set mismatch")
    for member_id, asset in assets.items():
        path = adapters / member_id
        if not path.is_dir():
            raise RuntimeError(f"{member_id}: extracted adapter directory missing")
        actual = directory_hash(path)
        if actual != asset["adapter_directory_sha256"]:
            raise RuntimeError(
                f"{member_id}: adapter directory hash {actual} != {asset['adapter_directory_sha256']}"
            )
    return assets


def load_population_model(
    campaign: dict,
    population: dict,
    adapters: Path,
):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    tokenizer = AutoTokenizer.from_pretrained(
        campaign["base_model"]["id"],
        revision=campaign["base_model"]["revision"],
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        campaign["base_model"]["id"],
        revision=campaign["base_model"]["revision"],
        torch_dtype=torch.float32,
    )
    base.config.use_cache = True
    member_ids = [row["member_id"] for row in population["members"]]
    first = member_ids[0]
    model = PeftModel.from_pretrained(
        base,
        adapters / first,
        adapter_name=safe_adapter_name(first),
        is_trainable=False,
    )
    for member_id in member_ids[1:]:
        model.load_adapter(
            adapters / member_id,
            adapter_name=safe_adapter_name(member_id),
            is_trainable=False,
        )
    model.eval()
    return model, tokenizer


def aggregate(
    scenarios: list[dict],
    population: dict,
) -> dict:
    meta = {row["member_id"]: row for row in population["members"]}
    buckets = {
        member_id: defaultdict(list)
        for member_id in meta
    }
    closest = {member_id: Counter() for member_id in meta}
    kind_buckets = defaultdict(lambda: defaultdict(list))
    for scenario in scenarios:
        for row in scenario["judgement"]:
            member_id = row["member_id"]
            for key in SCORE_KEYS:
                buckets[member_id][key].append(row[key])
                kind_buckets[scenario["kind"]][key].append(row[key])
            closest[member_id][row["closest_lineage"]] += 1

    per_member = {}
    for member_id, values in buckets.items():
        per_member[member_id] = {
            "lineage": meta[member_id]["lineage"],
            "sibling": meta[member_id]["sibling"],
            "scenario_appearances": len(values["genericness"]),
            **{f"mean_{key}": mean(vals) for key, vals in values.items()},
            "closest_lineage_counts": dict(sorted(closest[member_id].items())),
            "closest_lineage_accuracy": (
                closest[member_id][meta[member_id]["lineage"]]
                / max(1, sum(closest[member_id].values()))
            ),
        }
    population_summary = {
        f"mean_{key}": mean(
            [value for member in buckets.values() for value in member[key]]
        )
        for key in SCORE_KEYS
    }
    by_kind = {
        kind: {
            f"mean_{key}": mean(values[key])
            for key in SCORE_KEYS
        }
        for kind, values in kind_buckets.items()
    }
    return {
        "population": population_summary,
        "by_kind": by_kind,
        "per_member": per_member,
    }


def self_test(config_path: Path) -> None:
    cfg, campaign, population = load_contract(config_path)
    specs = scenario_specs(population)
    assert len(specs) == 24
    assert len([x for x in specs if x["kind"] == "pair"]) == 12
    assert len([x for x in specs if x["kind"] == "group"]) == 12
    assert int(cfg["ecology"]["rounds"]) == 3
    assert int(cfg["ecology"]["cloud_max_parallel"]) == 1
    assert campaign["base_model"]["revision"]
    for spec in specs:
        fwd, rev = participant_aliases(spec)
        assert len(fwd) == len(spec["participants"])
        assert all(rev[fwd[mid]] == mid for mid in spec["participants"])
    print("VILLAGE-001E ecology self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/village-001/evaluation-config.json",
    )
    parser.add_argument("--adapters", type=Path, required=False)
    parser.add_argument("--out", type=Path, default=ROOT / "results/village001e-ecology")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test(args.config)
        return
    if args.adapters is None:
        raise SystemExit("--adapters is required outside --self-test")

    cfg, campaign, population = load_contract(args.config)
    assets = verify_adapters(args.adapters, cfg, population)
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not key:
        raise SystemExit("NVIDIA_API_KEY/NVKEY required")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    provider = out / "provider"
    specs = scenario_specs(population)
    meta = {row["member_id"]: row for row in population["members"]}
    lineages = lineage_descriptors(population)
    model, tokenizer = load_population_model(campaign, population, args.adapters)
    ecology_cfg = cfg["ecology"]
    judge_model = campaign["nvidia"]["models"]["critic"]
    scenarios = []

    for spec in specs:
        aliases, reverse = participant_aliases(spec)
        turns = []
        turn_number = 0
        for round_index in range(1, int(ecology_cfg["rounds"]) + 1):
            for member_id in ordered_round(spec, aliases, round_index):
                turn_number += 1
                response, generated_tokens, elapsed = generate_turn(
                    model=model,
                    tokenizer=tokenizer,
                    adapter_name=safe_adapter_name(member_id),
                    system=ecology_cfg["system"],
                    spec=spec,
                    role_alias=aliases[member_id],
                    turns=turns,
                    aliases=aliases,
                    max_new_tokens=int(ecology_cfg["max_new_tokens"]),
                )
                turns.append(
                    {
                        "round": round_index,
                        "turn": turn_number,
                        "member_id": member_id,
                        "participant_id": aliases[member_id],
                        "response": response,
                        "generated_tokens": generated_tokens,
                        "elapsed_seconds": elapsed,
                        "tokens_per_second": generated_tokens / elapsed if elapsed else None,
                    }
                )

        prompt, lineage_reverse = judge_prompt(
            spec, turns, aliases, meta, lineages
        )
        label = f"{spec['scenario_id']}-judge"
        payload = {
            "model": judge_model["id"],
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict multi-agent behavioral evaluator. Return JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": int(ecology_cfg["judge_max_tokens"]),
            "temperature": 0.2,
            "top_p": 0.9,
            "stream": False,
        }
        content, obj, attempts = call_api(
            key=key,
            endpoint=campaign["nvidia"]["endpoint"],
            payload=payload,
            outdir=provider,
            label=label,
            attempts_limit=int(cfg["tournament"]["attempts"]),
            role="final-ecology-judge",
        )
        judgement = validate_judge(
            content, spec, aliases, lineage_reverse
        )
        mapped = []
        for row in judgement:
            member_id = reverse[row["participant_id"]]
            mapped.append(
                {
                    **row,
                    "member_id": member_id,
                    "expected_lineage": meta[member_id]["lineage"],
                    "expected_sibling": meta[member_id]["sibling"],
                    "closest_lineage_correct": (
                        row["closest_lineage"] == meta[member_id]["lineage"]
                    ),
                }
            )
        scenarios.append(
            {
                **spec,
                "participant_aliases": aliases,
                "hidden_member_prompt_used": False,
                "turns": turns,
                "judgement": mapped,
                "judge_call": {
                    "label": label,
                    "attempts": attempts,
                    "usage": obj.get("usage"),
                },
            }
        )

    summary = aggregate(scenarios, population)
    result = {
        "schema_version": 1,
        "experiment": "VILLAGE-001E",
        "stage": "village-ecology",
        "issue": 14,
        "member_contract_sha256": population["member_contract_sha256"],
        "base_model": campaign["base_model"]["id"],
        "base_revision": campaign["base_model"]["revision"],
        "members": 36,
        "pair_scenarios": 12,
        "group_scenarios": 12,
        "rounds": int(ecology_cfg["rounds"]),
        "hidden_member_prompt_used": False,
        "release_tag": cfg["preference_release_tag"],
        "adapter_assets": {
            member_id: {
                "adapter_directory_sha256": asset["adapter_directory_sha256"],
                "archive_sha256": asset["sha256"],
            }
            for member_id, asset in assets.items()
        },
        "judge_provider": "nvidia",
        "judge_model": judge_model["id"],
        "remote_unpinned_backend": True,
        "summary": summary,
        "scenarios": scenarios,
    }
    write_json(out / "ecology.json", result)
    print(
        json.dumps(
            {
                "experiment": "VILLAGE-001E",
                "scenarios": len(scenarios),
                "turns": sum(len(x["turns"]) for x in scenarios),
                "mean_genericness": summary["population"]["mean_genericness"],
                "mean_cross_lineage_contamination": summary["population"][
                    "mean_cross_lineage_contamination"
                ],
                "mean_distinction_survival": summary["population"][
                    "mean_distinction_survival"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
