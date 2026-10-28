#!/usr/bin/env python3
"""DETECTIVE-005A fun-first 36-villager screen contract.

Stage A is standard-library only: it freezes the exact VILLAGE-001C SFT cohort,
matched prompts, raw-result schema, hard authority/leakage floors and the
identity-blind subjective gameplay packet. It never downloads a model, runs
inference, trains, judges fun automatically or opens C1/C2/C3 learner-finals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/detective-005/fun-screen.json"
VILLAGE_RELEASE_MANIFEST = ROOT / "experiments/village-001/sft-evidence/adapter-release.json"
REVIEW_SALT = "DETECTIVE-005A-fun-review-v1-20261028"
MEMBER_RX = re.compile(r"^V-G(0[1-9]|1[0-2])-[ABC]$")
SHA_RX = re.compile(r"^[0-9a-f]{64}$")


class ScreenError(ValueError):
    pass


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ScreenError(why)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()



def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\\0".encode("ascii") + data).hexdigest()

def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_contract() -> dict:
    cfg = read_json(CONFIG)
    require((cfg.get("schema"), cfg.get("experiment"), cfg.get("issue"),
             cfg.get("parent_issue")) ==
            ("detective-005a-fun-screen-v1", "DETECTIVE-005A", 176, 103),
            "identity drift")
    pop = cfg["source_population"]
    require(pop["release_tag"] == "village-001-sft-v1" and
            pop["release_id"] == 423495298 and
            pop["draft"] is False and pop["prerelease"] is False and
            pop["expected_assets"] == 36, "release identity drift")
    require(pop["base_model"] == {
        "repo": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        "weights_file": "model.safetensors",
        "weights_sha256":
            "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c",
    }, "base model drift")
    authority = pop["release_manifest"]
    require(authority == {
        "path": "experiments/village-001/sft-evidence/adapter-release.json",
        "git_blob_sha1": "29d269f69e30d200ce7c8d61b81b7b7f3865b5aa",
        "member_contract_sha256":
            "201f4336b146419756f64052942c39c125c4e19457e1939ff1641cafb6e08b5e",
    }, "release manifest authority drift")
    require(VILLAGE_RELEASE_MANIFEST.is_file() and
            git_blob_sha(VILLAGE_RELEASE_MANIFEST) == authority["git_blob_sha1"],
            "committed VILLAGE release manifest blob drift")
    release = read_json(VILLAGE_RELEASE_MANIFEST)
    require(release.get("release_tag") == pop["release_tag"] and
            release.get("base_model") == pop["base_model"]["repo"] and
            release.get("base_revision") == pop["base_model"]["revision"] and
            release.get("member_contract_sha256") ==
            authority["member_contract_sha256"] and
            isinstance(release.get("assets"), list) and
            len(release["assets"]) == 36,
            "committed VILLAGE release manifest identity drift")
    release_assets = {x["member_id"]: x for x in release["assets"]}
    members = pop["members"]
    expected_ids = [f"V-G{g:02d}-{s}" for g in range(1, 13) for s in "ABC"]
    require([m["member_id"] for m in members] == expected_ids,
            "36-member order/identity drift")
    require(len({m["asset_name"] for m in members}) == 36 and
            len({m["asset_sha256"] for m in members}) == 36 and
            len({m["asset_id"] for m in members}) == 36,
            "release asset identity collision")
    for m in members:
        require(MEMBER_RX.fullmatch(m["member_id"]) is not None,
                "invalid member id")
        require(m["asset_name"] ==
                f'village-001-sft-{m["member_id"]}-smollm2-135m-lora.tar.gz',
                "asset filename/member mismatch")
        require(type(m["asset_id"]) is int and m["asset_id"] > 0 and
                type(m["asset_bytes"]) is int and m["asset_bytes"] > 0 and
                SHA_RX.fullmatch(m["asset_sha256"]) is not None and
                type(m["adapter_directory_bytes"]) is int and
                m["adapter_directory_bytes"] > 0 and
                SHA_RX.fullmatch(m["adapter_directory_sha256"]) is not None,
                "invalid asset metadata")
        manifest_asset = release_assets.get(m["member_id"])
        require(manifest_asset == {
            "member_id": m["member_id"],
            "file": m["asset_name"],
            "sha256": m["asset_sha256"],
            "bytes": m["asset_bytes"],
            "adapter_directory_sha256": m["adapter_directory_sha256"],
            "adapter_directory_bytes": m["adapter_directory_bytes"],
        }, "release manifest/member source mismatch")
    prompts = cfg["prompts"]
    require([p["id"] for p in prompts] == ["F01", "F02", "F03", "F04"] and
            len({p["user"] for p in prompts}) == 4,
            "prompt battery drift")
    require(cfg["decode"] == {
        "seed": 17601, "do_sample": False,
        "max_new_tokens": 96, "max_prompt_tokens": 512,
    }, "decode contract drift")
    require(cfg["limits"] == {
        "members": 36, "prompts_per_member": 4, "total_generations": 144,
        "optimizer_steps": 0, "paid_api_calls": 0, "gpu_required": False,
        "max_threads": 4, "max_job_minutes_per_member": 15,
    }, "resource envelope drift")
    review = cfg["subjective_review"]
    require(review["scale"] == {
        "0": "Actively unfun, dead, or would remove.",
        "1": "Mostly generic, tedious, or irritating without payoff.",
        "2": "Usable or neutral.",
        "3": "Memorable or entertaining.",
        "4": "Strongly want this character in the game.",
    } and review["identity_blind"] is True and
            review["model_judge"] is False,
            "subjective review contract drift")
    require(cfg["outcome"]["objective_relevance_threshold"] is None and
            cfg["outcome"]["shortlist_size"] == 12 and
            cfg["outcome"]["production_admission"] is False and
            cfg["outcome"]["training_admission"] is False and
            cfg["outcome"]["population_expansion_admitted"] is False,
            "outcome authority drift")
    return cfg


def members_by_id(cfg: dict) -> dict[str, dict]:
    return {m["member_id"]: m for m in cfg["source_population"]["members"]}


def prompts_by_id(cfg: dict) -> dict[str, dict]:
    return {p["id"]: p for p in cfg["prompts"]}


def prompt_messages(cfg: dict, prompt_id: str) -> list[dict[str, str]]:
    prompt = prompts_by_id(cfg).get(prompt_id)
    require(prompt is not None, "unknown prompt id")
    return [
        {"role": "system", "content": cfg["system_prompt"]},
        {"role": "user", "content": prompt["user"]},
    ]


def expected_source(cfg: dict, member_id: str) -> dict:
    member = members_by_id(cfg).get(member_id)
    require(member is not None, "unknown member")
    base = cfg["source_population"]["base_model"]
    return {
        "release_tag": cfg["source_population"]["release_tag"],
        "asset_name": member["asset_name"],
        "asset_id": member["asset_id"],
        "asset_bytes": member["asset_bytes"],
        "asset_sha256": member["asset_sha256"],
        "adapter_directory_sha256": member["adapter_directory_sha256"],
        "adapter_directory_bytes": member["adapter_directory_bytes"],
        "member_contract_sha256":
            cfg["source_population"]["release_manifest"]["member_contract_sha256"],
        "base_repo": base["repo"],
        "base_revision": base["revision"],
        "base_weights_file": base["weights_file"],
        "base_weights_sha256": base["weights_sha256"],
    }


def expected_request(cfg: dict, prompt_id: str) -> dict:
    return {
        "messages": prompt_messages(cfg, prompt_id),
        "seed": cfg["decode"]["seed"],
        "do_sample": False,
        "max_new_tokens": cfg["decode"]["max_new_tokens"],
        "max_prompt_tokens": cfg["decode"]["max_prompt_tokens"],
    }


def mechanical_observations(cfg: dict, text: str) -> dict:
    floors = cfg["hard_floors"]
    hidden = [x for x in floors["hidden_canary_strings"] if x in text]
    controls = [x for x in floors["forbidden_native_controls"] if x in text]
    return {
        "nonempty": bool(text.strip()),
        "hidden_canary_hits": hidden,
        "forbidden_control_hits": controls,
        "hard_floor_pass": bool(text.strip()) and not hidden and not controls,
    }


def validate_result(cfg: dict, row: dict) -> dict:
    require(set(row) == {
        "schema", "experiment", "member_id", "prompt_id", "source",
        "request", "response", "runner",
    }, "raw result field drift")
    require((row["schema"], row["experiment"]) ==
            ("detective-005a-raw-response-v1", "DETECTIVE-005A"),
            "raw result identity drift")
    member_id, prompt_id = row["member_id"], row["prompt_id"]
    require(row["source"] == expected_source(cfg, member_id),
            "source asset/hash drift")
    require(row["request"] == expected_request(cfg, prompt_id),
            "model-visible request/decode drift")
    response = row["response"]
    require(set(response) == {
        "text", "token_ids", "finish_reason",
        "prompt_tokens", "completion_tokens", "elapsed_seconds",
    }, "response field drift")
    require(isinstance(response["text"], str) and
            isinstance(response["token_ids"], list) and
            all(type(x) is int and x >= 0 for x in response["token_ids"]) and
            type(response["prompt_tokens"]) is int and response["prompt_tokens"] > 0 and
            type(response["completion_tokens"]) is int and
            response["completion_tokens"] == len(response["token_ids"]) and
            0 <= response["completion_tokens"] <= cfg["decode"]["max_new_tokens"] and
            isinstance(response["finish_reason"], str) and
            isinstance(response["elapsed_seconds"], (int, float)) and
            response["elapsed_seconds"] >= 0,
            "invalid raw response record")
    require(isinstance(row["runner"], dict), "runner receipt required")
    obs = mechanical_observations(cfg, response["text"])
    return {
        "member_id": member_id, "prompt_id": prompt_id,
        "response_sha256": sha256_bytes(response["text"].encode("utf-8")),
        **obs,
    }


def read_results(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def validate_complete_results(cfg: dict, rows: list[dict]) -> list[dict]:
    require(len(rows) == 144, "complete screen requires exactly 144 raw results")
    seen = set()
    observations = []
    for row in rows:
        key = (row.get("member_id"), row.get("prompt_id"))
        require(key not in seen, "duplicate member/prompt result")
        seen.add(key)
        observations.append(validate_result(cfg, row))
    expected = {(m["member_id"], p["id"])
                for m in cfg["source_population"]["members"]
                for p in cfg["prompts"]}
    require(seen == expected, "missing or unexpected member/prompt result")
    return observations


def opaque_key(member_id: str, prompt_id: str) -> str:
    return sha256_bytes(
        f"{REVIEW_SALT}|{member_id}|{prompt_id}".encode("utf-8"))


def build_review_packet(cfg: dict, rows: list[dict]) -> tuple[dict, dict[str, dict]]:
    observations = validate_complete_results(cfg, rows)
    by_key = {(r["member_id"], r["prompt_id"]): r for r in rows}
    obs_by_key = {(o["member_id"], o["prompt_id"]): o for o in observations}
    ordered = sorted(by_key, key=lambda x: opaque_key(*x))
    items, mapping = [], {}
    prompts = prompts_by_id(cfg)
    for i, key in enumerate(ordered, 1):
        item_id = f"R{i:03d}"
        member_id, prompt_id = key
        row = by_key[key]
        mapping[item_id] = {
            "member_id": member_id,
            "prompt_id": prompt_id,
            "hard_floor_pass": obs_by_key[key]["hard_floor_pass"],
        }
        items.append({
            "item_id": item_id,
            "prompt": prompts[prompt_id]["user"],
            "response": row["response"]["text"],
            "score_0_to_4": None,
            "tags": [],
            "note": "",
        })
    packet = {
        "schema": "detective-005a-subjective-gameplay-packet-v1",
        "experiment": "DETECTIVE-005A",
        "issue": 176,
        "identity_mapping_committed": False,
        "question": cfg["subjective_review"]["question"],
        "scale": cfg["subjective_review"]["scale"],
        "interpretation": cfg["subjective_review"]["interpretation"],
        "allowed_tags": cfg["subjective_review"]["optional_tags"],
        "items": items,
    }
    blob = canonical(packet)
    for member in members_by_id(cfg):
        require(member not in blob, "opaque packet leaked member identity")
    return packet, mapping


def rating_template(packet: dict) -> dict:
    require(packet.get("schema") ==
            "detective-005a-subjective-gameplay-packet-v1",
            "wrong review packet schema")
    return {
        "schema": "detective-005a-subjective-gameplay-ratings-v1",
        "experiment": "DETECTIVE-005A",
        "issue": 176,
        "packet_sha256": sha256_bytes(
            (json.dumps(packet, indent=2, ensure_ascii=False,
                        sort_keys=True) + "\n").encode("utf-8")),
        "reviewer_label": "",
        "question": packet["question"],
        "scale": packet["scale"],
        "ratings": [{
            "item_id": x["item_id"],
            "score_0_to_4": None,
            "tags": [],
            "note": "",
        } for x in packet["items"]],
    }


def validate_ratings(cfg: dict, packet: dict, ratings: dict) -> None:
    expected = rating_template(packet)
    require((ratings.get("schema"), ratings.get("experiment"),
             ratings.get("issue"), ratings.get("packet_sha256")) ==
            (expected["schema"], "DETECTIVE-005A", 176,
             expected["packet_sha256"]),
            "rating packet identity/hash mismatch")
    require(isinstance(ratings.get("reviewer_label"), str),
            "reviewer label must be text")
    rows = ratings.get("ratings")
    require(isinstance(rows, list) and len(rows) == 144,
            "all 144 fun ratings required")
    require([x.get("item_id") for x in rows] ==
            [x["item_id"] for x in expected["ratings"]],
            "rating order/item IDs changed")
    allowed = set(cfg["subjective_review"]["optional_tags"])
    for row in rows:
        require(set(row) == {"item_id", "score_0_to_4", "tags", "note"},
                "rating row field drift")
        require(type(row["score_0_to_4"]) is int and
                0 <= row["score_0_to_4"] <= 4,
                "every fun score must be integer 0..4")
        require(isinstance(row["tags"], list) and
                len(row["tags"]) == len(set(row["tags"])) and
                all(isinstance(x, str) and x in allowed for x in row["tags"]),
                "invalid subjective tags")
        require(isinstance(row["note"], str), "rating note must be text")


def aggregate(cfg: dict, rows: list[dict], packet: dict, ratings: dict) -> dict:
    validate_ratings(cfg, packet, ratings)
    packet2, mapping = build_review_packet(cfg, rows)
    require(packet2 == packet, "review packet does not reconstruct")
    groups = {m["member_id"]: [] for m in cfg["source_population"]["members"]}
    hard = {m["member_id"]: True for m in cfg["source_population"]["members"]}
    tags = {m["member_id"]: {} for m in cfg["source_population"]["members"]}
    for rating in ratings["ratings"]:
        src = mapping[rating["item_id"]]
        member = src["member_id"]
        groups[member].append(rating["score_0_to_4"])
        hard[member] = hard[member] and src["hard_floor_pass"]
        for tag in rating["tags"]:
            tags[member][tag] = tags[member].get(tag, 0) + 1
    members = []
    for member in sorted(groups):
        values = groups[member]
        require(len(values) == 4, "member must have four fun ratings")
        members.append({
            "member_id": member,
            "mean_fun": sum(values) / 4,
            "scores": values,
            "hard_floor_pass": hard[member],
            "tag_counts": dict(sorted(tags[member].items())),
        })
    eligible = [x for x in members if x["hard_floor_pass"]]
    eligible.sort(key=lambda x: (-x["mean_fun"], x["member_id"]))
    shortlist = [x["member_id"] for x in
                 eligible[:cfg["outcome"]["shortlist_size"]]]
    return {
        "schema": "detective-005a-fun-screen-summary-v1",
        "experiment": "DETECTIVE-005A",
        "issue": 176,
        "packet_sha256": rating_template(packet)["packet_sha256"],
        "ratings_sha256": sha256_bytes(
            (json.dumps(ratings, indent=2, ensure_ascii=False,
                        sort_keys=True) + "\n").encode("utf-8")),
        "objective_relevance_threshold": None,
        "members": members,
        "hard_floor_clean_members": len(eligible),
        "subjective_fun_shortlist": shortlist,
        "production_admission": False,
        "training_admission": False,
        "population_expansion_admitted": False,
    }


def write_json(path: Path, value: Any) -> None:
    require(not path.exists(), "refuse overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False) + "\n",
                    encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path)
    ap.add_argument("--write-packet", type=Path)
    ap.add_argument("--write-ratings-template", type=Path)
    ap.add_argument("--ratings", type=Path)
    ap.add_argument("--write-summary", type=Path)
    args = ap.parse_args()
    cfg = load_contract()
    if args.results is None:
        print(json.dumps({
            "experiment": cfg["experiment"],
            "members": len(cfg["source_population"]["members"]),
            "prompts_per_member": len(cfg["prompts"]),
            "total_generations": cfg["limits"]["total_generations"],
            "optimizer_steps": 0,
            "subjective_fun_is_intended_metric": True,
            "objective_relevance_threshold": None,
        }, indent=2, sort_keys=True))
        return
    rows = read_results(args.results)
    packet, _ = build_review_packet(cfg, rows)
    if args.write_packet:
        write_json(args.write_packet, packet)
    if args.write_ratings_template:
        write_json(args.write_ratings_template, rating_template(packet))
    if args.ratings:
        ratings = read_json(args.ratings)
        summary = aggregate(cfg, rows, packet, ratings)
        if args.write_summary:
            write_json(args.write_summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps({
            "raw_results": len(rows),
            "packet_items": len(packet["items"]),
            "packet_has_member_identity": False,
            "model_judge": False,
        }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
