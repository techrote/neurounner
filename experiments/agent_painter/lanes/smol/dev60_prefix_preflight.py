#!/usr/bin/env python3
"""SMOL exact-tokenizer 60-row synthetic prefix transport proof, NOT DEV60.

No actual DEVELOPMENT rows, teacher targets, FINAL files, model weights,
model inference, optimization or native Art are read/executed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

MODEL_REV = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
SHARED_SOURCE = "11ba1f1915362d3d127edfbe37fe9e30c57eb8b9"
FAMILIES = ("greebles", "pipework", "architecture")
MODEL_HASHES = {
    "config.json": "8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a",
    "tokenizer.json": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
    "tokenizer_config.json": "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df",
}
CONTEXT = 2048
LIMIT = 512  # Provisional transport-only bound; not a DEV60 ceiling.


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def fixture():
    """No actual designs or reference teachers; all labels are artificial."""
    requests = []
    for family in FAMILIES:
        for slot in range(1, 5):
            group = f"SYNTH-{family.upper()}-{slot:02d}"
            for form in range(5):
                requests.append({
                    "id": f"{group}-F{form}", "group_id": group,
                    "family": family, "formulation_id": f"F{form}",
                    "prompt": f"ENCODING-ONLY, NOT AN ART REQUEST: {family}; "
                              f"opaque group {slot:02d}; formulation {form}; "
                              f"unicode check ñ é; literal JSON characters {{}} [].",
                })
    return {"schema": "cybersand.smol-synthetic-prefix-fixture/1",
            "status": "SYNTHETIC-NOT-A-BENCHMARK", "requests": requests}


def validate_fixture(doc):
    require(type(doc) is dict and set(doc) == {"schema", "status", "requests"}
            and doc["schema"] == "cybersand.smol-synthetic-prefix-fixture/1"
            and doc["status"] == "SYNTHETIC-NOT-A-BENCHMARK", "not a synthetic fixture")
    rows = doc["requests"]
    require(type(rows) is list and len(rows) == 60, "expected 60 transport-only prompts")
    groups, identifiers, prompts = {}, set(), set()
    for row in rows:
        require(type(row) is dict and set(row) ==
                {"id", "group_id", "family", "formulation_id", "prompt"},
                "unexpected request or teacher fields")
        family, group, form, identifier, prompt = (row[x] for x in
            ("family", "group_id", "formulation_id", "id", "prompt"))
        require(family in FAMILIES and type(form) is str and form in
                {f"F{i}" for i in range(5)}, "unrecognized family/formulation")
        require(type(group) is str and re.fullmatch(
                r"SYNTH-(GREEBLES|PIPEWORK|ARCHITECTURE)-0[1-4]", group)
                and group.split("-")[1].lower() == family,
                "not a synthetic group")
        require(identifier == f"{group}-{form}" and identifier not in identifiers,
                "duplicate or noncanonical request ID")
        require(type(prompt) is str and prompt.startswith(
                "ENCODING-ONLY, NOT AN ART REQUEST:") and "\x00" not in prompt
                and 1 <= len(prompt.encode("utf-8")) <= 512,
                "not an artificial transport-only prompt")
        require(prompt not in prompts, "duplicate synthetic prompt")
        identifiers.add(identifier)
        prompts.add(prompt)
        groups.setdefault((family, group), set()).add(form)
    for family in FAMILIES:
        require(len([1 for fam, _ in groups if fam == family]) == 4,
                "incorrect structural slots per family")
    require(len(groups) == 12 and all(forms == {f"F{i}" for i in range(5)}
                                      for forms in groups.values()),
            "wrong synthetic 12x5 denominator")
    return rows


def read_system(shared):
    shared = Path(shared)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                  cwd=shared, text=True).strip()
    require(sha == SHARED_SOURCE, "shared frozen trainer identity changed")
    src = shared / "experiments/agent_painter/training/train.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    values = [node.value.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(name, ast.Name) and name.id == "SYSTEM"
                      for name in node.targets)
              and isinstance(node.value, ast.Constant)
              and isinstance(node.value.value, str)]
    require(len(values) == 1, "shared training system changed")
    return values[0]


def inspect(doc, tokenizer, system, *, context=CONTEXT, limit=LIMIT):
    rows = validate_fixture(doc)
    require(type(context) is int and type(limit) is int
            and 0 < limit < context, "invalid context or output limit")
    require(tokenizer.eos_token_id == 2 and isinstance(tokenizer.chat_template, str)
            and tokenizer.chat_template, "upstream authentic chat/EOS missing")
    records = []
    for row in rows:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": row["prompt"]}]
        prefix = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
        require(type(prefix) is str and prefix, "empty chat prefix")
        tokens = tokenizer.encode(prefix, add_special_tokens=False)
        require(type(tokens) is list and tokens
                and all(type(i) is int and i >= 0 for i in tokens),
                "bad token sequence")
        require(len(tokens) + limit <= context, "static context overflow")
        records.append({
            "id": row["id"], "family": row["family"],
            "group_id": row["group_id"], "formulation_id": row["formulation_id"],
            "prompt_sha256": digest(row["prompt"].encode("utf-8")),
            "prefix_sha256": digest(canonical(tokens)),
            "prefix_tokens": len(tokens),
            "headroom_after_provisional_limit": context - limit - len(tokens),
        })
    require(len(records) == 60, "missing encoded prefixes")
    return {"schema": "cybersand.smol-synthetic-prefix-preflight/1",
            "status": "SYNTHETIC-ONLY-NOT-DEV60",
            "model_revision": MODEL_REV, "shared_source_sha": SHARED_SOURCE,
            "upstream_eos_token_id": 2,
            "chat_template_sha256": digest(tokenizer.chat_template.encode("utf-8")),
            "system_sha256": digest(system.encode("utf-8")),
            "fixture_sha256": digest(canonical(doc)),
            "context_tokens": context, "provisional_max_new_tokens": limit,
            "independent_real_structures": 0, "real_dev60_requests": 0,
            "synthetic_requests": 60, "synthetic_groups": 12,
            "minimum_provisional_headroom": min(x["headroom_after_provisional_limit"] for x in records),
            "maximum_prefix_tokens": max(x["prefix_tokens"] for x in records),
            "reference_targets_checked": False, "native_execution": False,
            "training": False, "model_inference": False, "final_opened": False,
            "records": records}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--tokenizer", type=Path)
    ap.add_argument("--shared", type=Path)
    ap.add_argument("--source-sha")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.self_test:
        validate_fixture(fixture())
        print("Synthetic 12-group / 60-prompt fixture PASS; not DEV60")
        return
    env = os.environ
    require(args.tokenizer and args.shared and args.output
            and re.fullmatch(r"[0-9a-f]{40}", args.source_sha or "")
            and env.get("GITHUB_ACTIONS") == "true"
            and env.get("GITHUB_REPOSITORY") == "techrote/cybersand"
            and env.get("GITHUB_REF") == "refs/heads/codex/agent-painter-smol-568"
            and env.get("RUNNER_ENVIRONMENT") == "github-hosted"
            and env.get("GITHUB_SHA") == args.source_sha,
            "source-matched hosted tokenizer-only probe required")
    require(not args.output.exists(), "refuse overwrite original receipt")
    from transformers import AutoTokenizer
    for name, expected in MODEL_HASHES.items():
        path = args.tokenizer / name
        require(path.is_file() and digest(path.read_bytes()) == expected,
                f"unexpected upstream {name}")
    from json import loads
    profile_path = args.shared / "experiments/agent_painter/training/model_profiles.json"
    profile = loads(profile_path.read_text())["profiles"]["smol"]
    require(profile["revision"] == MODEL_REV
            and profile["max_sequence_tokens"] == CONTEXT
            and profile["required_hashes"]["tokenizer.json"] == MODEL_HASHES["tokenizer.json"],
            "model profile is not the qualified SMOL pin")
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer),
                                               local_files_only=True,
                                               trust_remote_code=False)
    result = inspect(fixture(), tokenizer, read_system(args.shared))
    result["lane_source_sha"] = args.source_sha
    result["pinned_tokenizer_file_sha256"] = MODEL_HASHES
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json.dumps(result, ensure_ascii=False, sort_keys=True,
                                       indent=2).encode("utf-8") + b"\n")
    print(json.dumps({key: result[key] for key in (
        "status", "synthetic_requests", "real_dev60_requests",
        "maximum_prefix_tokens", "minimum_provisional_headroom")}, sort_keys=True))


if __name__ == "__main__":
    main()
