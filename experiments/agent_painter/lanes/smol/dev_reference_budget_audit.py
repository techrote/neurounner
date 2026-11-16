#!/usr/bin/env python3
"""SMOL original DEVELOPMENT reference-token budget audit; no model inference or TRAIN.

Read full source633 artifact only as opaque SHA-256. Selectively decompress
manifest, DEVELOPMENT JSONL, and original visible native receipts/readbacks/PNGs.
Never open FINAL targets/images, even for ZIP CRC.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import zipfile

NATIVE_SOURCE = "633920373a3d406a2ede61ce4137c324f76be23b"
ZIP_HASH = "50a9ad9b255b2a57180edf5558ca2b47dab9affd9ef9d2438f9011d648f81f39"
MANIFEST_HASH = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
SHARED_SOURCE = "11ba1f1915362d3d127edfbe37fe9e30c57eb8b9"
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
SEEDS = (104729, 130363)
FAMILIES = ("greebles", "pipework", "architecture")


def check(value, error):
    if not value:
        raise ValueError(error)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(raw):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            check(key not in obj, "duplicate JSON key")
            obj[key] = value
        return obj
    def reject(value):
        raise ValueError("nonfinite JSON")
    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject)


def member(archive, name):
    check(isinstance(name, str) and name.startswith("corpus/")
          and "\\" not in name and "\x00" not in name and not name.endswith("/"),
          "invalid native corpus path")
    parts = name.split("/")
    check(not any(x in ("", ".", "..") or "final" in x.lower() for x in parts)
          and not PurePosixPath(name).is_absolute(), "sealed/traversal path refused")
    choices = [x for x in archive.infolist() if x.filename == name]
    check(len(choices) == 1, "missing or duplicate ZIP entry")
    info = choices[0]
    check(not stat.S_ISLNK(info.external_attr >> 16)
          and info.file_size <= 20 * 1024 * 1024
          and info.compress_size <= 20 * 1024 * 1024,
          "symlink/oversize ZIP member")
    return info


def extract_visible(original_zip, dest):
    check(sha_file(original_zip) == ZIP_HASH, "source633 original ZIP hash mismatch")
    dest = Path(dest)
    check(not dest.exists(), "refuse overwrite")
    dest.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(original_zip) as archive:
        def read_visible(name):
            info = member(archive, name)
            with archive.open(info) as fh:
                data = fh.read(20 * 1024 * 1024 + 1)
            check(len(data) == info.file_size and len(data) <= 20 * 1024 * 1024,
                  "bad selected member size/CRC")
            path = dest / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return data
        manifest_bytes = read_visible("corpus/admitted-corpus.json")
        check(sha_bytes(manifest_bytes) == MANIFEST_HASH, "manifest mismatch")
        manifest = strict_json(manifest_bytes)
        check(manifest.get("source_sha") == NATIVE_SOURCE
              and manifest.get("state") == "native-admitted"
              and set(manifest["structural_groups"]) == {"TRAIN", "DEVELOPMENT", "FINAL"},
              "native source/admission mismatch")
        split = manifest["split_files"]["DEVELOPMENT"]
        data = read_visible("corpus/" + split["path"])
        check(sha_bytes(data) == split["sha256"], "DEVELOPMENT hash mismatch")
        rows = [strict_json(line) for line in data.splitlines() if line.strip()]
        check(len(rows) == split["count"] == 12
              and all(r.get("split") == "DEVELOPMENT" for r in rows),
              "DEVELOPMENT denominator/split mismatch")
        paths = sorted({r["native_admission"]["paths"][name]
                        for r in rows for name in ("png", "readback", "receipt")})
        for name in paths:
            read_visible("corpus/" + name)
    return rows, manifest, paths


def group_requests(rows):
    groups = {}
    for row in rows:
        key = row["group_id"]
        check(row["structural_group"] == key and row["family"] in FAMILIES
              and row["seed"] in SEEDS, "unexpected structural group/seed")
        if key in groups:
            prior = groups[key][0]
            check(row["prompt"] == prior["prompt"]
                  and row["program"] == prior["program"]
                  and row["family"] == prior["family"]
                  and row["seed"] != prior["seed"],
                  "seed-specific prompt/teacher drift")
        groups.setdefault(key, []).append(row)
    check(len(groups) == 6 and all(len(items) == 2
                                  and {r["seed"] for r in items} == set(SEEDS)
                                  for items in groups.values()), "not six independent structures")
    for family in FAMILIES:
        check(sum(items[0]["family"] == family for items in groups.values()) == 2,
              "not two independent designs per family")
    return groups


def read_system(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    found = [n.value.value for n in tree.body
             if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "SYSTEM" for t in n.targets)
             and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)]
    check(len(found) == 1, "shared training system literal changed")
    return found[0]


def pinned_common(root):
    root = Path(root)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    check(sha == SHARED_SOURCE, "shared v2 trainer checkout changed")
    root = root / "experiments/agent_painter/training"
    spec = importlib.util.spec_from_file_location("pinned_smol_v2_common", root / "common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check(module.SMOL_SUPERVISION_POLICY_V2 == "smol-upstream-terminal-eos-v2",
          "SMOL terminal-EOS objective changed")
    return module, root


def inspect(rows, tokenizer, common, system):
    reports = []
    for group, views in sorted(group_requests(rows).items()):
        row = views[0]
        target = common.canonical(row["program"])
        encoded = common.encode_row(tokenizer,
            {"id":group, "prompt":row["prompt"], "completion":target},
            system, 2048, supervision_policy=common.SMOL_SUPERVISION_POLICY_V2)
        ids, boundary = encoded["input_ids"], encoded["boundary"]
        supervised = ids[boundary:]
        check(encoded["supervised_eos_terminal"] and supervised[-1] == 2
              and supervised.count(2) == 1 and len(ids) <= 2048,
              "true supervised terminal EOS2/context missing")
        prefix_len = len(encoded["prefix_ids"])
        reports.append({
            "structural_group":group, "family":row["family"],
            "prompt_sha256":sha_bytes(row["prompt"].encode("utf-8")),
            "reference_sha256":sha_bytes(target.encode("utf-8")),
            "prefix_tokens":prefix_len, "mask_boundary":boundary,
            "reference_tokens_including_eos":len(supervised),
            "full_reference_tokens":len(ids),
            "reference_fits_original_512_cap":len(supervised) <= 512,
            "context_headroom_after_prefix":2048-prefix_len,
        })
    return reports


def selftest():
    example = []
    for family in FAMILIES:
        for variant in range(2):
            for seed in SEEDS:
                example.append({"group_id":family+str(variant),
                                "structural_group":family+str(variant),
                                "family":family, "seed":seed, "prompt":"demo",
                                "program":{"version":2,"family":family,"variation":variant}})
    check(len(group_requests(example)) == 6, "positive split test failed")
    example[1]["prompt"] = "changed"
    try:
        group_requests(example)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted seed-specific teacher drift")
    for bad in (b'{"x":1,"x":2}', b'{"x":NaN}'):
        try:
            strict_json(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid JSON")
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as root:
        p = Path(root)/"fixture.zip"
        with zipfile.ZipFile(p, "w") as archive:
            archive.writestr("corpus/development.jsonl", "visible")
            archive.writestr("corpus/final.jsonl", "sealed")
        with zipfile.ZipFile(p) as archive:
            check(member(archive, "corpus/development.jsonl").file_size == 7,
                  "positive visible member check failed")
            for name in ("corpus/final.jsonl","corpus/../escape",
                         "corpus/missing","corpus/./development.jsonl"):
                try:
                    member(archive, name)
                except ValueError:
                    pass
                else:
                    raise AssertionError("accepted unsafe/invalid member")
    print("SMOL original DEV budget source-only/adversarial checks PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--zip", type=Path)
    ap.add_argument("--snapshot", type=Path)
    ap.add_argument("--shared", type=Path)
    ap.add_argument("--source")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.self_test:
        selftest()
        return
    env = os.environ
    check(args.zip and args.snapshot and args.shared and args.output
          and re.fullmatch(r"[0-9a-f]{40}", args.source or "")
          and env.get("GITHUB_ACTIONS") == "true"
          and env.get("RUNNER_ENVIRONMENT") == "github-hosted"
          and env.get("GITHUB_REF") == "refs/heads/codex/agent-painter-smol-568"
          and env.get("GITHUB_REPOSITORY") == "techrote/cybersand"
          and env.get("GITHUB_SHA") == args.source,
          "source-matched GitHub-hosted read-only budget audit required")
    visible = args.output.parent/"visible-development"
    rows, manifest, assets = extract_visible(args.zip, visible)
    # Existing immutable six-request validator independently verifies actual
    # positive native receipts, readback bytes, compile equality and hashes.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from contracts import load_development
    verified_rows, verified_manifest = load_development(visible/"corpus")
    check(rows == verified_rows and manifest == verified_manifest,
          "positive DEVELOPMENT source/native verification failed")
    common, training_dir = pinned_common(args.shared)
    profile = strict_json((training_dir/"model_profiles.json").read_bytes())["profiles"]["smol"]
    check(profile["revision"] == MODEL_REVISION
          and profile["trust_remote_code"] is False, "model pin changed")
    for file in ("config.json","tokenizer.json","tokenizer_config.json"):
        check(sha_file(args.snapshot/file) == profile["required_hashes"][file],
              "upstream tokenizer/config hash mismatch: "+file)
    check(not any(args.snapshot.rglob("*.safetensors")),
          "read-only tokenizer audit forbids model weights")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True,
                                        trust_remote_code=False)
    check(tok.eos_token_id == 2 and tok.eos_token == "<|im_end|>"
          and tok.chat_template, "authentic upstream SMOL template/EOS required")
    results = inspect(rows,tok,common,read_system(training_dir/"train.py"))
    report = {
        "schema":"cybersand.smol.original-development-budget/1",
        "status":"VISIBLE_DEVELOPMENT_REFERENCE_ONLY_NOT_MODEL_EVALUATION",
        "lane_source_sha":args.source, "shared_terminal_eos_v2_sha":SHARED_SOURCE,
        "original_native_source_sha":NATIVE_SOURCE,
        "original_native_zip_sha256":ZIP_HASH,
        "original_native_manifest_sha256":MANIFEST_HASH,
        "model_revision":MODEL_REVISION,
        "chat_template_sha256":sha_bytes(tok.chat_template.encode("utf-8")),
        "eos_token_id":tok.eos_token_id, "independent_requests":6,
        "native_seed_views":12, "visible_native_asset_members":len(assets),
        "reference_completion_token_range":[min(x["reference_tokens_including_eos"] for x in results),
                                            max(x["reference_tokens_including_eos"] for x in results)],
        "all_reference_completions_fit_512":all(x["reference_fits_original_512_cap"] for x in results),
        "reference_representability_only":True, "per_group":results,
        "new_optimizer_steps":0,"new_model_outputs":0,"new_native_paintings":0,
        "new_shared_train_admissions":0,"dev60_executed":False,"final_opened":False
    }
    check(not args.output.exists(), "refuse evidence overwrite")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="per_group"},sort_keys=True))


if __name__ == "__main__":
    main()
