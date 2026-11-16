"""Model-free, DEVELOPMENT-only contracts for the first bounded Smol lane."""
from __future__ import annotations

import os
from pathlib import Path
import re
import struct
import subprocess
import sys

LANE = Path(__file__).resolve().parent
SHARED = LANE.parents[1] / "training"
sys.path.insert(0, str(SHARED))
import common
from lane_adapters import adapter_identity

FOUNDATION_SOURCE = "633920373a3d406a2ede61ce4137c324f76be23b"
MANIFEST_SHA256 = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
DEVELOPMENT_SHA256 = "8c73edd82d9289a78721ef7968ab2f939f858b666a106731709bc2d80b347572"
TRAIN_SHA256 = "5cb437db0663d15496adced3c255c32a5a274d99a69dd6113938b2dbc15ddbb1"
COMPILER_SHA256 = "4d8559805df1ff5f3f23277a7bca98e47fa8feb53337aa79e8546d0bb89132f6"
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
TRAIN_SEED = 568
MAX_STEPS = 120
MAX_SECONDS = 1200


def hosted_source(expected, environment=None):
    """Refuse accidental worker-local model execution and dirty source claims."""
    env = os.environ if environment is None else environment
    if (env.get("GITHUB_ACTIONS") != "true"
            or env.get("RUNNER_ENVIRONMENT") != "github-hosted"
            or env.get("RUNNER_OS") != "Linux"):
        raise ValueError("model execution requires a Linux GitHub-hosted Actions runner")
    if not isinstance(expected, str) or not common.COMMIT.fullmatch(expected):
        raise ValueError("full source SHA required")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=common.ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=common.ROOT, text=True)
    if actual != expected or dirty.strip():
        raise ValueError("source SHA differs or tracked source is dirty")
    return actual


def frozen_manifest(directory):
    directory = Path(directory)
    path = common.verify_file(directory, "admitted-corpus.json", MANIFEST_SHA256)
    manifest = common.read_json(path)
    if (manifest.get("state") != "native-admitted" or manifest.get("source_sha") != FOUNDATION_SOURCE
            or manifest.get("compiler_sha256") != COMPILER_SHA256
            or common.sha256_file(common.ROOT / "compiler.py") != COMPILER_SHA256
            or manifest["split_files"]["DEVELOPMENT"]["sha256"] != DEVELOPMENT_SHA256
            or manifest["split_files"]["TRAIN"]["sha256"] != TRAIN_SHA256):
        raise ValueError("frozen native pilot identity mismatch")
    groups = manifest["structural_groups"]
    if (set(groups) != {"TRAIN", "DEVELOPMENT", "FINAL"}
            or sum(map(len, groups.values())) != len(set().union(*map(set, groups.values())))):
        raise ValueError("structural partition leakage")
    return manifest


def load_development(directory):
    """Verify frozen DEV custody; never open TRAIN or FINAL targets/evidence."""
    directory = Path(directory)
    manifest = frozen_manifest(directory)
    entry = manifest["split_files"]["DEVELOPMENT"]
    path = common.verify_file(directory, entry["path"], DEVELOPMENT_SHA256)
    rows = [common.strict_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 12 or entry["count"] != 12 or len({r["id"] for r in rows}) != 12:
        raise ValueError("the frozen DEVELOPMENT denominator is twelve seed views")
    compiler = common.compiler_module()
    for row in rows:
        if (row["split"] != "DEVELOPMENT" or row["group_id"] != row["structural_group"]
                or row["group_id"] not in manifest["structural_groups"]["DEVELOPMENT"]
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", row["id"])
                or not isinstance(row.get("prompt"), str) or not row["prompt"].strip()):
            raise ValueError("invalid DEVELOPMENT request identity")
        compiled, admission = row["compiled"], row["native_admission"]
        if (compiled["focus"] != [0, 0, 96, 96] or compiled["material"] != 58
                or row["seed"] not in (104729, 130363)
                or admission.get("state") != "PASS" or admission.get("native_execution") is not True
                or admission.get("verified") is not True or admission.get("source_sha") != FOUNDATION_SOURCE
                or admission.get("runtime_sha256") != manifest["runtime_sha256"]
                or admission.get("affected_cells", 0) <= 0):
            raise ValueError("positive frozen native admission required")
        for name in ("png", "readback", "receipt"):
            common.verify_file(directory, admission["paths"][name], admission[name + "_sha256"])
        recomputed = compiler.compile_program(row["program"], seed=row["seed"], focus=compiled["focus"],
                                               material=compiled["material"], allowed_materials=(58,))
        if (recomputed != compiled or row["family"] != row["program"]["family"]
                or admission["program_sha256"] != recomputed["program_sha256"]
                or admission["operations_sha256"] != recomputed["operations_sha256"]):
            raise ValueError("native admission differs from DEVELOPMENT teacher")
        receipt = common.read_json(common.relative_file(directory, admission["paths"]["receipt"]))
        if (receipt.get("id") != row["id"] or receipt.get("family") != row["family"]
                or receipt.get("seed") != row["seed"] or receipt.get("passed") is not True
                or receipt.get("program_sha256") != recomputed["program_sha256"]
                or receipt.get("operations_sha256") != recomputed["operations_sha256"]
                or receipt.get("crop_i32le_sha256") != admission["readback_sha256"]
                or receipt.get("outside_area_cells") != 0 or receipt.get("unexpected_material_cells") != 0
                or receipt.get("deterministic_replay_checked") is not True):
            raise ValueError("native DEVELOPMENT receipt binding failed")
        raw = common.relative_file(directory, admission["paths"]["readback"]).read_bytes()
        if len(raw) != 96 * 96 * 4:
            raise ValueError("native readback length mismatch")
        values = [value[0] for value in struct.iter_unpack("<i", raw)]
        if (values != compiler.rasterize_operations(compiled["operations"], compiled["focus"])
                or compiler.verify_structure(compiled, values)["passed"] is not True):
            raise ValueError("native DEVELOPMENT readback/structure mismatch")
    if len({r["group_id"] for r in rows}) != 6:
        raise ValueError("the frozen DEVELOPMENT denominator is six structural groups")
    return rows, manifest


def request_fields(row):
    """Only request/host metadata crosses into generation; no teacher completion."""
    return {**{key: row[key] for key in ("id", "group_id", "structural_group", "family", "design", "split", "prompt", "seed")},
            "focus": row["compiled"]["focus"], "material": row["compiled"]["material"]}


def generation_groups(requests):
    """Host seeds never enter the model prompt. Reuse identical greedy requests explicitly."""
    groups = {}
    for request in requests:
        key = request["prompt"]
        if key not in groups:
            groups[key] = {"id": f"generation-{len(groups) + 1:04d}", "prompt": key, "requests": []}
        groups[key]["requests"].append(request)
    return list(groups.values())


def validate_candidate(config, selected, source, numerics, model_files, system, template, packages, python_version):
    expected = {"profile": "smol", "model": selected, "source_sha": source,
                "native_source_sha": FOUNDATION_SOURCE, "corpus_manifest_sha256": MANIFEST_SHA256,
                "train_sha256": TRAIN_SHA256, "compiler_sha256": COMPILER_SHA256,
                "trainer_sha256": common.sha256_file(SHARED / "train.py"),
                "contract_sha256": common.sha256_file(SHARED / "common.py"),
                "model_files": model_files, "numerics": numerics, "packages": packages, "python": python_version,
                "device": "cpu", "dtype": "float32", "threads": 1, "seed": TRAIN_SEED,
                "max_length": 2048, "rank": 8, "alpha": 16, "dropout": 0.05, "learning_rate": 0.0005,
                "system": system, "system_sha256": common.digest_json(system),
                "chat_template_sha256": common.digest_json(template),
                "adapter_contract": adapter_identity(selected),
                "native_execution_by_trainer": False, "final_targets_opened": False}
    changed = [key for key, value in expected.items() if config.get(key) != value]
    if changed:
        raise ValueError("candidate checkpoint identity mismatch: " + ", ".join(changed))


def classify_output(tokenizer, token_ids, request, compiler, generation_failure=None):
    try:
        verdict = common.strict_completion(tokenizer, token_ids, compiler.validate_program)
    except (RecursionError, OverflowError) as error:
        tokens = list(token_ids)
        terminated = bool(tokens) and tokens[-1] == tokenizer.eos_token_id and tokenizer.eos_token_id not in tokens[:-1]
        verdict = {"raw_text": tokenizer.decode(tokens[:-1] if terminated else tokens, skip_special_tokens=False),
                   "token_ids": tokens, "eos_terminated": terminated, "json_valid": False,
                   "program_valid": False, "first_pass_accepted": False,
                   "failure": f"{type(error).__name__}: {error}"}
    verdict["eos_token_id"] = tokenizer.eos_token_id
    verdict["raw_text_with_special_tokens"] = tokenizer.decode(list(token_ids), skip_special_tokens=False)
    verdict["compile_valid"] = False
    verdict["requested_family_matches"] = False
    verdict["requested_layout_fulfillment"] = "not automatically assessed"
    program, compiled = None, None
    if generation_failure:
        verdict["generation_failure"] = generation_failure
        verdict["first_pass_accepted"] = False
    if verdict["first_pass_accepted"]:
        program = common.strict_json(verdict["raw_text"])
        verdict["generated_family"] = program["family"]
        verdict["requested_family_matches"] = program["family"] == request["family"]
        try:
            # Match the native replay parser's original-text byte bound too.
            # Never canonicalize or strip whitespace before this check.
            program = compiler.parse_program(verdict["raw_text"])
            compiled = compiler.compile_program(program, seed=request["seed"], focus=request["focus"],
                                                 material=request["material"], allowed_materials=(request["material"],))
            verdict["compile_valid"] = True
        except (ValueError, TypeError, KeyError) as error:
            verdict["compile_failure"] = str(error)
    verdict["accepted"] = verdict["first_pass_accepted"] and verdict["compile_valid"]
    if not verdict["eos_terminated"]:
        verdict.setdefault("failure", "missing unique terminal EOS")
    return verdict, program if verdict["accepted"] else None, compiled if verdict["accepted"] else None
