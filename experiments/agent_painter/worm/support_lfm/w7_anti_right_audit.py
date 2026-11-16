#!/usr/bin/env python3
"""W7 original first-pass audit and a NEW synthetic RIGHT-excluded local probe.

No new model inference/optimizer/native painting. Original control PNG bytes only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import io
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import zipfile

SUPPORT_ZIP_SHA = "458db32c6239536f6cbd4f421fc76356f734be4f9b7f9efb850a208875f42794"
PRIMARY_ZIP_SHA = "51f6b2cb5214b818864dc739a958da79f433075d7dc7c0b71e352aa0480ea6ff"
CHALLENGE_SHA = "f07fae41a0416d13d587a2be02bec54e23178fc46fc6bb66b3ed0aa710d20faf"
W0_SOURCE = "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
CODES = {"STRAIGHT":"0", "LEFT":"1", "RIGHT":"2", "BRANCH":"3", "RETRACE":"4", "STOP":"5"}
PROMPT_KEYS = ("family", "heading", "local_5x5", "legal", "last_actions", "tick")
SEEDS = {3101, 3103, 3109, 3119}
SCHEMA = "worm-lfm-support-w7-anti-right-probe-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verified_zip(path: Path, expected: str, *, support: bool):
    payload = path.read_bytes()
    if digest(payload) != expected:
        raise ValueError("original GitHub artifact ZIP SHA-256 mismatch")
    z = zipfile.ZipFile(io.BytesIO(payload))
    names = z.namelist()
    if len(names) != len(set(names)) or z.testzip() is not None:
        raise ValueError("archive duplicates / CRC corruption")
    if any(n.startswith("/") or ".." in PurePosixPath(n).parts or "\\" in n for n in names):
        raise ValueError("unsafe ZIP member path")
    if support:
        sums = z.read("SHA256SUMS.txt").decode().splitlines()
        expected_members = {}
        for line in sums:
            if len(line) < 68 or line[64:68] != "  ./":
                raise ValueError("malformed original SHA256SUMS entry")
            hash_value, name = line[:64], line[68:]
            if not all(c in "0123456789abcdef" for c in hash_value) or name in expected_members:
                raise ValueError("duplicate or invalid manifest hash")
            expected_members[name] = hash_value
        if set(expected_members) != set(names) - {"SHA256SUMS.txt"}:
            raise ValueError("incomplete original source member inventory")
        for name, hash_value in expected_members.items():
            if digest(z.read(name)) != hash_value:
                raise ValueError("original W0 member digest mismatch")
    return z


def make_probe(observed):
    """No right turns allowed in *synthetic prompt*; original W0 untouched."""
    if observed.get("kind") != "observed" or observed.get("observation_origin") != "actual-W0":
        raise ValueError("requires a genuine original W0 observation")
    source_obs = observed["observation"]
    if not all(k in source_obs for k in PROMPT_KEYS):
        raise ValueError("missing W0 observation field")
    if "RIGHT" not in source_obs["legal"] or "STOP" not in source_obs["legal"]:
        raise ValueError("no legal RIGHT/STOP contrast in source observation")
    if len(source_obs["legal"]) != len(set(source_obs["legal"])) or set(source_obs["legal"]) - set(CODES):
        raise ValueError("invalid source legal set")
    source_prompt = canonical({k: source_obs[k] for k in PROMPT_KEYS})
    if source_prompt != observed["prompt"] or digest(source_prompt.encode()) != observed["prompt_sha256"]:
        raise ValueError("unfaithful original W0 prompt identity")
    revised = copy.deepcopy(source_obs)
    revised["legal"] = [a for a in revised["legal"] if a != "RIGHT"]
    rewritten = canonical({k: revised[k] for k in PROMPT_KEYS})
    return {"id": observed["id"] + ":RIGHT-excluded",
            "source_observed_id": observed["id"],
            "source_observed_prompt_sha256": observed["prompt_sha256"],
            "source_seed": observed["source_seed"], "source_mode": observed["source_mode"],
            "source_tick": observed["source_tick"], "kind": "anti_RIGHT_legal_mask",
            "observation_origin": "synthetic-not-W0-executed",
            "removed_action": "RIGHT", "target_action": None,
            "observation": revised, "prompt": rewritten,
            "prompt_sha256": digest(rewritten.encode())}


def audit(support_zip: Path, primary_zip: Path, output: Path):
    if output.exists():
        raise ValueError("immutable output directory already exists")
    source = verified_zip(support_zip, SUPPORT_ZIP_SHA, support=True)
    primary = verified_zip(primary_zip, PRIMARY_ZIP_SHA, support=False)
    manifest = json.loads(source.read("challenge-v1/manifest.json"))
    challenge_bytes = source.read("challenge-v1/challenge.jsonl")
    if manifest["W0_source"] != W0_SOURCE or digest(challenge_bytes) != CHALLENGE_SHA:
        raise ValueError("wrong original W0 source or challenge bytes")
    rows = [json.loads(v) for v in challenge_bytes.splitlines()]
    observed = [r for r in rows if r["kind"] == "observed"]
    orig_masks = [r for r in rows if r["kind"] == "legal_subset"]
    if (len(rows), len(observed), len(orig_masks)) != (96, 24, 24):
        raise ValueError("original four-world paired prompt denominators differ")
    if ({r["source_seed"] for r in observed} != SEEDS or
            len({r["id"] for r in rows}) != 96):
        raise ValueError("original structural split identity mismatch")
    raw = [json.loads(v) for v in primary.read("sensitivity/raw-first-passes.jsonl").splitlines()]
    logits = [json.loads(v) for v in primary.read("sensitivity/six-code-logits.jsonl").splitlines()]
    summary = json.loads(primary.read("sensitivity/summary.json"))
    if (len(raw), len(logits), summary["requests"], summary["new_optimizer_updates"]) != (96, 96, 96, 0):
        raise ValueError("wrong original checkpoint-evaluation denominators")
    by_id = {r["id"]: r for r in rows}
    by_raw = {r["id"]: r for r in raw}
    by_logits = {r["id"]: r for r in logits}
    if len(by_raw) != 96 or len(by_logits) != 96 or set(by_raw) != set(by_id) or set(by_logits) != set(by_id):
        raise ValueError("duplicate/missing first-pass or logit IDs")
    true_eos_code, true_host_legal = 0, 0
    original_code_counts = {code: 0 for code in "012345"}
    for row in rows:
        answer, score = by_raw[row["id"]], by_logits[row["id"]]
        if answer["prompt_sha256"] != row["prompt_sha256"] or score["prompt_sha256"] != row["prompt_sha256"]:
            raise ValueError("original generated responses lack original prompt custody")
        ids = answer["token_ids"]
        if not isinstance(ids, list) or not ids or any(type(x) is not int for x in ids):
            raise ValueError("malformed first-pass token identities")
        if answer["eos_terminated"] != (ids[-1] == 7):
            raise ValueError("original EOS receipt inconsistency")
        code = answer["raw_text"]
        valid = answer["eos_terminated"] and code in original_code_counts
        legal = valid and next(action for action, digit in CODES.items()
                               if digit == code) in row["observation"]["legal"]
        if valid:
            original_code_counts[code] += 1
        true_eos_code += bool(valid)
        true_host_legal += bool(legal)
        if answer["raw_legal"] != legal:
            raise ValueError("primary raw-legal receipt differs from independent W0 mask")
        probs, scores = score["six_code_conditional_probabilities"], score["candidate_logits"]
        if set(probs) != set(CODES) or set(scores) != set(CODES) or any(
                not math.isfinite(x) for x in scores.values()):
            raise ValueError("missing/malformed original six-action model diagnostics")
        top = max(scores.values())
        expected = {a: math.exp(x - top) for a, x in scores.items()}
        denom = sum(expected.values())
        if any(abs(probs[a] - expected[a] / denom) > 1e-6 for a in CODES):
            raise ValueError("original six-code softmax doesn't match saved genuine logits")
    if original_code_counts != {"0":0, "1":0, "2":96, "3":0, "4":0, "5":0} or (
            true_eos_code, true_host_legal) != (96, 95):
        raise ValueError("original W7 collapse/legality evidence changed unexpectedly")
    probes = [make_probe(r) for r in observed]
    if len({p["prompt_sha256"] for p in probes}) != 15 or any(
            "RIGHT" in p["observation"]["legal"] for p in probes):
        raise ValueError("new anti-RIGHT challenge is incomplete")
    model_masked_top, right_probs = {}, []
    for p in probes:
        old_probs = by_logits[p["source_observed_id"]]["six_code_conditional_probabilities"]
        right_probs.append(old_probs["RIGHT"])
        # Model did NOT see p['prompt']; this is a legal-mask projection of OLD logits.
        choice = max(p["observation"]["legal"], key=lambda a: old_probs[a])
        model_masked_top[choice] = model_masked_top.get(choice, 0) + 1
    original_removed_right = sum(r.get("removed_action") == "RIGHT" for r in orig_masks)
    if original_removed_right != 1:
        raise ValueError("original weak anti-RIGHT coverage changed")
    output.mkdir(parents=True)
    gallery_dir = output / "original-W0-PNG-and-trace"
    gallery_dir.mkdir()
    gallery = []
    for p in probes:
        stem = p["source_observed_id"].removesuffix(":observed")
        for ext in ("png", "json"):
            name = "challenge-v1/w0-controls/" + stem + "." + ext
            payload = source.read(name)
            (gallery_dir / (stem + "." + ext)).write_bytes(payload)
            if ext == "png" and digest(payload) != manifest["original_png_sha256"][stem]:
                raise ValueError("source PNG does not match original W0 manifest")
        gallery.append((stem, p["source_seed"], p["observation"]["family"], p["source_mode"]))
    content = "".join(canonical(r) + "\n" for r in probes).encode()
    (output / "anti-right-24-provenance.jsonl").write_bytes(content)
    unique = {}
    for probe in probes:
        fingerprint = probe["prompt_sha256"]
        if fingerprint not in unique:
            unique[fingerprint] = {**probe, "provenance_case_ids": []}
        unique[fingerprint]["provenance_case_ids"].append(probe["id"])
    unique_bytes = "".join(canonical(r) + "\n" for r in unique.values()).encode()
    (output / "anti-right-15-unique.jsonl").write_bytes(unique_bytes)
    report = {
        "schema": SCHEMA, "status": "reproduced-original-W7-and-new-synthetic-prompt-only",
        "support_artifact_zip_sha256": SUPPORT_ZIP_SHA,
        "primary_artifact_zip_sha256": PRIMARY_ZIP_SHA, "source_W0_sha": W0_SOURCE,
        "original_challenge_sha256": CHALLENGE_SHA,
        "new_anti_right_24_provenance_sha256": digest(content),
        "new_anti_right_15_unique_sha256": digest(unique_bytes),
        "new_anti_right_unique_prompt_count": len(unique),
        "new_anti_right_repeated_provenance_cases": len(probes) - len(unique),
        "original_observed_W0_states": len(probes), "independent_obstacle_worlds": 4,
        "original_W0_PNGs_copied_byte_identical": len(gallery),
        "original_W7_first_pass_valid_code_and_EOS7": true_eos_code,
        "original_W7_raw_host_legal": true_host_legal,
        "original_W7_raw_RIGHT_count": original_code_counts["2"],
        "original_W7_primary_summary_uses_host_legal_as_valid_eos":
            summary["raw_valid_eos"] == true_host_legal,
        "original_legal_subset_masks_excluding_RIGHT": original_removed_right,
        "new_synthetic_masks_excluding_RIGHT": len(probes),
        "original_logits_RIGHT_probability_mean": statistics.mean(right_probs),
        "original_logits_host_masked_top_without_RIGHT": model_masked_top,
        "new_model_first_passes": 0, "new_optimizer_updates": 0,
        "new_native_operations": 0, "sealed_FINAL_opened": False,
        "interpretation": "Original W7: 96/96 valid code+EOS7, 95/96 host-legal. "
                          "New 24 prompts remove RIGHT from actual W0 observations, but "
                          "are synthetic edits, deduplicated to 15 unique prompts; NO "
                          "model inference or native W0 rollout on changed masks. "
                          "Masked-top counts use OLD model logits and are NOT new model "
                          "answers. Original PNGs are W0 controls, NOT trained LFM paintings."
    }
    (output / "report.json").write_text(canonical(report) + "\n")
    items = []
    for stem, seed, family, mode in gallery:
        items.append('<figure><img src="original-W0-PNG-and-trace/' +
                     html.escape(stem) + '.png" alt="Original W0 ' +
                     html.escape(stem) + '"><figcaption>' +
                     html.escape(f'{family} · world {seed} · {mode} · original W0 control') +
                     '</figcaption></figure>')
    document = ('<!doctype html><html lang="en"><meta charset="utf-8">'
                '<title>W7 anti-RIGHT source control gallery</title><style>'
                'body{font:15px system-ui;margin:2em;background:#181b1c;color:#e4e9e9}'
                'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1em}'
                'figure{background:#252a2c;padding:1em;margin:0}'
                'img{image-rendering:pixelated;width:100%;aspect-ratio:1}'
                'figcaption{margin-top:.7em;font-size:13px}</style>'
                '<h1>Actual W0 controls — anti-RIGHT probe</h1>'
                '<p>24 original W0 baseline PNGs from four obstacle worlds. '
                'No trained LFM art here. The NEW RIGHT-excluded legal masks '
                'are synthetic prompts ONLY, NOT yet model-evaluated or host-executed.</p>'
                '<main>' + ''.join(items) + '</main></html>')
    (output / "gallery.html").write_text(document)
    (output / "SHA256SUMS.txt").write_text(''.join(
        digest(file.read_bytes()) + '  ' + file.relative_to(output).as_posix() + '\n'
        for file in sorted(output.rglob('*'))
        if file.is_file() and file.name != "SHA256SUMS.txt"))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--support-zip", type=Path, required=True)
    p.add_argument("--primary-zip", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(canonical(audit(a.support_zip, a.primary_zip, a.out)))


if __name__ == "__main__":
    main()
