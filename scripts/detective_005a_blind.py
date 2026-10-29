#!/usr/bin/env python3
"""DETECTIVE-005A private human-review custody; NO MODEL / NO TRAINING.

This tool must run on a trusted custodian's machine with isolated filesystem
access. Never run it in a public GitHub Actions job or commit its private files.
The reviewer must not have access to this repository's raw source archive:
random item ordering alone cannot anonymize publicly matched response text.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import detective_005a_fun_screen as screen
from scripts import detective_005a_assemble as assemble

PREPARED = "detective-005a-private-blind-custody-v1"
FREEZE = "detective-005a-blind-ratings-freeze-v1"
REVEAL = "UNBLIND-AFTER-FROZEN-144-RATINGS"


def outside_repo(target: Path) -> Path:
    path = target.resolve()
    screen.require(not path.is_relative_to(ROOT.resolve()),
                   "reviewer/private custody must stay outside repository")
    return path


def disjoint(a: Path, b: Path) -> None:
    aa, bb = a.resolve(), b.resolve()
    screen.require(not (aa == bb or aa.is_relative_to(bb) or bb.is_relative_to(aa)),
                   "private and reviewer outputs must be disjoint")


def new_private_dir(path: Path) -> None:
    screen.require(not path.exists() and not path.is_symlink(),
                   "refuse overwrite of sealed review custody")
    outside_repo(path)
    path.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(path, 0o700)


def write_private_bytes(path: Path, data: bytes) -> None:
    screen.require(not path.exists() and not path.is_symlink(),
                   "refuse overwrite of sealed review file")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def encoded(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True,
                       ensure_ascii=False) + "\n").encode("utf-8")


def source_dataset(dataset: Path) -> tuple[dict, list[dict], dict]:
    manifest = screen.read_json(dataset / "manifest.json")
    verified = assemble.verify_dataset(dataset)
    screen.require(
        verified["raw_results"] == 144 and
        verified["review_packet_generated"] is False and
        manifest["subjective_fun_ratings_present"] is False,
        "private review requires 144 exact raw rows with no public packet")
    rows = screen.read_results(dataset / "raw" / "results.jsonl")
    cfg = screen.load_contract()
    screen.validate_complete_results(cfg, rows)
    screen.require(manifest["raw_results_sha256"] ==
                   screen.sha256_file(dataset / "raw" / "results.jsonl"),
                   "raw response hash mismatch")
    return manifest, rows, cfg


def packet_identity(packet: dict) -> str:
    return screen.sha256_bytes(encoded(packet))


def prepare(dataset: Path, reviewer_dir: Path, custody_dir: Path) -> dict:
    """Create a non-public permutation; publish only the unlabelled packet."""
    disjoint(reviewer_dir, custody_dir)
    disjoint(dataset, reviewer_dir)
    disjoint(dataset, custody_dir)
    outside_repo(reviewer_dir)
    outside_repo(custody_dir)
    screen.require(not reviewer_dir.exists() and not custody_dir.exists(),
                   "refuse overwrite of private/reviewer custody")
    manifest, rows, cfg = source_dataset(dataset)
    secret = secrets.token_bytes(32)
    screen.require_blinding_key(secret)
    packet, mapping = screen.build_review_packet(cfg, rows, secret)
    screen.require(len(mapping) == 144, "private mapping cardinality drift")
    packet_sha = packet_identity(packet)
    receipt = {
        "schema": PREPARED,
        "experiment": "DETECTIVE-005A",
        "issue": 176,
        "secret_hex": secret.hex(),
        "mapping": mapping,
        "private_mapping_commitment_sha256":
            packet["private_mapping_commitment_sha256"],
        "packet_sha256": packet_sha,
        "raw_results_sha256": manifest["raw_results_sha256"],
        "repository_head_sha": manifest["repository_head_sha"],
        "actions_run_id": manifest["actions_run_id"],
        "rating_status": "unscored",
        "model_judge_calls": 0,
        "optimizer_steps": 0,
        "production_admission": False,
        "population_expansion_admitted": False,
    }
    new_private_dir(custody_dir)
    write_private_bytes(custody_dir / "private-custody.json", encoded(receipt))
    new_private_dir(reviewer_dir)
    write_private_bytes(reviewer_dir / "fun-packet.json", encoded(packet))
    write_private_bytes(
        reviewer_dir / "fun-ratings-template.json",
        encoded(screen.rating_template(packet)))
    write_private_bytes(
        reviewer_dir / "README.md",
        ("# DETECTIVE-005A blind reviewer-only packet\n\n"
         "Rate all 144 responses without model/member labels, raw archives, "
         "source-run provenance or a permutation seed. Wrong-but-funny may "
         "score highly. Do not inspect source/raw responses or the private "
         "custodian files before ratings are frozen. This is operational "
         "blinding, not anonymity against a reviewer who independently "
         "possesses the exact source outputs.\n").encode("utf-8"))
    return {
        "packet_sha256": packet_sha,
        "private_mapping_commitment_sha256":
            packet["private_mapping_commitment_sha256"],
        "packet_items": len(packet["items"]),
        "member_labels_disclosed": False,
        "optimizer_steps": 0,
        "model_judge_calls": 0,
    }


def authenticated_packet(dataset: Path, reviewer_dir: Path,
                         custody_dir: Path) -> tuple[dict, dict, list[dict], dict]:
    """Reconstruct privately; fail closed on altered seed, rows or packet."""
    disjoint(reviewer_dir, custody_dir)
    disjoint(dataset, reviewer_dir)
    disjoint(dataset, custody_dir)
    outside_repo(reviewer_dir)
    outside_repo(custody_dir)
    manifest, rows, cfg = source_dataset(dataset)
    custody = screen.read_json(custody_dir / "private-custody.json")
    screen.require(custody.get("schema") == PREPARED and
                   custody.get("rating_status") == "unscored" and
                   custody.get("raw_results_sha256") ==
                   manifest["raw_results_sha256"] and
                   custody.get("repository_head_sha") ==
                   manifest["repository_head_sha"] and
                   custody.get("actions_run_id") == manifest["actions_run_id"],
                   "private custody source/identity drift")
    seed = custody.get("secret_hex")
    screen.require(isinstance(seed, str) and len(seed) == 64 and
                   all(c in "0123456789abcdef" for c in seed),
                   "private blinding seed missing or malformed")
    packet, mapping = screen.build_review_packet(cfg, rows, bytes.fromhex(seed))
    observed = screen.read_json(reviewer_dir / "fun-packet.json")
    screen.require(packet == observed and
                   packet_identity(observed) == custody.get("packet_sha256") and
                   custody.get("mapping") == mapping and
                   custody.get("private_mapping_commitment_sha256") ==
                   packet["private_mapping_commitment_sha256"] and
                   screen.read_json(
                       reviewer_dir / "fun-ratings-template.json") ==
                   screen.rating_template(packet),
                   "private mapping/packet commitment mismatch")
    return custody, packet, rows, cfg


def freeze(dataset: Path, reviewer_dir: Path, custody_dir: Path,
           ratings_path: Path, freeze_dir: Path) -> dict:
    """Freeze the original 144 human ratings without revealing any mapping."""
    outside_repo(freeze_dir)
    disjoint(custody_dir, reviewer_dir)
    disjoint(custody_dir, freeze_dir)
    disjoint(reviewer_dir, freeze_dir)
    disjoint(dataset, freeze_dir)
    screen.require(not freeze_dir.exists(), "refuse overwrite of frozen ratings")
    custody, packet, _, cfg = authenticated_packet(
        dataset, reviewer_dir, custody_dir)
    ratings = screen.read_json(ratings_path)
    screen.validate_ratings(cfg, packet, ratings)
    ratings_bytes = ratings_path.read_bytes()
    digest = screen.sha256_bytes(ratings_bytes)
    seal = {
        "schema": FREEZE,
        "experiment": "DETECTIVE-005A",
        "issue": 176,
        "packet_sha256": custody["packet_sha256"],
        "private_mapping_commitment_sha256":
            custody["private_mapping_commitment_sha256"],
        "ratings_file_sha256": digest,
        "ratings": 144,
        "reviewer_label": ratings["reviewer_label"],
        "mapping_disclosed": False,
        "model_judge_calls": 0,
        "optimizer_steps": 0,
        "production_admission": False,
    }
    new_private_dir(freeze_dir)
    write_private_bytes(freeze_dir / "ratings.json", ratings_bytes)
    write_private_bytes(freeze_dir / "freeze.json", encoded(seal))
    return {k: v for k, v in seal.items() if k != "reviewer_label"}


def reveal(dataset: Path, reviewer_dir: Path, custody_dir: Path,
           freeze_dir: Path, out_dir: Path, confirm: str) -> dict:
    """Only after the original 144 ratings SHA is frozen may identities appear."""
    screen.require(confirm == REVEAL, "unblinding requires explicit confirmation")
    outside_repo(out_dir)
    disjoint(custody_dir, reviewer_dir)
    disjoint(custody_dir, out_dir)
    disjoint(reviewer_dir, out_dir)
    disjoint(dataset, out_dir)
    disjoint(freeze_dir, out_dir)
    screen.require(not out_dir.exists(), "refuse overwrite of unblinded archive")
    custody, packet, rows, cfg = authenticated_packet(
        dataset, reviewer_dir, custody_dir)
    seal = screen.read_json(freeze_dir / "freeze.json")
    ratings_file = freeze_dir / "ratings.json"
    screen.require(
        seal.get("schema") == FREEZE and
        seal.get("ratings") == 144 and
        seal.get("mapping_disclosed") is False and
        seal.get("packet_sha256") == custody["packet_sha256"] and
        seal.get("private_mapping_commitment_sha256") ==
            custody["private_mapping_commitment_sha256"] and
        seal.get("ratings_file_sha256") == screen.sha256_file(ratings_file),
        "ratings are not sealed to source/packet commitment")
    ratings = screen.read_json(ratings_file)
    screen.validate_ratings(cfg, packet, ratings)
    secret = bytes.fromhex(custody["secret_hex"])
    summary = screen.aggregate(cfg, rows, packet, ratings, secret)
    screen.require(summary["ratings_sha256"] == screen.sha256_bytes(
        encoded(ratings)), "sealed ratings reserialization drift")
    new_private_dir(out_dir)
    write_private_bytes(out_dir / "summary.json", encoded(summary))
    write_private_bytes(out_dir / "revealed-item-map.json",
                        encoded(custody["mapping"]))
    write_private_bytes(out_dir / "freeze.json", encoded(seal))
    return {
        "ratings_file_sha256": seal["ratings_file_sha256"],
        "members": len(summary["members"]),
        "shortlist": len(summary["subjective_fun_shortlist"]),
        "unblinding_after_frozen_ratings": True,
        "production_admission": False,
        "training_admission": False,
        "population_expansion_admitted": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    actions = ap.add_subparsers(dest="action", required=True)
    for action in ("prepare", "freeze", "reveal"):
        sub = actions.add_parser(action)
        sub.add_argument("--dataset", type=Path, required=True)
        sub.add_argument("--reviewer-dir", type=Path, required=True)
        sub.add_argument("--custody-dir", type=Path, required=True)
        if action in ("freeze", "reveal"):
            sub.add_argument("--freeze-dir", type=Path, required=True)
        if action == "freeze":
            sub.add_argument("--ratings", type=Path, required=True)
        if action == "reveal":
            sub.add_argument("--out-dir", type=Path, required=True)
            sub.add_argument("--confirm", required=True)
    args = ap.parse_args()
    if args.action == "prepare":
        result = prepare(args.dataset, args.reviewer_dir, args.custody_dir)
    elif args.action == "freeze":
        result = freeze(args.dataset, args.reviewer_dir, args.custody_dir,
                        args.ratings, args.freeze_dir)
    else:
        result = reveal(args.dataset, args.reviewer_dir, args.custody_dir,
                        args.freeze_dir, args.out_dir, args.confirm)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
