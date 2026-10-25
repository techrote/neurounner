#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

B_DIRS = (
    "member-corpora",
    "ancestors",
    "quality",
)
B_PROVENANCE_DIRS = (
    "member-manifests",
)
B_PROVENANCE_FILES = (
    "source-artifacts.json",
)
DOWNSTREAM_DATASET_MANIFESTS = (
    "preference-manifest.json",
    "final-manifest.json",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_inventory(
    root: Path,
    destination: Path,
    *,
    exclude: set[str],
    exclude_prefixes: tuple[str, ...] = (),
) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel in exclude or any(rel.startswith(prefix) for prefix in exclude_prefixes):
            continue
        rows.append(f"{sha256_file(path)}  {rel}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rows) + ("\n" if rows else ""))


def verify_inventory(root: Path, inventory: Path) -> None:
    for number, raw in enumerate(inventory.read_text().splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            expected, rel = raw.split("  ", 1)
        except ValueError as exc:
            raise RuntimeError(
                f"{inventory}:{number}: malformed SHA-256 inventory row"
            ) from exc
        path = root / rel
        if not path.is_file():
            raise RuntimeError(
                f"{inventory}:{number}: missing inventoried file {rel}"
            )
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"{inventory}:{number}: {rel} hash {actual} != {expected}"
            )


def validate_candidate(candidate: Path) -> dict:
    manifest_path = candidate / "manifest.json"
    quality_path = candidate / "quality" / "validation.json"
    inventory_path = candidate / "provenance" / "hashes.sha256"
    required = (
        manifest_path,
        quality_path,
        candidate / "member-corpora",
        candidate / "provider",
        candidate / "quality" / "members",
        candidate / "ancestors",
        candidate / "provenance" / "member-manifests",
        candidate / "provenance" / "source-artifacts.json",
        inventory_path,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"candidate B archive incomplete: {missing}")

    manifest = load_json(manifest_path)
    quality = load_json(quality_path)
    if manifest.get("stage") != "VILLAGE-001B-member-corpora":
        raise RuntimeError(
            f"candidate stage {manifest.get('stage')!r} is not VILLAGE-001B"
        )
    counts = manifest.get("counts") or {}
    if counts.get("members") != 36 or counts.get("rows") != 5760:
        raise RuntimeError(f"candidate manifest counts invalid: {counts}")
    if quality.get("members") != 36:
        raise RuntimeError("candidate quality member count != 36")
    if quality.get("total_rows") != 5760:
        raise RuntimeError("candidate quality total rows != 5760")
    if quality.get("unique_example_ids") != 5760:
        raise RuntimeError("candidate unique example IDs != 5760")
    if quality.get("cloud_expansion_rows") != 2160:
        raise RuntimeError("candidate cloud expansion rows != 2160")

    contract = manifest.get("member_contract_sha256")
    if not isinstance(contract, str) or len(contract) != 64:
        raise RuntimeError("candidate member contract SHA-256 missing/invalid")

    member_files = sorted((candidate / "member-corpora").glob("V-G??-[ABC].jsonl"))
    if len(member_files) != 36:
        raise RuntimeError(
            f"candidate member-corpora count {len(member_files)} != 36"
        )

    verify_inventory(candidate, inventory_path)
    return manifest


def check_downstream_compatibility(
    candidate: Path,
    candidate_manifest: dict,
    live: Path,
    repo_root: Path,
) -> None:
    contract = candidate_manifest["member_contract_sha256"]

    for name in DOWNSTREAM_DATASET_MANIFESTS:
        path = live / name
        if path.is_file():
            doc = load_json(path)
            if doc.get("member_contract_sha256") != contract:
                raise RuntimeError(
                    f"{name}: member contract {doc.get('member_contract_sha256')!r} "
                    f"!= candidate {contract!r}"
                )

    sft_root = repo_root / "experiments" / "village-001" / "sft-evidence"
    if sft_root.exists():
        summary_path = sft_root / "summary.json"
        results_root = sft_root / "results"
        if not summary_path.is_file() or not results_root.is_dir():
            raise RuntimeError(
                "SFT evidence exists but lacks summary/results; refuse B repromotion"
            )
        summary = load_json(summary_path)
        if summary.get("members") != 36:
            raise RuntimeError("SFT evidence member count is not 36")

        for shard in sorted((candidate / "member-corpora").glob("V-G??-[ABC].jsonl")):
            member = shard.stem
            result_path = results_root / f"{member}.json"
            if not result_path.is_file():
                raise RuntimeError(
                    f"SFT evidence missing result for candidate member {member}"
                )
            result = load_json(result_path)
            expected = result.get("training_dataset_sha256")
            actual = sha256_file(shard)
            if expected != actual:
                raise RuntimeError(
                    f"{member}: candidate B shard hash {actual} != "
                    f"SFT training dataset hash {expected}; downstream ancestry differs"
                )

    # If a terminal manifest exists, earlier downstream evidence must also exist.
    if (live / "final-manifest.json").is_file() and not sft_root.exists():
        raise RuntimeError(
            "final-manifest.json exists without SFT evidence; ancestry cannot be "
            "independently reconciled, so B repromotion fails closed"
        )


def replace_path(source: Path, destination: Path) -> None:
    if destination.is_dir():
        shutil.rmtree(destination)
    elif destination.exists():
        destination.unlink()
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def apply_b_layer(candidate: Path, staged: Path) -> None:
    for rel in B_DIRS:
        replace_path(candidate / rel, staged / rel)

    staged_provider = staged / "provider"
    staged_provider.mkdir(parents=True, exist_ok=True)
    for old in list(staged_provider.glob("V-G??-[ABC]")):
        if old.is_dir():
            shutil.rmtree(old)
        else:
            old.unlink()
    for source in sorted((candidate / "provider").glob("V-G??-[ABC]")):
        shutil.copytree(source, staged_provider / source.name)

    staged_provenance = staged / "provenance"
    staged_provenance.mkdir(parents=True, exist_ok=True)
    for rel in B_PROVENANCE_DIRS:
        replace_path(
            candidate / "provenance" / rel,
            staged_provenance / rel,
        )
    for rel in B_PROVENANCE_FILES:
        replace_path(
            candidate / "provenance" / rel,
            staged_provenance / rel,
        )

    replace_path(candidate / "manifest.json", staged / "manifest.json")

    # README belongs to the newest promoted stage. Only use the B README when no
    # later-stage manifest has authority over the root description.
    if not (
        (staged / "preference-manifest.json").is_file()
        or (staged / "final-manifest.json").is_file()
    ):
        replace_path(candidate / "README.md", staged / "README.md")


def regenerate_b_inventory(staged: Path) -> None:
    """Write the B-stage inventory over B-owned authority only.

    Root README is deliberately excluded because D/E own it after later promotion.
    Downstream provider/preference/evaluation data is also excluded.
    """
    owned = []
    for rel in ("manifest.json",):
        path = staged / rel
        if path.is_file():
            owned.append(path)

    for prefix in ("member-corpora", "ancestors", "quality"):
        root = staged / prefix
        if root.is_dir():
            owned.extend(path for path in root.rglob("*") if path.is_file())

    provider = staged / "provider"
    if provider.is_dir():
        for member_dir in sorted(provider.glob("V-G??-[ABC]")):
            if member_dir.is_dir():
                owned.extend(
                    path for path in member_dir.rglob("*") if path.is_file()
                )

    provenance = staged / "provenance"
    for rel in ("source-artifacts.json",):
        path = provenance / rel
        if path.is_file():
            owned.append(path)
    manifests = provenance / "member-manifests"
    if manifests.is_dir():
        owned.extend(path for path in manifests.rglob("*") if path.is_file())

    rows = []
    for path in sorted(set(owned)):
        rel = path.relative_to(staged).as_posix()
        rows.append(f"{sha256_file(path)}  {rel}")
    destination = provenance / "hashes.sha256"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rows) + ("\n" if rows else ""))


def regenerate_downstream_inventories(staged: Path) -> None:
    if (staged / "preference-manifest.json").is_file():
        final_exists = (staged / "final-manifest.json").is_file()
        exclude = {
            "provenance/hashes.sha256",
            "preference-hashes.sha256",
            "final-hashes.sha256",
        }
        prefixes: tuple[str, ...] = ()
        if final_exists:
            # Preserve D's stage meaning after E has taken over root README and
            # added evaluation/final authority.
            exclude.update({"final-manifest.json", "README.md"})
            prefixes = ("evaluation/",)
        preference_inventory = staged / "preference-hashes.sha256"
        write_inventory(
            staged,
            preference_inventory,
            exclude=exclude,
            exclude_prefixes=prefixes,
        )
        verify_inventory(staged, preference_inventory)

    if (staged / "final-manifest.json").is_file():
        final_inventory = staged / "final-hashes.sha256"
        write_inventory(
            staged,
            final_inventory,
            exclude={"final-hashes.sha256"},
        )
        verify_inventory(staged, final_inventory)


def verify_promoted_b(staged: Path, candidate_manifest: dict) -> None:
    promoted = load_json(staged / "manifest.json")
    if promoted != candidate_manifest:
        raise RuntimeError("promoted B manifest differs from validated candidate")
    quality = load_json(staged / "quality" / "validation.json")
    if quality.get("unique_example_ids") != 5760:
        raise RuntimeError("promoted B quality validation is not complete")
    verify_inventory(staged, staged / "provenance" / "hashes.sha256")


def promote_candidate(
    candidate: Path,
    live: Path,
    *,
    repo_root: Path,
    validate: bool = True,
) -> None:
    candidate_manifest = (
        validate_candidate(candidate)
        if validate
        else load_json(candidate / "manifest.json")
    )

    if live.exists():
        check_downstream_compatibility(
            candidate,
            candidate_manifest,
            live,
            repo_root,
        )

    parent = live.parent
    staged = parent / f".{live.name}.b-promote-staged"
    backup = parent / f".{live.name}.b-promote-backup"
    for path in (staged, backup):
        if path.exists():
            shutil.rmtree(path)

    if live.exists():
        shutil.copytree(live, staged)
    else:
        staged.mkdir(parents=True)

    try:
        apply_b_layer(candidate, staged)
        regenerate_b_inventory(staged)
        regenerate_downstream_inventories(staged)
        verify_promoted_b(staged, candidate_manifest)

        if live.exists():
            live.rename(backup)
        try:
            staged.rename(live)
        except Exception:
            if backup.exists() and not live.exists():
                backup.rename(live)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
        if backup.exists() and live.exists():
            shutil.rmtree(backup)


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        candidate = root / "candidate"
        live = root / "live"
        candidate.mkdir()
        live.mkdir()

        contract = "a" * 64
        (candidate / "manifest.json").write_text(
            json.dumps(
                {
                    "stage": "VILLAGE-001B-member-corpora",
                    "member_contract_sha256": contract,
                }
            )
        )
        for rel in ("member-corpora", "ancestors", "quality", "provider", "provenance"):
            (candidate / rel).mkdir(parents=True, exist_ok=True)
        (candidate / "quality" / "validation.json").write_text(
            json.dumps({"unique_example_ids": 5760})
        )
        (candidate / "README.md").write_text("B README\n")
        (candidate / "member-corpora" / "V-G01-A.jsonl").write_text("candidate\n")
        (candidate / "provider" / "V-G01-A").mkdir()
        (candidate / "provider" / "V-G01-A" / "raw.json").write_text("{}\n")
        (candidate / "provenance" / "member-manifests").mkdir()
        (candidate / "provenance" / "source-artifacts.json").write_text("{}\n")
        (candidate / "provenance" / "hashes.sha256").write_text("")

        (live / "manifest.json").write_text(
            json.dumps(
                {
                    "stage": "VILLAGE-001B-member-corpora",
                    "member_contract_sha256": contract,
                }
            )
        )
        (live / "member-corpora").mkdir()
        (live / "member-corpora" / "V-G01-A.jsonl").write_text("old\n")
        (live / "provider" / "preference").mkdir(parents=True)
        (live / "provider" / "preference" / "downstream.json").write_text("{}\n")
        (live / "evaluation").mkdir()
        (live / "evaluation" / "marker.txt").write_text("keep-me\n")
        (live / "preference-manifest.json").write_text(
            json.dumps({"member_contract_sha256": contract})
        )
        (live / "README.md").write_text("downstream README\n")

        # Simulate already-trained SFT evidence that binds downstream models to
        # the exact B member-corpus bytes.
        sft_root = root / "experiments" / "village-001" / "sft-evidence"
        results_root = sft_root / "results"
        results_root.mkdir(parents=True)
        (sft_root / "summary.json").write_text(json.dumps({"members": 36}))
        for i in range(1, 13):
            for sibling in "ABC":
                member = f"V-G{i:02d}-{sibling}"
                shard = candidate / "member-corpora" / f"{member}.jsonl"
                if member != "V-G01-A":
                    shard.write_text(f"{member}\n")
                (results_root / f"{member}.json").write_text(
                    json.dumps(
                        {"training_dataset_sha256": sha256_file(shard)}
                    )
                )

        promote_candidate(
            candidate,
            live,
            repo_root=root,
            validate=False,
        )

        assert (live / "evaluation" / "marker.txt").read_text() == "keep-me\n"
        assert (
            live / "provider" / "preference" / "downstream.json"
        ).is_file()
        assert (live / "README.md").read_text() == "downstream README\n"
        assert (
            live / "member-corpora" / "V-G01-A.jsonl"
        ).read_text() == "candidate\n"

        bad = root / "bad-live"
        shutil.copytree(live, bad)
        (bad / "preference-manifest.json").write_text(
            json.dumps({"member_contract_sha256": "b" * 64})
        )
        try:
            promote_candidate(
                candidate,
                bad,
                repo_root=root,
                validate=False,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("incompatible downstream member contract was accepted")

        # Same contract is still insufficient when candidate bytes differ from
        # already-trained SFT ancestry.
        mismatch = root / "mismatch-candidate"
        shutil.copytree(candidate, mismatch)
        (mismatch / "member-corpora" / "V-G01-A.jsonl").write_text(
            "byte-different-candidate\n"
        )
        try:
            promote_candidate(
                mismatch,
                live,
                repo_root=root,
                validate=False,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("byte-different B corpus was accepted under SFT evidence")

    print("VILLAGE-001B non-destructive repromotion self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--live", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.candidate or not args.live:
        parser.error("--candidate and --live are required unless --self-test is used")

    promote_candidate(
        args.candidate,
        args.live,
        repo_root=ROOT,
    )
    print(
        json.dumps(
            {
                "candidate": str(args.candidate),
                "live": str(args.live),
                "status": "promoted",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
