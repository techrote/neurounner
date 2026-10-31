#!/usr/bin/env python3
"""Read-only NCR-001F original-PR composition on current main (NO MODEL)."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SOURCE_ANCHOR = "e53ab3a738de8f3313e8b1ca98a89c6194093ed7"
OLD_MAIN = "12f7bf0475aefb61227f676ea377134734fa5072"
AUTHORITY = {
    "experiments/ncr-001/protocol.json": "14ce5fbaa38f8806c8c8738f19c01b91385eebaf",
    "experiments/ncr-001/source-pins.json": "59fff1aed872b5563aa6898171983973e0cdb02c",
}
STATUS = "NCR001F_SOURCE_COMPOSITION_PASS__SCIENTIFIC_GATES_BLOCKED"
OWN_FILES = {
    ".github/workflows/ncr-001f.yml",
    ".github/workflows/ncr-001f-composition-gate.yml",
    "docs/NCR_001_F_OPTIMIZED_ASSERT_GUARD.md",
    "docs/NCR_001_F_COMPOSITION_GATE.md",
    "scripts/ncr_001f_plan.py",
    "scripts/ncr_001f_composition_gate.py",
    "tests/test_ncr_001f_controls.py",
    "tests/test_ncr_001f_composition_gate.py",
}


@dataclass(frozen=True)
class Stage:
    pr: int
    base: str
    head: str
    group: str


STAGES = (
    Stage(199, "2d5c0ed72c3d1c44f78ba58493bfffd8fab26d80",
          "2e2d7c86b066e96207afd6355ddbfbe535d3718e", "phase"),
    Stage(201, "2e2d7c86b066e96207afd6355ddbfbe535d3718e",
          "55026d914efd3ef4548da0644ccd27bb8de3074f", "phase"),
    Stage(209, "55026d914efd3ef4548da0644ccd27bb8de3074f",
          "28353d042fe479589e04fc7a85fbc56a90fb7707", "phase"),
    Stage(212, "28353d042fe479589e04fc7a85fbc56a90fb7707",
          "e25b84e9766ee2be9150ae79172821af32a56eaf", "phase"),
    Stage(227, "e25b84e9766ee2be9150ae79172821af32a56eaf",
          "4aa78b83b97b9a7623721905dab58381c1427730", "phase"),
    Stage(224, OLD_MAIN, "c597621d349e192e57fdfc18e5338ca9c9792ef2", "event"),
    Stage(226, OLD_MAIN, "8fb610af77538c28a4481235f5c7a266438eff94", "adapter"),
)


class Refusal(RuntimeError):
    """Source, dependency, object or source-only patch was not qualified."""


def git(root: Path, *args: str, data: bytes | None = None) -> bytes:
    p = subprocess.run(["git", "-C", str(root), *args], input=data,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if p.returncode:
        err = p.stderr.decode("utf-8", "replace")[-1400:].strip()
        raise Refusal(f"git {' '.join(args[:4])} exit {p.returncode}: {err}")
    return p.stdout


def changed_paths(root: Path, base: str, head: str) -> set[str]:
    return {p.decode("utf-8", "surrogateescape")
            for p in git(root, "diff", "--name-only", "-z", base, head).split(b"\0") if p}


def protected(path: str) -> bool:
    return (path in {*AUTHORITY, "RAG.md", "SCHEDUSWARM.md"}
            or path.startswith(("datasets/", "experiments/ncr-001/fixtures/",
                                "experiments/ncr-001/heldouts/", "weights/"))
            or ("sealed" in path.lower() and path.startswith("experiments/ncr-001/")))


def verify_stages(root: Path, stages: tuple[Stage, ...],
                  main: str, own_anchor: str,
                  own_files: set[str] | None,
                  authority: dict[str, str]) -> list[dict]:
    if not stages or stages[-1].group != "self":
        raise Refusal("Owned literal source must be the final stage")
    git(root, "merge-base", "--is-ancestor", own_anchor, stages[-1].head)
    git(root, "merge-base", "--is-ancestor", stages[-1].base, main)
    if git(root, "rev-parse", "HEAD").decode().strip() != stages[-1].head:
        raise Refusal("Current checkout is not literal owned source HEAD")
    records = []
    seen: dict[str, str] = {}
    previous_phase = None
    for stage in stages:
        git(root, "cat-file", "-e", f"{stage.base}^{{commit}}")
        git(root, "cat-file", "-e", f"{stage.head}^{{commit}}")
        git(root, "merge-base", "--is-ancestor", stage.base, stage.head)
        if stage.group == "phase":
            if previous_phase is not None and stage.base != previous_phase:
                raise Refusal(f"#{stage.pr}: original phase dependency changed")
            previous_phase = stage.head
        paths = changed_paths(root, stage.base, stage.head)
        if not paths:
            raise Refusal(f"#{stage.pr}: missing original source diff")
        for path in paths:
            if protected(path) or path in authority:
                raise Refusal(f"#{stage.pr}: protected path {path}")
            if path in seen and seen[path] != stage.group:
                raise Refusal(f"#{stage.pr}: independent source overlap {path} with {seen[path]}")
            seen[path] = stage.group
        if stage.group == "self" and own_files is not None and paths != own_files:
            raise Refusal(f"#{stage.pr}: owned path drift {sorted(paths ^ own_files)}")
        patch = git(root, "diff", "--binary", "--full-index", stage.base, stage.head)
        records.append({"pr": stage.pr, "group": stage.group,
                        "original_base": stage.base, "literal_source_head": stage.head,
                        "changed_paths": sorted(paths),
                        "patch_sha256": hashlib.sha256(patch).hexdigest(),
                        "patch_bytes": len(patch)})
    for path, blob in authority.items():
        actual = git(root, "rev-parse", f"{main}:{path}").decode().strip()
        if actual != blob:
            raise Refusal(f"Current main authority differs: {path} {actual}")
    return records


def build(root: Path, worktree: Path, main_ref: str, stages: tuple[Stage, ...],
          own_anchor: str = SOURCE_ANCHOR,
          own_files: set[str] | None = OWN_FILES,
          authority: dict[str, str] = AUTHORITY) -> dict:
    root, worktree = root.resolve(), worktree.resolve()
    if worktree.exists():
        raise Refusal("Worktree destination exists; refuse stale reuse")
    main = git(root, "rev-parse", main_ref).decode().strip()
    own = git(root, "rev-parse", "HEAD").decode().strip()
    records = verify_stages(root, stages, main, own_anchor, own_files, authority)
    created = False
    try:
        git(root, "worktree", "add", "--detach", str(worktree), main)
        created = True
        for stage in stages:
            patch = git(root, "diff", "--binary", "--full-index", stage.base, stage.head)
            try:
                git(worktree, "apply", "--3way", "--index", data=patch)
            except Refusal as exc:
                raise Refusal(f"PR #{stage.pr} source composition conflict: {exc}") from exc
        for path, blob in authority.items():
            if git(worktree, "hash-object", path).decode().strip() != blob:
                raise Refusal(f"Composed source authority changed: {path}")
        if git(root, "rev-parse", "HEAD").decode().strip() != own:
            raise Refusal("Composition moved the original source HEAD")
        changed = {p.decode("utf-8", "surrogateescape") for p in
                   git(worktree, "diff", "--cached", "--name-only", "-z").split(b"\0") if p}
        expected = {p for row in records for p in row["changed_paths"]}
        if changed != expected:
            raise Refusal(f"Composed path set drift: {sorted(changed ^ expected)}")
        return {"status": STATUS, "main_source_sha": main, "self_head": own,
                "original_stages": records, "composed_changed_paths": sorted(changed),
                "protocol_source_git_blob": authority["experiments/ncr-001/protocol.json"],
                "source_pins_git_blob": authority["experiments/ncr-001/source-pins.json"],
                "no_remote_writes": True, "original_model_executed": False,
                "optimizer_steps": 0, "sealed_heldouts_opened": False,
                "train_authorized": False, "full_72_conformance_pass": False,
                "actual_compute_budget_qualified": False,
                "actual_four_model_qualification_completed": False,
                "all_domain_tests_passed": None}
    except BaseException:
        if created:
            git(root, "worktree", "remove", "--force", str(worktree))
        raise


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--main-ref", default="refs/remotes/origin/main")
    p.add_argument("--worktree", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    root = args.repo.resolve()
    own = git(root, "rev-parse", "HEAD").decode().strip()
    stages = STAGES + (Stage(223, OLD_MAIN, own, "self"),)
    try:
        data = build(root, args.worktree, args.main_ref, stages)
        print(STATUS)
    except Refusal as exc:
        data = {"status": "REFUSED_SOURCE_COMPOSITION__NO_ADMISSION",
                "error": str(exc), "original_model_executed": False,
                "optimizer_steps": 0, "sealed_heldouts_opened": False,
                "train_authorized": False, "full_72_conformance_pass": False,
                "actual_compute_budget_qualified": False}
        print(f"COMPOSITION REFUSED: {exc}", file=sys.stderr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n",
                        encoding="utf-8")
    return 0 if data["status"] == STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
