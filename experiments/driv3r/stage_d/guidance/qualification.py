"""Fail-closed, CPU-only exact-checkout D-1B unit qualification.

Run from repository root:
    python3 -m experiments.driv3r.stage_d.guidance.qualification

Never launches model inference, native execution, training, APIs or formal A/B.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .m1_compat import M1_COMMIT as ADAPTER_M1_COMMIT
from .m1_compat import M1_SCHEMA_SHA256 as ADAPTER_SCHEMA_SHA256

VERSION = "d1b-exact-head-offline-qualification/1"
LANE = "experiments/driv3r/stage_d/guidance"
SCHEMA = "experiments/driv3r/d_common/v1/interface.schema.json"
M1_COMMIT = "8fa90de6be209ed0b7bd125039e6ec8b332fb1a6"
M1_SCHEMA_SHA256 = "ebe2fa211148602c614cba0f0a40f285cba797784b70efa035832d833877f80d"
# Exact-source structural inventory: 20 test modules, 224 methods.
# A global count of >=220 masked missing tests from the actual 223-method tree.
REQUIRED_METHOD_COUNTS = {
    "test_comparison.py": 9,
    "test_feedback_audit.py": 10,
    "test_feedback_edgecases.py": 6,
    "test_fleet_samples.py": 24,
    "test_fork_probe.py": 9,
    "test_geometry_guard.py": 5,
    "test_guidance.py": 18,
    "test_information.py": 11,
    "test_m1_compat.py": 16,
    "test_native_guidance_rank.py": 8,
    "test_native_pair_shadow.py": 18,
    "test_png_structure.py": 4,
    "test_qualification.py": 16,
    "test_shadow_policy.py": 5,
    "test_snapshot_identity.py": 5,
    "test_trajectory_audit.py": 13,
    "test_worm_morphology.py": 12,
    "test_worm_morphology_binding.py": 9,
    "test_worm_shift_nulls.py": 14,
    "test_worm_trace.py": 12,
}
MIN_TESTS = 224
REQUIRED_MODULES = frozenset(REQUIRED_METHOD_COUNTS)
SHA = re.compile(r"[a-f0-9]{40}\Z")
FOOTER = re.compile(r"^Ran ([0-9]+) tests? in [0-9.]+s$", re.MULTILINE)
SKIP = re.compile(r"\bskipped=([0-9]+)\b")


class QualificationError(ValueError):
    """Precondition failure: no suite result should be certified."""


def parse_unittest(log: str, returncode: int, minimum: int = MIN_TESTS) -> dict:
    """Never classify shortened, skipped or ambiguous runs as passed."""
    matches = list(FOOTER.finditer(log))
    if len(matches) != 1 or type(returncode) is not int or type(minimum) is not int:
        return {"passed": False, "test_count": None, "skipped": None,
                "reason": "missing_or_ambiguous_unittest_footer"}
    count = int(matches[0].group(1))
    tail = log[matches[0].end():]
    skipped_match = SKIP.search(tail)
    skipped = int(skipped_match.group(1)) if skipped_match else 0
    passed = (returncode == 0 and count >= minimum and skipped == 0
              and re.search(r"^OK$", tail, re.MULTILINE) is not None
              and "FAILED" not in tail)
    return {"passed": passed, "test_count": count, "skipped": skipped,
            "reason": "qualified" if passed else "failed_incomplete_or_skipped_suite"}


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(["git", *args], cwd=root, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise QualificationError("Git source verification unavailable") from exc


def _checked_git(root: Path, *args: str) -> str:
    result = _git(root, *args)
    if result.returncode != 0:
        raise QualificationError("Git query failed: " + " ".join(args[:2]))
    return result.stdout



def _validate_test_inventory(root: Path) -> int:
    """Structural test-source preflight; not an executed-suite result."""
    directory = root / LANE
    modules = {p.name for p in directory.glob("test_*.py")}
    if modules != set(REQUIRED_METHOD_COUNTS):
        raise QualificationError("Required test modules missing or unregistered")
    total = 0
    for filename, expected in sorted(REQUIRED_METHOD_COUNTS.items()):
        try:
            tree = ast.parse((directory / filename).read_text(encoding="utf-8"),
                             filename=filename)
        except (OSError, UnicodeError, SyntaxError) as exc:
            raise QualificationError("Invalid test inventory source: " + filename) from exc
        count = 0
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            methods = [m for m in node.body
                       if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and m.name.startswith("test_")]
            if len({m.name for m in methods}) != len(methods):
                raise QualificationError("Duplicate shadowed test inventory: " + filename)
            if any(isinstance(m, ast.AsyncFunctionDef) for m in methods):
                raise QualificationError("Async test inventory unsupported: " + filename)
            count += len(methods)
        if count != expected:
            raise QualificationError("Changed test inventory: " + filename +
                                     f" expected {expected}, found {count}")
        total += count
    if total != MIN_TESTS:
        raise QualificationError("Test inventory differs from declared minimum")
    return total


def source_fence(root: Path) -> tuple[str, int]:
    """Pin real source/schema bytes, ancestry, scope, status and discovery."""
    if (ADAPTER_M1_COMMIT != M1_COMMIT or
            ADAPTER_SCHEMA_SHA256 != M1_SCHEMA_SHA256):
        raise QualificationError("D-1B adapter constants disagree with frozen M1")
    head = _checked_git(root, "rev-parse", "HEAD").strip()
    if not SHA.fullmatch(head):
        raise QualificationError("Invalid Git HEAD")
    if _git(root, "merge-base", "--is-ancestor", M1_COMMIT, head).returncode:
        raise QualificationError("Owner-accepted immutable M1 is not an ancestor")
    schema = root / SCHEMA
    try:
        digest = hashlib.sha256(schema.read_bytes()).hexdigest()
    except OSError as exc:
        raise QualificationError("Frozen M1 schema bytes unavailable") from exc
    if digest != M1_SCHEMA_SHA256:
        raise QualificationError("Frozen M1 schema bytes differ from owner acceptance")
    # All imported project code is part of the exact-source claim. Checking only
    # the lane and schema would miss a dirty shared module imported by a test.
    dirty = _checked_git(root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise QualificationError("Dirty worktree invalidates exact-source qualification")
    # Disable rename detection so an out-of-lane deletion cannot masquerade
    # as an in-lane-only addition in --name-only output.
    changed = _checked_git(root, "diff", "--no-renames", "--name-only",
                           "--diff-filter=ACDMRTUXB", M1_COMMIT, head, "--")
    paths = [p for p in changed.splitlines() if p]
    if not paths or any(not p.startswith(LANE + "/") for p in paths):
        raise QualificationError("M1-to-head changes outside exclusive D-1B lane or empty")
    # Non-merge commits must never write out of lane, even when a later
    # revert hides that change from the final M1-to-HEAD tree diff.
    history = _checked_git(root, "log", "--no-merges", "--no-renames",
                           "--format=", "--name-only", f"{M1_COMMIT}..{head}", "--")
    historical_paths = [p for p in history.splitlines() if p]
    if not historical_paths or any(not p.startswith(LANE + "/") for p in historical_paths):
        raise QualificationError("commit history wrote outside exclusive D-1B lane or is empty")
    # This branch deliberately merged the owner-accepted M1 source. The naive
    # '-m' history check mistakes those approved incoming d_common files for
    # worker writes. Allow precisely that merge and compare its result against
    # M1, not against the D-1B first parent; reject any other merge ancestry.
    merges = [m for m in _checked_git(root, "rev-list", "--merges",
                                     f"{M1_COMMIT}..{head}").splitlines() if m]
    if len(merges) > 1:
        raise QualificationError("unapproved merge ancestry in D-1B history")
    for merge in merges:
        if not SHA.fullmatch(merge):
            raise QualificationError("invalid merge SHA in D-1B history")
        parents = _checked_git(root, "rev-list", "--parents", "-n", "1", merge).split()
        if len(parents) != 3 or parents[0] != merge or parents[2] != M1_COMMIT:
            raise QualificationError("merge is not exact owner-accepted M1 adoption")
        merge_paths = _checked_git(root, "diff", "--no-renames", "--name-only",
                                   M1_COMMIT, merge, "--").splitlines()
        if any(not p.startswith(LANE + "/") for p in merge_paths):
            raise QualificationError("M1 adoption merge changed frozen/shared paths")
    _validate_test_inventory(root)
    return head, len(paths)


def run_qualification(root: Path, timeout_seconds: int = 180) -> tuple[dict, str]:
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 600:
        raise QualificationError("Timeout must be an integer in 1..600 seconds")
    root = root.resolve()
    head, changed_count = source_fence(root)
    command = [sys.executable, "-B", "-m", "unittest", "discover", "-s", LANE,
               "-p", "test_*.py", "-v"]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    try:
        proc = subprocess.run(command, cwd=root, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=timeout_seconds,
                              check=False)
        log, code, timeout = proc.stdout, proc.returncode, False
    except subprocess.TimeoutExpired as exc:
        data = exc.stdout or b""
        log = data.decode("utf-8", "replace") if isinstance(data, bytes) else data
        code, timeout = 124, True
    result = parse_unittest(log, code)
    try:
        after, _ = source_fence(root)
        stable = head == after
    except QualificationError:
        stable = False
    result.update({
        "version": VERSION, "exact_head": head, "m1_commit": M1_COMMIT,
        "m1_schema_sha256": M1_SCHEMA_SHA256, "changed_lane_paths": changed_count,
        "minimum_expected_tests": MIN_TESTS, "test_command": command,
        "returncode": code, "timed_out": timeout, "source_stable": stable,
        "full_suite_passed": bool(result["passed"] and stable and not timeout),
        "test_log_sha256": hashlib.sha256(log.encode("utf-8")).hexdigest(),
        "native_qualified": False, "visual_d1_qualified": False,
        "actual_undo_verified": False, "five_record_m1_conformance_verified": False,
        "formal_ab_executed": False,
    })
    if not result["full_suite_passed"]:
        result["reason"] = ("source_changed_or_dirty" if not stable else
                            "timeout" if timeout else result["reason"])
    return result, log


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    try:
        report, log = run_qualification(root, args.timeout_seconds)
        sys.stderr.write(log)
    except QualificationError as exc:
        report = {"version": VERSION, "full_suite_passed": False,
                  "reason": "precondition_refused", "detail": str(exc),
                  "native_qualified": False, "five_record_m1_conformance_verified": False}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["full_suite_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
