"""Fake-Git source composition regression tests; no model or held-out."""
from __future__ import annotations
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ncr_001f_composition_gate import Stage, Refusal, build, protected, STATUS


def git(root, *args):
    p = subprocess.run(["git", "-C", str(root), *args],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, check=False)
    if p.returncode:
        raise RuntimeError(f"git {args}: {p.stderr}")
    return p.stdout.strip()


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-b", "main")
        git(self.repo, "config", "user.name", "NCR Source Test")
        git(self.repo, "config", "user.email", "ncr-test@example.invalid")
        self.commit("experiments/ncr-001/protocol.json", '{"train_authorized":false}\n')
        self.base = self.commit("experiments/ncr-001/source-pins.json", '{}\n')
        self.authority = {
            path: git(self.repo, "rev-parse", f"{self.base}:{path}")
            for path in ("experiments/ncr-001/protocol.json",
                         "experiments/ncr-001/source-pins.json")}

    def commit(self, path, content):
        f = self.repo / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")
        git(self.repo, "add", path)
        git(self.repo, "commit", "-m", f"test {path}")
        return git(self.repo, "rev-parse", "HEAD")

    def branch(self, branch, parent, path, content):
        git(self.repo, "checkout", "-b", branch, parent)
        return self.commit(path, content)

    def compose(self, stages, main="main", own_files=None):
        git(self.repo, "checkout", "own")
        return build(self.repo, Path(self.tmp.name) / "workspace", main,
                     tuple(stages), own_anchor=self.base,
                     own_files=own_files, authority=self.authority)

    def test_full_source_chain_and_two_disjoint_drafts(self):
        a = self.branch("phase", self.base, "scripts/phase.py", "phase=1\n")
        b = self.commit("scripts/phase.py", "phase=2\n")
        c = self.branch("event", self.base, "scripts/event.py", "event=1\n")
        d = self.branch("own", self.base, "scripts/own.py", "own=1\n")
        result = self.compose(
            (Stage(199, self.base, a, "phase"),
             Stage(201, a, b, "phase"),
             Stage(224, self.base, c, "event"),
             Stage(223, self.base, d, "self")), own_files={"scripts/own.py"})
        self.assertEqual(result["status"], STATUS)
        self.assertEqual(len(result["original_stages"]), 4)
        self.assertEqual(len(result["composed_changed_paths"]), 3)
        self.assertFalse(result["train_authorized"])
        self.assertIsNone(result["all_domain_tests_passed"])
        self.assertEqual((Path(self.tmp.name)/"workspace/scripts/phase.py").read_text(),
                         "phase=2\n")
        self.assertEqual(git(self.repo, "rev-parse", "HEAD"), d)

    def test_unrelated_overlap_fails_before_any_composition(self):
        a = self.branch("event", self.base, "scripts/shared.py", "event=1\n")
        d = self.branch("own", self.base, "scripts/shared.py", "own=1\n")
        with self.assertRaisesRegex(Refusal, "independent source overlap"):
            self.compose((Stage(224, self.base, a, "event"),
                          Stage(223, self.base, d, "self")),
                         own_files={"scripts/shared.py"})

    def test_wrong_dependency_ancestry_fails(self):
        a = self.branch("phase", self.base, "scripts/a.py", "a=1\n")
        b = self.branch("unrelated", self.base, "scripts/b.py", "b=1\n")
        d = self.branch("own", self.base, "scripts/own.py", "own=1\n")
        with self.assertRaisesRegex(Refusal, "ancestor|exit 1"):
            self.compose((Stage(199, self.base, a, "phase"),
                          Stage(201, a, b, "phase"),
                          Stage(223, self.base, d, "self")),
                         own_files={"scripts/own.py"})

    def test_protected_authority_patch_fails(self):
        a = self.branch("bad", self.base, "experiments/ncr-001/protocol.json",
                        '{"train_authorized":true}\n')
        d = self.branch("own", self.base, "scripts/own.py", "own=1\n")
        with self.assertRaisesRegex(Refusal, "protected path"):
            self.compose((Stage(224, self.base, a, "event"),
                          Stage(223, self.base, d, "self")),
                         own_files={"scripts/own.py"})

    def test_current_main_authority_drift_fails(self):
        self.branch("main-diverged", self.base, "experiments/ncr-001/source-pins.json",
                    '{"altered":true}\n')
        d = self.branch("own", self.base, "scripts/own.py", "own=1\n")
        with self.assertRaisesRegex(Refusal, "Current main authority"):
            self.compose((Stage(223, self.base, d, "self"),),
                         main="main-diverged", own_files={"scripts/own.py"})

    def test_owned_change_set_drift_fails(self):
        d = self.branch("own", self.base, "scripts/own.py", "own=1\n")
        with self.assertRaisesRegex(Refusal, "owned path drift"):
            self.compose((Stage(223, self.base, d, "self"),),
                         own_files={"scripts/not-own.py"})

    def test_disjoint_patch_conflicting_with_current_main_fails(self):
        seed = self.branch("seed", self.base, "scripts/shared.py", "value=0\n")
        a = self.branch("phase", seed, "scripts/shared.py", "value=phase\n")
        self.branch("main-diverged", seed, "scripts/shared.py", "value=main\n")
        d = self.branch("own", self.base, "scripts/own.py", "own=1\n")
        with self.assertRaisesRegex(Refusal, "composition conflict"):
            self.compose((Stage(199, seed, a, "phase"),
                          Stage(223, self.base, d, "self")),
                         main="main-diverged", own_files={"scripts/own.py"})

    def test_reject_protected_paths_without_file_access(self):
        for path in ("RAG.md", "SCHEDUSWARM.md",
                     "datasets/train-003/heldout.jsonl",
                     "experiments/ncr-001/protocol.json",
                     "experiments/ncr-001/fixtures/a.json",
                     "experiments/ncr-001/sealed/a.json"):
            with self.subTest(path=path):
                self.assertTrue(protected(path))
        self.assertFalse(protected("scripts/ncr_001f_controls.py"))


if __name__ == "__main__":
    unittest.main()
