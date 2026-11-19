"""Offline regressions for paired training invocation custody and native hooks."""
from __future__ import annotations

import argparse
import copy
import shutil
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import village_001f_worker_diagnostic as w


def execution(member=w.PANEL[0], host="first"):
    return {"repository": "techrote/neurounner", "run_id": "1", "run_attempt": "1", "job": "train",
        "member_id": member, "paired_job_nonce": "job-" + host, "source_head": "a" * 40,
        "runner_name": "runner-" + host, "boot_id": "boot-" + host, "hostname": "host-" + host,
        "cpu_model": "Fixture CPU", "cpu_count": 4, "platform": "Fixture Linux",
        "packages": {"python": "3.12.3", **w.RUNTIME}, "process_id": 1 if host == "first" else 2,
        "started_at_utc": "2000-01-01T00:00:00+00:00", "started_monotonic_ns": 1}


def traces(step):
    return {key: [float(index) / 100 for index in range(step)] for key in w.TRACE_KEYS}


def mutable_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(w.json_bytes(value))


class ProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.suite, cls.campaign, cls.pref = w.load_contract(w.DEFAULT_CONFIG)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="village001f-provenance-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "lane"
        self.root.mkdir()
        w.save(self.root / "training-identity.json", {"worker_config_sha256": w.digest(self.cfg),
            "member_id": w.PANEL[0], "mask_mode": w.MASKS[0]})

    def invocation(self, *, resumed=False, complete=False, host="first"):
        with patch.object(w, "training_execution_metadata", return_value=execution(host=host)):
            return w.TrainingInvocation(self.root, self.cfg, w.PANEL[0], w.MASKS[0],
                resumed=resumed, complete=complete)

    def checkpoint(self, step):
        mutable_json(self.root / "checkpoint/progress.json", {"next_step": step, "source_run_id": "1", **traces(step)})

    def completed(self):
        mutable_json(self.root / "result.json", {"training": {"optimizer_steps": 120, **traces(120)}})
        w.save(self.root / "training.json", {"training_result_sha256": w.sha(self.root / "result.json")})

    def fresh(self):
        invocation = self.invocation()
        for step in (30, 60, 90, 120):
            self.checkpoint(step)
            invocation.checkpoint()
        self.completed()
        invocation.finish("completed")
        return invocation

    def test_order_is_three_each_and_balanced_within_each_sibling(self):
        expected = {"V-G01-A", "V-G05-C", "V-G09-B"}
        historical = {m for m in w.PANEL if w.training_arm_order(self.cfg, m)[0] == w.MASKS[0]}
        self.assertEqual(historical, expected)
        self.assertEqual(sorted(m[-1] for m in historical), ["A", "B", "C"])
        modified = copy.deepcopy(self.cfg)
        modified["training"]["paired_execution"]["member_arm_order"][w.PANEL[0]].reverse()
        with self.assertRaisesRegex(RuntimeError, "order changed"):
            w.training_arm_order(modified, w.PANEL[0])

    def test_fresh_checkpoint_chain_covers_exactly_120_steps_and_all_files(self):
        self.fresh()
        summary, files = w.validate_training_provenance(self.root, self.cfg)
        self.assertEqual([(s["start_step"], s["end_step"], s["steps"]) for s in summary["optimizer_segments"]], [(0, 120, 120)])
        self.assertEqual(summary["unattributed_optimizer_segments"], [])
        self.assertEqual(summary["resume_invocations"], [])
        self.assertEqual(len(files), 6)
        self.assertEqual(set(files), {p.relative_to(self.root).as_posix() for p in (self.root / w.PROVENANCE_DIR).iterdir()})

    def test_completed_copy_appends_zero_steps_and_preserves_original_host_and_receipts(self):
        self.fresh()
        before = {name: (self.root / name).read_bytes() for name in ("result.json", "training.json")}
        original_files = {name: path.read_bytes() for name, path in w._provenance_files(self.root).items()}
        invocation = self.invocation(resumed=True, complete=True, host="second")
        invocation.finish("completed")
        summary, _ = w.validate_training_provenance(self.root, self.cfg)
        self.assertEqual(summary["zero_step_copies"], [2])
        self.assertEqual(summary["resume_invocations"], [2])
        self.assertEqual([s["execution"]["hostname"] for s in summary["optimizer_segments"]], ["host-first"])
        self.assertEqual(before, {name: (self.root / name).read_bytes() for name in before})
        self.assertTrue(all((self.root / name).read_bytes() == data for name, data in original_files.items()))

    def test_partial_interruption_retains_two_actual_optimizer_hosts(self):
        first = self.invocation()
        self.checkpoint(30)
        first.checkpoint()
        first.finish("interrupted", "KeyboardInterrupt")
        second = self.invocation(resumed=True, host="second")
        self.checkpoint(120)
        second.checkpoint()
        self.completed()
        second.finish("completed")
        summary, _ = w.validate_training_provenance(self.root, self.cfg)
        self.assertEqual([(s["start_step"], s["end_step"]) for s in summary["optimizer_segments"]], [(0, 30), (30, 120)])
        self.assertEqual([s["execution"]["hostname"] for s in summary["optimizer_segments"]], ["host-first", "host-second"])
        self.assertEqual(summary["unfinished_invocations"], [1])
        self.assertEqual(summary["unattributed_optimizer_segments"], [])

    def test_hard_kill_after_native_checkpoint_has_explicit_unattributed_steps(self):
        self.invocation()
        self.checkpoint(30)  # Native atomic rename survived; sidecar did not.
        before, _ = w.validate_training_provenance(self.root, self.cfg, require_complete=False)
        self.assertEqual(before["unattributed_optimizer_segments"][0]["steps"], 30)
        resumed = self.invocation(resumed=True, host="second")
        self.checkpoint(120)
        resumed.checkpoint()
        self.completed()
        resumed.finish("completed")
        summary, _ = w.validate_training_provenance(self.root, self.cfg)
        self.assertEqual([s["steps"] for s in summary["optimizer_segments"]], [90])
        self.assertEqual([s["steps"] for s in summary["unattributed_optimizer_segments"]], [30])
        self.assertEqual(summary["retained_optimizer_steps"], 120)

    def test_missing_checkpoint_receipt_is_detected_by_finish_hash(self):
        self.fresh()
        (self.root / w.PROVENANCE_DIR / "invocation-0001-checkpoint-000060.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "lost checkpoint provenance"):
            w.validate_training_provenance(self.root, self.cfg)

    def test_changed_native_trace_prefix_is_rejected(self):
        self.fresh()
        path = self.root / "result.json"
        result = w.read(path)
        result["training"]["losses"][0] += 1
        mutable_json(path, result)
        with self.assertRaisesRegex(RuntimeError, "trace changed"):
            w.validate_training_provenance(self.root, self.cfg)

    def test_missing_runner_identity_fails_before_start_receipt(self):
        value = execution()
        value["paired_job_nonce"] = None
        with patch.object(w, "training_execution_metadata", return_value=value), self.assertRaisesRegex(RuntimeError, "runner identity incomplete"):
            w.TrainingInvocation(self.root, self.cfg, w.PANEL[0], w.MASKS[0], resumed=False, complete=False)
        self.assertFalse((self.root / w.PROVENANCE_DIR).exists())

    def test_boolean_step_and_nonfinite_trace_are_rejected(self):
        for value in ({"next_step": True, **traces(1)}, {"next_step": 1, **traces(1), "losses": [float("nan")]}):
            with self.subTest(value=value):
                mutable_json(self.root / "checkpoint/progress.json", value)
                with self.assertRaises(RuntimeError):
                    w.validate_training_provenance(self.root, self.cfg, require_complete=False)

    def test_native_completed_copy_and_both_arm_mask_wiring(self):
        with redirect_stdout(StringIO()):
            w.mock_training_wiring(argparse.Namespace(config=w.DEFAULT_CONFIG), self.cfg, self.suite, self.campaign, self.pref)

    def test_native_interruption_hook_and_partial_resume_restore_every_engine_hook(self):
        member_id, arm = w.PANEL[0], w.MASKS[1]
        population, member = w.member_contract(self.campaign, member_id)
        parent_hash = w.source_result(member_id, "C")["adapter"]["sha256"]
        result_hash = "b" * 64
        parent = Path(self.temp.name) / "parent"
        parent.mkdir()
        engine = types.ModuleType("village_001d_preference_train")
        engine.__file__ = str(w.ROOT / "scripts/village_001d_preference_train.py")
        engine.write_json = mutable_json
        engine.pair_batch = lambda *a: None
        engine.run_heldout = lambda *a: [{"should_not_run": True}]
        engine.validate_complete = lambda *a: False
        engine.save_checkpoint = lambda **kw: mutable_json(kw["root"] / "progress.json",
            {"next_step": kw["next_step"], "source_run_id": "1", **{key: kw[key] for key in w.TRACE_KEYS}})
        first = True
        starts = []
        def main_fixture():
            options = dict(zip(sys.argv[1::2], sys.argv[2::2]))
            target = Path(options["--out"])
            shutil.copytree(Path(options["--resume"]), target, dirs_exist_ok=True)
            if engine.validate_complete(target, member, population, parent_hash, w.sha(w.pair_path(member_id)), self.campaign, self.pref):
                return
            starts.append(w._training_state(target)["step"])
            step = 30 if first else 120
            engine.save_checkpoint(root=target / "checkpoint", next_step=step, **traces(step))
            if first:
                raise KeyboardInterrupt("fixture interruption after durable checkpoint")
            (target / "adapter").mkdir()
            engine.write_json(target / "result.json", {"experiment": "VILLAGE-001D", "member_id": member_id,
                "member_seed": member["seed"], "member_contract_sha256": population["member_contract_sha256"],
                "parent_adapter_sha256": parent_hash, "preference_dataset_sha256": w.sha(w.pair_path(member_id)),
                "training_config": self.pref["preference_training"],
                "training": {"optimizer_steps": 120, "pair_count": 60, **traces(120)},
                "evaluation": {"responses": engine.run_heldout(None, None, None)}, "adapter": {"sha256": result_hash},
                "recovery": {"resumed": True, "resumed_from_step": 30}})
        engine.main = main_fixture
        args = argparse.Namespace(config=w.DEFAULT_CONFIG, member=member_id, parent_adapter=parent, mask_mode=arm,
            out=Path(self.temp.name) / "interrupted", resume=None)
        hooks = {name: getattr(engine, name) for name in ("write_json", "pair_batch", "run_heldout", "validate_complete", "save_checkpoint")}
        original_argv = list(sys.argv)
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"village_001d_preference_train": engine}))
            stack.enter_context(patch.object(w, "runtime_modules", return_value=(types.SimpleNamespace(), None, None)))
            stack.enter_context(patch.object(w, "directory_hash", side_effect=lambda path: parent_hash if path == parent else result_hash))
            stack.enter_context(patch.object(w, "adapter_fingerprint", return_value={"canonical_tensor_sha256": "fixture"}))
            metadata = stack.enter_context(patch.object(w, "training_execution_metadata", return_value=execution()))
            with self.assertRaises(KeyboardInterrupt):
                w.train(args, self.cfg, self.suite, self.campaign, self.pref)
            partial, _ = w.validate_training_provenance(args.out, self.cfg, require_complete=False)
            self.assertEqual(partial["invocations"][0]["finish"]["status"], "interrupted")
            self.assertEqual(partial["retained_optimizer_steps"], 30)
            self.assertEqual(sys.argv, original_argv)
            self.assertTrue(all(getattr(engine, name) is value for name, value in hooks.items()))
            source = args.out
            args.out, args.resume = Path(self.temp.name) / "resumed", source
            first = False
            metadata.return_value = execution(host="second")
            w.train(args, self.cfg, self.suite, self.campaign, self.pref)
        self.assertEqual(starts, [0, 30])
        self.assertEqual(sys.argv, original_argv)
        self.assertTrue(all(getattr(engine, name) is value for name, value in hooks.items()))
        completed, _ = w.validate_training_provenance(args.out, self.cfg)
        self.assertEqual([(s["start_step"], s["end_step"]) for s in completed["optimizer_segments"]], [(0, 30), (30, 120)])
        self.assertEqual(completed["zero_step_copies"], [])


if __name__ == "__main__":
    unittest.main()
