"""Pure SMOL fail-closed promotion tests; synthetic receipts, no model or FINAL."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "completion_gate.py"
spec = importlib.util.spec_from_file_location("smol_completion_gate", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
REAL_FROZEN_TRAIN_ORDER = gate.frozen_train_order
SHA = "a" * 40


def fixture(root):
    root = Path(root)
    cp = root / gate.CHECKPOINT
    (cp / "adapter").mkdir(parents=True)
    (cp / "adapter" / "adapter_model.safetensors").write_bytes(b"synthetic-test-only")
    (cp / "adapter" / "adapter_config.json").write_text('{"fixture":"model-free"}')
    (cp / "training_state.pt").write_bytes(b"synthetic optimizer fixture")
    config = {"source_sha": SHA, "profile": "smol",
              "native_source_sha": gate.NATIVE_SOURCE,
              "corpus_manifest_sha256": gate.CORPUS,
              "train_sha256": gate.contracts.TRAIN_SHA256,
              "compiler_sha256": gate.contracts.COMPILER_SHA256,
              "seed": 568, "device": "cpu", "dtype": "float32", "threads": 1,
              "max_length": 2048, "rank": 8, "alpha": 16, "dropout": 0.05,
              "learning_rate": 0.0005, "model": gate.contracts.common.profile("smol"),
              "numerics": {"profile": "portable-cpu-fp32-v1"},
              "native_execution_by_trainer": False, "final_targets_opened": False,
              "sample_order": [f"synthetic-{i}" for i in range(12)]}
    events = [{"global_step": i, "sample_cursor": i,
               "sample_id": config["sample_order"][(i-1) % 12],
               "loss": 5.0/i, "gradient_norm": 0.1,
               "loss_tokens": 12, "sequence_tokens": 24,
               "prompt_bytes": 20, "completion_bytes": 30,
               "seconds": 0.01} for i in range(1, 121)]
    (root / "progress.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events))
    runtime = {"synthetic": "model-free"}
    (cp / "receipt.json").write_text(json.dumps({
        "global_step": 120, "sample_cursor": 120, "config": config,
        "history": events, "numerical_runtime": runtime,
        "frozen_base_sha256": "1"*64}))
    inventory = {p.relative_to(cp).as_posix(): {
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
        for p in cp.rglob("*") if p.is_file()}
    (cp / "inventory.json").write_text(json.dumps(inventory))
    summary = {
        "status": "completed", "actual_training": True,
        "global_step": 120, "sample_cursor": 120,
        "optimizer_steps_this_job": 120, "checkpoint": gate.CHECKPOINT,
        "checkpoint_readback_exact": True, "frozen_base_unchanged": True,
        "adapter_changed": True, "resume": None,
        "native_execution": False, "model_quality_claim": False,
        "config": config,
        "sample_ids_this_job": [e["sample_id"] for e in events],
        "loss_tokens_this_job": 1440, "sequence_tokens_this_job": 2880,
        "bytes_this_job": 6000, "first_loss": 5.0, "last_loss": 5.0/120,
        "numerical_runtime": runtime, "frozen_base_sha256": "1"*64,
        "adapter_before_sha256": "2"*64, "adapter_after_sha256": "3"*64,
        "checkpoint_bytes": sum(p.stat().st_size for p in cp.rglob("*") if p.is_file()),
        "checkpoint_inventory_sha256": hashlib.sha256(
            (cp / "inventory.json").read_bytes()).hexdigest(),
    }
    lane = {"lane": "smol", "source_sha": SHA, "target_global_steps": 120,
            "actual_training": True, "actual_global_step": 120,
            "optimizer_steps_this_job": 120, "checkpoint": gate.CHECKPOINT,
            "checkpoint_readback_exact": True, "status": "completed",
            "exit_code": 0, "native_execution": False,
            "final_targets_opened": False}
    (root / "summary.json").write_text(json.dumps(summary))
    (root / "lane-receipt.json").write_text(json.dumps(lane))
    return root


def mutate_json(path, key, value):
    obj = json.loads(path.read_text())
    obj[key] = value
    path.write_text(json.dumps(obj))



def rehash_checkpoint(root):
    """Model-free adversarial fixture: deliberately keep modified bytes internally self-consistent."""
    cp = root / gate.CHECKPOINT
    inventory = json.loads((cp / "inventory.json").read_text())
    for name in ("receipt.json", "training_state.pt"):
        path = cp / name
        inventory[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "bytes": path.stat().st_size}
    (cp / "inventory.json").write_text(json.dumps(inventory))
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["checkpoint_inventory_sha256"] = hashlib.sha256(
        (cp / "inventory.json").read_bytes()).hexdigest()
    summary["checkpoint_bytes"] = sum(p.stat().st_size for p in cp.rglob("*") if p.is_file())
    summary_path.write_text(json.dumps(summary))


class Fresh120PromotionTests(unittest.TestCase):
    def setUp(self):
        # Model-free receipts stand in for native evidence in unit tests only.
        # Production always rechecks real native-admitted TRAIN rows.
        patcher = mock.patch.object(
            gate, "frozen_train_order",
            return_value=[f"synthetic-{i}" for i in range(12)])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_complete_fresh120_is_promotable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture(tmp)
            verdict = gate.verify_completion(root, 0, SHA, root)
            self.assertTrue(verdict["promotable"])
            self.assertFalse(verdict["training"])
            self.assertFalse(verdict["final_targets_accessed"])

    def test_partial_exit_two_does_not_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture(tmp)
            with self.assertRaisesRegex(ValueError, "exit must be 0"):
                gate.verify_completion(root, 2, SHA, root)

    def test_partial_checkpoint_with_zero_exit_is_still_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture(tmp)
            mutate_json(root / "summary.json", "global_step", 40)
            with self.assertRaisesRegex(ValueError, "global_step"):
                gate.verify_completion(root, 0, SHA, root)

    def test_completed_flag_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture(tmp)
            mutate_json(root / "summary.json", "status", "stopped-checkpoint-saved")
            with self.assertRaisesRegex(ValueError, "status"):
                gate.verify_completion(root, 0, SHA, root)

    def test_fake_bool_and_optimizer_shortfall_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture(tmp)
            mutate_json(root / "summary.json", "optimizer_steps_this_job", True)
            with self.assertRaisesRegex(ValueError, "optimizer_steps_this_job"):
                gate.verify_completion(root, 0, SHA, root)

    def test_lane_receipt_contradiction_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture(tmp)
            mutate_json(root / "lane-receipt.json", "status", "FAILED")
            with self.assertRaisesRegex(ValueError, "status"):
                gate.verify_completion(root, 0, SHA, root)

    def test_mutated_inventory_hash_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture(tmp)
            (root / gate.CHECKPOINT / "inventory.json").write_text("mutated")
            with self.assertRaisesRegex(ValueError, "checkpoint_inventory_sha256"):
                gate.verify_completion(root, 0, SHA, root)

    def test_checkpoint_receipt_contradiction_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture(tmp)
            mutate_json(root / gate.CHECKPOINT / "receipt.json", "sample_cursor", 119)
            rehash_checkpoint(root)
            with self.assertRaisesRegex(ValueError, "sample_cursor"):
                gate.verify_completion(root, 0, SHA, root)

    def test_missing_adapter_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture(tmp)
            (root / gate.CHECKPOINT / "adapter" / "adapter_model.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "missing/empty"):
                gate.verify_completion(root, 0, SHA, root)

    def test_wrong_source_and_changed_training_lineage_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture(tmp)
            with self.assertRaisesRegex(ValueError, "source_sha"):
                gate.verify_completion(root, 0, "b"*40, root)
            mutate_json(root / "summary.json", "resume", {"from_step": 16})
            with self.assertRaisesRegex(ValueError, "resume"):
                gate.verify_completion(root, 0, SHA, root)



    def test_truncated_progress_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture(tmp)
            progress = root / "progress.jsonl"
            progress.write_text("\n".join(progress.read_text().splitlines()[:119]) + "\n")
            with self.assertRaisesRegex(ValueError, "progress trace"):
                gate.verify_completion(root, 0, SHA, root)

    def test_corrupted_optimizer_bytes_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture(tmp)
            (root / gate.CHECKPOINT / "training_state.pt").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                gate.verify_completion(root, 0, SHA, root)

    def test_wrong_training_partition_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture(tmp)
            mutate_json(root / "summary.json", "config", {
                **json.loads((root / "summary.json").read_text())["config"],
                "train_sha256": "0"*64})
            # The mismatch between summary/receipt is already sufficient.
            with self.assertRaisesRegex(ValueError, "config|train_sha256"):
                gate.verify_completion(root, 0, SHA, root)

    def test_optimizer_progress_reordering_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture(tmp)
            progress = root / "progress.jsonl"
            lines = progress.read_text().splitlines()
            lines[2], lines[3] = lines[3], lines[2]
            progress.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "checkpoint history"):
                gate.verify_completion(root, 0, SHA, root)


    def test_zero_gradient_with_rehashed_receipts_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture(tmp)
            cp = root / gate.CHECKPOINT
            events = [json.loads(line) for line in (root / "progress.jsonl").read_text().splitlines()]
            events[10]["gradient_norm"] = 0.0
            (root / "progress.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
            mutate_json(cp / "receipt.json", "history", events)
            rehash_checkpoint(root)
            with self.assertRaisesRegex(ValueError, "gradient_norm"):
                gate.verify_completion(root, 0, SHA, root)

    def test_relabelled_model_revision_with_rehashed_receipt_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture(tmp)
            summary_path = root / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["config"]["model"]["revision"] = "0"*40
            # The checkpoint receipt independently retains the original pin.
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "config"):
                gate.verify_completion(root, 0, SHA, root)

    def test_self_consistent_fake_train_ids_cannot_promote(self):
        """The claimed config+progress order cannot authorize itself."""
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture(tmp)
            with mock.patch.object(gate, "frozen_train_order",
                                   return_value=[f"actual-native-{i}" for i in range(12)]):
                with self.assertRaisesRegex(ValueError, "frozen TRAIN sample order"):
                    gate.verify_completion(root, 0, SHA, root)

    def test_frozen_order_uses_verified_native_train_only(self):
        """Model-free stub checks the deterministic sampler and native intake call."""
        rows = [{"family": family, "id": f"{family}-{i}"}
                for family in ("greebles", "pipework", "architecture")
                for i in range(4)]
        admission = {"source_sha": gate.NATIVE_SOURCE, "state": "native-admitted"}
        with mock.patch.object(gate.contracts, "frozen_manifest", return_value=admission), \
             mock.patch.object(gate.contracts.common, "load_training_corpus",
                               return_value=(rows, admission)) as loader:
            expected = REAL_FROZEN_TRAIN_ORDER(Path("synthetic-corpus"))
            self.assertEqual(len(expected), 12)
            self.assertEqual(set(expected), {r["id"] for r in rows})
            loader.assert_called_once_with(Path("synthetic-corpus"))
        rows[0]["family"] = "unreviewed"
        with mock.patch.object(gate.contracts, "frozen_manifest", return_value=admission), \
             mock.patch.object(gate.contracts.common, "load_training_corpus",
                               return_value=(rows, admission)):
            with self.assertRaisesRegex(ValueError, "TRAIN family"):
                REAL_FROZEN_TRAIN_ORDER(Path("synthetic-corpus"))

    def test_direct_delivery_does_not_request_forbidden_release(self):
        workflow = (SCRIPT.parents[4] / ".github/workflows/agent-painter-smol-native.yml").read_text()
        self.assertNotIn("gh release create", workflow)
        self.assertNotIn("contents: write", workflow)
        self.assertIn("contents: read", workflow)
        self.assertIn("name: smol-release-", workflow)
        self.assertIn("name: smol-gallery-", workflow)
        self.assertIn("path: release/", workflow)
        self.assertIn("if-no-files-found: error", workflow)
        self.assertIn('delivery": "github-actions-artifact"', workflow)
        self.assertIn('github_release_published": False', workflow)

    def test_actions_delivery_receipt_matches_source_and_hashes(self):
        """Execute only the embedded pure receipt code with synthetic files."""
        import os
        import subprocess
        import sys
        import textwrap

        workflow = (SCRIPT.parents[4] / ".github/workflows/agent-painter-smol-native.yml").read_text()
        marker = "      - name: Record exact-source read-only Actions delivery"
        self.assertEqual(workflow.count(marker), 1)
        block = workflow.split(marker, 1)[1].split(
            "      - uses: actions/upload-artifact@", 1)[0]
        self.assertEqual(block.count("python3 - <<'PY'"), 1)
        code = textwrap.dedent(block.split("python3 - <<'PY'\n", 1)[1].split(
            "\n          PY", 1)[0])
        env = {**os.environ, "GITHUB_SHA": SHA, "GITHUB_RUN_ID": "12345",
               "GITHUB_RUN_ATTEMPT": "1"}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "release"
            out.mkdir()
            manifest = {"source_sha": SHA, "workflow_run_id": 12345,
                        "workflow_run_attempt": 1,
                        "producer": {"training": "synthetic-test-only"},
                        "publisher": {"source": SHA}}
            (out / "MANIFEST.json").write_text(json.dumps(manifest))
            (out / "SHA256SUMS.txt").write_text("abc  placeholder\n")
            (out / "smol-gallery.html").write_text("<html>fixture only</html>")
            run = subprocess.run([sys.executable, "-c", code], cwd=tmp, env=env,
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            record_path = out / "PUBLICATION.json"
            receipt = json.loads(record_path.read_text())
            self.assertEqual(receipt["source_sha"], SHA)
            self.assertEqual(receipt["delivery"], "github-actions-artifact")
            self.assertFalse(receipt["github_release_published"])
            self.assertEqual(receipt["retention_days"], 90)
            self.assertEqual(receipt["bundle_artifact"], f"smol-release-{SHA}-12345-1")
            lines = (out / "SHA256SUMS.txt").read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[1], hashlib.sha256(record_path.read_bytes()).hexdigest()
                             + "  PUBLICATION.json")
            # A contradictory source must refuse without creating a delivery receipt.
            record_path.unlink()
            manifest["source_sha"] = "b" * 40
            (out / "MANIFEST.json").write_text(json.dumps(manifest))
            failed = subprocess.run([sys.executable, "-c", code], cwd=tmp, env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("source/run identity differs", failed.stderr)
            self.assertFalse(record_path.exists())


if __name__ == "__main__":
    unittest.main()
