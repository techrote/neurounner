"""Model-free positive and adversarial checkpoint promotion controls for VOID."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from completion_gate import (CHECKPOINT, verify_completion as strict_verify,
                             TRAINING, CPU_EFFECTIVE, CORPUS, NATIVE,
                             TRAIN_SHA256, COMPILER_SHA256, TRAIN_IDS,
                             TRAIN_SEED_ORDER, TRAIN_SEED_ORDER_SHA256)

SOURCE = "d" * 40


def verify_completion(summary: dict, code: int, checkpoint: Path) -> dict:
    return strict_verify(summary, code, checkpoint, source_sha=SOURCE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root: Path) -> tuple[dict, Path]:
    checkpoint = root / CHECKPOINT
    (checkpoint / "adapter").mkdir(parents=True)
    (checkpoint / "adapter/adapter_model.safetensors").write_bytes(b"synthetic adapter fixture only")
    (checkpoint / "training_state.pt").write_bytes(b"synthetic optimizer fixture only")
    # Synthetic tensors and optimizer receipts, but the genuine native-admitted
    # seed-568 family-interleaved TRAIN order, not lexically sorted row IDs.
    order = list(TRAIN_SEED_ORDER)
    history = [{"global_step": i, "sample_cursor": i, "sample_id": order[(i-1) % 12],
                "gradient_norm": 1.0, "loss": 2.0-i/1000, "seconds": 0.01,
                "sequence_tokens": 750, "loss_tokens": 320,
                "prompt_bytes": 300, "completion_bytes": 319}
               for i in range(1, 121)]
    profile = json.loads((TRAINING / "model_profiles.json").read_text())["profiles"]["void"]
    config = {"schema_version": 1, "profile": "void", "source_sha": SOURCE,
              "native_source_sha": NATIVE, "corpus_manifest_sha256": CORPUS,
              "model": profile, "model_files": {
                  n: {"sha256": d, "bytes": 123} for n, d in profile["required_hashes"].items()},
              "trainer_sha256": sha(TRAINING / "train.py"),
              "contract_sha256": sha(TRAINING / "common.py"),
              "train_sha256": TRAIN_SHA256, "compiler_sha256": COMPILER_SHA256,
              "device": "cpu", "dtype": "float32", "threads": 1, "seed": 568,
              "max_length": 2048, "rank": 8, "alpha": 16, "dropout": 0.05,
              "learning_rate": 0.0005, "native_execution_by_trainer": False,
              "final_targets_opened": False, "sample_order": order,
              "numerics": {"profile": "portable-cpu-fp32-v1",
                           "effective": CPU_EFFECTIVE,
                           "numerics_sha256": sha(TRAINING / "numerics.py"),
                           "torch_version": "2.15.0+cpu",
                           "torch_build_sha256": "3"*64}}
    # An additional hash-verified upstream metadata entry is legitimate under
    # the shared model downloader's permissive audited-file allowlist.
    config["model_files"]["extra-metadata.json"] = {"sha256": "e"*64, "bytes": 64}
    summary = {"schema_version": 1, "status": "completed", "global_step": 120,
               "optimizer_steps_this_job": 120, "sample_cursor": 120,
               "checkpoint": CHECKPOINT, "actual_training": True,
               "checkpoint_readback_exact": True, "frozen_base_unchanged": True,
               "adapter_changed": True, "resume": None, "native_execution": False,
               "model_quality_claim": False, "config": config,
               "loss_tokens_this_job": 120*320, "sequence_tokens_this_job": 120*750,
               "bytes_this_job": 120*(300+319), "first_loss": history[0]["loss"],
               "last_loss": history[-1]["loss"],
               "numerical_runtime": {"synthetic": "model-free"},
               "frozen_base_sha256": "b" * 64,
               "adapter_before_sha256": "a" * 64, "adapter_after_sha256": "c" * 64,
               "sample_ids_this_job": [e["sample_id"] for e in history]}
    receipt = {"global_step": 120, "sample_cursor": 120, "config": summary["config"],
               "frozen_base_sha256": summary["frozen_base_sha256"], "history": history,
               "numerical_runtime": summary["numerical_runtime"]}
    (checkpoint / "receipt.json").write_text(json.dumps(receipt))
    (root / "progress.jsonl").write_text("".join(json.dumps(e, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n" for e in history))
    inventory = {p.relative_to(checkpoint).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
                 for p in sorted(checkpoint.rglob("*")) if p.is_file()}
    (checkpoint / "inventory.json").write_text(json.dumps(inventory))
    summary["checkpoint_inventory_sha256"] = sha(checkpoint / "inventory.json")
    return summary, checkpoint


class GateTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.summary, self.checkpoint = fixture(Path(self._temp.name))

    def test_positive_model_free_fixture(self):
        result = verify_completion(self.summary, 0, self.checkpoint)
        self.assertTrue(result["promotable"])
        self.assertEqual(result["verified_optimizer_receipts"], 120)
        self.assertEqual(result["verified_checkpoint_files"], 3)
        self.assertEqual(set(self.summary["config"]["sample_order"]), TRAIN_IDS)
        self.assertEqual(self.summary["config"]["sample_order"], list(TRAIN_SEED_ORDER))
        digest = hashlib.sha256(json.dumps(list(TRAIN_SEED_ORDER), sort_keys=True,
                               separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(digest, TRAIN_SEED_ORDER_SHA256)

    def test_frozen_native_train_and_compiler_identity_refused(self):
        # All digests are syntactically valid; no self-consistent summary may
        # substitute a different native TRAIN split or compiler.
        for field in ("train_sha256", "compiler_sha256"):
            with self.subTest(field=field):
                bad = json.loads(json.dumps(self.summary))
                bad["config"][field] = "f" * 64
                with self.assertRaisesRegex(ValueError, "frozen native TRAIN/compiler"):
                    verify_completion(bad, 0, self.checkpoint)
        for change in (
                lambda a: a.__setitem__(0, "unknown-train-id"),
                lambda a: a.__setitem__(0, a[1]),
                lambda a: a.pop(0),
        ):
            with self.subTest(change=change.__code__.co_firstlineno):
                bad = json.loads(json.dumps(self.summary))
                change(bad["config"]["sample_order"])
                with self.assertRaisesRegex(ValueError, "native-admitted frozen TRAIN rows"):
                    verify_completion(bad, 0, self.checkpoint)

    def test_self_consistent_swapped_native_train_order_refused(self):
        # A malicious or defective producer can self-consistently rewrite all
        # 120 history events and progress bytes to a permutation of real IDs.
        # That must still be impossible for the frozen seed568 shared sampler.
        bad = json.loads(json.dumps(self.summary))
        swapped = list(TRAIN_SEED_ORDER)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        self.assertEqual(set(swapped), TRAIN_IDS)
        bad["config"]["sample_order"] = swapped
        bad["sample_ids_this_job"] = [swapped[i % 12] for i in range(120)]
        receipt_path = self.checkpoint / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["config"] = bad["config"]
        for i, event in enumerate(receipt["history"]):
            event["sample_id"] = bad["sample_ids_this_job"][i]
        receipt_path.write_text(json.dumps(receipt))
        progress_path = self.checkpoint.parent / "progress.jsonl"
        progress_path.write_text("".join(
            json.dumps(event, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n"
            for event in receipt["history"]))
        inventory_path = self.checkpoint / "inventory.json"
        inventory = json.loads(inventory_path.read_text())
        inventory["receipt.json"] = {
            "sha256": sha(receipt_path), "bytes": receipt_path.stat().st_size}
        inventory_path.write_text(json.dumps(inventory))
        bad["checkpoint_inventory_sha256"] = sha(inventory_path)
        self.assertEqual(inventory["receipt.json"]["sha256"], sha(receipt_path))
        with self.assertRaisesRegex(ValueError, "native seed568 TRAIN sampler order"):
            verify_completion(bad, 0, self.checkpoint)

    def test_partial_and_nonzero_exit_refused(self):
        for code in (1, 2, 124, 137):
            with self.subTest(code=code), self.assertRaisesRegex(ValueError, "partial/failure"):
                verify_completion(self.summary, code, self.checkpoint)
        partial = dict(self.summary, status="stopped-checkpoint-saved", global_step=40,
                       sample_cursor=40, optimizer_steps_this_job=40, checkpoint="checkpoint-000040")
        with self.assertRaisesRegex(ValueError, "status"):
            verify_completion(partial, 0, self.checkpoint)

    def test_all_summary_promotion_fields_mandatory(self):
        overrides = {"schema_version": 2, "status": "incomplete", "global_step": 119,
                     "optimizer_steps_this_job": 119, "sample_cursor": 119,
                     "checkpoint": "checkpoint-000040", "actual_training": False,
                     "checkpoint_readback_exact": False, "frozen_base_unchanged": False,
                     "adapter_changed": False, "resume": {"from_step": 40},
                     "native_execution": True, "model_quality_claim": True}
        for key, value in overrides.items():
            with self.subTest(key=key):
                bad = dict(self.summary, **{key: value})
                with self.assertRaisesRegex(ValueError, key):
                    verify_completion(bad, 0, self.checkpoint)
                del bad[key]
                with self.assertRaisesRegex(ValueError, key):
                    verify_completion(bad, 0, self.checkpoint)

    def test_boolean_or_forged_adapter_change_refused(self):
        for key in ("global_step", "sample_cursor", "optimizer_steps_this_job"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_completion(dict(self.summary, **{key: True}), 0, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "adapter change"):
            verify_completion(dict(self.summary, adapter_after_sha256="a"*64), 0, self.checkpoint)

    def test_missing_or_symlinked_checkpoint_refused(self):
        with self.assertRaisesRegex(ValueError, "directory"):
            verify_completion(self.summary, 0, self.checkpoint.parent / "checkpoint-000040")
        alias = self.checkpoint.parent / "alias"
        alias.symlink_to(self.checkpoint, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "directory"):
            verify_completion(self.summary, 0, alias)

    def test_truncated_or_mutated_adapter_and_state_refused(self):
        for name in ("adapter/adapter_model.safetensors", "training_state.pt"):
            with self.subTest(name=name):
                p = self.checkpoint / name
                previous = p.read_bytes()
                p.write_bytes(previous + b"mutated")
                with self.assertRaisesRegex(ValueError, "hash or length"):
                    verify_completion(self.summary, 0, self.checkpoint)
                p.write_bytes(previous)

    def test_inventory_summary_mismatch_refused(self):
        bad = dict(self.summary, checkpoint_inventory_sha256="0"*64)
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            verify_completion(bad, 0, self.checkpoint)

    def test_manifest_path_traversal_or_extra_file_refused(self):
        extra = self.checkpoint / "unexpected.dat"
        extra.write_bytes(b"unknown")
        with self.assertRaisesRegex(ValueError, "inventory disagree"):
            verify_completion(self.summary, 0, self.checkpoint)
        extra.unlink()
        inventory = json.loads((self.checkpoint / "inventory.json").read_text())
        inventory["../escaping"] = {"bytes": 1, "sha256": "f"*64}
        (self.checkpoint / "inventory.json").write_text(json.dumps(inventory))
        self.summary["checkpoint_inventory_sha256"] = sha(self.checkpoint / "inventory.json")
        with self.assertRaises(ValueError):
            verify_completion(self.summary, 0, self.checkpoint)

    def test_receipt_history_mismatch_or_nonfinite_gradient_refused(self):
        p = self.checkpoint / "receipt.json"
        orig = p.read_text()
        for transform in (lambda r: r["history"].pop(),
                          lambda r: r["history"][119].update(global_step=119),
                          lambda r: r["history"][0].update(gradient_norm=0),
                          lambda r: r.update(sample_cursor=40)):
            receipt = json.loads(orig)
            transform(receipt)
            p.write_text(json.dumps(receipt))
            inventory = json.loads((self.checkpoint / "inventory.json").read_text())
            inventory["receipt.json"] = {"sha256": sha(p), "bytes": p.stat().st_size}
            (self.checkpoint / "inventory.json").write_text(json.dumps(inventory))
            self.summary["checkpoint_inventory_sha256"] = sha(self.checkpoint / "inventory.json")
            with self.assertRaises(ValueError):
                verify_completion(self.summary, 0, self.checkpoint)
        p.write_text(orig)



    def test_additional_hashed_upstream_metadata_allowed(self):
        # The audited downloader permits extra matched public metadata files.
        self.assertTrue(verify_completion(self.summary, 0, self.checkpoint)["promotable"])
        self.summary["config"]["model_files"]["extra-metadata.json"]["sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "model snapshot"):
            verify_completion(self.summary, 0, self.checkpoint)

    def test_source_model_native_and_numerical_substitutions_refused(self):
        changes = [("source_sha", "0"*40), ("native_source_sha", "0"*40),
                   ("corpus_manifest_sha256", "0"*64), ("trainer_sha256", "0"*64),
                   ("threads", 2), ("final_targets_opened", True)]
        for key, new in changes:
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, "identity|contract"):
                bad = json.loads(json.dumps(self.summary))
                bad["config"][key] = new
                verify_completion(bad, 0, self.checkpoint)
        for section, key, replacement in (
                ("model", "revision", "0"*40), ("model_files", "model.safetensors", None),
                ("numerics", "profile", "fake"), ("numerics", "numerics_sha256", "0"*64)):
            with self.subTest(field=key), self.assertRaises(ValueError):
                bad = json.loads(json.dumps(self.summary))
                if section == "model_files":
                    bad["config"][section][key]["sha256"] = "0"*64
                else:
                    bad["config"][section][key] = replacement
                verify_completion(bad, 0, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "source"):
            strict_verify(self.summary, 0, self.checkpoint, source_sha="wrong")

    def test_missing_modified_or_short_progress_refused(self):
        p = self.checkpoint.parent / "progress.jsonl"
        original = p.read_text()
        p.unlink()
        with self.assertRaisesRegex(ValueError, "progress"):
            verify_completion(self.summary, 0, self.checkpoint)
        p.write_text(original)
        rows = [json.loads(line) for line in original.splitlines()]
        rows[5]["loss"] += 0.5
        p.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "differs"):
            verify_completion(self.summary, 0, self.checkpoint)
        p.write_text("\n".join(original.splitlines()[:119]) + "\n")
        with self.assertRaisesRegex(ValueError, "exactly 120"):
            verify_completion(self.summary, 0, self.checkpoint)
        p.write_text(original)

    def test_semantically_equal_but_noncanonical_optimizer_bytes_refused(self):
        p = self.checkpoint.parent / "progress.jsonl"
        original = p.read_text()
        # Whitespace-changing rewrite is semantically equal, but is not the
        # exact canonical line emitted by the qualified shared trainer.
        p.write_text("".join(json.dumps(json.loads(line)) + "\n"
                             for line in original.splitlines()))
        with self.assertRaisesRegex(ValueError, "bytes"):
            verify_completion(self.summary, 0, self.checkpoint)
        p.write_text(original)

    def test_missing_loss_and_supervised_token_receipts_refused(self):
        p = self.checkpoint / "receipt.json"
        baseline = p.read_text()
        for change in (lambda x: x["history"][0].pop("loss_tokens"),
                       lambda x: x["history"][0].update(loss=float("nan")),
                       lambda x: x["history"][0].update(loss_tokens=0),
                       lambda x: x["history"][0].update(sample_id="wrong")):
            r = json.loads(baseline)
            change(r)
            p.write_text(json.dumps(r))
            inventory = self.checkpoint / "inventory.json"
            manifest = json.loads(inventory.read_text())
            manifest["receipt.json"] = {"sha256": sha(p), "bytes": p.stat().st_size}
            inventory.write_text(json.dumps(manifest))
            bad = dict(self.summary, checkpoint_inventory_sha256=sha(inventory))
            with self.assertRaises(ValueError):
                verify_completion(bad, 0, self.checkpoint)
        p.write_text(baseline)

if __name__ == "__main__":
    unittest.main()
