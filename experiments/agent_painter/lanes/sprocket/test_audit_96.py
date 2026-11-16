#!/usr/bin/env python3
"""Synthetic model-free adversarial tests for the immutable Sprocket fresh96 audit."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_96 import (SOURCE, MODEL, REVISION, NATIVE_SOURCE, CORPUS_SHA,
                      TRAIN_SHA, COMPILER_SHA, RUN, STEPS, validate, expected_native_sampler)


def fixture():
    order = [f"TRAIN-{i:02d}" for i in range(12)]
    config = {
        "source_sha": SOURCE, "profile": "sprocket",
        "native_source_sha": NATIVE_SOURCE, "corpus_manifest_sha256": CORPUS_SHA,
        "train_sha256": TRAIN_SHA, "compiler_sha256": COMPILER_SHA,
        "model": {"model_id": MODEL, "revision": REVISION, "subfolder": "", "trust_remote_code": False},
        "device": "cpu", "dtype": "float32", "threads": 1, "seed": 568,
        "max_length": 2048, "rank": 8, "alpha": 16, "dropout": 0.05,
        "numerics": {"profile": "portable-cpu-fp32-v1"},
        "native_execution_by_trainer": False, "final_targets_opened": False,
        "sample_order": order,
    }
    progress = [{"global_step": i + 1, "sample_cursor": i + 1,
                 "sample_id": order[i % len(order)], "loss": 1.0 / (i + 1),
                 "gradient_norm": 0.25, "loss_tokens": 100, "sequence_tokens": 200}
                for i in range(STEPS)]
    summary = {
        "schema_version": 1, "status": "completed", "actual_training": True,
        "global_step": STEPS, "optimizer_steps_this_job": STEPS,
        "sample_cursor": STEPS, "resume": None, "checkpoint": "checkpoint-000096",
        "checkpoint_readback_exact": True, "adapter_changed": True,
        "frozen_base_unchanged": True, "adapter_before_sha256": "a" * 64,
        "adapter_after_sha256": "b" * 64, "frozen_base_sha256": "c" * 64,
        "native_execution": False, "model_quality_claim": False,
        "config": config, "model_contract": {"trainable_parameters": 4739072},
        "numerical_runtime": {"version": "v1"},
        "sample_ids_this_job": [x["sample_id"] for x in progress],
        "loss_tokens_this_job": 9600, "sequence_tokens_this_job": 19200,
        "first_loss": 1.0, "last_loss": 1.0 / STEPS,
        "runner": {"github_run_id": RUN},
    }
    preflight = {"config": config, "resume": None, "train_rows": 12,
                 "trainable_parameters": 4739072, "model_contract": summary["model_contract"],
                 "numerical_runtime": summary["numerical_runtime"]}
    checkpoint = {"config": config, "global_step": STEPS,
                  "sample_cursor": STEPS, "frozen_base_sha256": "c" * 64,
                  "history": progress}
    campaign = {
        "schema": "cybersand.agent-painter.sprocket-campaign/1",
        "experiment": "sprocket-fresh-96-001", "source_sha": SOURCE,
        "model": MODEL, "revision": REVISION, "corpus_manifest_sha256": CORPUS_SHA,
        "target_global_steps": STEPS, "actual_global_steps": STEPS,
        "optimizer_steps_this_job": STEPS, "result_code": 0,
        "failure": None, "status": "completed", "actual_training": True,
        "checkpoint": "checkpoint-000096", "checkpoint_readback_exact": True,
        "native_execution": False, "model_quality_claim": False,
        "final_targets_opened": False,
    }
    return summary, preflight, checkpoint, progress, campaign


class HistoricalProofGate(unittest.TestCase):
    def test_genuine_complete_shape_passes(self):
        validate(*fixture())

    def test_partial_or_falsely_complete_rejected(self):
        for key, value in [("status", "stopped-checkpoint-saved"),
                           ("actual_training", False), ("global_step", 80),
                           ("optimizer_steps_this_job", 80), ("sample_cursor", 80),
                           ("checkpoint_readback_exact", False),
                           ("adapter_changed", False), ("resume", {"from_step": 16}),
                           ("native_execution", True), ("model_quality_claim", True)]:
            v = deepcopy(fixture())
            v[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(*v)

    def test_source_model_corpus_and_numerical_substitution_rejected(self):
        for key, value in [
            ("source_sha", "0" * 40), ("native_source_sha", "1" * 40),
            ("corpus_manifest_sha256", "2" * 64),
            ("train_sha256", "1" * 64),
            ("compiler_sha256", "2" * 64),
            ("model", {"model_id": "substituted", "revision": REVISION, "subfolder": ""}),
            ("numerics", {"profile": "faster-than-reviewed"}),
            ("final_targets_opened", True), ("threads", 2),
        ]:
            v = deepcopy(fixture())
            for obj in (v[0]["config"], v[1]["config"], v[2]["config"]):
                obj[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(*v)

    def test_step_history_and_sample_order_rejected(self):
        for kind in ["truncated", "extra", "loss_nan", "zero_gradient",
                     "wrong_step", "different_sample", "disagree_checkpoint",
                     "incorrect_token_total", "missing_loss_tokens"]:
            v = deepcopy(fixture())
            if kind == "truncated":
                v[3].pop()
            elif kind == "extra":
                v[3].append(dict(v[3][-1]))
            elif kind == "loss_nan":
                v[3][0]["loss"] = float("nan")
            elif kind == "zero_gradient":
                v[3][8]["gradient_norm"] = 0
            elif kind == "wrong_step":
                v[3][6]["global_step"] = 99
            elif kind == "different_sample":
                v[3][2]["sample_id"] = "unexpected"
            elif kind == "disagree_checkpoint":
                v[2]["history"].pop()
            elif kind == "incorrect_token_total":
                v[0]["loss_tokens_this_job"] = 0
            else:
                v[3][0]["loss_tokens"] = 0
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate(*v)

    def test_false_wrapper_success_and_cross_run_rejected(self):
        for key, value in [("result_code", 2), ("status", "stopped-checkpoint-saved"),
                           ("actual_global_steps", 72), ("failure", {"timeout": True}),
                           ("native_execution", True), ("model_quality_claim", True)]:
            v = deepcopy(fixture())
            v[4][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(*v)
        v = fixture()
        v[0]["runner"]["github_run_id"] = "43529482590"
        with self.assertRaises(ValueError):
            validate(*v)

    def test_original_native_sampler_and_relabel_refusal(self):
        rows = [
            {"id": f"{family}-{group}-s{seed}", "family": family,
             "split": "TRAIN", "group_id": f"{family}-{group}"}
            for family in ("greebles", "pipework", "architecture")
            for group in ("first", "second") for seed in (104729, 130363)
        ]
        first = expected_native_sampler(rows, 568)
        self.assertEqual(len(first), 12)
        self.assertEqual(len(set(first)), 12)
        self.assertEqual(first, expected_native_sampler(list(reversed(rows)), 568))
        self.assertNotEqual(first, expected_native_sampler(rows, 569))
        bad = deepcopy(rows)
        bad[0]["split"] = "FINAL"
        with self.assertRaises(ValueError):
            expected_native_sampler(bad, 568)
        bad = deepcopy(rows)
        bad[0]["id"] = bad[1]["id"]
        with self.assertRaises(ValueError):
            expected_native_sampler(bad, 568)
        bad = deepcopy(rows)
        bad[0]["family"] = "architecture"
        with self.assertRaises(ValueError):
            expected_native_sampler(bad, 568)

    def test_historical_workflow_is_model_free_read_only(self):
        path = Path(__file__).resolve().parents[4] / ".github/workflows/agent-painter-sprocket-audit.yml"
        s = path.read_text(encoding="utf-8")
        self.assertIn("audit_96.py", s)
        self.assertIn("test_audit_96.py", s)
        self.assertIn("43529482591", s)
        self.assertIn("43506373969", s)
        self.assertIn("agent-painter-native-633920373a3d406a2ede61ce4137c324f76be23b", s)
        self.assertIn("--corpus evidence/original-native/corpus", s)
        self.assertIn("contents: read", s)
        self.assertNotIn("download_model.py", s)
        self.assertNotIn("train.py", s)
        self.assertNotIn("contents: write", s)
        self.assertNotIn("cancel-in-progress: true", s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
