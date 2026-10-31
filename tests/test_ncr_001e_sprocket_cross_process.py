"""Offline mutation sentinels for the Sprocket independent-process handoff.

No model execution, no sealed fixture, no optimizer and no checkpoint required.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ncr_001e_sprocket_cross_process import (
    PINS, PROTOCOL, SOURCE_REPO, SOURCE_REV, SOURCE_SPECS,
    filehash, guard_manifest,
)


class CrossProcessSourceGuardTests(unittest.TestCase):
    def setUp(self):
        self.pins = json.loads(PINS.read_text(encoding="utf-8"))["models"]
        self.variant = "chat"
        self.pin = self.pins["sprocket_chat"]
        self.spec = SOURCE_SPECS["chat"]
        self.producer_pid = os.getpid() + 123456
        self.manifest = {
            "schema": 1,
            "variant": self.variant,
            "repo": SOURCE_REPO,
            "revision": SOURCE_REV,
            "checkpoint_file": self.pin["hf_weight_file"],
            "checkpoint_sha256": self.pin["weight_declared_lfs_sha256_not_downloaded"],
            "implementation_commit": self.spec["implementation_commit"],
            "protocol_sha256": filehash(PROTOCOL),
            "pins_sha256": filehash(PINS),
            "producer_pid": self.producer_pid,
            "training": False,
            "optimizer_steps": 0,
            "heldout_evaluation": False,
            "tool_and_memory_behavior_qualified": False,
            "adapter_sha256": "a" * 64,
            "metadata_sha256": "b" * 64,
            "reference_sha256": "c" * 64,
            "token_ids_sha256": "d" * 64,
        }

    def validate(self, manifest=None):
        guard_manifest(
            manifest if manifest is not None else self.manifest,
            variant=self.variant, pin=self.pin,
            implementation_commit=self.spec["implementation_commit"],
            producer_pid=self.producer_pid,
        )

    def test_positive_identity_is_preliminary_not_training(self):
        self.assertIs(json.loads(PROTOCOL.read_text())["train_authorized"], False)
        self.assertNotEqual(
            self.pins["sprocket_chat"]["weight_declared_lfs_sha256_not_downloaded"],
            self.pins["sprocket_instruct"]["weight_declared_lfs_sha256_not_downloaded"],
        )
        self.validate()

    def test_wrong_variant_or_weight_file_is_rejected(self):
        for field, value in (
            ("variant", "instruct"), ("checkpoint_file", "model.safetensors"),
            ("checkpoint_sha256", self.pins["sprocket_instruct"][
                "weight_declared_lfs_sha256_not_downloaded"]),
            ("repo", "unverified/sprocket"), ("revision", "main"),
        ):
            with self.subTest(field=field):
                changed = dict(self.manifest, **{field: value})
                with self.assertRaises(RuntimeError):
                    self.validate(changed)

    def test_original_hook_or_protocol_drift_is_rejected(self):
        for field in ("implementation_commit", "protocol_sha256", "pins_sha256"):
            with self.subTest(field=field):
                with self.assertRaises(RuntimeError):
                    self.validate(dict(self.manifest, **{field: "0" * 40}))

    def test_same_interpreter_is_never_a_cold_restart(self):
        with self.assertRaises(RuntimeError):
            guard_manifest(
                dict(self.manifest, producer_pid=os.getpid()),
                variant="chat", pin=self.pin,
                implementation_commit=self.spec["implementation_commit"],
                producer_pid=os.getpid(),
            )

    def test_false_training_or_semantic_claim_is_rejected(self):
        for field, value in (
            ("training", True), ("optimizer_steps", 1),
            ("heldout_evaluation", True),
            ("tool_and_memory_behavior_qualified", True),
        ):
            with self.subTest(field=field):
                with self.assertRaises(RuntimeError):
                    self.validate(dict(self.manifest, **{field: value}))

    def test_missing_or_malformed_artifact_digest_is_rejected(self):
        for field in (
            "adapter_sha256", "metadata_sha256", "reference_sha256",
            "token_ids_sha256",
        ):
            for value in ("x" * 64, "0" * 63, None):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(RuntimeError):
                        self.validate(dict(self.manifest, **{field: value}))


if __name__ == "__main__":
    unittest.main()
