"""Source-only adversarial tests for the post-lease real-host witness.

No downloaded model or optimizer. Original model execution is in the separate
draft-PR Actions producer and cold consumer jobs.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ncr_001_postlease_sprocket import (  # noqa: E402
    LANE_WEIGHTS, PINNED, REQUIRED_LANES, verify_contract,
)


class PostLeaseSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (ROOT / "experiments/ncr-001/protocol.json").read_text())
        cls.pins = json.loads(
            (ROOT / "experiments/ncr-001/source-pins.json").read_text())

    def verify(self, variant, protocol=None, pins=None):
        return verify_contract(variant, protocol or self.protocol,
                               pins or self.pins)

    def test_real_chat_and_distinct_root_instruct(self):
        self.assertEqual(set(self.protocol["models"]), REQUIRED_LANES)
        self.assertEqual(set(self.pins["models"]), REQUIRED_LANES)
        self.assertIs(self.protocol["train_authorized"], False)
        for variant in LANE_WEIGHTS:
            with self.subTest(variant=variant):
                spec = self.verify(variant)
                self.assertEqual(spec["weight_path"], LANE_WEIGHTS[variant][0])
        self.assertNotEqual(LANE_WEIGHTS["chat"][1],
                            LANE_WEIGHTS["instruct"][1])

    def test_refuse_open_training_gate(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["train_authorized"] = True
        for variant in LANE_WEIGHTS:
            with self.assertRaisesRegex(RuntimeError, "Training gate"):
                verify_contract(variant, protocol, self.pins)

    def test_refuse_sibling_checkpoint_substitution(self):
        pins = copy.deepcopy(self.pins)
        pins["models"]["sprocket_chat"]["weight_declared_lfs_sha256_not_downloaded"] = (
            LANE_WEIGHTS["instruct"][1])
        with self.assertRaisesRegex(RuntimeError, "checkpoint"):
            verify_contract("chat", self.protocol, pins)
        pins = copy.deepcopy(self.pins)
        pins["models"]["sprocket_instruct"]["hf_weight_file"] = (
            "chat/model.safetensors")
        with self.assertRaisesRegex(RuntimeError, "checkpoint"):
            verify_contract("instruct", self.protocol, pins)

    def test_refuse_incorrect_taps_and_geometry(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["models"]["sprocket_instruct"]["taps_after_block_zero_based"] = [
            7, 15, 25]
        with self.assertRaisesRegex(RuntimeError, "geometry"):
            verify_contract("instruct", protocol, self.pins)
        protocol = copy.deepcopy(self.protocol)
        protocol["models"]["sprocket_chat"]["expected_config"]["hidden_size"] = 1024
        with self.assertRaisesRegex(RuntimeError, "geometry"):
            verify_contract("chat", protocol, self.pins)

    def test_refuse_missing_fourth_lane_and_wrong_identity(self):
        pins = copy.deepcopy(self.pins)
        del pins["models"]["void1"]
        with self.assertRaisesRegex(RuntimeError, "Source pins"):
            verify_contract("chat", self.protocol, pins)
        with self.assertRaisesRegex(RuntimeError, "Unknown Sprocket"):
            self.verify("void1")

    def test_refuse_mutated_source_revision(self):
        pins = copy.deepcopy(self.pins)
        pins["models"]["sprocket_chat"]["revision"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "source contract"):
            verify_contract("chat", self.protocol, pins)

    def test_no_training_semantics_in_runner(self):
        text = (ROOT / "scripts/ncr_001_postlease_sprocket.py").read_text()
        self.assertIn('"optimizer_steps": 0', text)
        self.assertIn('"train_authorized": False', text)
        self.assertIn('"heldout_opened": False', text)
        self.assertIn('"actual_host_final_admission": False', text)
        self.assertNotIn("optimizer.step(", text)
        self.assertNotIn(".backward(", text)


if __name__ == "__main__":
    unittest.main()
