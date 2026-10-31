"""Negative source/authorization tests for actual void.1 nontraining MLP prepass.

No checkpoint/tokenizer or held-out evaluation data are loaded in these tests.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ncr_001f_real_void import PINNED, source_contract, SOURCE_FILES, REVIEW_SHA


class PinnedVoidSourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (ROOT / "experiments/ncr-001/protocol.json").read_text(encoding="utf-8")
        )
        cls.pins = json.loads(
            (ROOT / "experiments/ncr-001/source-pins.json").read_text(encoding="utf-8")
        )

    def assert_refused(self, edit):
        protocol, pins = copy.deepcopy(self.protocol), copy.deepcopy(self.pins)
        edit(protocol, pins)
        with self.assertRaises(RuntimeError):
            source_contract(protocol, pins)

    def test_original_source_contract(self):
        pin = source_contract(self.protocol, self.pins)
        self.assertEqual(pin["variant"], "base_byte")
        self.assertEqual(pin["hf_weight_file"], "model.safetensors")
        self.assertEqual(pin["weight_bytes"], 360622744)
        self.assertEqual(PINNED["void1"]["model_type"], "qwen3")
        self.assertEqual(list(PINNED["void1"]["taps"]), [7, 15, 23])
        self.assertEqual(len(SOURCE_FILES), 6)
        self.assertEqual(REVIEW_SHA, "1fdc2e7e5c7b7b77923dbc4b47dcafa132aef47a")

    def test_refuse_admitted_training(self):
        self.assert_refused(lambda protocol, pins: protocol.__setitem__("train_authorized", True))

    def test_refuse_missing_fourth_lane(self):
        self.assert_refused(lambda protocol, pins: protocol["models"].pop("sprocket_instruct"))

    def test_refuse_sprocket_checkpoint_swap(self):
        self.assert_refused(lambda protocol, pins: pins["models"]["void1"].__setitem__(
            "weight_declared_lfs_sha256_not_downloaded",
            pins["models"]["sprocket_instruct"]["weight_declared_lfs_sha256_not_downloaded"]))

    def test_refuse_wrong_checkpoint_file(self):
        self.assert_refused(lambda protocol, pins: pins["models"]["void1"].__setitem__(
            "hf_weight_file", "chat/model.safetensors"))

    def test_refuse_wrong_variant(self):
        self.assert_refused(lambda protocol, pins: pins["models"]["void1"].__setitem__(
            "variant", "chat"))

    def test_refuse_wrong_taps(self):
        self.assert_refused(lambda protocol, pins: protocol["models"]["void1"].__setitem__(
            "taps_after_block_zero_based", [8, 16, 23]))

    def test_refuse_mismatched_tokenizer_sha(self):
        self.assert_refused(lambda protocol, pins: pins["models"]["void1"].__setitem__(
            "tokenization_byte_py_sha256", "0" * 64))

    def test_refuse_legacy_tokenizer_source_substitution(self):
        self.assert_refused(lambda protocol, pins: protocol["models"]["void1"][
            "tokenizer_reference"].__setitem__("train002b_source_sha256", "f" * 64))

    def test_refuse_changed_checkpoint_length(self):
        self.assert_refused(lambda protocol, pins: pins["models"]["void1"].__setitem__(
            "weight_bytes", 1002207984))


if __name__ == "__main__":
    unittest.main()
