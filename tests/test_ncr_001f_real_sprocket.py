"""NCR-001F Sprocket actual-host runner fail-closed source contract (offline).

These small sentinels do NOT claim any actual model conformance.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ncr_001f_real_sprocket import validate_source_contract
from ncr_001f_frozen_hooks import PINNED


class SprocketPinnedContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
        cls.pins = json.loads((ROOT / "experiments/ncr-001/source-pins.json").read_text())

    def run_bad(self, variant, patch):
        p = copy.deepcopy(self.protocol)
        s = copy.deepcopy(self.pins)
        patch(p, s)
        with self.assertRaises((RuntimeError, ValueError)):
            validate_source_contract(variant, p, s)

    def test_both_actual_checkpoint_variants_distinct_and_matched(self):
        for variant in ("chat", "instruct"):
            spec = validate_source_contract(variant, self.protocol, self.pins)
            self.assertEqual(spec["width"], 1280)
            self.assertEqual(spec["taps"], (7, 16, 25))
            self.assertEqual(spec["weight_path"].startswith("chat/"), variant == "chat")
        self.assertNotEqual(PINNED["sprocket_chat"]["weight_sha256"],
                            PINNED["sprocket_instruct"]["weight_sha256"])

    def test_training_admission_refused(self):
        self.run_bad("chat", lambda p, s: p.update(train_authorized=True))
        self.run_bad("chat", lambda p, s: p.pop("train_authorized"))

    def test_sibling_weight_substitution_refused(self):
        self.run_bad("chat", lambda p, s: s["models"]["sprocket_chat"].update(
            weight_declared_lfs_sha256_not_downloaded=s["models"]["sprocket_instruct"][
                "weight_declared_lfs_sha256_not_downloaded"]))
        self.run_bad("instruct", lambda p, s: s["models"]["sprocket_instruct"].update(
            weight_declared_lfs_sha256_not_downloaded=s["models"]["sprocket_chat"][
                "weight_declared_lfs_sha256_not_downloaded"]))

    def test_variant_and_source_path_refused(self):
        self.run_bad("chat", lambda p, s: p["models"]["sprocket_chat"].update(variant="instruct"))
        self.run_bad("instruct", lambda p, s: s["models"]["sprocket_instruct"].update(
            hf_weight_file="chat/model.safetensors"))

    def test_tap_geometry_or_architecture_refused(self):
        self.run_bad("instruct", lambda p, s: p["models"]["sprocket_instruct"].update(
            taps_after_block_zero_based=[6, 16, 25]))
        self.run_bad("chat", lambda p, s: s["models"]["sprocket_chat"]["architecture"].update(
            hidden=576))

    def test_wrong_revision_or_extra_model_refused(self):
        self.run_bad("chat", lambda p, s: s["models"]["sprocket_chat"].update(revision="main"))
        self.run_bad("chat", lambda p, s: p["models"].update(phantom={"name": "bad"}))

    def test_not_a_source_authority_for_unknown_variant(self):
        with self.assertRaises(ValueError):
            validate_source_contract("unknown", self.protocol, self.pins)


if __name__ == "__main__":
    unittest.main()
