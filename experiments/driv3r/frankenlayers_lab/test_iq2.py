"""Source-only tests for calibrated IQ2_XS with immutable Q8 and calibration provenance."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.driv3r.backend import PINS
from experiments.driv3r.frankenlayers_lab.calibration import ACTIONS, CHUNKS, ROWS, SEEDS, build_calibration
from experiments.driv3r.frankenlayers_lab.infer_iq2 import MODES, validate_iq2_receipt

class IQ2ScreenTests(unittest.TestCase):
    def test_screen_excludes_unqualified_layers(self):
        self.assertEqual(MODES, ("aero4_no10", "aero4_no2"))

    def test_calibration_is_deterministic_train_only_and_legally_scoped(self):
        data = build_calibration()
        self.assertEqual(data, build_calibration())
        self.assertEqual(ROWS, 96)
        self.assertEqual(CHUNKS, 8)
        self.assertTrue(set(SEEDS).isdisjoint({570, 571}))
        self.assertGreater(len(data.encode()), 15000)
        self.assertEqual(data.count("<|im_start|>"), ROWS)
        self.assertIn("legal_actions", data)
        self.assertIn("protected_cells", data)
        self.assertIn("RETRACE", data)
        self.assertIn("BRANCH", data)
        self.assertNotIn("greebles", data.lower())
        self.assertNotIn("pipework", data.lower())
        self.assertNotIn("architecture", data.lower())
        self.assertEqual(len(ACTIONS), 6)

    def test_receipt_checks_all_model_and_calibration_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            q8 = Path(tmp) / "q8.gguf"
            iq2 = Path(tmp) / "iq2.gguf"
            imatrix = Path(tmp) / "imatrix.gguf"
            calibration = Path(tmp) / "calibration.txt"
            q8.write_bytes(b"immutable-test-donor")
            with iq2.open("wb") as f:
                f.truncate(1_000_001)
            imatrix.write_bytes(b"i" * 1001)
            calibration.write_bytes(build_calibration().encode())
            receipt = {
                "schema": "frankenlayers-iq2-requant/2",
                "quant": "IQ2_XS", "allow_requantize": True,
                "q8_sha256": PINS["model"]["sha256"],
                "iq2_sha256": "test-iq2-bytes",
                "imatrix_source": "Q8 donor, train-only synthetic game prompts",
                "calibration_chunks": CHUNKS,
                "imatrix_sha256": hashlib.sha256(imatrix.read_bytes()).hexdigest(),
                "calibration_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
            }
            def digest(path):
                if path == q8: return PINS["model"]["sha256"]
                if path == iq2: return "test-iq2-bytes"
                return hashlib.sha256(path.read_bytes()).hexdigest()
            with patch("experiments.driv3r.frankenlayers_lab.infer_iq2.sha256_file",
                       side_effect=digest):
                validate_iq2_receipt(receipt, q8, iq2, imatrix, calibration)
                for field,bad in (("quant", "Q4_K_M"),("allow_requantize", False),
                    ("q8_sha256", "wrong"), ("iq2_sha256", "wrong"),
                    ("schema", "frankenlayers-iq2-requant/1"),
                    ("imatrix_source", "unverified"), ("calibration_chunks", 0),
                    ("imatrix_sha256", "wrong"), ("calibration_sha256", "wrong")):
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        validate_iq2_receipt(dict(receipt, **{field:bad}),
                                             q8, iq2, imatrix, calibration)
                imatrix.unlink()
                with self.assertRaises(ValueError):
                    validate_iq2_receipt(receipt, q8, iq2, imatrix, calibration)

if __name__ == "__main__":
    unittest.main()
