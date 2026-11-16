"""MZ-016 deterministic model-surgery unit tests (no checkpoint download)."""
import unittest

import torch

from experiments.driv3r.model_zoo.spatial_paint_mz013.run import models
from experiments.driv3r.model_zoo.ternary_audit_mz016.audit import quantize, convert


class QuantizerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_ternary_rowwise_and_nonmutation(self):
        w = torch.tensor([[0.0, 1.0, -3.0, 5.0], [-2.0, 0.0, 0.0, 2.0]])
        saved = w.clone()
        q = quantize(w, bits=2, threshold=1.0)
        self.assertTrue(torch.equal(w, saved))
        for row in q:
            self.assertLessEqual(len(set(row.abs().tolist()) - {0.0}), 1)
        self.assertEqual(float(q[0, 0]), 0.0)
        self.assertEqual(float(q[0, 1]), 0.0)
        self.assertLess(float(q[0, 2]), 0)
        self.assertGreater(float(q[0, 3]), 0)

    def test_zero_rows_and_int_precision(self):
        w = torch.zeros(3, 7)
        for bits in (2, 4, 8):
            self.assertTrue(torch.isfinite(quantize(w, bits=bits)).all())
            self.assertEqual(int(torch.count_nonzero(quantize(w, bits=bits))), 0)
        with self.assertRaises(ValueError):
            quantize(w, bits=3)

    def test_independent_copy_and_scope(self):
        model = models(torch)["diagonal"]
        saved = {k:v.clone() for k,v in model.state_dict().items()}
        quant, details = convert(model, "diagonal", bits=2,
                                 threshold=0.5, selection="recurrent_1")
        self.assertEqual(details["matrix_names"], ["cells.1.proj.weight"])
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, saved[name]), name)
        self.assertTrue(torch.equal(quant.state_dict()["cells.0.proj.weight"],
                                          saved["cells.0.proj.weight"]))
        self.assertFalse(torch.equal(quant.state_dict()["cells.1.proj.weight"],
                                           saved["cells.1.proj.weight"]))

    def test_gru_state_gate_weights_are_selected(self):
        model = models(torch)["gru"]
        _, info = convert(model, "gru", bits=2, threshold=0.5,
                          selection="state_recurrence")
        self.assertEqual(len(info["matrix_names"]), 2)
        self.assertEqual(info["quantized_values"], 2*384*128)


if __name__ == "__main__":
    unittest.main()
