"""Invariants for the MZ-014 legal-action coordinate-frame intervention."""
import unittest
import torch
from experiments.driv3r.model_zoo.spatial_paint_mz013 import run as parent
from experiments.driv3r.model_zoo.frame_audit_mz014.audit import transform, MODES


class FrameAuditTests(unittest.TestCase):
    def test_transform_is_bijective_on_action_and_patch(self):
        w = parent.World(7000, 1, 0, ood=True)
        for _, (h, v, _) in MODES.items():
            z, perm = transform(w.observe(), w, h, v)
            self.assertEqual(sorted(perm), list(range(8)))
            mask = z[131:139]
            original = w.legal()
            for a in range(8):
                self.assertEqual(int(mask[perm[a]]), original[a])
            self.assertEqual(tuple(z.shape), (parent.D,))
            self.assertTrue(torch.equal(z[147:155],
                                         torch.tensor(w.observe()[147:155])))
    def test_180_double_transform_recovers_categorical_patch(self):
        w = parent.World(7001, 0, 1, ood=True)
        original = w.observe()
        z, _ = transform(original, w, True, True)
        patch = z[:125].reshape(5, 5, 5).flip(0).flip(1)
        self.assertTrue(torch.equal(patch.flatten(),
                                    torch.tensor(original[:125])))
    def test_parent_provenance(self):
        from experiments.driv3r.model_zoo.frame_audit_mz014.audit import PARENT_SHA
        self.assertEqual(PARENT_SHA,
                         "66b30e6fd3799bc331f2ce066ed106720bbcb262")
        self.assertEqual(parent.SCHEMA, "MZ013-SPATIAL-PAINT-v1")


if __name__ == "__main__":
    unittest.main()
