"""NCR-001E isolated SmolLM2 shared-backbone regression; no original model or training."""
from __future__ import annotations

import threading
import unittest

import torch

from test_ncr_001_smollm2 import ToyLM
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ncr_001_smollm2 import FrozenSmolLM2DepthNCA


class SharedBackboneIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(20261028)
        self.backbone = ToyLM()
        self.a = FrozenSmolLM2DepthNCA(self.backbone)
        self.b = FrozenSmolLM2DepthNCA(self.backbone)
        self.ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    def test_wrappers_share_exact_backbone_lock(self):
        self.assertIs(self.a._backbone_lock, self.b._backbone_lock)
        other = FrozenSmolLM2DepthNCA(ToyLM())
        self.assertIsNot(self.a._backbone_lock, other._backbone_lock)

    def test_active_owner_blocks_other_enabled_disabled_and_generate(self):
        with self.a._claim_backbone():
            for enabled in (True, False):
                with self.subTest(enabled=enabled):
                    with self.assertRaisesRegex(RuntimeError, "Shared frozen SmolLM2"):
                        self.b(self.ids, enabled=enabled)
            for enabled in (True, False):
                with self.subTest(generate_enabled=enabled):
                    with self.assertRaisesRegex(RuntimeError, "Shared frozen SmolLM2"):
                        self.b.generate(self.ids, enabled=enabled, max_new_tokens=1)
        self.assertTrue(torch.isfinite(self.b(self.ids, enabled=False).logits).all())
        self.assertTrue(torch.isfinite(self.b(self.ids).logits).all())

    def test_reference_holder_blocks_instrumented_wrapper(self):
        with self.a._claim_backbone():
            with self.assertRaisesRegex(RuntimeError, "Shared frozen SmolLM2"):
                self.b(self.ids)
        source = self.a(self.ids, enabled=False).logits
        no_op = self.b(self.ids).logits
        self.assertTrue(torch.equal(source, no_op))

    def test_distinct_original_model_instances_are_independent(self):
        independent = FrozenSmolLM2DepthNCA(ToyLM())
        with self.a._claim_backbone():
            self.assertTrue(torch.isfinite(independent(self.ids).logits).all())
            self.assertTrue(torch.isfinite(independent(self.ids, enabled=False).logits).all())

    def test_nested_same_wrapper_is_refused_and_later_recovers(self):
        with self.a._claim_backbone():
            with self.assertRaisesRegex(RuntimeError, "Shared frozen SmolLM2"):
                self.a(self.ids, enabled=False)
            with self.assertRaisesRegex(RuntimeError, "Shared frozen SmolLM2"):
                self.a(self.ids)
        self.assertTrue(torch.isfinite(self.a(self.ids).logits).all())

    def test_real_overlapping_forward_hook_window_rejects_reference(self):
        entered = threading.Event()
        release = threading.Event()
        errors = []
        old_forward = self.backbone.forward

        def blocking_forward(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=10):
                raise TimeoutError("Synthetic blocking frozen-model callback")
            return old_forward(*args, **kwargs)

        self.backbone.forward = blocking_forward
        thread = threading.Thread(
            target=lambda: self._thread_forward(errors), daemon=True
        )
        thread.start()
        try:
            self.assertTrue(entered.wait(timeout=5))
            self.assertGreater(len(self.backbone.model._forward_hooks), 0)
            with self.assertRaisesRegex(RuntimeError, "Shared frozen SmolLM2"):
                self.b(self.ids, enabled=False)
        finally:
            release.set()
            thread.join(timeout=10)
            self.backbone.forward = old_forward
        self.assertFalse(thread.is_alive(), "Instrumented thread did not finish")
        self.assertEqual(errors, [])
        self.assertEqual(len(self.backbone.model._forward_hooks), 0)
        self.assertEqual(len(self.backbone.model._forward_pre_hooks), 0)
        self.assertTrue(torch.isfinite(self.b(self.ids, enabled=False).logits).all())

    def _thread_forward(self, errors):
        try:
            self.a(self.ids)
        except Exception as exc:
            errors.append(repr(exc))

    def test_truncated_decoder_exception_releases_both_locks(self):
        self.backbone.model.truncate_after = 9
        with self.assertRaisesRegex(RuntimeError, "observed"):
            self.a(self.ids)
        self.backbone.model.truncate_after = None
        self.assertEqual(len(self.backbone.model._forward_hooks), 0)
        self.assertEqual(len(self.backbone.model._forward_pre_hooks), 0)
        self.assertTrue(torch.isfinite(self.b(self.ids).logits).all())


if __name__ == "__main__":
    unittest.main()
