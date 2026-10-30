"""NCR-001F no-training cross-wrapper ownership regressions.

Toy frozen backbones only. These guards do not admit #72 or model training.
"""
from __future__ import annotations

import unittest

import torch

from test_ncr_001f_frozen_hooks import FrozenToyLM, wrap


class SharedFrozenBackboneOwnershipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(20261028)
        self.ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    def test_second_wrapper_cannot_install_overlapping_hooks(self):
        base = FrozenToyLM("smollm2")
        first = wrap("smollm2", base)
        second = wrap("smollm2", base)
        self.assertIs(first.backbone, second.backbone)
        self.assertIs(first._backbone_lock, second._backbone_lock)

        with first._instrument():
            self.assertEqual(len(base.model._forward_hooks), 1)
            with self.assertRaisesRegex(RuntimeError, "Shared frozen backbone"):
                second(self.ids)
            self.assertEqual(len(base.model._forward_hooks), 1)
            self.assertEqual(len(base.model._forward_pre_hooks), 1)

        self.assertFalse(base.model._forward_hooks)
        self.assertFalse(base.model._forward_pre_hooks)
        self.assertTrue(torch.isfinite(second(self.ids).logits).all())
        self.assertEqual(second.last_hook_trace,
                         [{"tap_count": 3, "lane": "smollm2"}])

    def test_disabled_reference_paths_also_refuse_foreign_active_hooks(self):
        base = FrozenToyLM("sprocket_instruct")
        first = wrap("sprocket_instruct", base)
        second = wrap("sprocket_instruct", base)
        with first._instrument():
            with self.assertRaisesRegex(RuntimeError, "Shared frozen backbone"):
                second(self.ids, enabled=False)
            with self.assertRaisesRegex(RuntimeError, "Shared frozen backbone"):
                second.generate(self.ids, enabled=False, max_new_tokens=1)
            # Even a baseline of the current wrapper cannot run through
            # its own active instrumentation without risk of contamination.
            with self.assertRaisesRegex(RuntimeError, "Concurrent or nested"):
                first(self.ids, enabled=False)

        frozen = second(self.ids, enabled=False).logits
        zero_control = second(self.ids).logits
        self.assertTrue(torch.equal(frozen, zero_control))
        self.assertFalse(base.model._forward_hooks)

    def test_distinct_frozen_backbones_remain_independently_usable(self):
        a = wrap("void1", FrozenToyLM("void1"))
        b = wrap("void1", FrozenToyLM("void1"))
        self.assertIsNot(a._backbone_lock, b._backbone_lock)
        with a._instrument():
            result = b(self.ids).logits
            self.assertTrue(torch.isfinite(result).all())
            frozen = b(self.ids, enabled=False).logits
            self.assertTrue(torch.equal(result, frozen))
        self.assertFalse(a.backbone.model._forward_hooks)
        self.assertFalse(b.backbone.model._forward_hooks)

    def test_exception_always_releases_shared_owner_and_detaches_hooks(self):
        base = FrozenToyLM("smollm2")
        first, second = wrap("smollm2", base), wrap("smollm2", base)
        with self.assertRaisesRegex(ValueError, "test interruption"):
            with first._instrument():
                raise ValueError("test interruption")
        self.assertFalse(first._backbone_lock.locked())
        self.assertFalse(base.model._forward_hooks)
        self.assertTrue(torch.isfinite(second(self.ids).logits).all())
        self.assertFalse(second._backbone_lock.locked())

    def test_same_wrapper_lock_behavior_preserved(self):
        first = wrap("smollm2")
        first._lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "Concurrent or nested"):
                first(self.ids)
            with self.assertRaisesRegex(RuntimeError, "Concurrent or nested"):
                first.generate(self.ids, enabled=False, max_new_tokens=1)
        finally:
            first._lock.release()
        self.assertTrue(torch.isfinite(first(self.ids).logits).all())


if __name__ == "__main__":
    unittest.main()
