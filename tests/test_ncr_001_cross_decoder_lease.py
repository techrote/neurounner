"""NCR-001 source-only NCA/MLP decoder-root lease composition regressions.

No HF original-model download, supervised loss, optimizer step, comparative
training, held-out dataset, tokenizer execution or final #72 admission.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys
import threading
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ncr_001_smollm2 import FrozenSmolLM2DepthNCA  # noqa: E402
from ncr_001_sprocket_chat import (  # noqa: E402
    FrozenSprocketChatDepthNCA, PINNED_CHAT_SAFETENSORS_SHA256)
from ncr_001_sprocket_instruct import (  # noqa: E402
    FrozenSprocketInstructDepthNCA, PINNED_ROOT_SAFETENSORS_SHA256)
from ncr_001_void1 import FrozenVoid1DepthNCA, PINNED_SAFETENSORS_SHA256  # noqa: E402
from ncr_001f_frozen_hooks import PINNED  # noqa: E402
from test_ncr_001f_frozen_hooks import FrozenToyLM, wrap  # noqa: E402

NCA_CLASSES = {
    "smollm2": FrozenSmolLM2DepthNCA,
    "sprocket_chat": FrozenSprocketChatDepthNCA,
    "sprocket_instruct": FrozenSprocketInstructDepthNCA,
    "void1": FrozenVoid1DepthNCA,
}
NCA_SHAS = {
    "sprocket_chat": PINNED_CHAT_SAFETENSORS_SHA256,
    "sprocket_instruct": PINNED_ROOT_SAFETENSORS_SHA256,
    "void1": PINNED_SAFETENSORS_SHA256,
}


def wrap_nca(lane, backbone):
    cls = NCA_CLASSES[lane]
    if lane == "smollm2":
        return cls(backbone)
    return cls(backbone, verified_weight_sha256=NCA_SHAS[lane])


class CrossAdapterLeaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(20261028)
        self.ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    def pair(self, lane):
        backbone = FrozenToyLM(lane)
        return backbone, wrap_nca(lane, backbone), wrap(lane, backbone)

    def test_protocol_training_closed_and_four_checkpoint_identities_distinct(self):
        p = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
        self.assertIs(p["train_authorized"], False)
        self.assertEqual(set(p["models"]), set(NCA_CLASSES))
        self.assertEqual(set(PINNED), set(NCA_CLASSES))
        self.assertNotEqual(PINNED["sprocket_chat"]["weight_sha256"],
                            PINNED["sprocket_instruct"]["weight_sha256"])
        for lane in NCA_SHAS:
            self.assertEqual(NCA_SHAS[lane], PINNED[lane]["weight_sha256"])

    def test_one_decoder_lock_shared_by_all_nca_and_mlp_variants(self):
        for lane in NCA_CLASSES:
            with self.subTest(lane=lane):
                base, nca, mlp = self.pair(lane)
                self.assertIs(nca.backbone, base)
                self.assertIs(nca._backbone_lock, mlp._backbone_lock)
                other = wrap_nca(lane, FrozenToyLM(lane))
                self.assertIsNot(nca._backbone_lock, other._backbone_lock)

    def test_nca_active_window_blocks_enabled_disabled_mlp_forward_generate(self):
        for lane in NCA_CLASSES:
            with self.subTest(lane=lane):
                base, nca, mlp = self.pair(lane)
                with nca._instrument():
                    self.assertTrue(base.model._forward_hooks)
                    for enabled in (False, True):
                        with self.assertRaisesRegex(RuntimeError, "Shared frozen"):
                            mlp(self.ids, enabled=enabled)
                        with self.assertRaisesRegex(RuntimeError, "Shared frozen"):
                            mlp.generate(self.ids, enabled=enabled, max_new_tokens=1)
                self.assertFalse(base.model._forward_hooks)
                self.assertFalse(base.model._forward_pre_hooks)
                with torch.no_grad():
                    self.assertTrue(torch.equal(
                        nca(self.ids, enabled=False).logits, mlp(self.ids).logits))

    def test_mlp_active_window_blocks_enabled_disabled_nca_forward_generate(self):
        for lane in NCA_CLASSES:
            with self.subTest(lane=lane):
                base, nca, mlp = self.pair(lane)
                with mlp._instrument():
                    self.assertTrue(base.model._forward_hooks)
                    for enabled in (False, True):
                        with self.assertRaisesRegex(RuntimeError, "Shared frozen"):
                            nca(self.ids, enabled=enabled)
                        with self.assertRaisesRegex(RuntimeError, "Shared frozen"):
                            nca.generate(self.ids, enabled=enabled, max_new_tokens=1)
                self.assertFalse(base.model._forward_hooks)
                self.assertFalse(base.model._forward_pre_hooks)
                with torch.no_grad():
                    self.assertTrue(torch.equal(
                        mlp(self.ids, enabled=False).logits, nca(self.ids).logits))

    def test_two_outer_models_aliasing_same_decoder_are_rejected(self):
        left = FrozenToyLM("smollm2")
        right = FrozenToyLM("smollm2")
        right.model = left.model  # Different outer nn.Modules share *one* hook root.
        nca = wrap_nca("smollm2", right)
        mlp = wrap("smollm2", left)
        self.assertIsNot(nca.backbone, mlp.backbone)
        self.assertIs(nca._backbone_lock, mlp._backbone_lock)
        with mlp._instrument():
            with self.assertRaisesRegex(RuntimeError, "Shared frozen"):
                nca(self.ids, enabled=False)

    def test_nonaliased_decoders_remain_independent(self):
        _, nca, _ = self.pair("void1")
        other = wrap("void1", FrozenToyLM("void1"))
        with nca._instrument():
            with torch.no_grad():
                out = other(self.ids, enabled=False).logits
            self.assertTrue(torch.isfinite(out).all())

    def test_same_wrapper_nested_reference_is_rejected(self):
        _, nca, mlp = self.pair("sprocket_chat")
        with nca._instrument():
            with self.assertRaisesRegex(RuntimeError, "Shared frozen"):
                nca(self.ids, enabled=False)
        with mlp._instrument():
            with self.assertRaisesRegex(RuntimeError, "Concurrent or nested"):
                mlp(self.ids, enabled=False)

    def test_incomplete_depth_releases_lease_and_hooks(self):
        base, nca, mlp = self.pair("sprocket_instruct")
        base.model.truncate_after = 7
        with self.assertRaisesRegex(RuntimeError, "observed"):
            nca(self.ids)
        self.assertFalse(nca._backbone_lock.locked())
        self.assertFalse(base.model._forward_hooks)
        self.assertFalse(base.model._forward_pre_hooks)
        base.model.truncate_after = None
        with torch.no_grad():
            self.assertTrue(torch.isfinite(mlp(self.ids).logits).all())
            self.assertTrue(torch.isfinite(nca(self.ids).logits).all())

    def test_real_thread_overlap_refuses_foreign_disabled_reference(self):
        base, nca, mlp = self.pair("smollm2")
        entered, release = threading.Event(), threading.Event()
        old_forward = base.forward
        failures = []

        def blocked(*args, **kwargs):
            entered.set()
            if not release.wait(10):
                raise TimeoutError("Synthetic active-window watchdog")
            return old_forward(*args, **kwargs)

        def worker():
            try:
                with torch.no_grad():
                    nca(self.ids)
            except Exception as exc:
                failures.append(repr(exc))

        base.forward = blocked
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            self.assertTrue(entered.wait(5))
            self.assertTrue(base.model._forward_hooks)
            with self.assertRaisesRegex(RuntimeError, "Shared frozen"):
                mlp(self.ids, enabled=False)
        finally:
            release.set()
            thread.join(10)
            base.forward = old_forward
        self.assertFalse(thread.is_alive())
        self.assertFalse(failures)
        self.assertFalse(base.model._forward_hooks)
        self.assertFalse(nca._backbone_lock.locked())

    def test_wrong_sprocket_variant_hash_is_not_accepted(self):
        base = FrozenToyLM("sprocket_chat")
        with self.assertRaisesRegex(ValueError, "chat checkpoint"):
            FrozenSprocketChatDepthNCA(
                base, verified_weight_sha256=PINNED_ROOT_SAFETENSORS_SHA256)


if __name__ == "__main__":
    unittest.main()
