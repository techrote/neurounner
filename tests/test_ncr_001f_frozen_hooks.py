"""NCR-001F frozen-source matched MLP integration smoke, with no LM training."""
from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ncr_001f_frozen_hooks import FrozenDepthGatedMLP, PINNED  # noqa: E402


class Decoder(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.embed_tokens = nn.Embedding(64, width)
        self.layers = nn.ModuleList(nn.Identity() for _ in range(depth))
        self.truncate_after = None

    def forward(self, input_ids, **kwargs):
        x = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.truncate_after == i:
                break
        return x


class FrozenToyLM(nn.Module):
    def __init__(self, lane):
        super().__init__()
        s = PINNED[lane]
        self.config = SimpleNamespace(model_type=s["model_type"],
                                      hidden_size=s["width"],
                                      num_hidden_layers=s["layers"],
                                      vocab_size=64)
        self.model = Decoder(s["width"], s["layers"])
        self.lm_head = nn.Linear(s["width"], 64, bias=False)

    def forward(self, input_ids, labels=None, **kwargs):
        logits = self.lm_head(self.model(input_ids, **kwargs))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.flatten(0, 1), labels.flatten()
            )
        return SimpleNamespace(logits=logits, loss=loss)

    def generate(self, input_ids, max_new_tokens=3, **kwargs):
        x = input_ids
        for _ in range(max_new_tokens):
            nxt = self.forward(x, **kwargs).logits[:, -1].argmax(-1, keepdim=True)
            x = torch.cat((x, nxt), dim=-1)
        return x


def wrap(lane, backbone=None, **override):
    s = PINNED[lane]
    source = dict(lane=lane, verified_weight_sha256=s["weight_sha256"],
                  source_revision=s["revision"])
    source.update(override)
    return FrozenDepthGatedMLP(backbone or FrozenToyLM(lane), **source)


class FourLaneMatchedHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(20261027)
        self.ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    def test_all_four_real_model_layouts_and_exact_initial_noop(self):
        protocol = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
        self.assertFalse(protocol["train_authorized"])
        self.assertEqual(set(PINNED), set(protocol["models"]))
        for lane, spec in PINNED.items():
            with self.subTest(lane=lane):
                self.assertEqual(list(spec["taps"]),
                                 protocol["models"][lane]["taps_after_block_zero_based"])
                m = wrap(lane)
                self.assertTrue(all(not p.requires_grad for p in m.backbone.parameters()))
                self.assertFalse(m.backbone.training)
                self.assertEqual(m.control.trainable_parameters(),
                                 514 * spec["width"] + 512)
                original = m(self.ids, enabled=False).logits
                got = m(self.ids).logits
                self.assertTrue(torch.equal(original, got))
                self.assertEqual(m.last_hook_trace, [{"tap_count": 3, "lane": lane}])
                self.assertEqual(len(m.backbone.model._forward_hooks), 0)
                self.assertEqual(len(m.backbone.model._forward_pre_hooks), 0)
                self.assertTrue(all(
                    len(m.backbone.model.layers[i]._forward_hooks) == 0
                    for i in spec["taps"]
                ))
                self.assertFalse(hasattr(m.control, "field"))
                self.assertFalse(hasattr(m.control, "state"))

    def test_each_lane_generation_reset_and_nontrained_active_control(self):
        for lane in PINNED:
            with self.subTest(lane=lane):
                m = wrap(lane)
                expected = m.generate(self.ids, enabled=False, max_new_tokens=2)
                actual = m.generate(self.ids, max_new_tokens=2)
                self.assertTrue(torch.equal(expected, actual))
                self.assertEqual(len(m.last_hook_trace), 2)
                with torch.no_grad():
                    m.control.down.weight[0, 0] = 0.1
                a = m(self.ids).logits
                b = m(self.ids, enabled=False).logits
                self.assertGreater(float((a - b).abs().max()), 0)
                self.assertTrue(torch.equal(a, m(self.ids).logits))
                alternative = self.ids.clone()
                alternative[:, -1] += 1
                c = m(alternative).logits
                self.assertTrue(torch.equal(a[:, :-1], c[:, :-1]))

    def test_chat_instruct_weights_and_checkpoint_metadata_cannot_cross(self):
        chat = PINNED["sprocket_chat"]
        instruct = PINNED["sprocket_instruct"]
        self.assertNotEqual(chat["weight_sha256"], instruct["weight_sha256"])
        self.assertNotEqual(chat["weight_path"], instruct["weight_path"])
        with self.assertRaisesRegex(ValueError, "source/variant"):
            wrap("sprocket_chat", verified_weight_sha256=instruct["weight_sha256"])
        with self.assertRaisesRegex(ValueError, "source/variant"):
            wrap("sprocket_instruct", verified_weight_sha256=chat["weight_sha256"])
        with self.assertRaisesRegex(ValueError, "source/variant"):
            wrap("void1", source_revision=chat["revision"])
        with self.assertRaises(ValueError):
            FrozenDepthGatedMLP(FrozenToyLM("smollm2"), lane="nonexistent",
                                verified_weight_sha256="", source_revision="")
        mismatched = FrozenToyLM("sprocket_instruct")
        mismatched.config.model_type = "qwen3"
        with self.assertRaisesRegex(ValueError, "configuration"):
            wrap("sprocket_instruct", mismatched)

    def test_adapter_only_gradients_and_base_tensor_integrity_no_optimizer(self):
        m = wrap("smollm2")
        m.train()
        self.assertFalse(m.backbone.training)
        original = {k: p.detach().clone() for k, p in m.backbone.named_parameters()}
        loss = m(self.ids, labels=self.ids).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(m.control.down.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in m.backbone.parameters()))
        self.assertTrue(all(torch.equal(p, original[n])
                            for n, p in m.backbone.named_parameters()))
        self.assertFalse(hasattr(m, "optimizer"))
        m.zero_grad(set_to_none=True)

    def test_adapter_cold_new_instance_with_variant_refusal(self):
        for lane in PINNED:
            with self.subTest(lane=lane):
                torch.manual_seed(171)
                first = wrap(lane)
                with torch.no_grad():
                    first.control.down.weight[0, 0] = 0.05
                before = first(self.ids).logits
                torch.manual_seed(171)
                second = wrap(lane)
                with tempfile.TemporaryDirectory() as directory:
                    first.save_adapter(directory)
                    second.load_adapter(directory)
                    self.assertTrue(torch.equal(before, second(self.ids).logits))
                    metadata = Path(directory) / "metadata.json"
                    good = json.loads(metadata.read_text())
                    bad = dict(good, lane="sprocket_chat" if lane != "sprocket_chat"
                               else "sprocket_instruct")
                    metadata.write_text(json.dumps(bad))
                    with self.assertRaisesRegex(RuntimeError, "metadata"):
                        second.load_adapter(directory)
                    metadata.write_text(json.dumps(good))
                    bad = dict(good, adapter_kind="NCR-001B_NCA")
                    metadata.write_text(json.dumps(bad))
                    with self.assertRaisesRegex(RuntimeError, "metadata"):
                        second.load_adapter(directory)

    def test_partial_decoder_exception_cleanup_then_recovery(self):
        m = wrap("smollm2")
        m.backbone.model.truncate_after = 9
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            m(self.ids)
        self.assertFalse(m.backbone.model._forward_hooks)
        self.assertFalse(m.backbone.model._forward_pre_hooks)
        self.assertFalse(m.backbone.model.layers[9]._forward_hooks)
        m.backbone.model.truncate_after = None
        self.assertTrue(torch.isfinite(m(self.ids).logits).all())
        self.assertEqual(len(m.last_hook_trace), 1)

    def test_concurrent_instrumentation_rejected_and_unlocks(self):
        m = wrap("void1")
        m._lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "Concurrent"):
                m(self.ids)
        finally:
            m._lock.release()
        self.assertTrue(torch.isfinite(m(self.ids).logits).all())


if __name__ == "__main__":
    unittest.main()
