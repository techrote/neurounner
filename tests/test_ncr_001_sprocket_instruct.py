"""Offline test of hook lifetime and invariants with a frozen 26-layer Sprocket toy."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
from ncr_001_sprocket_instruct import FrozenSprocketInstructDepthNCA, PINNED_ROOT_SAFETENSORS_SHA256  # noqa: E402


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(64, 1280)
        self.layers = nn.ModuleList(nn.Identity() for _ in range(26))
        self.truncate_after = None

    def forward(self, input_ids, **_kwargs):
        hidden = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if self.truncate_after == i:
                break
        return hidden


class ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="llama",
            hidden_size=1280,
            num_hidden_layers=26,
        )
        self.model = ToyDecoder()
        self.lm_head = nn.Linear(1280, 64, bias=False)

    def forward(self, input_ids, labels=None, **kwargs):
        hidden = self.model(input_ids, **kwargs)
        logits = self.lm_head(hidden)
        if labels is None:
            return SimpleNamespace(logits=logits)
        # Finite scalar loss for the synthetic backward-only test.
        return SimpleNamespace(
            logits=logits,
            loss=nn.functional.cross_entropy(
                logits.reshape(-1, 64), labels.reshape(-1)
            ),
        )

    def generate(self, input_ids, max_new_tokens=2, **kwargs):
        output = input_ids
        for _ in range(max_new_tokens):
            next_token = self.forward(output, **kwargs).logits[:, -1].argmax(-1)
            output = torch.cat((output, next_token[:, None]), -1)
        return output


class DepthHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(20261026)
        self.adapter = FrozenSprocketInstructDepthNCA(ToyLM(), verified_weight_sha256=PINNED_ROOT_SAFETENSORS_SHA256)
        self.tokens = torch.tensor([[1, 2, 3, 4],
                                    [5, 6, 7, 8]], dtype=torch.long)

    def test_identity_traces_and_all_hooks_removed(self):
        base = self.adapter(self.tokens, enabled=False).logits
        ours = self.adapter(self.tokens).logits
        self.assertTrue(torch.equal(base, ours))
        self.assertEqual(len(self.adapter.last_hook_trace), 1)
        self.assertEqual(self.adapter.last_hook_trace[0]["tap_count"], 3)
        self.assertEqual(self.adapter.last_hook_trace[0]["state_shape"],
                         [2, 4, 4, 8, 8])
        self.assertTrue(self.adapter.last_hook_trace[0]["finite"])
        self.assertEqual(len(self.adapter.backbone.model._forward_hooks), 0)
        self.assertEqual(len(self.adapter.backbone.model._forward_pre_hooks), 0)
        for index in (7, 16, 25):
            self.assertEqual(len(self.adapter.backbone.model.layers[index]._forward_hooks), 0)

    def test_generate_resets_at_every_forward_and_disabled_path(self):
        reference = self.adapter.generate(self.tokens, enabled=False, max_new_tokens=3)
        result = self.adapter.generate(self.tokens, max_new_tokens=3)
        self.assertTrue(torch.equal(reference, result))
        self.assertEqual(len(self.adapter.last_hook_trace), 3)
        for i, row in enumerate(self.adapter.last_hook_trace):
            self.assertEqual(row["state_shape"][1], self.tokens.shape[1] + i)
            self.assertEqual(row["tap_count"], 3)

    def test_chat_checkpoint_cannot_impersonate_instruct(self):
        with self.assertRaisesRegex(ValueError, "never substitute chat"):
            FrozenSprocketInstructDepthNCA(ToyLM(), verified_weight_sha256="551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97")

    def test_wrong_backbone_or_tap_configuration_rejected(self):
        for key, value in (
            ("model_type", "qwen3"), ("hidden_size", 640), ("num_hidden_layers", 24)
        ):
            base = ToyLM()
            setattr(base.config, key, value)
            with self.assertRaises(ValueError):
                FrozenSprocketInstructDepthNCA(base, verified_weight_sha256=PINNED_ROOT_SAFETENSORS_SHA256)
        with self.assertRaises(ValueError):
            FrozenSprocketInstructDepthNCA(ToyLM(), verified_weight_sha256=PINNED_ROOT_SAFETENSORS_SHA256, taps=(6, 16, 25))

    def test_frozen_gradients_and_adapter_only_checkpoint(self):
        self.adapter.train()
        self.assertFalse(self.adapter.backbone.training)
        self.assertTrue(self.adapter.nca.training)
        self.adapter.verify_frozen_base()
        base_params = {
            k: v.detach().clone()
            for k, v in self.adapter.backbone.named_parameters()
        }
        self.assertGreater(self.adapter.frozen_parameters(), 0)
        self.assertEqual(self.adapter.nca.trainable_parameters(), 658904)
        self.adapter(input_ids=self.tokens, labels=self.tokens).loss.backward()
        self.assertGreater(self.adapter.nca.readout.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in self.adapter.backbone.parameters()))
        optimizer = torch.optim.AdamW(self.adapter.adapter_parameters(), lr=0.01)
        optimizer.step()
        self.assertTrue(all(torch.equal(p, base_params[k]) for k, p in
                            self.adapter.backbone.named_parameters()))
        self.assertFalse(torch.equal(
            self.adapter(input_ids=self.tokens).logits,
            self.adapter(input_ids=self.tokens, enabled=False).logits
        ))
        with tempfile.TemporaryDirectory() as temp:
            self.adapter.save_adapter(temp)
            new = FrozenSprocketInstructDepthNCA(ToyLM(), verified_weight_sha256=PINNED_ROOT_SAFETENSORS_SHA256)
            new.load_adapter(temp)
            for name, p in self.adapter.nca.named_parameters():
                self.assertTrue(torch.equal(p, dict(new.nca.named_parameters())[name]))
            file = Path(temp) / "metadata.json"
            bad = file.read_text().replace(
                "kandivault/sprocket-500m", "kandivault/WRONG-checkpoint"
            )
            file.write_text(bad)
            with self.assertRaises(RuntimeError):
                new.load_adapter(temp)

    def test_partial_tap_failure_fails_closed_then_recovers(self):
        self.adapter.backbone.model.truncate_after = 9
        with self.assertRaises(RuntimeError):
            self.adapter(self.tokens)
        self.assertEqual(len(self.adapter.backbone.model._forward_hooks), 0)
        self.assertEqual(len(self.adapter.backbone.model._forward_pre_hooks), 0)
        self.adapter.backbone.model.truncate_after = None
        self.assertEqual(len(self.adapter.last_hook_trace), 0)
        logits = self.adapter(self.tokens).logits
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(len(self.adapter.last_hook_trace), 1)

    def test_concurrent_instrumentation_is_rejected_without_deadlock(self):
        self.adapter._instrument_lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "Concurrent"):
                self.adapter(self.tokens)
        finally:
            self.adapter._instrument_lock.release()
        logits = self.adapter(self.tokens).logits
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
