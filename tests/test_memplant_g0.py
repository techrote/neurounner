"""Offline structural and gradient tests for G0. Tiny synthetic models only."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.memplant.dreamgraft001.contract import (FIXED_QUESTION, FIXTURES,
    ensure_no_source_in_query, metadata)


class ContractTests(unittest.TestCase):
    def test_query_does_not_leak_donor(self):
        ensure_no_source_in_query()
        self.assertTrue(all(m.donor_text not in FIXED_QUESTION for m in FIXTURES))
        with self.assertRaises(ValueError):
            ensure_no_source_in_query(FIXTURES[0].donor_text)

    def test_source_hashes_unique_and_deterministic(self):
        x = metadata()
        self.assertEqual(len({v["sha256"] for v in x}), len(x))
        self.assertEqual(metadata(), x)


try:
    import torch
    from torch import nn
    from experiments.memplant.dreamgraft001.bridge import (
        source_fingerprint, read_cache, DreamPrefix, answer_loss)
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class BridgeTests(unittest.TestCase):
    def cache(self, value_offset=0):
        class FakeLayer:
            def __init__(self, i):
                self.keys = torch.arange(1, 1 + 1 * 2 * 12 * 4).float().reshape(1, 2, 12, 4) + i
                self.values = torch.randn(1, 2, 12, 4) + value_offset + i
        class FakeCache:
            layers = [FakeLayer(i) for i in range(8)]
        return FakeCache()

    def test_real_cache_shape_and_binning(self):
        cache = self.cache()
        pairs = read_cache(cache)
        self.assertEqual(len(pairs), 8)
        f, g = source_fingerprint(cache)
        self.assertEqual(tuple(f.shape), (4 * 4 * 2 * 4,))
        self.assertTrue(g["keys_nonzero"] and g["values_nonzero"])
        self.assertEqual(len(g["feature_sha256"]), 64)
        self.assertEqual(g["selected_layers"][0], 0)

    def test_rejects_impostor_cache(self):
        class Invalid:
            layers = [type("L", (), {"keys": torch.zeros(1, 2, 12, 4),
                                       "values": torch.ones(1, 2, 12, 4)})()] * 8
        with self.assertRaises(ValueError):
            source_fingerprint(Invalid())

    def test_adapter_and_gradients(self):
        torch.manual_seed(4)
        class ToyReceiver(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(15, 12)
                self.h = nn.Linear(12, 15)
            def get_input_embeddings(self):
                return self.emb
            def forward(self, inputs_embeds, attention_mask, use_cache, return_dict):
                class O: pass
                result = O()
                result.logits = self.h(inputs_embeds + inputs_embeds.mean(dim=1, keepdim=True))
                return result
        b=ToyReceiver().eval()
        for p in b.parameters():p.requires_grad_(False)
        a=DreamPrefix(128, 12, slots=8, bottleneck=16)
        features=torch.randn(1, 128)
        prefix=a(features)
        loss=answer_loss(b, torch.tensor([[1,2,3]]), torch.tensor([[4,5]]), prefix)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters()))
        self.assertTrue(all(p.grad is None for p in b.parameters()))

if __name__ == "__main__":
    unittest.main()
