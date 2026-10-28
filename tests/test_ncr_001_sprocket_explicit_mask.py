"""Offline sentinels for the Sprocket genuine-end-marker vs synthetic-pad contract.

These tests do not download any checkpoint, run any optimizer, or open heldout
fixtures. The pinned actual-host witness is a separate draft PR CI job.
"""
from pathlib import Path
import json
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ncr_001_sprocket_explicit_mask import (  # noqa: E402
    mask_authority, generated,
)


class FakePinnedTokenizer:
    pad_token_id = 5
    padding_side = "right"

    def __init__(self, *, mask_by_id=False, no_real_end=False, equal_length=False):
        self.mask_by_id = mask_by_id
        self.no_real_end = no_real_end
        self.equal_length = equal_length

    def __call__(self, texts, *, add_special_tokens=False, padding=False,
                 return_attention_mask=False, return_tensors="pt"):
        assert not add_special_tokens and return_tensors == "pt"
        sequences = {
            "short": [3, 12, 5, 4],
            "long": [3, 13, 5, 4, 20, 21, 5, 3, 21, 5, 4],
        }
        if self.equal_length:
            sequences["long"] = [3, 12, 5, 4]
        if self.no_real_end:
            sequences = {k: [44 if x == 5 else x for x in v]
                         for k, v in sequences.items()}
        single = isinstance(texts, str)
        xs = [texts] if single else texts
        rows = [sequences[x][:] for x in xs]
        width = max(len(x) for x in rows)
        ids, masks = [], []
        for row in rows:
            missing = width - len(row) if padding else 0
            ids.append([5] * missing + row)
            masks.append([0] * missing + [1] * len(row))
        a = torch.tensor(ids)
        b = torch.tensor(masks)
        if self.mask_by_id:
            b = (a != 5).long()
        return {"input_ids": a, "attention_mask": b}


def format_source(rows, *, variant):
    assert variant in ("chat", "instruct")
    return "short" if len(rows) == 1 else "long"


class RecordingWrapper:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        assert "attention_mask" in kwargs
        assert kwargs["attention_mask"].shape == kwargs["input_ids"].shape
        assert kwargs["min_new_tokens"] == kwargs["max_new_tokens"] == 3
        return torch.cat(
            [kwargs["input_ids"],
             torch.full((kwargs["input_ids"].shape[0], 3), 8, dtype=torch.long)],
            dim=1,
        )


class ExplicitSprocketMaskTests(unittest.TestCase):
    def test_both_variants_preserve_real_end_marker_with_synthetic_left_padding(self):
        for variant in ("chat", "instruct"):
            with self.subTest(variant=variant):
                original, ids, mask, observed = mask_authority(
                    FakePinnedTokenizer(), format_source, variant)
                self.assertEqual(len(original), 2)
                self.assertTrue(observed[0]["synthetic_pad"] > 0)
                self.assertTrue(observed[0]["real_end_marker_id_5_visible"])
                self.assertTrue(observed[1]["real_end_marker_id_5_visible"])
                self.assertEqual(mask[0, 0].item(), 0)
                self.assertEqual(ids[0, 0].item(), 5)
                self.assertTrue(((ids == 5) & (mask == 1)).any())

    def test_incorrect_pad_id_derived_mask_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "Real source tokens are masked"):
            mask_authority(FakePinnedTokenizer(mask_by_id=True),
                           format_source, "chat")

    def test_missing_source_end_marker_refused(self):
        with self.assertRaisesRegex(RuntimeError, "end"):
            mask_authority(FakePinnedTokenizer(no_real_end=True),
                           format_source, "instruct")

    def test_negative_sentinel_requires_genuine_padding(self):
        with self.assertRaisesRegex(RuntimeError, "left-padding witness"):
            mask_authority(FakePinnedTokenizer(equal_length=True),
                           format_source, "chat")

    def test_generate_always_forwards_explicit_mask_and_cache_selection(self):
        x = torch.tensor([[5, 3, 12, 5, 4]])
        m = torch.tensor([[0, 1, 1, 1, 1]])
        wrapper = RecordingWrapper()
        y = generated(wrapper, x, m, enabled=False, cache=True)
        self.assertEqual(list(y.shape), [1, 8])
        self.assertTrue(torch.equal(wrapper.calls[0]["attention_mask"], m))
        self.assertFalse(wrapper.calls[0]["enabled"])
        self.assertTrue(wrapper.calls[0]["use_cache"])
        generated(wrapper, x, m, enabled=True, cache=False)
        self.assertTrue(wrapper.calls[1]["enabled"])
        self.assertFalse(wrapper.calls[1]["use_cache"])

    def test_protocol_and_source_do_not_claim_training_or_tool_semantics(self):
        protocol = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
        self.assertIs(protocol["train_authorized"], False)
        self.assertEqual(set(protocol["models"]),
                         {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"})
        src = (ROOT / "scripts/ncr_001_sprocket_explicit_mask.py").read_text()
        self.assertIn('"tool_memory_semantics_qualified": False', src)
        self.assertIn('"full_72_conformance_pass": False', src)
        self.assertNotIn("optimizer.step(", src)
        self.assertNotIn(".backward(", src)


if __name__ == "__main__":
    unittest.main()
