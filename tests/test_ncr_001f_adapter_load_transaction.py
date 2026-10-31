"""Toy-only adversarial checks for NCR-001F atomic matched-MLP adapter import."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from test_ncr_001f_frozen_hooks import FrozenToyLM, wrap
from ncr_001f_frozen_hooks import PINNED

ROOT = Path(__file__).resolve().parents[1]


def snapshot(adapter):
    return {k: p.detach().clone() for k, p in adapter.control.named_parameters()}


def assert_unchanged(test, adapter, before):
    for name, parameter in adapter.control.named_parameters():
        test.assertTrue(torch.equal(parameter, before[name]), name)
    test.assertFalse(adapter._lock.locked())
    test.assertFalse(adapter._backbone_lock.locked())
    test.assertTrue(all(not p.requires_grad for p in adapter.backbone.parameters()))


class AtomicMatchedControlImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(20261031)
        self.ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    def test_all_four_geometries_valid_roundtrip_no_model_or_optimizer(self):
        protocol = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
        self.assertIs(protocol["train_authorized"], False)
        self.assertEqual(set(PINNED), set(protocol["models"]))
        for lane in PINNED:
            with self.subTest(lane=lane), tempfile.TemporaryDirectory() as directory:
                source, target = wrap(lane), wrap(lane)
                with torch.no_grad():
                    source.control.down.weight[0, 0] = 0.125
                before_base = {k: v.detach().clone()
                               for k, v in target.backbone.named_parameters()}
                source.save_adapter(directory)
                target.load_adapter(directory)
                source_params = dict(source.control.named_parameters())
                for name, value in target.control.named_parameters():
                    self.assertTrue(torch.equal(value, source_params[name]), name)
                for name, value in target.backbone.named_parameters():
                    self.assertTrue(torch.equal(value, before_base[name]), name)
                    self.assertFalse(value.requires_grad)
                self.assertTrue(torch.isfinite(target(self.ids).logits).all())
                self.assertFalse(target._backbone_lock.locked())

    def refuse(self, mutate, error):
        with tempfile.TemporaryDirectory() as directory:
            adapter = wrap("smollm2")
            adapter.save_adapter(directory)
            file = Path(directory) / "adapter.pt"
            payload = torch.load(file, map_location="cpu", weights_only=True)
            mutate(payload)
            torch.save(payload, file)
            before = snapshot(adapter)
            logits = adapter(self.ids).logits.detach().clone()
            with self.assertRaisesRegex(RuntimeError, error):
                adapter.load_adapter(directory)
            assert_unchanged(self, adapter, before)
            self.assertTrue(torch.equal(logits, adapter(self.ids).logits))
            self.assertFalse(adapter.backbone.model._forward_hooks)

    def test_valid_early_tensor_plus_invalid_late_shape_is_atomic(self):
        def change(state):
            state["feature_gate"] = torch.full_like(state["feature_gate"], 123.0)
            state["up.weight"] = torch.ones(5, dtype=state["up.weight"].dtype)
        self.refuse(change, "tensor metadata mismatch")

    def test_missing_and_extra_keys_are_refused(self):
        self.refuse(lambda s: s.pop("down.weight"), "state keys")
        self.refuse(lambda s: s.update(unexpected=torch.ones(1)), "state keys")

    def test_dtype_nan_inf_refusals(self):
        self.refuse(
            lambda s: s.__setitem__("up.bias", s["up.bias"].to(torch.float64)),
            "tensor metadata mismatch")
        def nan_case(s):
            s["feature_gate"][0] = float("nan")
        def inf_case(s):
            s["down.weight"][0, 0] = float("inf")
        self.refuse(nan_case, "nonfinite values")
        self.refuse(inf_case, "nonfinite values")

    def test_extra_state_and_container_schema_refusals(self):
        def invalid_extra(s):
            s["_extra_state"] = copy.deepcopy(s["_extra_state"])
            s["_extra_state"]["cross_depth_recurrence"] = True
        self.refuse(invalid_extra, "extra-state mismatch")
        with tempfile.TemporaryDirectory() as directory:
            adapter = wrap("void1")
            adapter.save_adapter(directory)
            file = Path(directory) / "adapter.pt"
            state = torch.load(file, weights_only=True)
            torch.save(list(state.items()), file)
            before = snapshot(adapter)
            with self.assertRaisesRegex(RuntimeError, "state keys"):
                adapter.load_adapter(directory)
            assert_unchanged(self, adapter, before)

    def test_wrong_chat_variant_metadata_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = wrap("sprocket_chat")
            adapter.save_adapter(directory)
            meta = Path(directory) / "metadata.json"
            wrong = json.loads(meta.read_text(encoding="utf-8"))
            wrong["lane"] = "sprocket_instruct"
            meta.write_text(json.dumps(wrong), encoding="utf-8")
            before = snapshot(adapter)
            with self.assertRaisesRegex(RuntimeError, "wrong model variant"):
                adapter.load_adapter(directory)
            assert_unchanged(self, adapter, before)

    def test_active_wrapper_blocks_cross_wrapper_and_nested_import(self):
        with tempfile.TemporaryDirectory() as directory:
            base = FrozenToyLM("sprocket_instruct")
            owner, other = wrap("sprocket_instruct", base), wrap("sprocket_instruct", base)
            owner.save_adapter(directory)
            before = snapshot(other)
            with owner._instrument():
                with self.assertRaisesRegex(RuntimeError, "Shared frozen backbone"):
                    other.load_adapter(directory)
                with self.assertRaisesRegex(RuntimeError, "Concurrent or nested"):
                    owner.load_adapter(directory)
                self.assertTrue(owner._backbone_lock.locked())
            assert_unchanged(self, other, before)
            other.load_adapter(directory)
            # A successful import intentionally replaces this wrapper's
            # independently initialized weights with the owner's snapshot.
            owner_params = dict(owner.control.named_parameters())
            for name, parameter in other.control.named_parameters():
                self.assertTrue(torch.equal(parameter, owner_params[name]), name)
            self.assertFalse(other._lock.locked())
            self.assertFalse(other._backbone_lock.locked())

    def test_injected_late_load_exception_restores_prior_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = wrap("smollm2")
            adapter.save_adapter(directory)
            before = snapshot(adapter)
            expected_logits = adapter(self.ids).logits.detach().clone()
            def fail_after_copy(_state, *, strict):
                self.assertTrue(strict)
                with torch.no_grad():
                    adapter.control.norm.weight.fill_(91.0)
                    adapter.control.up.weight.fill_(92.0)
                raise RuntimeError("injected mid-commit failure")
            with patch.object(adapter.control, "load_state_dict", side_effect=fail_after_copy):
                with self.assertRaisesRegex(RuntimeError, "injected mid-commit"):
                    adapter.load_adapter(directory)
            assert_unchanged(self, adapter, before)
            self.assertTrue(torch.equal(expected_logits, adapter(self.ids).logits))

    def test_corrupt_file_releases_lease_and_preserves_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = wrap("smollm2")
            adapter.save_adapter(directory)
            (Path(directory) / "adapter.pt").write_bytes(b"not a valid torch checkpoint")
            before = snapshot(adapter)
            with self.assertRaises(Exception):
                adapter.load_adapter(directory)
            assert_unchanged(self, adapter, before)
            self.assertTrue(torch.isfinite(adapter(self.ids).logits).all())


if __name__ == "__main__":
    unittest.main()
