"""Pure offline negative tests for pinned void.1 two-process custody.

These never fetch a model, import PyTorch, run the remote tokenizer, or open a
sealed held-out fixture. They enforce source/permission checks before loading.
"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ncr_001e_void_process_guard import (
    REPO, PIN, ORIGINAL_HOOK, WEIGHT, BYTE_SOURCE, LANES, LENGTH, PREFIX,
    validate_protocol, validate_tokenizer_units, validate_manifest,
)

SHA = "b" * 64


def protocol():
    return {
        "train_authorized": False,
        "models": {
            key: {
                "repo_id": REPO if key == "void1" else key,
                "revision": PIN if key == "void1" else key,
                "weight_file": "model.safetensors",
                "taps_after_block_zero_based": [7, 15, 23],
                "expected_config": {"max_position_embeddings": 2048},
            }
            for key in LANES
        },
        "conformance": {
            "full_vs_cached_prefill_atol": 0.0005,
            "full_vs_cached_prefill_rtol": 0.0005,
        },
    }


def pins():
    return {"models": {"void1": {
        "repo_id": REPO, "revision": PIN,
        "hf_weight_file": "model.safetensors",
        "weight_declared_lfs_sha256_not_downloaded": WEIGHT,
        "weight_bytes": 360622744,
        "tokenization_byte_py_sha256": BYTE_SOURCE,
    }}}


def manifest():
    return {
        "schema": 1, "model_key": "void1", "repo": REPO, "revision": PIN,
        "checkpoint_file": "model.safetensors", "checkpoint_sha256": WEIGHT,
        "byte_tokenizer_source_sha256": BYTE_SOURCE,
        "original_hook_commit": ORIGINAL_HOOK,
        "protocol_sha256": SHA, "source_pins_sha256": SHA,
        "input_token_ids_sha256": SHA, "producer_pid": 321,
        "adapter_sha256": SHA, "adapter_metadata_sha256": SHA,
        "reference_sha256": SHA,
        "model_training_performed": False, "optimizer_steps": 0,
        "train_authorized": False, "heldout_evaluation": False,
        "full_four_model_conformance_pass": False,
        "historical_byte_eos_impact_measured": False,
    }


def validate(m, pid=654):
    return validate_manifest(m, current_pid=pid, protocol_sha=SHA,
                             pins_sha=SHA, token_ids_sha=SHA)


class FakeIDs:
    ndim = 2
    shape = (1, LENGTH)

    def __init__(self, token_ids):
        self.token_ids = token_ids

    def flatten(self):
        return self.token_ids


class FakeTokenizer:
    vocab_size = 259
    pad_token_id = 256
    bos_token_id = 257
    eos_token_id = 258

    def encode(self, s, add_special_tokens=True):
        return list(s.encode("utf-8"))


class VoidTwoProcessCustodyTests(unittest.TestCase):
    def test_locked_four_variants_no_training(self):
        self.assertEqual(set(LANES), {
            "smollm2", "sprocket_chat", "sprocket_instruct", "void1",
        })
        self.assertEqual((LENGTH, PREFIX), (256, 240))
        validate_protocol(protocol(), pins())
        bad = protocol()
        bad["train_authorized"] = True
        with self.assertRaises(RuntimeError):
            validate_protocol(bad, pins())
        bad = protocol()
        bad["models"].pop("sprocket_instruct")
        with self.assertRaises(RuntimeError):
            validate_protocol(bad, pins())

    def test_source_revision_and_byte_identity_fail_closed(self):
        for location, field, wrong in (
            ("lane", "revision", "head"),
            ("lane", "repo_id", "somewhere_else"),
            ("lane", "taps_after_block_zero_based", [8, 15, 23]),
            ("pin", "tokenization_byte_py_sha256", "0" * 64),
            ("pin", "weight_declared_lfs_sha256_not_downloaded", "0" * 64),
            ("pin", "weight_bytes", 42),
        ):
            with self.subTest(location=location, field=field):
                p, q = protocol(), pins()
                target = (p["models"]["void1"] if location == "lane" else
                          q["models"]["void1"])
                target[field] = wrong
                with self.assertRaises(RuntimeError):
                    validate_protocol(p, q)

    def test_causal_limit_cannot_be_relaxed(self):
        for key in ("full_vs_cached_prefill_atol", "full_vs_cached_prefill_rtol"):
            p = protocol()
            p["conformance"][key] = 0.05
            with self.assertRaises(RuntimeError):
                validate_protocol(p, pins())

    def test_reviewed_utf8_and_literal_eos_distinction(self):
        good = FakeIDs(list(range(256)))
        validate_tokenizer_units(FakeTokenizer(), good,
                                 "日本語 café 🔧 " * 50)
        tok = FakeTokenizer()
        tok.eos_token_id = 62
        with self.assertRaises(RuntimeError):
            validate_tokenizer_units(tok, good, "日本語 " * 100)
        broken = FakeIDs([258] + [32] * 255)
        with self.assertRaises(RuntimeError):
            validate_tokenizer_units(FakeTokenizer(), broken,
                                     "日本語 " * 100)
        with self.assertRaises(RuntimeError):
            validate_tokenizer_units(FakeTokenizer(), good, "ascii " * 150)

    def test_manifest_source_and_bytes(self):
        validate(manifest())
        for field in (
            "model_key", "repo", "revision", "checkpoint_file",
            "checkpoint_sha256", "byte_tokenizer_source_sha256",
            "original_hook_commit", "protocol_sha256",
            "source_pins_sha256", "input_token_ids_sha256",
        ):
            bad = manifest()
            bad[field] = "unverified"
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                validate(bad)

    def test_manifest_rejects_training_and_false_admission(self):
        for key, wrong in (
            ("train_authorized", True),
            ("optimizer_steps", 1),
            ("model_training_performed", True),
            ("heldout_evaluation", True),
            ("full_four_model_conformance_pass", True),
            ("historical_byte_eos_impact_measured", True),
        ):
            bad = manifest()
            bad[key] = wrong
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate(bad)

    def test_fresh_process_and_sha_required(self):
        with self.assertRaises(RuntimeError):
            validate(manifest(), pid=321)
        for value in (None, 0, "321", -1):
            bad = manifest()
            bad["producer_pid"] = value
            with self.subTest(producer_pid=value), self.assertRaises(RuntimeError):
                validate(bad)
        for key in ("adapter_sha256", "adapter_metadata_sha256", "reference_sha256"):
            for bad_hash in ("z" * 64, "a" * 63, ""):
                bad = manifest()
                bad[key] = bad_hash
                with self.subTest(key=key, bad_hash=bad_hash), self.assertRaises(RuntimeError):
                    validate(bad)


if __name__ == "__main__":
    unittest.main()
