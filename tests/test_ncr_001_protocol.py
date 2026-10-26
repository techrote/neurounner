"""No-network NCR-001A protocol and chat-source safety checks."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/ncr_001_protocol.py"
SPEC = importlib.util.spec_from_file_location("ncr_001_protocol", PATH)
assert SPEC and SPEC.loader
ncr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ncr)


class ProtocolGuards(unittest.TestCase):
    def test_current_preregistration_cannot_train(self):
        result = ncr.validate()
        self.assertEqual(result["status"], "PASS_WITH_EXPLICIT_TRAINING_BLOCKERS")
        self.assertFalse(result["training_authorized"])
        self.assertGreater(len(result["unresolved"]), 1)

    def test_spatial_taps_are_deterministic(self):
        self.assertEqual(ncr.taps_for_layers(30), [9, 19, 29])
        self.assertEqual(ncr.taps_for_layers(24), [7, 15, 23])
        with self.assertRaises(ValueError):
            ncr.taps_for_layers(4)

    def test_no_training_permission_can_be_smuggled_in(self):
        p = ncr.json_load(ncr.PROTOCOL)
        p["train_authorized"] = True
        with tempfile.TemporaryDirectory() as td:
            altered = Path(td) / "protocol.json"
            altered.write_text(json.dumps(p), encoding="utf-8")
            with patch.object(ncr, "PROTOCOL", altered):
                with self.assertRaises(AssertionError):
                    ncr.validate()

    def test_sprocket_stays_chat_not_root_instruct(self):
        p = ncr.json_load(ncr.PROTOCOL)
        s = p["models"]["sprocket_chat"]
        self.assertEqual(s["variant"], "chat")
        self.assertEqual(s["weight_file"], "chat/model.safetensors")
        self.assertNotEqual(s["weight_file"], "model.safetensors")
        self.assertEqual(s["taps_after_block_zero_based"], [7, 16, 25])

    def test_missing_chat_safetensors_is_explicit_blocker(self):
        """A valid root instruct checkpoint must not qualify Sprocket chat."""
        p = ncr.json_load(ncr.PROTOCOL)
        expected = p["models"]["sprocket_chat"]["expected_config"].copy()
        expected.update({"num_hidden_layers": 26, "hidden_size": 1280})
        files = {
            "config.json": json.dumps(expected),
            "tokenizer.json": "{}",
            "tokenizer_config.json": json.dumps({
                "chat_template": "{% for msg in messages %}{{msg.content}}{% endfor %}"
            }),
        }
        with tempfile.TemporaryDirectory() as td:
            for name, content in files.items():
                (Path(td) / name).write_text(content, encoding="utf-8")

            class Sibling:
                def __init__(self, path):
                    self.rfilename = path
                    self.size = 1024
                    self.blob_id = "test"

            class Info:
                sha = p["models"]["sprocket_chat"]["revision"]
                siblings = [Sibling(x) for x in [
                    "config.json", "tokenizer.json", "tokenizer_config.json",
                    "model.safetensors", "sprocket-500m-chat-f16.gguf"
                ]]

            fake_hub = types.SimpleNamespace(
                HfApi=lambda: types.SimpleNamespace(
                    model_info=lambda repo_id, revision, files_metadata: Info()
                ),
                hf_hub_download=lambda repo_id, revision, filename: str(
                    Path(td) / filename
                ),
            )
            with patch.dict("sys.modules", {"huggingface_hub": fake_hub}):
                result = ncr._source_preflight("sprocket_chat", p["models"]["sprocket_chat"])

        self.assertEqual(result["status"], "BLOCKED_REQUIRES_SOURCE_REVIEW")
        self.assertTrue(result["root_instruct_safetensors_present"])
        self.assertFalse(result["weight_file_present"])
        self.assertEqual(result["tap_indices"], [7, 16, 25])
        self.assertTrue(any("Missing requested trainable HF checkpoint" in reason
                            for reason in result["blockers"]))
        self.assertTrue(all("model.safetensors" not in key
                            for key in result["metadata_sha256"]))
        self.assertTrue(any(x["path"].endswith("-f16.gguf")
                            for x in result["known_reference_gguf"]))


if __name__ == "__main__":
    unittest.main()
