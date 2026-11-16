"""Offline VOID byte/serialization contracts; optional meta-only architecture check.

Default discovery needs only Python. VOID_META_TESTS=1 enables native Transformers
and PEFT shape inspection on the meta device, never inference or optimization.
VOID_TRAIN_CORPUS enables verification/budgeting of the admitted TRAIN partition;
common.load_training_corpus never reads FINAL targets or images.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "training"))
spec = importlib.util.spec_from_file_location("void_contract_adapter", HERE / "adapter.py")
adapter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = adapter
spec.loader.exec_module(adapter)
PROFILE = json.loads((HERE / "profile.json").read_text())


class ByteContract(unittest.TestCase):
    def setUp(self):
        self.tokenizer = adapter.CyberSandVoidByteTokenizer.from_pretrained(HERE / "upstream")

    def test_import_does_not_load_numerical_libraries(self):
        code = ("import importlib.util,sys;"
                f"s=importlib.util.spec_from_file_location('void_check',{str(HERE / 'adapter.py')!r});"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                "assert 'torch' not in sys.modules and 'transformers' not in sys.modules and 'peft' not in sys.modules")
        subprocess.run([sys.executable, "-c", code], check=True, timeout=15)

    def test_preserved_source_hashes(self):
        for name, expected in PROFILE["required_hashes"].items():
            if name != "model.safetensors":
                self.assertEqual(hashlib.sha256((HERE / "upstream" / name).read_bytes()).hexdigest(), expected)
        self.assertEqual(PROFILE["upstream_tokenizer_auto_map"],
                         {"AutoTokenizer": ["tokenization_byte.ByteTokenizer", None]})
        config = json.loads((HERE / "upstream/config.json").read_text())
        self.assertNotIn("auto_map", config)

    def test_utf8_not_bpe_or_codepoints(self):
        for text in ("", "one two", "\x00\t\n\r", "é", "日本語", "🤖", "e\u0301", "é 日本語 🤖"):
            ids = self.tokenizer.encode(text)
            self.assertEqual(ids, list(text.encode("utf-8")))
            self.assertEqual(self.tokenizer.decode(ids), text)
            self.assertEqual(self.tokenizer.encode(text, add_special_tokens=True), ids)
        self.assertEqual(self.tokenizer.encode("🤖"), [240, 159, 164, 150])

    def test_all_259_symbols_and_literal_special_strings(self):
        self.assertEqual(set(self.tokenizer.get_vocab().values()), set(range(259)))
        self.assertEqual(len(self.tokenizer), 259)
        self.assertIsNone(self.tokenizer.chat_template)
        self.assertEqual(self.tokenizer.decode([256, 257, 258]), "<pad><bos><eos>")
        text = "literal <pad> <bos> <eos> and <0x41>"
        self.assertEqual(self.tokenizer.encode(text), list(text.encode()))
        self.assertTrue(all(value < 256 for value in self.tokenizer.encode(text)))
        self.assertEqual(self.tokenizer.decode([256, 65, 258], skip_special_tokens=True), "A")

    def test_malformed_utf8_and_unicode_are_refused(self):
        for values in ([255], [128], [194], [192, 128], [237, 160, 128], [244, 144, 128, 128], [194, 258, 162]):
            with self.subTest(values=values), self.assertRaises(UnicodeDecodeError):
                self.tokenizer.decode(values)
        for text in ("\ud800", "a\udfffz"):
            with self.assertRaises(UnicodeEncodeError):
                self.tokenizer.encode(text)
        for values in ([-1], [259], [True], [65.0], ["65"]):
            with self.assertRaises(ValueError):
                self.tokenizer.decode(values)
        with self.assertRaises(ValueError):
            self.tokenizer.decode([65], clean_up_tokenization_spaces=True)

    def test_checkpoint_serialization_and_corruption_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.tokenizer.save_pretrained(directory)
            self.assertEqual({Path(path).name for path in paths},
                             {"byte_vocab.json", "special_tokens_map.json", "tokenizer_config.json"})
            loaded = adapter.CyberSandVoidByteTokenizer.from_pretrained(directory)
            self.assertEqual(loaded.get_vocab(), self.tokenizer.get_vocab())
            self.assertEqual(loaded.encode("字 <eos>\n"), self.tokenizer.encode("字 <eos>\n"))
            config_path = Path(directory) / "tokenizer_config.json"
            config = json.loads(config_path.read_text())
            self.assertNotIn("auto_map", config)
            self.assertNotIn("chat_template", config)
            config["eos_token"] = "fake"
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "saved tokenizer contract"):
                adapter.CyberSandVoidByteTokenizer.from_pretrained(directory)
            self.tokenizer.save_pretrained(directory)
            vocab_path = Path(directory) / "byte_vocab.json"
            vocab = json.loads(vocab_path.read_text())
            vocab["<0x41>"] = 66
            vocab_path.write_text(json.dumps(vocab))
            with self.assertRaisesRegex(ValueError, "byte vocabulary"):
                adapter.CyberSandVoidByteTokenizer.from_pretrained(directory)

    def test_framing_boundary_supervises_complete_bytes_and_eos(self):
        row = {"id": "contract", "prompt": "Draw é <eos>\nCyberSand", "completion": '{"message":"🤖 <eos>"}'}
        encoded = adapter.encode_row(self.tokenizer, row, "Only JSON", 2048)
        prefix = adapter.encode_prompt(self.tokenizer, row["prompt"], "Only JSON")
        expected_prefix = 'cybersand-void-task-v1\n{"instruction":"Only JSON","request":"Draw é <eos>\\nCyberSand"}\n'
        self.assertEqual(self.tokenizer.decode(prefix), expected_prefix)
        self.assertEqual(encoded["input_ids"][:encoded["boundary"]], prefix)
        self.assertEqual(encoded["prefix_ids"], prefix)
        self.assertEqual(encoded["labels"][:encoded["boundary"]], [-100] * len(prefix))
        self.assertEqual(encoded["labels"][encoded["boundary"]:], list(row["completion"].encode()) + [258])
        self.assertEqual(encoded["input_ids"].count(258), 1)
        self.assertEqual(encoded["labels"][-1], 258)
        self.assertNotIn(257, encoded["input_ids"])

    def test_exact_context_limit_and_no_truncation(self):
        row = {"id": "budget", "prompt": "draw", "completion": '{"v":2}'}
        encoded = adapter.encode_row(self.tokenizer, row, "Only JSON", 2048)
        exact = len(encoded["input_ids"])
        self.assertEqual(adapter.encode_row(self.tokenizer, row, "Only JSON", exact), encoded)
        with self.assertRaisesRegex(ValueError, "no truncation"):
            adapter.encode_row(self.tokenizer, row, "Only JSON", exact - 1)
        for cap in (0, 2049, True):
            with self.assertRaises(ValueError):
                adapter.encode_row(self.tokenizer, row, "Only JSON", cap)
        with self.assertRaisesRegex(ValueError, "no truncation"):
            adapter.encode_prompt(self.tokenizer, "x" * 2048, "Only JSON")

    @unittest.skipUnless(os.environ.get("VOID_TRAIN_CORPUS"), "set VOID_TRAIN_CORPUS for admitted TRAIN custody/budgets")
    def test_actual_native_admitted_train_budget(self):
        from common import load_training_corpus, sha256_file
        from train import SYSTEM
        directory = Path(os.environ["VOID_TRAIN_CORPUS"])
        rows, manifest = load_training_corpus(directory)
        expected = json.loads((HERE / "train-byte-budget.json").read_text())
        self.assertEqual(sha256_file(directory / "admitted-corpus.json"), expected["corpus_manifest_sha256"])
        self.assertEqual(manifest["source_sha"], expected["native_source_sha"])
        measurements = []
        for row in rows:
            encoded = adapter.encode_row(self.tokenizer, row, SYSTEM, 2048)
            measurements.append({"id": row["id"], "family": row["family"],
                                 "prompt_bytes": encoded["prompt_bytes"], "prefix_bytes": encoded["boundary"],
                                 "completion_bytes": encoded["completion_bytes"],
                                 "sequence_tokens": len(encoded["input_ids"]), "target_tokens": encoded["target_tokens"]})
            self.assertEqual(encoded["labels"][-1], 258)
            self.assertEqual(self.tokenizer.decode(encoded["labels"][encoded["boundary"]:-1]), row["completion"])
        self.assertEqual(measurements, expected["rows"])
        self.assertEqual(len(rows), 12)


@unittest.skipUnless(os.environ.get("VOID_META_TESTS") == "1", "set VOID_META_TESTS=1 for meta-only Transformers/PEFT inspection")
class MetaArchitectureContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from numerics import configure_cpu
        configure_cpu(1)
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from peft import LoraConfig, get_peft_model
        with torch.device("meta"):
            base = Qwen3ForCausalLM(Qwen3Config.from_pretrained(HERE / "upstream", local_files_only=True))
            cls.model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16,
                                                       lora_dropout=0.05, bias="none",
                                                       target_modules=PROFILE["lora_targets"]))

    def test_native_qwen3_lora_shape_and_freeze_contract(self):
        diagnostics = adapter._trainable_inventory(self.model, PROFILE, allow_meta=True)
        self.assertEqual(diagnostics["target_modules"], 168)
        self.assertEqual(diagnostics["trainable_tensors"], 336)
        self.assertEqual(diagnostics["trainable_parameters"], 1984512)
        self.assertGreater(diagnostics["meta_parameters"], 0)
        self.assertTrue(all(parameter.device.type == "meta" for parameter in self.model.parameters()))

    def test_production_validation_refuses_meta_parameters(self):
        with self.assertRaisesRegex(ValueError, "materialized CPU float32"):
            adapter.validate_trainable(self.model, PROFILE)

    def test_unfrozen_base_weights_are_refused(self):
        embeddings = self.model.get_base_model().get_input_embeddings().weight
        embeddings.requires_grad_(True)
        try:
            with self.assertRaisesRegex(ValueError, "unauthorized base"):
                adapter._trainable_inventory(self.model, PROFILE, allow_meta=True)
        finally:
            embeddings.requires_grad_(False)

    def test_missing_actual_lora_projection_is_refused(self):
        layer = self.model.get_base_model().model.layers[0].self_attn
        original = layer.q_proj
        layer.q_proj = original.base_layer
        try:
            with self.assertRaisesRegex(ValueError, "target set"):
                adapter._trainable_inventory(self.model, PROFILE, allow_meta=True)
        finally:
            layer.q_proj = original

    def test_valid_utf8_matches_exact_reviewed_upstream_implementation(self):
        source = HERE / "upstream/tokenization_byte.py"
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), adapter.SOURCE_TOKENIZER_SHA256)
        module_spec = importlib.util.spec_from_file_location("void_pinned_test_tokenizer", source)
        upstream = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = upstream
        module_spec.loader.exec_module(upstream)
        original = upstream.ByteTokenizer.from_pretrained(HERE / "upstream", local_files_only=True,
                                                          trust_remote_code=False)
        strict = adapter.CyberSandVoidByteTokenizer.from_pretrained(HERE / "upstream")
        for text in ("", "abc\x00\r\n", "é 日本語 🤖 e\u0301", "<eos><pad><bos><0x41>",
                     "".join(chr(value) for value in range(256))):
            self.assertEqual(original.encode(text, add_special_tokens=False), strict.encode(text))
            self.assertEqual(original.decode(strict.encode(text)), text)
        self.assertEqual(original.decode([255]), "\ufffd")
        with self.assertRaises(UnicodeDecodeError):
            strict.decode([255])


if __name__ == "__main__":
    unittest.main()
