"""Sprocket contract tests; real-tokenizer/meta tests require explicit local inputs.

No test loads weight tensors, runs a forward pass or performs optimization.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

LANE = Path(__file__).resolve().parent
TRAINING = LANE.parents[1] / "training"
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(LANE))
import adapter
from common import load_training_corpus, verify_file

MODEL = os.environ.get("SPROCKET_TOKENIZER_DIR")
CORPUS = os.environ.get("SPROCKET_TRAIN_CORPUS")
META = os.environ.get("SPROCKET_META_INSPECTION") == "1"
SYSTEM = "Return only one valid CyberSand painter program JSON object. Follow the requested structure. Do not set host seeds, material permissions or focus."


class ProfileContract(unittest.TestCase):
    def test_import_does_not_initialize_torch(self):
        code = "import sys;sys.path[:0]=sys.argv[1:];import adapter;assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-c", code, str(LANE), str(TRAINING)], check=True)

    def test_refuses_chat_checkpoint_remote_code_and_broad_lora(self):
        for key, value in (("subfolder", "chat"), ("trust_remote_code", True),
                           ("lora_targets", "all-linear"), ("revision", "main")):
            with self.subTest(key=key):
                selected = copy.deepcopy(adapter.PROFILE)
                selected[key] = value
                with self.assertRaisesRegex(ValueError, "profile mismatch"):
                    adapter.validate_profile(selected)
        selected = copy.deepcopy(adapter.PROFILE)
        selected["required_hashes"]["model.safetensors"] = "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97"
        with self.assertRaisesRegex(ValueError, "required_hashes"):
            adapter.validate_profile(selected)


class RealTokenizerContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL:
            raise unittest.SkipTest("set --model for the actual pinned root tokenizer")
        from numerics import configure_cpu
        if "torch" not in sys.modules:
            configure_cpu(1)
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        adapter.validate_tokenizer(cls.tokenizer)

    def test_every_small_upstream_file_matches_reviewed_hash(self):
        for name, expected in adapter.PROFILE["required_hashes"].items():
            if name != "model.safetensors":
                with self.subTest(name=name):
                    verify_file(MODEL, name, expected)

    def test_framing_unicode_and_complete_loss_boundary(self):
        row = {"id": "unicode-framing", "prompt": "Draw a panel — with λ and café labels.",
               "completion": json.dumps({"text": "λ café", "value": 1}, ensure_ascii=False, separators=(",", ":"))}
        encoded = adapter.encode_row(self.tokenizer, row, SYSTEM, 2048)
        expected_prefix = "<|system|>" + SYSTEM + "<|end|><|user|>" + row["prompt"] + "<|end|><|assistant|>"
        self.assertEqual(self.tokenizer.decode(encoded["prefix_ids"], skip_special_tokens=False), expected_prefix)
        self.assertEqual(encoded["prefix_ids"], adapter.encode_prompt(self.tokenizer, row["prompt"], SYSTEM))
        target = encoded["input_ids"][encoded["boundary"]:]
        self.assertEqual(target.count(5), 1)
        self.assertEqual(target[-1], 5)
        self.assertEqual(self.tokenizer.decode(target[:-1], skip_special_tokens=False), row["completion"])
        self.assertEqual(encoded["labels"][:encoded["boundary"]], [-100] * encoded["boundary"])
        self.assertEqual(encoded["labels"][encoded["boundary"]:], target)
        self.assertNotIn(0, encoded["input_ids"])
        self.assertNotIn(1, encoded["input_ids"])

    def test_refuses_short_budget_instead_of_truncating_json_or_eos(self):
        row = {"id": "complete-budget", "prompt": "Draw a service panel.", "completion": '{"complete":true}'}
        full = adapter.encode_row(self.tokenizer, row, SYSTEM, 2048)
        self.assertEqual(adapter.encode_row(self.tokenizer, row, SYSTEM, len(full["input_ids"])), full)
        with self.assertRaisesRegex(ValueError, "no truncation"):
            adapter.encode_row(self.tokenizer, row, SYSTEM, len(full["input_ids"]) - 1)
        with self.assertRaisesRegex(ValueError, "1..2048"):
            adapter.encode_row(self.tokenizer, row, SYSTEM, 2049)

    def test_refuses_control_tokens_in_system_prompt_and_completion(self):
        for marker in ("<|assistant|>", "<|end|>", "<think>", "<|reserved_4|>"):
            with self.subTest(marker=marker):
                with self.assertRaisesRegex(ValueError, "control or unknown"):
                    adapter.encode_prompt(self.tokenizer, "Draw " + marker, SYSTEM)
                with self.assertRaisesRegex(ValueError, "control or unknown"):
                    adapter.encode_prompt(self.tokenizer, "Draw pipes.", SYSTEM + marker)
                with self.assertRaisesRegex(ValueError, "control or unknown"):
                    adapter.encode_row(self.tokenizer, {"id": "injection", "prompt": "Draw pipes.", "completion": '{"x":"' + marker + '"}'}, SYSTEM, 2048)

    def test_refuses_template_or_eos_drift(self):
        template = self.tokenizer.chat_template
        try:
            self.tokenizer.chat_template = template + " "
            with self.assertRaisesRegex(ValueError, "exact upstream"):
                adapter.encode_prompt(self.tokenizer, "Draw pipes.", SYSTEM)
        finally:
            self.tokenizer.chat_template = template
        eos = self.tokenizer.eos_token
        try:
            self.tokenizer.eos_token = self.tokenizer.pad_token
            with self.assertRaisesRegex(ValueError, "BOS/PAD/EOS"):
                adapter.encode_prompt(self.tokenizer, "Draw pipes.", SYSTEM)
        finally:
            self.tokenizer.eos_token = eos

    def test_every_native_admitted_train_row_fits_and_roundtrips(self):
        if not CORPUS:
            self.skipTest("set --corpus for native-admitted TRAIN verification")
        rows, _ = load_training_corpus(CORPUS)
        self.assertEqual(len(rows), 12)
        lengths = []
        for row in rows:
            with self.subTest(row=row["id"]):
                encoded = adapter.encode_row(self.tokenizer, row, SYSTEM, 2048)
                self.assertEqual(encoded["prefix_ids"], adapter.encode_prompt(self.tokenizer, row["prompt"], SYSTEM))
                self.assertEqual(encoded["labels"].count(5), 1)
                self.assertEqual(encoded["labels"][-1], 5)
                lengths.append(len(encoded["input_ids"]))
        self.assertEqual((min(lengths), max(lengths)), (185, 286))


class MetaProjectionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL or not META:
            raise unittest.SkipTest("set --model and --meta for shape-only model inspection")
        from numerics import configure_cpu
        if "torch" not in sys.modules:
            configure_cpu(1)
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        from peft import LoraConfig, TaskType, get_peft_model
        config = AutoConfig.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        with torch.device("meta"):
            base = AutoModelForCausalLM.from_config(config, dtype=torch.float32, attn_implementation="eager")
            cls.model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8,
                lora_alpha=16, lora_dropout=0.05, target_modules=adapter.PROFILE["lora_targets"]), low_cpu_mem_usage=True)

    def test_exact_actual_meta_lora_shapes_and_frozen_base(self):
        result = adapter._trainable_inventory(self.model, adapter.PROFILE)
        self.assertEqual(result["lora_modules"], 182)
        self.assertEqual(result["trainable_tensors"], 364)
        self.assertEqual(result["trainable_parameters"], 4739072)
        self.assertEqual(result["frozen_parameters"], 501090560)
        self.assertTrue(all(parameter.device.type == "meta" for parameter in self.model.parameters()))
        with self.assertRaisesRegex(ValueError, "materialized CPU"):
            adapter.validate_trainable(self.model, adapter.PROFILE)

    def test_rejects_training_any_frozen_embedding(self):
        embedding = self.model.get_base_model().get_input_embeddings().weight
        try:
            embedding.requires_grad_(True)
            with self.assertRaisesRegex(ValueError, "trainable tensors"):
                adapter._trainable_inventory(self.model, adapter.PROFILE)
        finally:
            embedding.requires_grad_(False)

    def test_rejects_missing_lora_gradient_parameter(self):
        parameter = next(parameter for parameter in self.model.parameters() if parameter.requires_grad)
        try:
            parameter.requires_grad_(False)
            with self.assertRaisesRegex(ValueError, "trainable tensors"):
                adapter._trainable_inventory(self.model, adapter.PROFILE)
        finally:
            parameter.requires_grad_(True)

    def test_rejects_peft_recipe_changes(self):
        config = self.model.peft_config["default"]
        changes = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.0, "bias": "all",
                   "task_type": "FEATURE_EXTRACTION", "modules_to_save": ["lm_head"],
                   "use_dora": True, "use_rslora": True, "rank_pattern": {"q_proj": 16},
                   "alpha_pattern": {"q_proj": 32}, "lora_bias": True}
        for key, value in changes.items():
            original = getattr(config, key)
            with self.subTest(key=key):
                try:
                    setattr(config, key, value)
                    with self.assertRaisesRegex(ValueError, "exact PEFT recipe"):
                        adapter._trainable_inventory(self.model, adapter.PROFILE)
                finally:
                    setattr(config, key, original)

    def test_rejects_module_recipe_drift_even_with_unchanged_config(self):
        module = self.model.get_base_model().get_submodule("model.layers.0.self_attn.q_proj")
        original = module.scaling["default"]
        try:
            module.scaling["default"] = 4.0
            with self.assertRaisesRegex(ValueError, "actual LoRA"):
                adapter._trainable_inventory(self.model, adapter.PROFILE)
        finally:
            module.scaling["default"] = original

    def test_training_mode_disables_cache(self):
        self.model.eval()
        self.model.config.use_cache = True
        adapter.set_training_mode(self.model)
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.config.use_cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="pinned root small-file or complete snapshot directory")
    parser.add_argument("--corpus", help="native-admitted corpus directory; only TRAIN targets are read")
    parser.add_argument("--meta", action="store_true", help="inspect actual model/LoRA shapes without allocating weights")
    args, remaining = parser.parse_known_args()
    MODEL, CORPUS, META = args.model or MODEL, args.corpus or CORPUS, args.meta or META
    unittest.main(argv=[sys.argv[0]] + remaining)
