"""No-forward regression checks; optional official small-file and meta probes.

Set LFM25_SNAPSHOT to a hash-matched root snapshot to exercise the real processor
and meta-only PEFT structure. Set PAINTER_CORPUS for positive-admission TRAIN
framing. These tests never run model inference, loss, backward or an optimizer.
"""

import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("lfm25_lane_under_test", HERE / "adapter.py")
lane = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lane)
SNAPSHOT = os.environ.get("LFM25_SNAPSHOT")
CORPUS = os.environ.get("PAINTER_CORPUS")
SYSTEM = "Return only one valid CyberSand painter program JSON object. Follow the requested structure. Do not set host seeds, material permissions or focus."


class ProfileContractTests(unittest.TestCase):
    def test_import_does_not_initialize_torch(self):
        script = (
            "import importlib.util,sys; "
            f"s=importlib.util.spec_from_file_location('isolated_lane',{str(HERE / 'adapter.py')!r}); "
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m); "
            "assert 'torch' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_substituted_text_model_and_generic_targets_are_refused(self):
        for key, value in (("model_id", "LiquidAI/LFM2.5-350M"),
                           ("architecture", "Lfm2ForCausalLM"),
                           ("subfolder", "text"), ("trust_remote_code", True),
                           ("lora_targets", ["q_proj"])):
            with self.subTest(field=key):
                selected = copy.deepcopy(lane.PROFILE)
                selected[key] = value
                with self.assertRaisesRegex(ValueError, "profile mismatch"):
                    lane._selected(selected)

    def test_whitelist_is_complete_and_excludes_image_paths(self):
        targets = lane.PROFILE["lora_target_paths"]
        self.assertEqual(len(targets), 92)
        self.assertEqual(len(set(targets)), 92)
        self.assertTrue(all(name.startswith(lane.LANGUAGE_PREFIX) for name in targets))
        self.assertEqual(sum(".feed_forward." in name for name in targets), 48)
        self.assertEqual(sum(".self_attn." in name for name in targets), 24)
        self.assertEqual(sum(".conv." in name for name in targets), 20)
        self.assertFalse(any("vision" in name or "projector" in name for name in targets))
        pattern = lane.PROFILE["lora_targets"]
        self.assertTrue(all(re.fullmatch(pattern, name) for name in targets))
        for wrong in ("model.vision_tower.encoder.layers.0.self_attn.q_proj",
                      "model.multi_modal_projector.linear_1", "lm_head",
                      "model.language_model.layers.0.self_attn.q_proj",
                      "model.language_model.layers.2.conv.in_proj"):
            self.assertIsNone(re.fullmatch(pattern, wrong))

    def test_weight_and_processor_provenance_are_both_required(self):
        hashes = lane.PROFILE["required_hashes"]
        self.assertEqual(set(hashes), {"model.safetensors", "config.json", "generation_config.json",
                                      "tokenizer.json", "tokenizer_config.json", "processor_config.json",
                                      "chat_template.jinja", "LICENSE", "README.md"})
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))
        self.assertEqual(lane.PROFILE["license"], "lfm1.0")
        self.assertEqual(lane.PROFILE["card_license"], "other")


@unittest.skipUnless(SNAPSHOT, "set LFM25_SNAPSHOT for official processor/meta checks")
class OfficialProcessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = lane.load_processor(SNAPSHOT)

    def test_exact_prefix_and_single_supervised_eos(self):
        row = {"id": "unicode", "prompt": "Draw a pipe.",
               "completion": '{"label":"café – 窓","family":"pipework"}'}
        encoded = lane.encode_row(self.processor, row, SYSTEM, 2048)
        self.assertEqual(encoded["prefix_ids"], lane.encode_prompt(self.processor, row["prompt"], SYSTEM))
        self.assertEqual(encoded["input_ids"][:encoded["boundary"]], encoded["prefix_ids"])
        self.assertEqual(encoded["labels"][:encoded["boundary"]], [-100] * encoded["boundary"])
        supervised = encoded["labels"][encoded["boundary"]:]
        self.assertEqual(supervised[-1], 7)
        self.assertEqual(supervised.count(7), 1)
        self.assertEqual(self.processor.decode(supervised[:-1], skip_special_tokens=False), row["completion"])

    def test_oversize_target_is_refused_without_truncation(self):
        row = {"id": "oversize", "prompt": "Pipe.", "completion": '{"label":"' + " pipe" * 3000 + '"}'}
        with self.assertRaisesRegex(ValueError, "no truncation"):
            lane.encode_row(self.processor, row, SYSTEM, 2048)

    def test_image_input_and_forged_eos_are_refused(self):
        with self.assertRaisesRegex(ValueError, "image tokens"):
            lane.encode_prompt(self.processor, "Inspect <image>.", SYSTEM)
        row = {"id": "bad-eos", "prompt": "Pipe.", "completion": '{"x":"<|im_end|>"}'}
        with self.assertRaisesRegex(ValueError, "single true EOS"):
            lane.encode_row(self.processor, row, SYSTEM, 2048)

    def test_real_processor_serialization_and_attribution(self):
        from transformers import Lfm2VlProcessor
        with tempfile.TemporaryDirectory() as directory:
            self.processor.save_pretrained(directory)
            root = Path(directory)
            restored = Lfm2VlProcessor.from_pretrained(root, local_files_only=True, trust_remote_code=False)
            self.assertEqual(type(restored.image_processor).__name__, "Lfm2VlImageProcessor")
            self.assertEqual(restored.chat_template, self.processor.chat_template)
            sample = "é pipe – 窓"
            self.assertEqual(restored.tokenizer.encode(sample), self.processor.tokenizer.encode(sample))
            self.assertEqual(lane._hash_file(root / "LICENSE"), lane.PROFILE["required_hashes"]["LICENSE"])
            self.assertEqual(lane._hash_file(root / "UPSTREAM_README.md"), lane.PROFILE["required_hashes"]["README.md"])
            self.assertEqual(json.loads((root / "LFM_ADAPTER_NOTICE.json").read_text())["revision"], lane.REVISION)

    @unittest.skipUnless(CORPUS, "set PAINTER_CORPUS for positively admitted TRAIN rows")
    def test_all_admitted_train_rows_without_opening_final(self):
        common_path = HERE.parents[1] / "training" / "common.py"
        spec = importlib.util.spec_from_file_location("lfm25_custody_under_test", common_path)
        common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(common)
        original_open = Path.open
        touched = []

        def guarded_open(path, *args, **kwargs):
            touched.append(str(path))
            self.assertNotEqual(path.name.lower(), "final.jsonl")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            rows, _ = common.load_training_corpus(CORPUS)
        self.assertGreater(len(rows), 0)
        for row in rows:
            encoded = lane.encode_row(self.processor, row, SYSTEM, 2048)
            with self.subTest(id=row["id"]):
                self.assertEqual(encoded["input_ids"][-1], 7)
                self.assertEqual(encoded["input_ids"][:encoded["boundary"]], encoded["prefix_ids"])
                self.assertEqual(self.processor.decode(encoded["input_ids"][encoded["boundary"]:-1]), row["completion"])
                self.assertLessEqual(len(encoded["input_ids"]), lane.PROFILE["max_sequence_tokens"])
        self.assertFalse(any(Path(name).name.lower() == "final.jsonl" for name in touched))


@unittest.skipUnless(SNAPSHOT, "set LFM25_SNAPSHOT for real-class meta-only PEFT checks")
class MultimodalMetaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import Lfm2VlConfig, Lfm2VlForConditionalGeneration
        from peft import LoraConfig, get_peft_model
        config = Lfm2VlConfig.from_pretrained(SNAPSHOT, local_files_only=True)
        config._attn_implementation = "eager"
        with torch.device("meta"):
            base = Lfm2VlForConditionalGeneration(config).to(dtype=torch.float32)
            cls.model = get_peft_model(base, LoraConfig(
                r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                target_modules=lane.PROFILE["lora_targets"],
            ))
        lane.set_training_mode(cls.model)

    def test_full_model_has_only_expected_language_adapters(self):
        result = lane._validate_trainable(self.model, lane.PROFILE, expected_device="meta")
        self.assertTrue(result["meta_only_non_training_check"])
        self.assertEqual(result["frozen_base_parameters"], 448718848)
        self.assertEqual(result["trainable_parameters"], 2998272)
        self.assertEqual(result["frozen_vision_parameters"], 85843200)
        self.assertEqual(result["frozen_connector_parameters"], 8391680)
        self.assertFalse(result["vision_training"])
        self.assertFalse(result["connector_training"])

    def test_public_gate_cannot_misrepresent_meta_as_training(self):
        with self.assertRaisesRegex(ValueError, "device/dtype"):
            lane.validate_trainable(self.model, lane.PROFILE)

    def test_trainable_vision_is_refused(self):
        parameter = next(self.model.get_base_model().model.vision_tower.parameters())
        parameter.requires_grad_(True)
        try:
            with self.assertRaisesRegex(ValueError, "trainable allowlist"):
                lane._validate_trainable(self.model, lane.PROFILE, expected_device="meta")
            with self.assertRaisesRegex(ValueError, "vision/connector is trainable"):
                lane.set_training_mode(self.model)
        finally:
            parameter.requires_grad_(False)
            lane.set_training_mode(self.model)

    def test_recipe_changes_are_refused(self):
        config = self.model.peft_config["default"]
        for name, changed in (("r", 4), ("lora_alpha", 8), ("lora_dropout", 0.0),
                              ("bias", "all"), ("use_dora", True), ("use_rslora", True),
                              ("rank_pattern", {"q_proj": 4}), ("modules_to_save", ["lm_head"])):
            previous = getattr(config, name)
            setattr(config, name, changed)
            try:
                with self.subTest(field=name), self.assertRaisesRegex(ValueError, "LoRA recipe"):
                    lane._validate_trainable(self.model, lane.PROFILE, expected_device="meta")
            finally:
                setattr(config, name, previous)


if __name__ == "__main__":
    unittest.main(verbosity=2)
