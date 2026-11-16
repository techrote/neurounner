"""Model-free regressions; these tests never import Torch or execute a model."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

LANE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LANE))
import contracts
from contracts import common
import evaluate_native as evaluate

spec = importlib.util.spec_from_file_location("smol_lane_train_wrapper", LANE / "train.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


class CharacterTokenizer:
    eos_token_id = 0

    def decode(self, tokens, skip_special_tokens=False):
        return "".join("<eos>" if token == 0 else chr(token - 1) for token in tokens)


def tokens(text, eos=True):
    return [ord(character) + 1 for character in text] + ([0] if eos else [])


PROGRAM = {"version": 2, "family": "greebles", "variation": 0,
           "requirements": {"negative_space_min_percent": 45},
           "components": [{"kind": "panel", "box": [8, 8, 24, 24], "detail": "vent",
                           "density": 2, "border": 1, "fasteners": True}]}
REQUEST = {"id": "request-s104729", "group_id": "greebles/test/v1", "structural_group": "greebles/test/v1",
           "family": "greebles", "design": "test", "split": "DEVELOPMENT", "prompt": "Draw one panel.",
           "seed": 104729, "focus": [0, 0, 96, 96], "material": 58}


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.compiler = common.compiler_module()
        self.tokenizer = CharacterTokenizer()

    def classify(self, text, eos=True, request=None, failure=None):
        return contracts.classify_output(self.tokenizer, tokens(text, eos), request or REQUEST,
                                         self.compiler, failure)

    def test_valid_json_without_true_eos_is_retained_and_refused(self):
        text = common.canonical(PROGRAM)
        verdict, program, compiled = self.classify(text, eos=False)
        self.assertEqual(verdict["raw_text"], text)
        self.assertTrue(verdict["json_valid"])
        self.assertTrue(verdict["program_valid"])
        self.assertFalse(verdict["accepted"])
        self.assertIsNone(program)
        self.assertIsNone(compiled)

    def test_duplicate_json_and_model_owned_host_seed_are_refused(self):
        duplicate = common.canonical(PROGRAM).replace('"version":2', '"version":2,"version":2')
        for text in (duplicate, common.canonical({**PROGRAM, "seed": 123})):
            with self.subTest(text=text):
                verdict, program, compiled = self.classify(text)
                self.assertFalse(verdict["accepted"])
                self.assertEqual(verdict["raw_text"], text)
                self.assertEqual(verdict["token_ids"], tokens(text))
                self.assertIsNone(program)
                self.assertIsNone(compiled)

    def test_excessively_nested_json_remains_a_raw_refusal(self):
        text = "[" * 2000 + "0" + "]" * 2000
        verdict, program, compiled = self.classify(text)
        self.assertFalse(verdict["accepted"])
        self.assertEqual(verdict["raw_text"], text)
        self.assertEqual(verdict["token_ids"], tokens(text))
        self.assertTrue(verdict["failure"])
        self.assertIsNone(program)
        self.assertIsNone(compiled)

    def test_raw_program_byte_limit_matches_native_parser_without_whitespace_repair(self):
        text = " " * self.compiler.MAX_PROGRAM_BYTES + common.canonical(PROGRAM)
        verdict, program, compiled = self.classify(text)
        self.assertTrue(verdict["json_valid"])
        self.assertTrue(verdict["program_valid"])
        self.assertFalse(verdict["accepted"])
        self.assertEqual(verdict["raw_text"], text)
        self.assertIn("byte budget", verdict["compile_failure"])
        self.assertIsNone(program)
        self.assertIsNone(compiled)

    def test_valid_wrong_family_is_separate_from_request_agreement(self):
        request = {**REQUEST, "family": "pipework"}
        verdict, program, compiled = self.classify(common.canonical(PROGRAM), request=request)
        self.assertTrue(verdict["accepted"])
        self.assertFalse(verdict["requested_family_matches"])
        self.assertEqual(program["family"], compiled["family"])
        self.assertEqual(compiled["seed"], request["seed"])
        self.assertEqual(compiled["material"], request["material"])

    def test_deadline_preserves_all_twelve_views_without_synthetic_generations(self):
        requests = [{**REQUEST, "id": f"request-{group}-s{seed}", "prompt": f"Request {group}",
                     "group_id": f"group-{group}", "seed": seed}
                    for group in range(6) for seed in (104729, 130363)]
        groups = contracts.generation_groups(requests)
        self.assertEqual(len(groups), 6)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            cases = []
            for group in groups:
                generation = {"id": group["id"], "request_ids": [r["id"] for r in group["requests"]],
                              "attempted": False, "failure": "not attempted: total evaluation deadline"}
                for request in group["requests"]:
                    verdict, program, compiled = contracts.classify_output(self.tokenizer, [], request,
                                                                           self.compiler, generation["failure"])
                    cases.append(evaluate.emit_case(output, request, generation, verdict, program, compiled, "a" * 64))
            self.assertEqual(len(cases), 12)
            self.assertEqual(sum(case["generation_reused"] for case in cases), 6)
            self.assertFalse(any(case["generation_attempted"] or case["accepted"] for case in cases))
            self.assertEqual(len(list(output.glob("cases/*/receipt.json"))), 12)
            for case in cases:
                self.assertEqual(case["verdict"]["token_ids"], [])
                for name, path in case["paths"].items():
                    self.assertEqual(common.sha256_file(output / path), case["sha256"][name])
                expected_receipt = copy.deepcopy(case)
                del expected_receipt["sha256"]["receipt"]
                self.assertEqual(common.read_json(output / case["paths"]["receipt"]), expected_receipt)

    def test_generation_exception_cannot_accept_a_valid_partial_stream(self):
        verdict, program, _compiled = self.classify(common.canonical(PROGRAM), failure="RuntimeError: interrupted")
        self.assertFalse(verdict["accepted"])
        self.assertTrue(verdict["eos_terminated"])
        self.assertIsNone(program)

    def test_stream_preserves_partial_tokens_and_rejects_wrong_prompt(self):
        class Value:
            def __init__(self, values):
                self.values = values

            def tolist(self):
                return self.values

        recorder = evaluate.TokenRecorder([3, 4])
        recorder.put(Value([[3, 4]]))
        recorder.put(Value([5]))
        recorder.put(Value([6]))
        self.assertEqual(recorder.tokens, [5, 6])
        with self.assertRaisesRegex(ValueError, "prompt differs"):
            evaluate.TokenRecorder([3, 4]).put(Value([[7, 8]]))

    def test_cli_requires_baseline_without_adapter_candidate_with_adapter(self):
        base = ["--source-sha", "a" * 40, "--model", "model", "--corpus", "corpus", "--output", "out"]
        self.assertIsNone(evaluate.arguments(base + ["--arm", "baseline"]).checkpoint)
        self.assertEqual(evaluate.arguments(base + ["--arm", "candidate", "--checkpoint", "adapter"]).checkpoint,
                         Path("adapter"))
        for invalid in (["--arm", "candidate"], ["--arm", "baseline", "--checkpoint", "adapter"],
                        ["--arm", "baseline", "--max-new-tokens", "1025"]):
            with self.subTest(invalid=invalid), self.assertRaises(SystemExit):
                evaluate.arguments(base + invalid)

    def test_training_wrapper_keeps_the_qualified_recipe_and_budget(self):
        base = ["--source-sha", "a" * 40, "--model", "model", "--corpus", "corpus", "--output", "out"]
        args = wrapper.arguments(base)
        self.assertEqual((args.steps, args.checkpoint_every, args.max_seconds, args.seed,
                          args.max_length, args.threads, args.profile), (120, 40, 1200, 568, 2048, 1, "smol"))
        for invalid in (["--steps", "121"], ["--max-seconds", "1201"], ["--checkpoint-every", "41"]):
            with self.subTest(invalid=invalid), self.assertRaises(SystemExit):
                wrapper.arguments(base + invalid)

    def test_checkpoint_requires_current_shared_smol_framing_adapter_identity(self):
        from lane_adapters import adapter_for
        selected = common.profile("smol", require_supported=True)
        self.assertIsNone(adapter_for(selected))
        identity = contracts.adapter_identity(selected)
        self.assertEqual(identity["lane"], "smol")
        self.assertEqual(identity["framing"], "upstream-chat-template")
        # Protocol-only metadata fixture; no checkpoint tensors or model exist.
        config = {"profile": "smol", "model": selected, "source_sha": "a" * 40,
                  "native_source_sha": contracts.FOUNDATION_SOURCE,
                  "corpus_manifest_sha256": contracts.MANIFEST_SHA256,
                  "train_sha256": contracts.TRAIN_SHA256, "compiler_sha256": contracts.COMPILER_SHA256,
                  "trainer_sha256": common.sha256_file(contracts.SHARED / "train.py"),
                  "contract_sha256": common.sha256_file(contracts.SHARED / "common.py"),
                  "model_files": {}, "numerics": {}, "packages": {}, "python": "metadata-only",
                  "device": "cpu", "dtype": "float32", "threads": 1, "seed": 568,
                  "max_length": 2048, "rank": 8, "alpha": 16, "dropout": 0.05, "learning_rate": 0.0005,
                  "system": "system", "system_sha256": common.digest_json("system"),
                  "chat_template_sha256": common.digest_json("template"), "adapter_contract": identity,
                  "native_execution_by_trainer": False, "final_targets_opened": False}
        arguments = (selected, "a" * 40, {}, {}, "system", "template", {}, "metadata-only")
        contracts.validate_candidate(config, *arguments)
        for contract in (None, {**identity, "framing": "different-template"},
                         {**identity, "source_files": {"training/lane_adapters.py": "0" * 64}}):
            with self.subTest(contract=contract), self.assertRaisesRegex(ValueError, "adapter_contract"):
                contracts.validate_candidate({**config, "adapter_contract": contract}, *arguments)

    def test_local_model_execution_is_refused_before_importing_torch(self):
        with self.assertRaisesRegex(ValueError, "GitHub-hosted"):
            contracts.hosted_source("a" * 40, {})
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)


if __name__ == "__main__":
    unittest.main()
