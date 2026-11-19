import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from village_001e_generation_checkpoint import GenerationCheckpoint, json_bytes, validate_token_custody, write_immutable
from village_001e_build_final_suite import build_suite, collect_forbidden_prompts, sha256_file
from village_001e_ecology_plan import build_sessions
from village_001e_generate_final_responses import run_heldout
from village_001e_ecology_run import generate_turn


class FakeTokens:
    def __init__(self, tokens):
        self.tokens = tokens
        self.shape = (1, len(tokens))
    def __getitem__(self, key):
        return FakeTokens(self.tokens[key[1]]) if isinstance(key, tuple) else self
    def numel(self):
        return len(self.tokens)
    def tolist(self):
        return list(self.tokens)


class FakeTokenizer:
    eos_token_id = 0
    def __init__(self):
        self.requests = []
    def apply_chat_template(self, messages, **kwargs):
        self.requests.append(messages)
        return FakeTokens([20, 21])
    def decode(self, tokens, **kwargs):
        return "decoded " + ",".join(str(x) for x in tokens.tolist())


class FakeModel:
    def __init__(self):
        self.config = type("Config", (), {"use_cache": False})()
        self.calls = 0
        self.adapters = []
    def eval(self):
        pass
    def set_adapter(self, name):
        self.adapters.append(name)
    def generate(self, ids, **kwargs):
        self.calls += 1
        return FakeTokens(ids.tolist() + [30, 31])


class GenerationCustodyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def checkpoint(self, path="first", resume=None, identity=None):
        def validate(key, row, prior):
            if row["id"] != key:
                raise RuntimeError("wrong source row")
        return GenerationCheckpoint(out=self.root / path, resume=resume,
            identity=identity or {"model": "pinned-revision", "suite": "frozen-bytes"},
            keys=["F01", "F02"], terminal_name="result.json", validate_row=validate)

    def test_partial_resume_preserves_first_row_bytes_and_appends(self):
        original = self.checkpoint()
        original.append("F01", {"id": "F01", "response": "first answer"})
        first_bytes = (original.out / "rows/001-F01.json").read_bytes()
        resumed = self.checkpoint("resumed", original.out)
        self.assertEqual(resumed.rows, original.rows)
        resumed.append("F02", {"id": "F02", "response": "second answer"})
        resumed.finish({"responses": resumed.rows}, "responses")
        terminal_bytes = (resumed.out / "result.json").read_bytes()
        replay = self.checkpoint("replay", resumed.out)
        self.assertEqual(replay.terminal("responses")["responses"], resumed.rows)
        self.assertEqual((replay.out / "result.json").read_bytes(), terminal_bytes)
        self.assertEqual((replay.out / "rows/001-F01.json").read_bytes(), first_bytes)
        self.assertFalse((original.out / "rows/002-F02.json").exists())

    def test_source_identity_drift_refuses_resume(self):
        original = self.checkpoint()
        original.append("F01", {"id": "F01"})
        with self.assertRaisesRegex(RuntimeError, "replace existing evidence"):
            self.checkpoint("resumed", original.out, {"model": "changed"})

    def test_hashed_recovery_receipts_survive_resume_and_reject_tampering(self):
        import hashlib
        original = self.checkpoint()
        data = json_bytes({"run_id": 1, "artifact_sha256": "a" * 64})
        name = "resume-artifact-custody-" + hashlib.sha256(data).hexdigest() + ".json"
        write_immutable(original.out / name, data)
        resumed = self.checkpoint("resumed", original.out)
        self.assertEqual((resumed.out / name).read_bytes(), data)
        (resumed.out / name).write_bytes(data + b" ")
        with self.assertRaisesRegex(RuntimeError, "custody hash mismatch"):
            self.checkpoint("tampered", resumed.out)

    def test_existing_output_cannot_be_silently_overwritten(self):
        original = self.checkpoint()
        with self.assertRaisesRegex(RuntimeError, "existing output"):
            self.checkpoint()
        with self.assertRaisesRegex(RuntimeError, "disjoint"):
            self.checkpoint("first/child", original.out)
        with self.assertRaisesRegex(RuntimeError, "disjoint"):
            self.checkpoint("first", original.out)

    def test_changed_prior_row_breaks_chain(self):
        original = self.checkpoint()
        original.append("F01", {"id": "F01", "response": "first"})
        original.append("F02", {"id": "F02", "response": "second"})
        path = original.out / "rows/001-F01.json"
        data = json.loads(path.read_text())
        data["row"]["response"] = "different answer"
        path.write_bytes(json_bytes(data))
        with self.assertRaisesRegex(RuntimeError, "previous_checkpoint_sha256 mismatch"):
            self.checkpoint("resumed", original.out)

    def test_nonprefix_and_out_of_order_rows_are_rejected(self):
        original = self.checkpoint()
        with self.assertRaisesRegex(RuntimeError, "out of order"):
            original.append("F02", {"id": "F02"})
        original.append("F01", {"id": "F01"})
        (original.out / "rows/001-F01.json").rename(original.out / "rows/002-F02.json")
        with self.assertRaisesRegex(RuntimeError, "consecutive prefix"):
            self.checkpoint("resumed", original.out)

    def test_symlink_resume_and_terminal_corruption_are_rejected(self):
        original = self.checkpoint()
        original.append("F01", {"id": "F01"})
        original.append("F02", {"id": "F02"})
        original.finish({"responses": original.rows}, "responses")
        path = original.out / "result.json"
        data = json.loads(path.read_text())
        data["responses"][0]["id"] = "tampered"
        path.write_bytes(json_bytes(data))
        resumed = self.checkpoint("resumed", original.out)
        with self.assertRaisesRegex(RuntimeError, "differs"):
            resumed.terminal("responses")
        (original.out / "injected").symlink_to(path)
        with self.assertRaisesRegex(RuntimeError, "unsafe resume entry"):
            self.checkpoint("linked", original.out)

    def test_native_token_budget_and_count_custody(self):
        row = {"input_token_ids": [1, 2], "generated_token_ids": [3, 4], "generated_tokens": 2, "rendered_prompt": "native prompt"}
        validate_token_custody(row, maximum=2)
        with self.assertRaisesRegex(RuntimeError, "budget"):
            validate_token_custody(row, maximum=1)
        with self.assertRaisesRegex(RuntimeError, "count/custody"):
            validate_token_custody({**row, "generated_tokens": 1}, maximum=2)
        with self.assertRaisesRegex(RuntimeError, "invalid token"):
            validate_token_custody({**row, "input_token_ids": [True]}, maximum=2)

    def test_final_suite_is_immutable_and_matches_frozen_D(self):
        recipe = ROOT / "experiments/village-001/final-eval.json"
        expected = build_suite(recipe=json.loads(recipe.read_text()), recipe_sha256=sha256_file(recipe),
            preference_summary_sha256=sha256_file(ROOT / "experiments/village-001/preference-evidence/summary.json"),
            preference_release_sha256=sha256_file(ROOT / "experiments/village-001/preference-evidence/adapter-release.json"),
            forbidden=collect_forbidden_prompts())
        frozen = ROOT / "experiments/village-001/qualification-20261119/final-eval.json"
        self.assertEqual(frozen.read_bytes(), json_bytes(expected))
        path = self.root / "frozen.json"
        write_immutable(path, frozen.read_bytes())
        write_immutable(path, frozen.read_bytes())
        with self.assertRaisesRegex(RuntimeError, "replace"):
            write_immutable(path, b"changed suite")

    def test_ecology_matrix_handles_two_and_three_participant_sessions(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/village_001e_ecology_plan.py"), "--github-output"], capture_output=True, text=True, check=True)
        matrix = json.loads(result.stdout.split("=", 1)[1])["include"]
        self.assertEqual(len(matrix), 24)
        self.assertEqual(sum(not row["m3"] for row in matrix), 12)
        self.assertEqual(sum(bool(row["m3"]) for row in matrix), 12)
        config = json.loads((ROOT / "experiments/village-001/tournament-config.json").read_text())
        sessions = build_sessions(config)
        self.assertEqual(sum(len(session["members"]) * config["ecology"]["rounds"] for session in sessions), 180)

    def test_heldout_generator_calls_only_missing_scenarios(self):
        checkpoint = GenerationCheckpoint(out=self.root / "local", resume=None,
            identity={"fixed": "source"}, keys=["F01", "F02"],
            terminal_name="result.json", validate_row=lambda key, row, prior: None)
        first = {"scenario_id": "F01", "response": "archived first answer"}
        checkpoint.append("F01", first)
        model, tok = FakeModel(), FakeTokenizer()
        suite = {"system": "fixed public instruction", "max_new_tokens": 160,
            "do_sample": False, "scenarios": [{"id": "F01", "family": "mundane", "prompt": "first"}, {"id": "F02", "family": "mundane", "prompt": "second"}]}
        rows = run_heldout(model, tok, suite, checkpoint)
        self.assertEqual(model.calls, 1)
        self.assertEqual(rows[0], first)
        self.assertEqual(tok.requests[0][1]["content"], "second")
        self.assertEqual(rows[1]["input_token_ids"], [20, 21])
        self.assertEqual(rows[1]["generated_token_ids"], [30, 31])
        self.assertIs(model.config.use_cache, False)

    def test_ecology_turn_uses_only_public_transcript_and_opaque_handles(self):
        model, tok = FakeModel(), FakeTokenizer()
        row = generate_turn(model, tok, "M2", "M2", {"text": "A public situation"},
            [{"handle": "M1", "member_id": "V-G01-A", "text": "Prior public words"}], 96)
        prompt = json.dumps(tok.requests)
        self.assertIn("Prior public words", prompt)
        self.assertNotIn("V-G01-A", prompt)
        self.assertEqual(model.adapters, ["M2"])
        self.assertEqual(row["generated_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
