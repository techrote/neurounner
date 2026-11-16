"""Artificial encoder unit tests; never substitute for upstream hosted tokenizer."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "smol_probe", Path(__file__).with_name("dev60_prefix_preflight.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class FakeTokenizer:
    eos_token_id = 2
    chat_template = "fake-test-only-not-upstream"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "SYS:" + messages[0]["content"] + ";USER:" + messages[1]["content"] + ";ASSISTANT:"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


class Check(unittest.TestCase):
    def test_count_and_source_only(self):
        doc = probe.fixture()
        self.assertEqual(len(probe.validate_fixture(doc)), 60)
        result = probe.inspect(doc, FakeTokenizer(), "Fixed upstream system")
        self.assertEqual((result["synthetic_requests"], result["synthetic_groups"]), (60, 12))
        self.assertEqual(result["real_dev60_requests"], 0)
        self.assertFalse(result["final_opened"] or result["model_inference"])
        self.assertGreater(result["minimum_provisional_headroom"], 0)

    def assert_refuses(self, doc):
        with self.assertRaises(ValueError):
            probe.validate_fixture(doc)

    def test_duplicate_id(self):
        doc = probe.fixture()
        doc["requests"][1]["id"] = doc["requests"][0]["id"]
        self.assert_refuses(doc)

    def test_duplicate_prompt(self):
        doc = probe.fixture()
        doc["requests"][1]["prompt"] = doc["requests"][0]["prompt"]
        self.assert_refuses(doc)

    def test_real_development_forbidden(self):
        doc = probe.fixture()
        doc["status"] = "DEVELOPMENT"
        self.assert_refuses(doc)

    def test_teacher_field_forbidden(self):
        doc = probe.fixture()
        doc["requests"][0]["reference_program"] = {"version": 2}
        self.assert_refuses(doc)

    def test_missing_group_form(self):
        doc = probe.fixture()
        doc["requests"][1]["formulation_id"] = "F0"
        self.assert_refuses(doc)

    def test_bad_family(self):
        doc = probe.fixture()
        doc["requests"][0]["family"] = "final"
        self.assert_refuses(doc)

    def test_too_long(self):
        doc = probe.fixture()
        doc["requests"][0]["prompt"] += "x" * 500
        self.assert_refuses(doc)

    def test_context_overflow(self):
        with self.assertRaisesRegex(ValueError, "static context overflow"):
            probe.inspect(probe.fixture(), FakeTokenizer(), "system", context=200, limit=150)

    def test_missing_upstream_eos(self):
        t = FakeTokenizer()
        t.eos_token_id = 99
        with self.assertRaisesRegex(ValueError, "chat/EOS"):
            probe.inspect(probe.fixture(), t, "system")

    def test_hash_stability(self):
        doc = probe.fixture()
        a = probe.inspect(doc, FakeTokenizer(), "Fixed upstream system")
        b = probe.inspect(doc, FakeTokenizer(), "Fixed upstream system")
        self.assertEqual(probe.canonical(a), probe.canonical(b))
        self.assertEqual(len({x["prompt_sha256"] for x in a["records"]}), 60)


if __name__ == "__main__":
    unittest.main()
