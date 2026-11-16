"""Model-free adversarial tests; all 60 prompts are SYNTHETIC."""
from __future__ import annotations
import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dev60_prefix_probe_v1 as gate


def sample():
    rows = []
    for family in gate.FAMILIES:
        for g in range(4):
            for form in sorted(gate.FORMS):
                prompt = f"FAKE {family} structure {g} formulation {form}: é <eos>\n"
                rows.append({"id": f"{family}-{g}-{form}", "family": family,
                             "group_id": f"{family}-{g}", "formulation_id": form,
                             "prompt": prompt,
                             "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()})
    return rows


class ProvisionalPrefixTests(unittest.TestCase):
    def setUp(self):
        self.rows = sample()

    def test_genuine_adapter_and_denominators(self):
        receipt = gate.audit(self.rows, 1024)
        self.assertEqual((receipt["requests"], receipt["groups"]), (60, 12))
        self.assertEqual(receipt["framing"], "cybersand-void-task-v1")
        self.assertEqual(receipt["terminal_eos_id"], 258)
        self.assertFalse(receipt["automatic_bos"])
        self.assertFalse(receipt["target_representability_checked"])
        self.assertFalse(receipt["dev60_frozen"])
        self.assertFalse(receipt["model_inference"])
        self.assertEqual(len(receipt["prefixes"]), 60)
        self.assertEqual(len({r["prefix_ids_sha256"] for r in receipt["prefixes"]}), 60)

    def test_immutable_reviewed_source_bytes(self):
        for filename, expected in (("adapter.py", gate.PINNED_ADAPTER_BLOB_SHA1),
                                   ("profile.json", gate.PINNED_PROFILE_BLOB_SHA1)):
            source = gate.HERE / filename
            self.assertEqual(gate._pinned_file_bytes(source, expected), source.read_bytes())
            with tempfile.TemporaryDirectory() as directory:
                changed = Path(directory) / filename
                changed.write_bytes(source.read_bytes() + b"\n# intentional drift\n")
                with self.assertRaisesRegex(ValueError, "unreviewed VOID pinned source"):
                    gate._pinned_file_bytes(changed, expected)

    def test_cached_module_cannot_substitute_for_reviewed_adapter(self):
        # A preimported module must not replace the source that was hashed.
        impostor = type(sys)("adapter")
        impostor.MODEL_ID = "unreviewed/foreign"
        sentinel = object()
        previous = sys.modules.get("adapter", sentinel)
        sys.modules["adapter"] = impostor
        try:
            result = gate.audit(self.rows, 1024)
            self.assertEqual(result["model_id"], "dotlabs/void.1")
            self.assertEqual(result["framing"], "cybersand-void-task-v1")
        finally:
            if previous is sentinel:
                sys.modules.pop("adapter", None)
            else:
                sys.modules["adapter"] = previous

    def test_canonical_request_set_identity(self):
        original = gate.audit(self.rows, 1024)
        reordered = gate.audit(list(reversed(self.rows)), 1024)
        self.assertEqual(original["request_set_sha256"], reordered["request_set_sha256"])
        self.assertEqual(original["adapter_git_blob_sha1"], gate.PINNED_ADAPTER_BLOB_SHA1)
        self.assertEqual(original["profile_git_blob_sha1"], gate.PINNED_PROFILE_BLOB_SHA1)
        other = copy.deepcopy(self.rows)
        other[0]["prompt"] += " changed"
        other[0]["prompt_sha256"] = hashlib.sha256(other[0]["prompt"].encode("utf-8")).hexdigest()
        self.assertNotEqual(original["request_set_sha256"],
                            gate.audit(other, 1024)["request_set_sha256"])

    def test_boundary_and_overshoot(self):
        max_prefix = gate.audit(self.rows, 1)["max_prefix_tokens"]
        self.assertEqual(gate.audit(self.rows, 2048 - max_prefix)["min_headroom"], 0)
        with self.assertRaisesRegex(ValueError, "context budget"):
            gate.audit(self.rows, 2049 - max_prefix)

    def test_nonpositive_boolean_and_excessive_ceiling(self):
        for ceiling in (0, -1, True, 2049):
            with self.subTest(ceiling=ceiling), self.assertRaises(ValueError):
                gate.audit(self.rows, ceiling)

    def test_group_form_denominators(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["formulation_id"] = rows[1]["formulation_id"]
        with self.assertRaisesRegex(ValueError, "duplicate group form"):
            gate.audit(rows, 1024)
        rows = copy.deepcopy(self.rows)
        rows[0]["group_id"] = "fifth-group"
        with self.assertRaises(ValueError):
            gate.audit(rows, 1024)

    def test_duplicate_id_and_missing_row(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["id"] = rows[1]["id"]
        with self.assertRaisesRegex(ValueError, "duplicate request ID"):
            gate.audit(rows, 1024)
        with self.assertRaisesRegex(ValueError, "exactly 60"):
            gate.audit(self.rows[:-1], 1024)

    def test_target_leakage_forbidden(self):
        for name in ("completion", "teacher", "target_png", "final_path"):
            rows = copy.deepcopy(self.rows)
            rows[0][name] = "not allowed"
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "targets forbidden"):
                gate.audit(rows, 1024)

    def test_strict_utf8_and_no_surrogate_repair(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["prompt"] = "\ud800"
        with self.assertRaises(UnicodeEncodeError):
            gate.audit(rows, 1024)

    def test_prompt_digest_mismatch(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["prompt"] = "other request"
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            gate.audit(rows, 1024)

    def test_duplicate_json_nonfinite_bom_bad_bytes(self):
        for raw in (b'{"rows":[],"rows":[]}', b'{"rows":NaN}',
                    b'\xef\xbb\xbf{"rows":[]}', b'{"rows":"\xff"}'):
            with self.subTest(raw=raw), self.assertRaises((ValueError, UnicodeDecodeError)):
                gate.strict_json(raw)

    def test_wrong_type_and_empty_request(self):
        for v in (None, True, 1, {}, [], ""):
            rows = copy.deepcopy(self.rows)
            rows[0]["prompt"] = v
            with self.subTest(value=v), self.assertRaises(ValueError):
                gate.audit(rows, 1024)


if __name__ == "__main__":
    unittest.main()
