"""Synthetic custody fixtures only: no model inference or native/image evidence."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest

LANE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LANE))
import contracts
import evaluate_native as evaluate
import native_gallery
import native_replay
from contracts import common

SOURCE = "a" * 40
PROGRAM = {"version": 2, "family": "greebles", "variation": 0,
           "requirements": {"negative_space_min_percent": 45},
           "components": [{"kind": "panel", "box": [8, 8, 24, 24], "detail": "vent",
                           "density": 2, "border": 1, "fasteners": True}]}


class Tokenizer:
    eos_token_id = 0

    def decode(self, tokens, skip_special_tokens=False):
        return "".join("<eos>" if token == 0 else chr(token - 1) for token in tokens)


def fixture(directory, *, arm="baseline", failure=None, attempted=True, raw_text=None, request_family="greebles"):
    """Exercise real evaluator serialization with an explicitly synthetic token fixture."""
    directory.mkdir(parents=True)
    requests = [{"id": f"fixture-{group}-s{seed}", "group_id": f"greebles/fixture_{group}/v1",
                 "structural_group": f"greebles/fixture_{group}/v1", "family": request_family,
                 "design": f"fixture_{group}", "split": "DEVELOPMENT", "prompt": f"Synthetic protocol request {group}",
                 "seed": seed, "focus": [0, 0, 96, 96], "material": 58}
                for group in range(6) for seed in (104729, 130363)]
    comparison = {"source_sha": SOURCE, "fixture": "model-free unit test"}
    checkpoint = None if arm == "baseline" else {"source_sha": SOURCE, "global_step": 120}
    identity = common.digest_json({"comparison": comparison, "arm": arm, "checkpoint": checkpoint})
    document = {"schema_version": 1, "arm": arm, "source_sha": SOURCE,
                "model_id": contracts.MODEL_ID, "model_revision": contracts.MODEL_REVISION,
                "compiler_sha256": contracts.COMPILER_SHA256,
                "checkpoint": checkpoint, "comparison_identity": comparison,
                "comparison_identity_sha256": common.digest_json(comparison),
                "evaluation_identity_sha256": identity, "cases": [], "generations": []}
    candidates = []
    text = raw_text if raw_text is not None else common.canonical(PROGRAM)
    token_ids = [ord(character) + 1 for character in text] + [0] if attempted else []
    compiler = common.compiler_module()
    for group in contracts.generation_groups(requests):
        generation = {"id": group["id"], "prompt": group["prompt"], "attempted": attempted, "failure": failure,
                      "request_ids": [r["id"] for r in group["requests"]],
                      "token_ids_sha256": common.digest_json(token_ids), "output_tokens": len(token_ids)}
        document["generations"].append(generation)
        for request in group["requests"]:
            verdict, program, compiled = contracts.classify_output(Tokenizer(), token_ids, request, compiler, failure)
            case = evaluate.emit_case(directory, request, generation, verdict, program, compiled, identity)
            document["cases"].append(case)
            if case["accepted"]:
                candidates.append({**request, "family": program["family"], "request_family": request["family"],
                                   "program": program, "compiled": compiled})
    cases = document["cases"]
    document["counts"] = {"case_views": 12, "structural_groups": 6, "distinct_prompts": 6,
                          "model_generations_attempted": 6 if attempted else 0,
                          "attempted_case_views": 12 if attempted else 0,
                          "accepted_case_views": len(candidates),
                          "accepted_generations": len({c["generation_id"] for c in cases if c["accepted"]}),
                          "family_matching_accepted_case_views": sum(c["accepted"] and c["requested_family_matches"] for c in cases)}
    common.write_json(directory / "accepted-candidates.json", {"schema_version": 1, "source_sha": SOURCE,
                      "compiler_sha256": contracts.COMPILER_SHA256, "evaluation_identity_sha256": identity,
                      "examples": candidates})
    document["accepted_candidates_sha256"] = common.sha256_file(directory / "accepted-candidates.json")
    common.write_json(directory / "evaluation.json", document)
    return requests, document


def save_changed_case(directory, document, index=0):
    """Re-sign test fixture metadata so deeper semantic checks are exercised."""
    case = document["cases"][index]
    for key, path in case["paths"].items():
        if key != "receipt":
            case["sha256"][key] = common.sha256_file(directory / path)
    receipt = {**case, "sha256": {key: value for key, value in case["sha256"].items() if key != "receipt"}}
    common.write_json(directory / case["paths"]["receipt"], receipt)
    case["sha256"]["receipt"] = common.sha256_file(directory / case["paths"]["receipt"])
    common.write_json(directory / "evaluation.json", document)


class NativeOutputCustodyTests(unittest.TestCase):
    def test_actual_evaluator_wire_format_recompiles_raw_programs_and_preserves_seed_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "baseline"
            requests, _ = fixture(path)
            result = native_replay.validate_evaluation(path, "baseline", requests, SOURCE)
            self.assertEqual(result["actual_generations"], 6)
            self.assertEqual(len(result["examples"]), 12)
            self.assertEqual({row["seed"] for row in result["examples"]}, {104729, 130363})
            self.assertTrue(all(row["id"].startswith("baseline-") for row in result["examples"]))

    def test_no_case_can_be_dropped_or_relabelled_as_a_different_host_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "baseline"
            requests, document = fixture(path)
            original = copy.deepcopy(document)
            document["cases"].pop()
            common.write_json(path / "evaluation.json", document)
            with self.assertRaisesRegex(ValueError, "twelve"):
                native_replay.validate_evaluation(path, "baseline", requests, SOURCE)
            original["cases"][0]["seed"] = 99
            common.write_json(path / "evaluation.json", original)
            with self.assertRaisesRegex(ValueError, "host fields"):
                native_replay.validate_evaluation(path, "baseline", requests, SOURCE)

    def test_raw_completion_hash_changes_and_repaired_program_files_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "baseline"
            requests, document = fixture(path)
            case = document["cases"][0]
            completion = path / case["paths"]["completion"]
            original = completion.read_bytes()
            completion.write_bytes(original + b" commentary")
            with self.assertRaisesRegex(ValueError, "custody hash"):
                native_replay.validate_evaluation(path, "baseline", requests, SOURCE)
            completion.write_bytes(original)
            common.write_json(path / case["paths"]["program"], {**PROGRAM, "variation": 1})
            save_changed_case(path, document)
            with self.assertRaisesRegex(ValueError, "repaired, substituted"):
                native_replay.validate_evaluation(path, "baseline", requests, SOURCE)

    def test_valid_looking_output_from_a_failed_generation_is_never_replayed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "baseline"
            requests, _ = fixture(path, failure="RuntimeError: interrupted")
            result = native_replay.validate_evaluation(path, "baseline", requests, SOURCE)
            self.assertEqual(result["examples"], [])
            self.assertEqual(result["actual_generations"], 6)
            self.assertTrue(all("interrupted" in row["refusal"] for row in result["outcomes"]))

    def test_total_deadline_keeps_twelve_views_but_reports_zero_actual_generations(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "baseline"
            requests, _ = fixture(path, attempted=False, failure="not attempted: total evaluation deadline")
            result = native_replay.validate_evaluation(path, "baseline", requests, SOURCE)
            self.assertEqual(result["actual_generations"], 0)
            self.assertEqual(result["examples"], [])
            self.assertEqual(len(result["outcomes"]), 12)

    def test_wrong_requested_family_retains_actual_native_program_family(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "baseline"
            requests, _ = fixture(path, request_family="pipework")
            result = native_replay.validate_evaluation(path, "baseline", requests, SOURCE)
            self.assertTrue(all(row["family"] == "greebles" and row["request_family"] == "pipework" for row in result["examples"]))
            self.assertFalse(any(row["family_agreement"] for row in result["outcomes"]))

    def test_symlink_evidence_and_receipt_relabelling_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "baseline"
            requests, document = fixture(path)
            case = document["cases"][0]
            receipt_path = path / case["paths"]["receipt"]
            receipt = common.read_json(receipt_path)
            receipt["seed"] = 999
            common.write_json(receipt_path, receipt)
            case["sha256"]["receipt"] = common.sha256_file(receipt_path)
            common.write_json(path / "evaluation.json", document)
            with self.assertRaisesRegex(ValueError, "receipt/verdict"):
                native_replay.validate_evaluation(path, "baseline", requests, SOURCE)
            save_changed_case(path, document)
            program = path / case["paths"]["program"]
            external = Path(temporary) / "external.json"
            external.write_bytes(program.read_bytes())
            program.unlink()
            program.symlink_to(external)
            with self.assertRaisesRegex(ValueError, "symlink"):
                native_replay.validate_evaluation(path, "baseline", requests, SOURCE)

    def test_zero_native_gallery_escapes_raw_html_and_displays_all_twenty_four_refusals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arms = {}
            for arm in native_replay.ARMS:
                path = root / arm
                requests, _ = fixture(path, arm=arm, raw_text='<script>alert("injected")</script>')
                arms[arm] = native_replay.validate_evaluation(path, arm, requests, SOURCE)
            (root / "native").mkdir()
            common.write_json(root / "native/runtime-custody.json", {"source_sha": SOURCE, "source_tree": "b" * 40,
                              "native_sha256": "c" * 64, "godot_version": "unit-fixture-no-runtime",
                              "godot_sha256": "d" * 64, "rapier_sha256": "e" * 64})
            report = {"passed": False, "cases": [], "global_gates": {}, "failures": ["Synthetic unit fixture; native not executed"]}
            common.write_json(root / "native/native-verification.json", report)
            summary = native_replay.summarize(SOURCE, {"source_sha": contracts.FOUNDATION_SOURCE}, arms, report, root)
            common.write_json(root / "model-native-summary.json", summary)
            manifest, page, copies = native_gallery.render(root, {"summary": summary, "arms": arms, "native_report": report})
            self.assertEqual(len(manifest["items"]), 24)
            self.assertEqual(page.count("<figure>"), 24)
            self.assertNotIn("<img", page)
            self.assertNotIn('<script>alert("injected")</script>', page)
            self.assertIn("&lt;script&gt;", page)
            self.assertEqual(copies, {})
            self.assertEqual(manifest["counts"]["native_verified_views"], 0)


if __name__ == "__main__":
    unittest.main()
