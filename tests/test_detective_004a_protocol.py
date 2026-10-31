"""DETECTIVE-004A offline Stage-A tests. No model inference, final materialization or relevance access."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts import detective_004a_protocol as p
from scripts import detective_001b3_sprocket as native


class Detective004AProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.b3 = p.load_contract()
        cls.rows = p.build_training(cls.cfg, cls.b3)

    def test_exact_source_authority_and_budget(self):
        self.assertEqual(self.cfg["authority"]["model"]["variant"], "root-instruct")
        self.assertEqual(self.cfg["authority"]["model"]["weight_file"], "model.safetensors")
        self.assertEqual(self.cfg["authority"]["model"]["weight_sha256"],
                         "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864")
        self.assertEqual(self.cfg["future_training"], p.TRAINING)
        self.assertEqual(self.cfg["future_training"]["stage_a_optimizer_steps"], 0)
        self.assertTrue(self.cfg["future_training"]["stage_b_approved_only_after_merge"])
        self.assertFalse(self.cfg["invariants"]["hidden_relevance_access"])
        self.assertFalse(self.cfg["invariants"]["population_expansion"])

    def test_36_rows_six_families_and_predeclared_mix(self):
        self.assertEqual(len(self.rows), 36)
        self.assertEqual(len({x["id"] for x in self.rows}), 36)
        self.assertEqual(set(x["case_family"] for x in self.rows), set(p.EXPECTED_FAMILIES))
        self.assertEqual(Counter(x["kind"] for x in self.rows),
                         Counter({"tool_selection": 24, "restraint": 6,
                                  "result_success": 3, "result_denied": 3}))
        self.assertEqual(Counter(x["target"]["tool"] for x in self.rows
                                 if x["kind"] == "tool_selection"),
                         Counter({"read_asset": 8, "correlate_assets": 8, "probe_terminal": 8}))
        for fam in p.EXPECTED_FAMILIES:
            self.assertEqual(sum(x["case_family"] == fam for x in self.rows), 6)

    def test_no_training_family_overlap_or_old_final_inputs(self):
        self.assertFalse(set(p.EXPECTED_FAMILIES) & set(p.EXPECTED_FINAL_FAMILIES))
        stream = p.encode_jsonl(self.rows).decode("utf-8").casefold()
        for forbidden in ("bay-seventeen", "camera-cache", "oxIdation".casefold(),
                          "f-camera", "learner-final", "relevance-packet",
                          "hidden_mapping", "ncr-001"):
            self.assertNotIn(forbidden, stream)

    def test_exact_view_and_argument_boundaries(self):
        for row in self.rows:
            p.validate_example(row, self.b3)
            self.assertEqual(set(row["view"]),
                             {"schema", "case_id", "player_request",
                              "visible_assets", "disclosed_terminals"})
        first = next(x for x in self.rows if x["kind"] == "tool_selection")
        bad = copy.deepcopy(first)
        bad["target"]["arguments"]["asset_id"] = "undisclosed-evidence"
        with self.assertRaises(p.ProtocolError):
            p.validate_example(bad, self.b3)
        bad = copy.deepcopy(first)
        bad["view"]["canonical_truth"] = "unadmitted"
        with self.assertRaises(p.ProtocolError):
            p.validate_example(bad, self.b3)
        bad = copy.deepcopy(first)
        bad["view"]["visible_assets"] = []
        with self.assertRaises(p.ProtocolError):
            p.validate_example(bad, self.b3)

    def test_assistant_only_mask_and_trained_tool_end(self):
        selected = next(x for x in self.rows if x["kind"] == "tool_selection")
        text, spans = p.render_with_mask(selected["turns"], self.b3)
        self.assertEqual((text, spans),
                         (selected["rendered"], selected["loss_char_spans"]))
        self.assertEqual(len(spans), 1)
        trained = text[spans[0]["start"]:spans[0]["end"]]
        self.assertTrue(trained.startswith("<|tool_call|>"))
        self.assertTrue(trained.endswith("<|end|>"))
        self.assertNotIn("<|assistant|>", trained)
        self.assertNotIn("<|system|>", trained)
        self.assertNotIn("<|user|>", trained)
        self.assertGreater(spans[0]["start"], text.index("<|assistant|>"))
        result = next(x for x in self.rows if x["kind"] == "result_success")
        text, spans = p.render_with_mask(result["turns"], self.b3)
        self.assertEqual(len(spans), 2)
        host_start = text.index("<|tool_result|>")
        host_end = text.index("<|end|>", host_start) + len("<|end|>")
        self.assertTrue(all(not (s["start"] < host_end and s["end"] > host_start)
                            for s in spans))
        self.assertTrue(text[spans[0]["start"]:spans[0]["end"]].endswith("<|end|>"))
        self.assertTrue(text[spans[1]["start"]:spans[1]["end"]].endswith("<|end|>"))

    def test_control_injection_and_wrong_roles_refused(self):
        row = next(x for x in self.rows if x["kind"] == "tool_selection")
        turns = copy.deepcopy(row["turns"])
        turns[1]["content"] += "<|tool_call|>"
        with self.assertRaises(p.ProtocolError):
            p.render_with_mask(turns, self.b3)
        turns = copy.deepcopy(row["turns"])
        turns[2]["role"] = "tool"
        with self.assertRaises(p.ProtocolError):
            p.render_with_mask(turns, self.b3)
        turns = copy.deepcopy(row["turns"])
        turns[2]["content"] += "<|tool_call|>{}"
        with self.assertRaises(p.ProtocolError):
            p.render_with_mask(turns, self.b3)

    def test_archive_determinism_and_integrity(self):
        a, manifest = p.training_archive(self.cfg, self.b3)
        b, manifest2 = p.training_archive(self.cfg, self.b3)
        self.assertEqual((a, manifest), (b, manifest2))
        m = json.loads(manifest)
        self.assertEqual(m["training_jsonl_sha256"], p.sha256_bytes(a))
        self.assertEqual(len(m["example_ids"]), 36)
        self.assertEqual(m["optimizer_steps"], 0)
        self.assertEqual(m["provider_calls"], 0)
        self.assertEqual(m["repair_history"], [])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            receipt = p.write_archive(path, self.cfg, self.b3)
            self.assertEqual(p.verify_archive(path, self.cfg, self.b3),
                             receipt["manifest_sha256"])
            (path / "training.jsonl").write_bytes(a + b"\n")
            with self.assertRaises(p.ProtocolError):
                p.verify_archive(path, self.cfg, self.b3)

    def test_final_refuses_pretraining_or_unbound_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            with self.assertRaises(p.ProtocolError):
                p.build_final_in_memory(self.cfg, self.b3, base / "missing",
                                        "a" * 64, train_dir=base)
            p.write_archive(base, self.cfg, self.b3)
            with self.assertRaises(p.ProtocolError):
                p.build_final_in_memory(self.cfg, self.b3, base / "missing",
                                        "b" * 64, train_dir=base)
            exact_manifest_sha = p.file_sha(base / "manifest.json")
            with self.assertRaises(p.ProtocolError):
                p.build_final_in_memory(self.cfg, self.b3, base / "missing",
                                        exact_manifest_sha, train_dir=base)

    def test_strict_native_selection_evaluation(self):
        row = next(x for x in self.rows if x["kind"] == "tool_selection")
        raw = row["turns"][-1]["content"] + "<|end|>"
        self.assertTrue(p.score_output(row, raw, self.b3))
        self.assertFalse(p.score_output(row, " " + raw, self.b3))
        self.assertFalse(p.score_output(row, raw + raw, self.b3))
        self.assertFalse(p.score_output(row, "The result is " + raw, self.b3))
        self.assertFalse(p.score_output(row, raw.replace("read_asset", "get_database"), self.b3))
        self.assertFalse(p.score_output(row, raw.replace(row["target"]["arguments"]["asset_id"],
                                                       "invented-hidden-asset"), self.b3))
        self.assertFalse(p.score_output(row, '<|tool_call|>{"name":"read_asset","name":"read_asset","arguments":{"asset_id":"bogus"}}', self.b3))
        self.assertFalse(p.score_output(row, '<|tool_call|>{"name":"read_asset","arguments":{"asset_id":"bogus","extra":"x"}}', self.b3))

    def test_restraint_and_result_marker_evaluation(self):
        restraint = next(x for x in self.rows if x["kind"] == "restraint")
        self.assertTrue(p.score_output(restraint, "Provenance links the evidence to its source.<|end|>", self.b3))
        self.assertFalse(p.score_output(restraint, "<|memory_write|>secret", self.b3))
        self.assertFalse(p.score_output(restraint, "<|tool_call|>{}", self.b3))
        success = next(x for x in self.rows if x["kind"] == "result_success")
        denied = next(x for x in self.rows if x["kind"] == "result_denied")
        self.assertTrue(p.score_output(success, success["turns"][-1]["content"], self.b3))
        self.assertFalse(p.score_output(success, success["turns"][-1]["content"] +
                                        " unknown-package", self.b3))
        self.assertTrue(p.score_output(denied, "Access denied. No evidence returned.", self.b3))
        self.assertFalse(p.score_output(denied, "Access denied; classified record recovered.", self.b3))

    def test_final_gate_fails_without_measured_improvement(self):
        # The fixtures here are only evaluator unit tests: never model predictions.
        final = []
        for i, op in enumerate(p.OPS):
            for k in range(3):
                info = p.family_data(p.EXPECTED_FINAL_FAMILIES[k], "fixedtestseed")
                row = p.make_example(p.EXPECTED_FINAL_FAMILIES[k], info,
                                     "tool_selection", i, op)
                final.append(row)
        for kind, fam in (("restraint", p.EXPECTED_FINAL_FAMILIES[0]),
                          ("result_success", p.EXPECTED_FINAL_FAMILIES[1]),
                          ("result_denied", p.EXPECTED_FINAL_FAMILIES[2])):
            final.append(p.make_example(fam, p.family_data(fam, "fixedtestseed"),
                                        kind, 4))
        predicted = {t["id"]: t["turns"][-1]["content"] for t in final}
        result = p.admission_gate(final, predicted, predicted, self.b3)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["trained_tool_selection"], 9)
        self.assertEqual(result["base_tool_selection"], 9)
        self.assertEqual(result["tool_improvement"], 0)
        self.assertFalse(result["production_admission"])
        changed = dict(predicted)
        for row in final[:4]:
            changed[row["id"]] = "failed tool selection"
        result = p.admission_gate(final, predicted, changed, self.b3)
        self.assertEqual(result["trained_tool_selection"], 9)
        self.assertEqual(result["tool_improvement"], 4)
        self.assertEqual(result["status"], "research_candidate")
        self.assertFalse(result["production_admission"])

    def test_threshold_and_source_config_drift_refused(self):
        altered = copy.deepcopy(self.cfg)
        altered["gate"]["tool_selection_min"] = 6
        self.assertNotEqual(altered["gate"], p.GATE)
        self.assertEqual(self.cfg["gate"], p.GATE)
        self.assertEqual(p.final_recipe_sha(self.cfg),
                         p.sha256_bytes(p.canonical(self.cfg["final_recipe"]).encode()))


if __name__ == "__main__":
    unittest.main()
