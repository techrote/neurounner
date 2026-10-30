"""No-model, no-network, fail-closed #72 void.1 diagnostic evidence reconciliation.

The archived original-host one-step diagnostic is NOT comparative training or
an independently accepted scientific model qualification.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ncr_001_void_postdiagnostic_reconciliation as check  # noqa: E402
import ncr_001_void_nca_original_custody as custody  # noqa: E402


class PostDiagnosticVoid1ReconciliationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy = check.read(check.LEGACY)
        cls.original = custody.audit()
        cls.manifest = check.read(check.MANIFEST)
        cls.protocol = check.read(check.PROTOCOL)

    def test_source_bound_report_is_reproducible_from_original_archives(self):
        self.assertEqual(check.build_report(), check.read(check.OUT))

    def test_previous_historical_matrix_is_not_rewritten(self):
        old = self.legacy
        self.assertFalse(old["per_model_gate"])
        self.assertEqual(set(old["blocking_required_checks"]), set(check.THREE))
        self.assertEqual(old["optimizer_steps_observed"], 0)
        self.assertFalse(old["backward_executed"])
        self.assertEqual(old["evidence"]["max_context_generation_records"], 12)

    def test_new_diagnostic_does_not_open_scientific_or_training_gate(self):
        result = check.build_report()
        self.assertEqual(result["model_key"], "void1")
        self.assertTrue(result["source_runtime_pretraining_subgate"])
        self.assertFalse(result["per_model_gate"])
        self.assertFalse(result["full_72_conformance_pass"])
        self.assertFalse(result["train_authorized"])
        self.assertFalse(result["independent_scientific_review"])
        self.assertFalse(result["comparative_nca_mlp_training_performed"])
        self.assertFalse(result["heldouts_opened"])
        self.assertFalse(result["trained_checkpoint_created"])
        self.assertEqual(set(result["unaccepted_required_checks"]),
                         set(check.THREE))
        self.assertEqual(set(result["candidate_required_check_evidence"]),
                         set(check.THREE))
        self.assertTrue(all(value == check.OBSERVED for value in
                            result["candidate_required_check_evidence"].values()))

    def test_source_and_frozen_original_host_are_distinct_from_old_audit(self):
        result = check.build_report()
        self.assertEqual(result["original_physical_run"], 40765025332)
        self.assertEqual(result["original_custody_ci_run"], 40766551985)
        self.assertEqual(result["original_source_zip_sha256"],
                         "b153d177497f15227545bfe6b717a6430219a662f2b0b868e7a68b89682a90d0")
        self.assertEqual(result["original_host_zip_sha256"],
                         "8e9152042fb909f18b13589e6e21552c40f3104c4cc292cf9a643c600fbf08b5")
        self.assertEqual(result["diagnostic_observation"]["terminal_eos_id"], 258)
        self.assertEqual(result["diagnostic_observation"]["optimizer_steps"], 1)
        self.assertEqual(result["diagnostic_observation"]["original_source_tests"], 11)
        self.assertEqual(result["diagnostic_observation"]
                         ["custody_adversarial_mutations_refused"], 14)

    def test_eos_nonemission_and_unmeasured_legacy_impact_retained(self):
        result = check.build_report()["byte_eos_negative_results"]
        self.assertEqual(result["max_context_four_step_no_eos_records"], 12)
        self.assertFalse(result["learned_eos_stopping_qualified"])
        self.assertFalse(result["historical_train002b_eos_effect_measured"])

    def test_additional_cross_receipt_mutations_are_refused(self):
        def wrong_policy(l, r, m, p):
            p["train_authorized"] = True
        def collapsed_variants(l, r, m, p):
            p["models"]["sprocket_chat"] = p["models"]["sprocket_instruct"]
        def old_pass(l, r, m, p):
            l["per_model_gate"] = True
        def old_grad_promotion(l, r, m, p):
            l["matrix"]["nonzero_adapter_gradient_after_warmup"]["status"] = "PASS"
        def missing_negative_eos(l, r, m, p):
            l["evidence"]["max_context_generation_records"] = 0
        def cross_weight(l, r, m, p):
            r["actual_source_files"]["model.safetensors"]["sha256"] = "f" * 64
        def false_scientific_review(l, r, m, p):
            r["independent_scientific_review"] = True
        def multiple_steps(l, r, m, p):
            r["diagnostic_optimizer_steps"] = 2
        def altered_runner(l, r, m, p):
            r["runner"]["run_id"] = "40765025333"
        def corrupted_eos(l, r, m, p):
            r["synthetic_sample"]["real_eos_id"] = 62
        def noncausal_supervision(l, r, m, p):
            r["synthetic_sample"]["supervised_after_shift"] = 22
        def upstream_gradient_missing(l, r, m, p):
            r["post_warmup"]["gradient_l1"]["perception.weight"] = 0
        def readout_not_updated(l, r, m, p):
            r["warmup"]["readout_after_sha256"] = r["warmup"]["readout_before_sha256"]
        def nonfinite_loss(l, r, m, p):
            r["post_warmup"]["native_causal_eos_loss"] = float("nan")
        def field_reused(l, r, m, p):
            r["post_warmup"]["ordered_field_trace"][1]["field_sha256"] = (
                r["post_warmup"]["ordered_field_trace"][0]["field_sha256"])
        def frozen_base_changed(l, r, m, p):
            r["backbone_after_sha256"] = "0" * 64
        def switched_run(l, r, m, p):
            m["original_run"] = 40765025333
        def missing_check(l, r, m, p):
            del r["checks"]["native_shifted_byte_eos_loss"]

        mutations = (
            wrong_policy, collapsed_variants, old_pass, old_grad_promotion,
            missing_negative_eos, cross_weight, false_scientific_review,
            multiple_steps, altered_runner, corrupted_eos, noncausal_supervision,
            upstream_gradient_missing, readout_not_updated, nonfinite_loss,
            field_reused, frozen_base_changed, switched_run, missing_check,
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate.__name__):
                original = [
                    copy.deepcopy(self.legacy), copy.deepcopy(self.original),
                    copy.deepcopy(self.manifest), copy.deepcopy(self.protocol),
                ]
                mutate(*original)
                with self.assertRaisesRegex(RuntimeError, "REFUSED"):
                    check.check_inputs(*original)

    def test_typed_numeric_and_field_evidence_rejects_bool_and_string_forgery(self):
        # Python bool is an int subclass: True==1 and False==0. Neither is
        # an original measured loss, gradient, count or ordered field index.
        mutations = (
            ("bool_step", lambda r: r.__setitem__("diagnostic_optimizer_steps", True)),
            ("bool_readout", lambda r: r["warmup"]["gradient_l1"].__setitem__("readout.weight", True)),
            ("bool_zero_gradient", lambda r: r["warmup"]["gradient_l1"].__setitem__("rule.weight", False)),
            ("bool_post_gradient", lambda r: r["post_warmup"]["gradient_l1"].__setitem__("observe.weight", True)),
            ("numeric_string_gradient", lambda r: r["post_warmup"]["gradient_l1"].__setitem__("rule.weight", "1.0")),
            ("boolean_eos", lambda r: r["synthetic_sample"].__setitem__("real_eos_id", True)),
            ("boolean_tap", lambda r: r["post_warmup"]["ordered_field_trace"][1].__setitem__("tap_index", True)),
            ("boolean_next_tap", lambda r: r["post_warmup"]["ordered_field_trace"][0].__setitem__("next_tap", True)),
            ("boolean_shape", lambda r: r["post_warmup"]["ordered_field_trace"][0]["shape"].__setitem__(0, True)),
            ("invalid_digest", lambda r: r["post_warmup"]["ordered_field_trace"][0].__setitem__("field_sha256", "not-a-hash")),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                arguments = [copy.deepcopy(self.legacy),
                             copy.deepcopy(self.original),
                             copy.deepcopy(self.manifest),
                             copy.deepcopy(self.protocol)]
                mutate(arguments[1])
                with self.assertRaisesRegex(RuntimeError, "REFUSED"):
                    check.check_inputs(*arguments)

    def test_independent_original_utf8_eos_tensor_oracle(self):
        # An independent stdlib int64 witness for the original #172 public
        # unicode diagnostic, not a new tokenizer or physical-model run.
        expected = check.original_synthetic_oracle()
        self.assertEqual(expected["prompt_bytes"], 10)
        self.assertEqual(expected["source_text_utf8_bytes"], 30)
        self.assertEqual(expected["total_units"], 31)
        self.assertEqual(expected["supervised_after_shift"], 21)
        self.assertEqual(expected["real_eos_id"], 258)
        self.assertEqual(expected["ids_sha256"],
                         "ee6fa2a893301b858b7fdac7fe2b9190cd1fc786ddff1ffdf906da757488c09e")
        self.assertEqual(expected["labels_sha256"],
                         "260dec3e7622aeada51e0483e005ad9d59cbae8f186d41432c381f2c35835241")
        self.assertEqual(expected["mask_sha256"],
                         "88986881ba52ad36b430e41cd24497b9cf9a64f8e6a979290ebdee36b1b6f85c")
        observed = self.original["synthetic_sample"]
        self.assertEqual({k: observed[k] for k in expected}, expected)

    def test_original_int64_tensor_and_mask_mutations_refused(self):
        # In particular mask_sha256 was recorded by original #172 but NOT
        # independently reconstructed in the earlier custody #173 validator.
        mutations = (
            ("ids_sha256", "0" * 64),
            ("labels_sha256", "0" * 64),
            ("mask_sha256", "0" * 64),
            ("source_text_utf8_bytes", 29),
            ("source_text_utf8_bytes", True),
        )
        for field, forged in mutations:
            with self.subTest(field=field, forged=forged):
                original = copy.deepcopy(self.original)
                original["synthetic_sample"][field] = forged
                with self.assertRaisesRegex(RuntimeError, "REFUSED"):
                    check.check_inputs(self.legacy, original,
                                       self.manifest, self.protocol)

    def test_all_original_decoded_logs_match_git_blobs(self):
        # The four decoded files include both failed original custody runs.
        # Original ZIP SHA-256 checks alone never protect these separate logs.
        self.assertEqual(len(check.ORIGINAL_DECODED_LOG_BLOBS), 4)
        check.assert_original_decoded_git_blobs()

    def test_all_original_decoded_logs_refuse_single_byte_and_append_mutations(self):
        # Preserve the existing run/head/status text when corrupting a tail
        # byte. Substring-only original custody could otherwise pass.
        with tempfile.TemporaryDirectory() as temporary:
            replica = Path(temporary)
            originals = {}
            for name in check.ORIGINAL_DECODED_LOG_BLOBS:
                raw = (custody.EVIDENCE / name).read_bytes()
                self.assertTrue(raw, name)
                (replica / name).write_bytes(raw)
                originals[name] = raw
            check.assert_original_decoded_git_blobs(replica)
            for name, raw in originals.items():
                with self.subTest(log=name, mutation="last-byte"):
                    (replica / name).write_bytes(
                        raw[:-1] + bytes([raw[-1] ^ 1]))
                    with self.assertRaisesRegex(RuntimeError,
                                                "original decoded job-log Git blob drift"):
                        check.assert_original_decoded_git_blobs(replica)
                    (replica / name).write_bytes(raw)
                with self.subTest(log=name, mutation="append"):
                    (replica / name).write_bytes(raw + b"forged\\n")
                    with self.assertRaisesRegex(RuntimeError,
                                                "original decoded job-log Git blob drift"):
                        check.assert_original_decoded_git_blobs(replica)
                    (replica / name).write_bytes(raw)
            check.assert_original_decoded_git_blobs(replica)

    def test_identical_content_symlink_cannot_impersonate_committed_log(self):
        # Previous content-only blob recomputation followed symlinks. The
        # original committed Git entry could be a symlink yet pass the hash.
        with tempfile.TemporaryDirectory() as temporary:
            replica = Path(temporary)
            for name in check.ORIGINAL_DECODED_LOG_BLOBS:
                (replica / name).write_bytes((custody.EVIDENCE / name).read_bytes())
            check.assert_original_decoded_git_blobs(replica)
            name = next(iter(check.ORIGINAL_DECODED_LOG_BLOBS))
            shadow = replica / "identical-original-content.log"
            shadow.write_bytes((replica / name).read_bytes())
            (replica / name).unlink()
            (replica / name).symlink_to(shadow.name)
            with self.assertRaisesRegex(RuntimeError, "not a regular file"):
                check.assert_original_decoded_git_blobs(replica)

    def test_git_head_tree_refuses_mode_blob_and_missing_entry_mutations(self):
        # Files with unchanged *working-tree bytes* must still refuse a
        # different committed tree object or non-ordinary-file Git mode.
        name, blob = next(iter(check.ORIGINAL_DECODED_LOG_BLOBS.items()))
        rel = (custody.EVIDENCE / name).relative_to(check.ROOT).as_posix()
        cases = (
            ("executable_mode", f"100755 blob {blob}\t{rel}\n", 0),
            ("symlink_mode", f"120000 blob {blob}\t{rel}\n", 0),
            ("different_tree_blob", f"100644 blob {'0' * 40}\t{rel}\n", 0),
            ("missing_tree_entry", "", 0),
            ("git_lookup_failure", "", 1),
        )
        for label, output, returncode in cases:
            with self.subTest(mutation=label):
                result = SimpleNamespace(returncode=returncode, stdout=output)
                with mock.patch.object(check.subprocess, "run", return_value=result):
                    with self.assertRaisesRegex(RuntimeError, "Git tree object/mode drift"):
                        check.assert_original_decoded_git_blobs()

    def test_unmodified_archives_pass_cross_receipt_validator(self):
        check.check_inputs(self.legacy, self.original,
                           self.manifest, self.protocol)


if __name__ == "__main__":
    unittest.main()
