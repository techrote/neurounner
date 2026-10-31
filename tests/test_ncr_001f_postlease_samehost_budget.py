"""Source/schedule refusal tests only; NEVER download frozen models or train."""
from __future__ import annotations
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001f_postlease_samehost_budget as budget


class SameHostBudgetTests(unittest.TestCase):
    def test_real_four_lane_protocol_and_distinct_sprocket_sources(self):
        chat, p, a = budget.check_contract("chat")
        instruct, q, b = budget.check_contract("instruct")
        self.assertIs(p["train_authorized"], False)
        self.assertEqual(set(p["models"]), {"smollm2", "sprocket_chat",
                                            "sprocket_instruct", "void1"})
        self.assertEqual(p, q)
        self.assertEqual(chat["weight_path"], "chat/model.safetensors")
        self.assertEqual(instruct["weight_path"], "model.safetensors")
        self.assertNotEqual(chat["weight_sha256"], instruct["weight_sha256"])
        self.assertEqual(a["hf_weight_file"], chat["weight_path"])
        self.assertEqual(b["hf_weight_file"], instruct["weight_path"])

    def test_refuse_any_unregistered_measurement_lengths_or_rounds(self):
        bad = [
            ((128, 2048), 1, 3), ((512, 1024), 1, 3),
            ((2048, 512), 1, 3), ((512, 2048), 0, 3),
            ((512, 2048), 1, 2), ((512, 2048), 1, 4),
            ((512, 2048), True, 3), ((512, 2048), 1, True),
            ([512, 2048], 1, 3),
        ]
        for lengths, warmups, repeats in bad:
            with self.subTest(lengths=lengths, warmups=warmups, repeats=repeats):
                with self.assertRaisesRegex(ValueError, "REFUSED"):
                    budget.validate_budget(lengths, warmups, repeats)

    def test_exact_three_path_interleaving_and_raw_sample_counts(self):
        sequence = []
        funcs = {
            kind: (lambda kind=kind: sequence.append(kind))
            for kind in budget.KINDS
        }
        result = budget.measure_schedule(funcs)
        expected = list(budget.KINDS) + [
            kind for order in result["forward_call_order"] for kind in order
        ]
        self.assertEqual(sequence, expected)
        self.assertEqual(result["forward_call_order"], [
            ["frozen", "nca", "mlp"],
            ["nca", "mlp", "frozen"],
            ["mlp", "frozen", "nca"],
        ])
        for kind in budget.KINDS:
            self.assertEqual(len(result["raw_complete_frozen_host_forward_ns"][kind]), 3)
            self.assertGreater(result["median_complete_host_forward_ms"][kind], 0)
        self.assertTrue(result["wall_time_includes_full_backbone"])
        self.assertFalse(result["measured_flops"])
        self.assertFalse(result["actual_compute_budget_qualified"])

    def test_reject_incomplete_or_forged_path_inventory(self):
        funcs = {kind: (lambda: None) for kind in budget.KINDS}
        for mutation in ("missing", "extra", "noncallable"):
            changed = dict(funcs)
            if mutation == "missing":
                changed.pop("mlp")
            elif mutation == "extra":
                changed["void1"] = lambda: None
            else:
                changed["mlp"] = object()
            with self.subTest(mutation=mutation):
                with self.assertRaisesRegex(ValueError, "REFUSED"):
                    budget.measure_schedule(changed)

    def test_original_shared_model_and_lease_are_mandatory(self):
        lock = object()
        class Wrapper:
            def __init__(self, model, lock):
                self.backbone = model
                self._backbone_lock = lock
                self.nca = self.control = None
        class Model:
            is_gradient_checkpointing = False
            def parameters(self):
                return iter(())
        model = Model()
        nca, mlp = Wrapper(model, lock), Wrapper(model, lock)
        nca.nca = type("NCA", (), {"trainable_parameters":lambda _:658904})()
        mlp.control = type("MLP", (), {"trainable_parameters":lambda _:658432})()
        budget.require_same_backbone(model, nca, mlp, "chat")
        with self.assertRaisesRegex(ValueError, "SAME frozen model"):
            budget.require_same_backbone(Model(), nca, mlp, "chat")
        mlp._backbone_lock = object()
        with self.assertRaisesRegex(ValueError, "decoder-root ownership"):
            budget.require_same_backbone(model, nca, mlp, "chat")
        mlp._backbone_lock = lock
        mlp.control = type("MLP", (), {"trainable_parameters":lambda _:1})()
        with self.assertRaisesRegex(ValueError, "parameter counts"):
            budget.require_same_backbone(model, nca, mlp, "chat")

    def test_train_and_scientific_acceptance_always_false_at_start(self):
        for variant in ("chat", "instruct"):
            r = budget.base_report(variant)
            for key in ("train_authorized", "full_72_conformance_pass",
                        "actual_compute_budget_qualified", "scientific_admission",
                        "heldouts_opened", "training_performed",
                        "comparative_nca_mlp_training",
                        "generated_tool_memory_semantics_qualified"):
                self.assertIs(r[key], False, key)
            self.assertEqual(r["optimizer_steps"], 0)
            self.assertFalse(r["frozen_host_loaded"])

    def test_effective_timing_threads_not_misreported_from_startup(self):
        r = budget.base_report("chat")
        self.assertIsNone(r["runner"]["cpu_threads"])
        self.assertIsInstance(r["runner"]["cpu_threads_at_report_start"], int)
        r["runner"]["cpu_threads"] = 4
        with patch.object(budget.torch, "get_num_threads", return_value=4):
            budget.require_runtime_threads(r)
        with patch.object(budget.torch, "get_num_threads", return_value=2):
            with self.assertRaisesRegex(ValueError, "four-thread"):
                budget.require_runtime_threads(r)
        r["runner"]["cpu_threads"] = 2
        with patch.object(budget.torch, "get_num_threads", return_value=4):
            with self.assertRaisesRegex(ValueError, "four-thread"):
                budget.require_runtime_threads(r)

    def test_wrong_variant_and_modified_protocol_are_refused(self):
        for variant in ("void1", "root", "sprocket_chat"):
            with self.subTest(variant=variant):
                with self.assertRaisesRegex(ValueError, "REFUSED"):
                    budget.check_contract(variant)
        real = budget.gitblob
        with patch.object(budget, "gitblob", return_value="0" * 40):
            with self.assertRaisesRegex(ValueError, "Git blob drift"):
                budget.check_contract("chat")
        self.assertTrue(callable(real))

    def test_failed_callable_never_returns_a_success_report(self):
        funcs = {kind: (lambda: None) for kind in budget.KINDS}
        def fail():
            raise RuntimeError("EXPECTED synthetic runtime failure")
        funcs["nca"] = fail
        with self.assertRaisesRegex(RuntimeError, "EXPECTED synthetic"):
            budget.measure_schedule(funcs)


if __name__ == "__main__":
    unittest.main()
