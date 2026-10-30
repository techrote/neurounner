"""Adversarial four-lane source-only compute/accounting preflight. NO TRAIN."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import ncr_001f_fourlane_phase_budget as gate

def protocol():
    return json.loads((ROOT/"experiments/ncr-001/protocol.json").read_text(encoding="utf-8"))

class FourLaneBudget(unittest.TestCase):
    def test_original_archival_provenance_and_missing_actual_training(self):
        x=gate.build()  # Parent independently audits all 3 original ZIPs + 3 logs.
        self.assertEqual(x["parent_source_head"],gate.PARENT)
        self.assertEqual(x["original_corrected_physical_run"],40904839128)
        self.assertEqual(x["status"],gate.STATUS)
        self.assertEqual(set(x["lanes"]),set(gate.WIDTHS))
        self.assertEqual(x["preregistered_updates_per_seed_per_arm"],192)
        self.assertEqual(x["planned_updates_per_arm_not_executed"],576)
        self.assertEqual(x["actual_optimizer_updates_in_this_run"],0)
        self.assertEqual(x["required_physical_phase_samples"],
                         ["forward_ns","backward_ns","optimizer_ns"])
        for field in ("actual_compute_budget_qualified",
                      "train_authorized","full_72_conformance_pass",
                      "reported_source_mac_are_measured_flops",
                      "forward_only_is_matched_training_compute",
                      "cross_model_performance_pooled",
                      "original_sealed_fixture_contents_opened",
                      "original_physical_execution_in_this_run"):
            self.assertIs(x[field],False)
        self.assertIs(x["no_actual_training_phase_receipts_present"],True)
        for name,item in x["lanes"].items():
            self.assertEqual(item["model_key"],name)
            self.assertEqual(item["checkpoint_weight_file"],gate.WEIGHT_FILES[name])
            self.assertEqual(item["taps_post_block_zero_based"],gate.TAPS[name])
            self.assertIsNone(item["matched_training_wall_cost_ratio"])
            self.assertIsNone(item["actual_training_tokens_observed"])
            self.assertIsNone(item["measured_forward_backward_optimizer_training_cost_ns"])
            self.assertFalse(item["matched_training_compute_qualified"])

    def test_four_distinct_in_model_costs_not_cross_model_scores(self):
        cases={
          576:(884736,1290240,1019904,297048,296576),
          640:(983040,1388544,1118208,329944,329472),
          1280:(1966080,2371584,2101248,658904,658432),
        }
        for hidden, expected in cases.items():
            with self.subTest(hidden=hidden):
                x=gate.estimate(hidden)
                m=x["counted_mac_per_token_not_flops"]
                self.assertEqual((
                    m["parameter_matched_gated_mlp"],
                    m["persistent_nca_3_steps"],
                    m["persistent_nca_1_step"],
                    x["parameter_count"]["nca_all_steps"],
                    x["parameter_count"]["gated_mlp"],
                ),expected)
                self.assertEqual(m["reset_each_tap_nca_3_steps"],
                                 m["persistent_nca_3_steps"])
                self.assertEqual(m["permuted_neighborhood_nca_3_steps"],
                                 m["persistent_nca_3_steps"])
                self.assertEqual(m["frozen_base_adapter_only"],0)
                self.assertIs(x["measured_hardware_flops"],False)
                self.assertIs(x["measured_training_compute"],False)
                self.assertEqual(x["parameter_count"]["absolute_difference"],472)
                self.assertEqual(x["extra_primary_local_conv_mac_per_token"],405504)

    def test_same_width_does_not_make_chat_root_same_checkpoint(self):
        x=gate.build()["lanes"]
        a,b=x["sprocket_chat"],x["sprocket_instruct"]
        self.assertEqual(a["source_cost_proxy"],b["source_cost_proxy"])
        self.assertNotEqual(a["checkpoint_weight_file"],b["checkpoint_weight_file"])
        self.assertNotEqual(a["original_physical_job_id"],b["original_physical_job_id"])
        self.assertTrue(a["original_untrained_complete_host_forward_samples_available"])
        self.assertTrue(b["original_untrained_complete_host_forward_samples_available"])
        self.assertFalse(x["void1"]["original_untrained_complete_host_forward_samples_available"])
        self.assertFalse(x["smollm2"]["original_untrained_complete_host_forward_samples_available"])

    def test_reject_unregistered_boolean_or_zero_hidden(self):
        for hidden in (True,False,0,128,512,1024,2048,576.0,"640",None):
            with self.subTest(hidden=hidden),self.assertRaisesRegex(ValueError,"REFUSED"):
                gate.estimate(hidden)

    def test_original_locked_protocol_passes_without_reading_heldouts(self):
        gate.assert_protocol(protocol())

    def test_refuse_changed_gate_controls_or_comparative_training_flag(self):
        for variant in ("training_on","wrong_seed","wrong_steps","wrong_optimizer",
                        "nonfrozen","threads","geometry","tap_count","tap_iterations",
                        "wrong_arms","missing_arms","allow_crossmodel","param_tolerance",
                        "wrong_cache_gate"):
            row=copy.deepcopy(protocol())
            if variant=="training_on":row["train_authorized"]=True
            elif variant=="wrong_seed":row["training"]["seed_list"][1]=999
            elif variant=="wrong_steps":row["training"]["primary_steps_per_seed"]=193
            elif variant=="wrong_optimizer":row["training"]["optimizer"]="SGD"
            elif variant=="nonfrozen":row["training"]["trainable_backbone"]=True
            elif variant=="threads":row["training"]["torch_threads"]=2
            elif variant=="geometry":row["architecture"]["state_geometry"]["height"]=16
            elif variant=="tap_count":row["architecture"]["tap_count"]=2
            elif variant=="tap_iterations":row["architecture"]["nca_steps_per_tap"]=1
            elif variant=="wrong_arms":row["comparisons"][-1]="extra_unknown_control"
            elif variant=="missing_arms":row["comparisons"].pop()
            elif variant=="allow_crossmodel":row["match_constraints"]["comparison_within_each_model"]=False
            elif variant=="param_tolerance":row["match_constraints"]["adapter_trainable_param_tolerance_fraction"]=0.50
            elif variant=="wrong_cache_gate":row["conformance"]["required_checks"].pop()
            with self.subTest(variant=variant),self.assertRaisesRegex(
                    (ValueError,KeyError),"REFUSED"):
                gate.assert_protocol(row)

    def test_refuse_variant_swap_or_bool_hidden(self):
        for variant in ("missing","chat_as_root","root_as_chat","wrong_void_taps",
                        "smol_width","bool_width","void_fake_weights"):
            row=copy.deepcopy(protocol())
            if variant=="missing":del row["models"]["void1"]
            elif variant=="chat_as_root":
                row["models"]["sprocket_chat"]["weight_file"]="model.safetensors"
            elif variant=="root_as_chat":
                row["models"]["sprocket_instruct"]["weight_file"]="chat/model.safetensors"
            elif variant=="wrong_void_taps":
                row["models"]["void1"]["taps_after_block_zero_based"]=[7,16,23]
            elif variant=="smol_width":
                row["models"]["smollm2"]["expected_config"]["hidden_size"]=640
            elif variant=="bool_width":
                row["models"]["void1"]["expected_config"]["hidden_size"]=True
            elif variant=="void_fake_weights":
                row["models"]["void1"]["weight_file"]="chat/model.safetensors"
            with self.subTest(variant=variant),self.assertRaisesRegex(ValueError,"REFUSED"):
                gate.assert_protocol(row)

if __name__=="__main__":
    unittest.main()
