"""Full synthetic population assembly, without model inference or provider calls."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import village_001e_assemble as a
from village_001e_generation_checkpoint import GenerationCheckpoint

REPO = Path(__file__).resolve().parents[1]


class FullAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.members = [{"member_id": f"V-G{i:02d}-{s}", "lineage": f"G{i:02d}", "sibling": s}
                        for i in range(1, 13) for s in "ABC"]
        self.contract = "a" * 64
        self.population = {"members": self.members, "member_contract_sha256": self.contract}
        self.campaign = {"base_model": {"id": "fixture/model", "revision": "b" * 40},
                         "archive": {"dataset_root": "dataset"},
                         "nvidia": {"models": {k: {"id": k} for k in ("compiler", "critic", "adjudicator")}}}
        self.cfg = json.loads((REPO / "experiments/village-001/tournament-config.json").read_text())
        self.cfg.update(campaign_config="campaign.json", preference_config="preference-config.json",
                        preference_evidence_root="D", final_eval_config="final.json", final_eval_recipe="recipe.json")
        for path, obj in (("config.json", self.cfg), ("campaign.json", self.campaign), ("preference-config.json", {})):
            a.write_json(self.root / path, obj)
        for name in ("village_001e_generation_checkpoint.py", "village_001_manifest.py",
                     "village_001e_generate_final_responses.py", "village_001e_ecology_run.py", "village_001e_ecology_plan.py"):
            (self.root / "scripts").mkdir(exist_ok=True)
            (self.root / "scripts" / name).write_text("# Synthetic source identity for " + name + "\n")
        self.assets = {m["member_id"]: {"member_id": m["member_id"], "file": m["member_id"] + ".tar.gz",
                       "sha256": f"{index:064x}", "adapter_directory_sha256": f"{index+100:064x}",
                       "parent_adapter_directory_sha256": f"{index+200:064x}", "qualified_for_population_tournament": True}
                       for index, m in enumerate(self.members, 1)}
        release = {"release_tag": "frozen-D-recovery", "release_url": "https://example.test/release/D",
                   "member_contract_sha256": self.contract, "assets": list(self.assets.values())}
        summary = {"members": 36, "qualified_for_population_tournament": list(self.assets)}
        a.write_json(self.root / "D/adapter-release.json", release)
        a.write_json(self.root / "D/summary.json", summary)
        for member, asset in self.assets.items():
            a.write_json(self.root / f"D/results/{member}.json", {"member_id": member})
            a.write_json(self.root / f"D/before-after/{member}.json", {
                "member_id": member, "member_contract_sha256": self.contract,
                "parent_adapter_sha256": asset["parent_adapter_directory_sha256"],
                "preference_adapter_sha256": asset["adapter_directory_sha256"],
                "after": {"task_relevance": 1, "genericness": 1},
                "gates": {"qualified_for_population_tournament": True},
            })
        for name in ("manifest.json", "preference-manifest.json", "provenance/hashes.sha256", "preference-hashes.sha256"):
            path = self.root / "dataset" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}\n')
        a.write_json(self.root / "experiments/persona-001/evidence/summary.json", {
            "raw_blind": {"control_scores": {"genericness": 3.83}},
            "behavior_blind": {"control_scores": {"genericness": 3.5}},
        })
        recipe = json.loads((REPO / "experiments/village-001/final-eval.json").read_text())
        a.write_json(self.root / "recipe.json", recipe)
        self.suite = a.build_suite(recipe=recipe, recipe_sha256=a.sha256_file(self.root / "recipe.json"),
            preference_summary_sha256=a.sha256_file(self.root / "D/summary.json"),
            preference_release_sha256=a.sha256_file(self.root / "D/adapter-release.json"), forbidden=set())
        a.write_json(self.root / "final.json", self.suite)
        self.common_sources = {
            "tournament_config": a.sha256_file(self.root / "config.json"),
            "campaign_config": a.sha256_file(self.root / "campaign.json"),
            "final_eval": a.sha256_file(self.root / "final.json"),
            "preference_summary": a.sha256_file(self.root / "D/summary.json"),
            "preference_release": a.sha256_file(self.root / "D/adapter-release.json"),
            "checkpoint_script": a.sha256_file(self.root / "scripts/village_001e_generation_checkpoint.py"),
            "manifest_script": a.sha256_file(self.root / "scripts/village_001_manifest.py"),
        }
        self.runtime = {"torch": "fixture", "transformers": "fixture", "peft": "fixture"}
        self.make_responses()
        self.make_tournaments()
        self.make_ecology()

    def native(self, lane, identity, rows, keys, row_field, filename, result):
        cp = GenerationCheckpoint(out=lane, resume=None, identity=identity, keys=keys,
                                  terminal_name=filename, validate_row=lambda *_: None)
        for key, row in zip(keys, rows):
            row.update(input_token_ids=[1, 2], generated_token_ids=[3], generated_tokens=1, rendered_prompt="Fixture native prompt.")
            cp.append(key, row)
        result[row_field] = cp.rows
        cp.finish(result, row_field)

    def provider(self, lane, filename, doc):
        a.write_json(lane / "provider-identity.json", {"fixture": "Synthetic provider output; never a measured result."})
        for suffix in ("request.json", "provider.json", "attempts.json"):
            a.write_json(lane / f"provider/call-{suffix}", {})
        (lane / "provider/call-content.txt").write_text('{}\n')
        doc.update(provider_identity_sha256=a.sha256_file(lane / "provider-identity.json"), provider_call_ids=["call"])
        files = {p.relative_to(lane).as_posix(): {"sha256": a.sha256_file(p), "bytes": p.stat().st_size}
                 for p in lane.rglob("*") if p.is_file()}
        a.write_json(lane / "provider-evidence-hashes.json", {"schema_version": 1, "files": files})
        a.write_json(lane / filename, doc)

    def make_responses(self):
        for meta in [*self.members, {"member_id": "BASE", "lineage": None, "sibling": None}]:
            member = meta["member_id"]
            base = member == "BASE"
            sources = {**self.common_sources, "generator_script": a.sha256_file(self.root / "scripts/village_001e_generate_final_responses.py")}
            if not base:
                sources["preference_result"] = a.sha256_file(self.root / f"D/results/{member}.json")
            adapter = None if base else self.assets[member]["adapter_directory_sha256"]
            identity = {"member_id": member, "member_contract_sha256": self.contract, "base_model": self.campaign["base_model"],
                        "preference_adapter_sha256": adapter, "runtime_contract": self.runtime, "sources": sources}
            result = {"experiment": "VILLAGE-001E", "stage": "fresh-final-heldout-responses", **meta,
                      "member_contract_sha256": self.contract, "base_model": "fixture/model", "base_revision": "b" * 40,
                      "base_control": base, "preference_release_tag": "frozen-D-recovery",
                      "preference_adapter_sha256": adapter,
                      "preference_adapter_archive_sha256": None if base else self.assets[member]["sha256"],
                      "final_eval_config_sha256": self.common_sources["final_eval"], "fresh_after_preference_training": True,
                      "exact_training_input_collision_count": 0, "hidden_member_prompt_used": False,
                      "system": self.suite["system"], "max_new_tokens": 160, "do_sample": False, "packages": self.runtime}
            rows = [{"scenario_id": s["id"], "family": s["family"], "prompt": s["prompt"], "response": "Fixture response."}
                    for s in self.suite["scenarios"]]
            self.native(self.root / "responses" / member, identity, rows, [s["id"] for s in self.suite["scenarios"]], "responses", "result.json", result)

    def make_tournaments(self):
        response_shas = {m: a.sha256_file(self.root / f"responses/{m}/result.json") for m in [*self.assets, "BASE"]}
        for scenario in self.suite["scenarios"]:
            sid = scenario["id"]
            members = sorted(self.members, key=lambda m: hashlib.sha256(f"{sid}:{m['member_id']}".encode()).hexdigest())
            records = []
            for index, member in enumerate(members, 1):
                prediction = {"predicted_lineage": member["lineage"], "predicted_sibling": member["sibling"],
                              "task_relevance": 3, "genericness": 1, "lineage_confidence": 3, "sibling_confidence": 3}
                records.append({**member, "opaque_id": f"R{index:03d}", "adapter_sha256": self.assets[member["member_id"]]["adapter_directory_sha256"],
                    "response": "Fixture response.", "behavior_summary": "Fixture neutral behavior.", "raw_prediction": prediction,
                    "behavior_prediction": prediction, "raw_sibling_prediction": {"predicted_sibling": member["sibling"]},
                    "behavior_sibling_prediction": {"predicted_sibling": member["sibling"]}, "super_audit": None})
            doc = {"experiment": "VILLAGE-001E", "stage": "hierarchical-blind-tournament-scenario", "scenario_id": sid,
                   "scenario": scenario, "fresh_after_preference_training": True, "final_eval_config_sha256": self.common_sources["final_eval"],
                   "providers_saw_member_ids": False, "normalizer_model": "compiler", "classifier_model": "critic", "adjudicator_model": "adjudicator",
                   "tournament_config_sha256": self.common_sources["tournament_config"], "campaign_config_sha256": self.common_sources["campaign_config"],
                   "preference_release_sha256": self.common_sources["preference_release"], "source_response_sha256": response_shas, "records": records,
                   "untuned_base_control": {"response": "Fixture response.", "behavior_summary": "Fixture neutral behavior.",
                                            "raw": {"task_relevance": 3, "genericness": 3}, "behavior": {"task_relevance": 3, "genericness": 3}}}
            self.provider(self.root / "tournament" / sid, "scenario.json", doc)

    def make_ecology(self):
        for expected in a.build_sessions(self.cfg):
            sid, handles = expected["session_id"], expected["handles"]
            adapter_shas = {m: self.assets[m]["adapter_directory_sha256"] for m in expected["members"]}
            identity = {"session": expected, "base_model": self.campaign["base_model"], "member_contract_sha256": self.contract,
                        "adapter_sha256": adapter_shas, "runtime_contract": self.runtime,
                        "sources": {**self.common_sources, "generator_script": a.sha256_file(self.root / "scripts/village_001e_ecology_run.py"),
                                    "ecology_plan_script": a.sha256_file(self.root / "scripts/village_001e_ecology_plan.py")}}
            session = {"experiment": "VILLAGE-001E", "stage": "ecology-session", **{k: v for k, v in expected.items() if k != "handles"},
                       "rounds": 3, "public_handles": sorted(handles), "handle_member_mapping": handles, "adapter_sha256": adapter_shas,
                       "provider_or_model_prompts_saw_member_ids": False, "base_model": "fixture/model", "base_revision": "b" * 40,
                       "member_contract_sha256": self.contract, "preference_release_tag": "frozen-D-recovery",
                       "final_eval_config_sha256": self.common_sources["final_eval"], "packages": self.runtime}
            keys = [f"R{r}-{h}" for r in range(1, 4) for h in sorted(handles)]
            rows = [{"round": r, "handle": h, "member_id": handles[h], "text": "Fixture public action."} for r in range(1, 4) for h in sorted(handles)]
            self.native(self.root / "ecology" / sid, identity, rows, keys, "turns", "session.json", session)
            lineage_audit = None
            if expected["kind"] == "cross_lineage_group":
                lineage_audit = {"agents": [{"handle": h, "member_id": m, "expected_lineage_fit": 2,
                                           "cross_lineage_contamination": 1, "distinction_survival": 2} for h, m in handles.items()]}
            doc = {"experiment": "VILLAGE-001E", "stage": "ecology-judgment", "session_id": sid,
                   "kind": expected["kind"], "scenario": expected["scenario"], "provider_prompts_saw_member_ids": False,
                   "agents_received_hidden_lineage_prompts": False, "lineage_audit_received_expected_lineage_descriptors": lineage_audit is not None,
                   "critic_model": "critic", "adjudicator_model": "adjudicator", "metrics": {key: 2 for key in a.ECOLOGY_METRICS},
                   "agents": [{"handle": h, "member_id": m, "distinctiveness": 2, "generic_assistant_tendency": 1, "dominance": 2, "deference": 1} for h, m in handles.items()],
                   "lineage_audit": lineage_audit, "super_audit": None,
                   "tournament_config_sha256": self.common_sources["tournament_config"], "campaign_config_sha256": self.common_sources["campaign_config"],
                   "preference_release_sha256": self.common_sources["preference_release"],
                   "source_session_sha256": a.sha256_file(self.root / f"ecology/{sid}/session.json")}
            self.provider(self.root / "judgments" / sid, "judgment.json", doc)

    def run_assembly(self):
        argv = ["assemble", "--config", str(self.root / "config.json"), "--out", str(self.root / "final-evidence"),
                "--response-artifacts", str(self.root / "responses"), "--tournament-artifacts", str(self.root / "tournament"),
                "--ecology-artifacts", str(self.root / "ecology"), "--ecology-judgment-artifacts", str(self.root / "judgments"),
                "--release-tag", "isolated-E-fixture"]
        with patch.object(a, "ROOT", self.root), patch.object(a, "build_manifest", return_value=self.population), patch.object(sys, "argv", argv):
            a.main()
        return json.loads((self.root / "final-evidence/summary.json").read_text())

    def test_full_population_promotes_exact_custody_and_isolated_release(self):
        summary = self.run_assembly()
        self.assertTrue(summary["usable_village"])
        self.assertEqual(summary["individual_quality_pass_count"], 36)
        self.assertEqual(summary["qualification_evidence_release"], "isolated-E-fixture")
        self.assertEqual(summary["final_adapter_release"], "frozen-D-recovery")
        self.assertEqual(summary["blind_response_records"], 864)
        self.assertEqual(summary["ecology"]["sessions"], 24)
        self.assertTrue((self.root / "dataset/evaluation/final-responses/BASE/rows/024-F24.json").is_file())
        self.assertTrue((self.root / "dataset/evaluation/ecology/judgments/E-P-G01/provider/call-content.txt").is_file())

    def test_failed_population_is_published_without_replacing_members(self):
        for path in (self.root / "tournament").glob("*/scenario.json"):
            doc = json.loads(path.read_text())
            for row in doc["records"]:
                row["raw_prediction"]["task_relevance"] = 0
            a.write_json(path, doc)
        summary = self.run_assembly()
        self.assertFalse(summary["usable_village"])
        self.assertEqual(summary["individual_quality_pass_count"], 0)
        self.assertEqual(len(summary["individual_quality_fail"]), 36)
        self.assertTrue(all("raw_task_relevance" in row["failed_individual_gates"] for row in summary["members_table"]))
        self.assertEqual(len(json.loads((self.root / "final-evidence/model-registry.json").read_text())["models"]), 36)


if __name__ == "__main__":
    unittest.main()
