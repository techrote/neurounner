"""Offline tests only: no model downloads, paid APIs or external host calls."""

import copy
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.detective_001_contract import (  # noqa: E402
    ContractError,
    execute_program,
    parse_program,
    validate_case,
    witness_view,
)

CASE_PATH = ROOT / "experiments" / "detective-001" / "pilot-case.json"


class CaseContractTests(unittest.TestCase):
    def setUp(self):
        self.case = json.loads(CASE_PATH.read_text(encoding="utf-8"))

    def test_fixture_is_valid(self):
        validate_case(self.case)
        self.assertEqual(len(self.case["npcs"]), 2)

    def test_public_non_dialogue_path_is_required(self):
        variant = copy.deepcopy(self.case)
        variant["leads"][0]["routes"] = variant["leads"][0]["routes"][:1]
        with self.assertRaises(ContractError):
            validate_case(variant)
        variant = copy.deepcopy(self.case)
        variant["assets"][0]["public"] = False
        with self.assertRaises(ContractError):
            validate_case(variant)

    def test_actor_must_actually_have_the_claimed_knowledge(self):
        variant = copy.deepcopy(self.case)
        variant["leads"][0]["routes"][0]["source"] = "V-G09-C"
        with self.assertRaises(ContractError):
            validate_case(variant)

    def test_different_route_ids_cannot_mask_same_source(self):
        variant = copy.deepcopy(self.case)
        variant["leads"][0]["routes"].append(
            {"id": "W-SECOND", "kind": "witness", "source": "V-G02-A"}
        )
        with self.assertRaises(ContractError):
            validate_case(variant)

    def test_interview_projection_does_not_expose_world_secrets(self):
        view = witness_view(self.case, "V-G02-A", "What went wrong with the clock?")
        payload = json.dumps(view, sort_keys=True)
        self.assertEqual([m["id"] for m in view["admitted_memories"]], ["F-CLOCK", "F-SWAP"])
        self.assertNotIn("F-PRIVATE", payload)
        self.assertNotIn("The foreman's private key", payload)
        self.assertNotIn("F-CAMERA", payload)
        self.assertNotIn("OXIDATION", payload)
        self.assertNotIn("terminals", view)
        self.assertNotIn("leads", view)

    def test_scripter_does_not_mutate_canonical_case(self):
        before = copy.deepcopy(self.case)
        view = witness_view(self.case, "V-G02-A", "Who owns the canister?")
        view["admitted_memories"].clear()
        plan = {"schema": "detective-script-v1", "steps": [
            {"op": "probe_terminal", "terminal_id": "camera-console", "credential": "OXIDATION"},
            {"op": "read_asset", "asset_id": "camera-cache"}
        ]}
        result = execute_program(self.case, plan, ["transfer-card"])
        self.assertEqual(self.case, before)
        self.assertEqual(result["events"][0]["result"]["status"], "GRANTED")
        self.assertEqual(result["events"][1]["result"]["status"], "OK")

    def test_correlate_and_receipt_are_repeatable(self):
        plan = {"schema": "detective-script-v1", "steps": [
            {"op": "read_asset", "asset_id": "dock-clock-ledger"},
            {"op": "correlate_assets", "left_id": "dock-clock-ledger", "right_id": "transfer-card"}
        ]}
        a = execute_program(self.case, plan, ["dock-clock-ledger", "transfer-card"])
        b = execute_program(self.case, plan, ["dock-clock-ledger", "transfer-card"])
        self.assertEqual(a, b)
        self.assertEqual(len(a["sha256"]), 64)
        self.assertEqual(a["events"][1]["result"]["shared_fact_ids"], [])

    def test_locked_asset_refuses_even_if_known_by_the_model(self):
        result = execute_program(
            self.case,
            {"schema": "detective-script-v1", "steps": [
                {"op": "read_asset", "asset_id": "camera-cache"},
                {"op": "probe_terminal", "terminal_id": "camera-console", "credential": "wrong"}
            ]},
            ["dock-clock-ledger"],
        )
        self.assertEqual([x["result"]["status"] for x in result["events"]],
                         ["ACCESS_DENIED", "DENIED"])
        self.assertNotIn("camera-cache", result["visible_asset_ids"])
        self.assertNotIn("OXIDATION", json.dumps(result))

    def test_model_cannot_send_shell_code_or_extra_fields(self):
        for step in [
            {"op": "shell", "cmd": "echo no"},
            {"op": "read_asset", "asset_id": "transfer-card", "path": "/etc/passwd"},
            {"op": "probe_terminal", "terminal_id": "camera-console", "credential": "OXIDATION",
             "output_file": "archive.txt"},
        ]:
            with self.subTest(step=step), self.assertRaises(ContractError):
                execute_program(self.case, {"schema": "detective-script-v1", "steps": [step]},
                                ["transfer-card"])

    def test_duplicate_json_keys_and_large_programme_refuse(self):
        with self.assertRaises(ContractError):
            parse_program('{"schema":"detective-script-v1","schema":"other","steps":[]}')
        with self.assertRaises(ContractError):
            parse_program({"schema": "detective-script-v1",
                           "steps": [{"op": "read_asset", "asset_id": "transfer-card"}] * 9})

    def test_invalid_actor_or_permission_list_refuses(self):
        with self.assertRaises(ContractError):
            witness_view(self.case, "V-G99-A", "Where is the ledger?")
        with self.assertRaises(ContractError):
            execute_program(self.case, {"schema": "detective-script-v1", "steps": []},
                            ["transfer-card", "transfer-card"])
        with self.assertRaises(ContractError):
            execute_program(self.case, {"schema": "detective-script-v1", "steps": []},
                            ["mystery-asset"])


if __name__ == "__main__":
    unittest.main()
