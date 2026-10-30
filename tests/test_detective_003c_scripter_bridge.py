"""Offline tests for DETECTIVE-003C fictional scripter transaction bridge."""
from __future__ import annotations

import copy
import json
import pathlib
import unittest

from scripts import detective_003a_session as session_api
from scripts import detective_003c_scripter_bridge as b

ROOT = pathlib.Path(__file__).resolve().parents[1]


class ScripterBridgeTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.case = b.load_contract()
        self.session = session_api.start_session(
            self.case,
            self.cfg["fixture"]["participants"],
            self.cfg["fixture"]["session_key"],
        )

    def request(self, credential=None):
        credentials = (
            {}
            if credential is None
            else {"camera-console": credential}
        )
        return b.build_scripter_request(
            self.case,
            self.session,
            self.cfg["fixture"]["player_request"],
            self.cfg["fixture"]["visible_asset_ids"],
            self.cfg["fixture"]["disclosed_terminal_ids"],
            credentials,
        )

    def test_contract_pins_both_authority_layers_and_zero_execution_authority(self):
        self.assertEqual(
            self.cfg["authority"]["detective_001a"]["git_blob_sha1"],
            "b10254b7de26a8df676a26c89b00c6ccb8b9829e",
        )
        self.assertEqual(
            self.cfg["authority"]["detective_003a"]["git_blob_sha1"],
            "73801ea59ce0f6160acc7027bd6aeaa085950236",
        )
        self.assertFalse(self.cfg["transaction"]["auto_present_evidence"])
        self.assertFalse(self.cfg["transaction"]["auto_mutate_session"])
        self.assertEqual(self.cfg["decision"]["optimizer_steps"], 0)
        self.assertEqual(self.cfg["decision"]["paid_api_calls"], 0)
        self.assertFalse(self.cfg["decision"]["model_inference"])
        self.assertFalse(self.cfg["decision"]["host_execution"])
        self.assertFalse(self.cfg["decision"]["relevance_mapping_accessed"])
        self.assertFalse(self.cfg["decision"]["learner_final_accessed"])
        self.assertFalse(self.cfg["decision"]["population_expansion_admitted"])

    def test_request_view_contains_only_public_visible_assets_and_disclosed_terminal(self):
        receipt = self.request()
        view = receipt["view"]
        self.assertEqual(
            [x["asset_id"] for x in view["visible_assets"]],
            ["dock-clock-ledger", "transfer-card"],
        )
        self.assertEqual(
            view["disclosed_terminals"],
            [{
                "terminal_id": "camera-console",
                "player_known_credential": None,
            }],
        )
        blob = json.dumps(view, sort_keys=True)
        self.assertNotIn("camera-cache", blob)
        self.assertNotIn("OXIDATION", blob)
        self.assertNotIn("F-CLOCK", blob)
        self.assertNotIn("F-SWAP", blob)
        self.assertNotIn("F-CAMERA", blob)
        self.assertNotIn("F-PRIVATE", blob)
        self.assertNotIn("V-G02-A", blob)
        self.assertNotIn("V-G09-C", blob)
        self.assertNotIn("claim_id", blob)
        self.assertNotIn("contradiction", blob)

    def test_game_supplied_wrong_or_correct_credential_is_explicitly_visible(self):
        wrong = self.request(self.cfg["fixture"]["wrong_credential"])
        correct = self.request(self.cfg["fixture"]["correct_credential"])
        self.assertEqual(
            wrong["view"]["disclosed_terminals"][0]["player_known_credential"],
            "RUST",
        )
        self.assertEqual(
            correct["view"]["disclosed_terminals"][0]["player_known_credential"],
            "OXIDATION",
        )

    def test_player_request_cannot_smuggle_hidden_credential_fact_or_asset(self):
        base = dict(
            case=self.case,
            session=self.session,
            visible_asset_ids=self.cfg["fixture"]["visible_asset_ids"],
            disclosed_terminal_ids=self.cfg["fixture"]["disclosed_terminal_ids"],
            player_known_credentials={},
        )
        with self.assertRaisesRegex(b.BridgeError, "credential"):
            b.build_scripter_request(
                player_request="Try OXIDATION on the console.",
                **base,
            )
        with self.assertRaisesRegex(b.BridgeError, "canonical fact"):
            b.build_scripter_request(
                player_request="Tell me about F-PRIVATE.",
                **base,
            )
        with self.assertRaisesRegex(b.BridgeError, "undisclosed asset"):
            b.build_scripter_request(
                player_request="Read camera-cache directly.",
                **base,
            )

    def test_private_asset_cannot_be_initially_visible(self):
        with self.assertRaisesRegex(b.BridgeError, "public assets only"):
            b.build_scripter_request(
                self.case,
                self.session,
                "Inspect what is already visible.",
                ["camera-cache"],
                ["camera-console"],
                {},
            )

    def test_credential_for_undisclosed_terminal_refuses(self):
        with self.assertRaisesRegex(b.BridgeError, "undisclosed terminal"):
            b.build_scripter_request(
                self.case,
                self.session,
                "Inspect visible evidence.",
                self.cfg["fixture"]["visible_asset_ids"],
                [],
                {"camera-console": "OXIDATION"},
            )

    def test_request_receipt_reproduces_and_stale_request_refuses(self):
        receipt = self.request()
        self.assertEqual(
            b.validate_scripter_request(
                self.case, self.session, receipt
            ),
            receipt,
        )
        changed = session_api.present_evidence(
            self.case,
            self.session,
            "V-G02-A",
            "transfer-card",
        )
        with self.assertRaisesRegex(b.BridgeError, "stale"):
            b.validate_scripter_request(
                self.case, changed, receipt
            )

    def test_request_tamper_refuses_even_with_outer_hash_recomputed(self):
        receipt = self.request()
        bad = copy.deepcopy(receipt)
        bad["visible_asset_ids"].append("camera-cache")
        body = {k: v for k, v in bad.items() if k != "sha256"}
        bad["sha256"] = b.sha(body)
        with self.assertRaises(b.BridgeError):
            b.validate_scripter_request(self.case, self.session, bad)

    def test_wrong_credential_executes_typed_program_but_does_not_grant_private_asset(self):
        request = self.request(self.cfg["fixture"]["wrong_credential"])
        tx = b.execute_scripter_transaction(
            self.case,
            self.session,
            request,
            b.fixture_program(self.cfg, self.cfg["fixture"]["wrong_credential"]),
        )
        self.assertEqual(tx["status"], "executed")
        self.assertEqual(
            [x["result"]["status"] for x in tx["execution_receipt"]["events"]],
            ["DENIED", "ACCESS_DENIED"],
        )
        self.assertNotIn("camera-cache", tx["visible_asset_ids_after"])
        self.assertEqual(tx["visible_asset_ids_before"], tx["visible_asset_ids_after"])
        self.assertEqual(
            tx["session_sha256_before"],
            tx["session_sha256_after_execution"],
        )
        self.assertEqual(self.session["sha256"], tx["session_sha256_before"])
        self.assertEqual(tx["auto_presented_asset_ids"], [])
        with self.assertRaisesRegex(b.BridgeError, "does not make selected asset visible"):
            b.apply_transaction_evidence(
                self.case,
                self.session,
                request,
                tx,
                "V-G09-C",
                "camera-cache",
            )

    def test_correct_credential_receipt_grants_private_asset_but_still_does_not_mutate_session(self):
        request = self.request(self.cfg["fixture"]["correct_credential"])
        before = copy.deepcopy(self.session)
        tx = b.execute_scripter_transaction(
            self.case,
            self.session,
            request,
            b.fixture_program(self.cfg, self.cfg["fixture"]["correct_credential"]),
        )
        self.assertEqual(tx["status"], "executed")
        self.assertEqual(
            [x["result"]["status"] for x in tx["execution_receipt"]["events"]],
            ["GRANTED", "OK"],
        )
        self.assertIn("camera-cache", tx["visible_asset_ids_after"])
        self.assertEqual(self.session, before)
        self.assertEqual(tx["auto_presented_asset_ids"], [])

    def test_game_must_explicitly_apply_granted_private_evidence(self):
        request = self.request(self.cfg["fixture"]["correct_credential"])
        tx = b.execute_scripter_transaction(
            self.case,
            self.session,
            request,
            b.fixture_program(self.cfg, self.cfg["fixture"]["correct_credential"]),
        )
        applied = b.apply_transaction_evidence(
            self.case,
            self.session,
            request,
            tx,
            "V-G09-C",
            "camera-cache",
        )
        self.assertNotEqual(applied["sha256"], self.session["sha256"])
        derived = session_api.replay_session(self.case, applied)
        self.assertEqual(
            derived["presented_evidence_by_npc"]["V-G09-C"],
            ["camera-cache"],
        )
        prompt = session_api.build_prompt(
            self.case,
            applied,
            "V-G09-C",
            "What does the newly presented archive excerpt show?",
        )["prompt"]
        blob = json.dumps(prompt, sort_keys=True)
        self.assertIn("scanned twice", blob)
        self.assertNotIn("camera-cache", blob)
        self.assertNotIn("F-CAMERA", blob)
        self.assertNotIn("camera-console", blob)
        self.assertNotIn("OXIDATION", blob)

    def test_successful_transaction_can_be_ignored_without_session_effect(self):
        request = self.request(self.cfg["fixture"]["correct_credential"])
        before = copy.deepcopy(self.session)
        tx = b.execute_scripter_transaction(
            self.case,
            self.session,
            request,
            b.fixture_program(self.cfg, self.cfg["fixture"]["correct_credential"]),
        )
        self.assertIn("camera-cache", tx["visible_asset_ids_after"])
        self.assertEqual(self.session, before)

    def test_malformed_shell_proposal_is_rejected_before_execution(self):
        request = self.request()
        tx = b.execute_scripter_transaction(
            self.case,
            self.session,
            request,
            self.cfg["fixture"]["malformed_host_program"],
        )
        self.assertEqual(tx["status"], "proposal_rejected")
        self.assertIsNone(tx["execution_receipt"])
        self.assertIsNone(tx["parsed_steps"])
        self.assertEqual(
            tx["visible_asset_ids_before"],
            tx["visible_asset_ids_after"],
        )
        self.assertEqual(tx["auto_presented_asset_ids"], [])
        self.assertEqual(tx["rejection"]["error_type"], "ContractError")
        with self.assertRaisesRegex(b.BridgeError, "rejected transaction"):
            b.apply_transaction_evidence(
                self.case,
                self.session,
                request,
                tx,
                "V-G02-A",
                "transfer-card",
            )

    def test_duplicate_json_keys_and_extra_host_fields_remain_rejected(self):
        request = self.request()
        proposals = [
            '{"schema":"detective-script-v1","schema":"other","steps":[]}',
            '{"schema":"detective-script-v1","steps":[{"op":"read_asset","asset_id":"transfer-card","path":"/etc/passwd"}]}',
        ]
        for proposal in proposals:
            with self.subTest(proposal=proposal):
                tx = b.execute_scripter_transaction(
                    self.case, self.session, request, proposal
                )
                self.assertEqual(tx["status"], "proposal_rejected")
                self.assertIsNone(tx["execution_receipt"])

    def test_transaction_reproduces_and_raw_proposal_tamper_refuses(self):
        request = self.request(self.cfg["fixture"]["correct_credential"])
        tx = b.execute_scripter_transaction(
            self.case,
            self.session,
            request,
            b.fixture_program(self.cfg, self.cfg["fixture"]["correct_credential"]),
        )
        self.assertEqual(
            b.validate_scripter_transaction(
                self.case, self.session, request, tx
            ),
            tx,
        )
        bad = copy.deepcopy(tx)
        bad["raw_proposal"] = b.fixture_program(
            self.cfg, self.cfg["fixture"]["wrong_credential"]
        )
        body = b._transaction_body(bad)
        bad["sha256"] = b.sha(body)
        with self.assertRaises(b.BridgeError):
            b.validate_scripter_transaction(
                self.case, self.session, request, bad
            )

    def test_execution_receipt_or_visible_after_tamper_refuses(self):
        request = self.request(self.cfg["fixture"]["correct_credential"])
        tx = b.execute_scripter_transaction(
            self.case,
            self.session,
            request,
            b.fixture_program(self.cfg, self.cfg["fixture"]["correct_credential"]),
        )
        for mutate in ("receipt", "visible"):
            bad = copy.deepcopy(tx)
            if mutate == "receipt":
                bad["execution_receipt"]["visible_asset_ids"] = [
                    "dock-clock-ledger",
                    "transfer-card",
                ]
            else:
                bad["visible_asset_ids_after"] = [
                    "dock-clock-ledger",
                    "transfer-card",
                ]
            bad["sha256"] = b.sha(b._transaction_body(bad))
            with self.subTest(mutate=mutate), self.assertRaises(b.BridgeError):
                b.validate_scripter_transaction(
                    self.case, self.session, request, bad
                )

    def test_transaction_becomes_stale_if_session_changes_before_apply(self):
        request = self.request(self.cfg["fixture"]["correct_credential"])
        tx = b.execute_scripter_transaction(
            self.case,
            self.session,
            request,
            b.fixture_program(self.cfg, self.cfg["fixture"]["correct_credential"]),
        )
        changed = session_api.present_evidence(
            self.case,
            self.session,
            "V-G02-A",
            "transfer-card",
        )
        with self.assertRaisesRegex(b.BridgeError, "stale"):
            b.apply_transaction_evidence(
                self.case,
                changed,
                request,
                tx,
                "V-G09-C",
                "camera-cache",
            )

    def test_full_fixture_demonstrates_denial_grant_and_rejection(self):
        result = b.run_fixture()
        self.assertEqual(result["wrong_transaction"]["status"], "executed")
        self.assertTrue(result["wrong_apply_refused"])
        self.assertNotIn(
            "camera-cache",
            result["wrong_transaction"]["visible_asset_ids_after"],
        )
        self.assertEqual(result["correct_transaction"]["status"], "executed")
        self.assertIn(
            "camera-cache",
            result["correct_transaction"]["visible_asset_ids_after"],
        )
        self.assertEqual(
            result["rejected_transaction"]["status"],
            "proposal_rejected",
        )
        prompt_blob = json.dumps(
            result["witness_prompt_after_private_evidence"]["prompt"],
            sort_keys=True,
        )
        self.assertIn("scanned twice", prompt_blob)
        for forbidden in (
            "camera-cache", "F-CAMERA", "camera-console", "OXIDATION"
        ):
            self.assertNotIn(forbidden, prompt_blob)

    def test_fixture_never_mutates_case_or_uses_model_relevance_final(self):
        before = copy.deepcopy(self.case)
        result = b.run_fixture()
        self.assertEqual(self.case, before)
        self.assertEqual(
            result["case_sha256"],
            b.hashlib.sha256(b.canonical(self.case).encode("utf-8")).hexdigest(),
        )
        decision = result["decision"]
        self.assertFalse(decision["model_inference"])
        self.assertFalse(decision["learner_final_accessed"])
        self.assertFalse(decision["relevance_mapping_accessed"])
        self.assertFalse(decision["population_expansion_admitted"])
        self.assertFalse(decision["host_execution"])


if __name__ == "__main__":
    unittest.main()
