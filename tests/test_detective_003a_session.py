"""DETECTIVE-003A event-sourced multi-NPC interview ledger tests.

Offline only: no model inference, network, training, relevance-map access, or host execution.
"""
from __future__ import annotations

import copy
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

from scripts import detective_003a_session as s
from scripts.detective_001_contract import execute_program, validate_case

CASE_PATH = ROOT / "experiments/detective-001/pilot-case.json"
FIXTURE_PATH = ROOT / "experiments/detective-003/session-fixture.json"


class SessionLedgerTests(unittest.TestCase):
    def setUp(self):
        self.case = json.loads(CASE_PATH.read_text(encoding="utf-8"))
        self.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        validate_case(self.case)

    def start(self):
        return s.start_session(
            self.case,
            self.fixture["participants"],
            self.fixture["session_key"],
        )

    def run_fixture(self):
        session = self.start()
        receipts = []
        for step in self.fixture["steps"]:
            if step["action"] == "turn":
                prompt = s.build_prompt(
                    self.case,
                    session,
                    step["npc_id"],
                    step["question"],
                    include_history=step["include_history"],
                )
                receipts.append(prompt)
                session = s.record_turn(
                    self.case,
                    session,
                    prompt,
                    step["raw_response"],
                    step["annotations"],
                )
            elif step["action"] == "evidence":
                session = s.present_evidence(
                    self.case,
                    session,
                    step["npc_id"],
                    step["asset_id"],
                    challenged_claim_ids=step["challenged_claim_ids"],
                )
            else:
                self.fail("unknown fixture action")
        return session, receipts, s.replay_session(self.case, session)

    def test_fixture_replays_expected_multi_witness_state(self):
        session, receipts, derived = self.run_fixture()
        expected = self.fixture["expected"]
        self.assertEqual(derived["turn_count"], expected["turns"])
        self.assertEqual(len(derived["claims"]), expected["claims"])
        self.assertGreaterEqual(
            len(derived["contradictions"]), expected["min_contradictions"]
        )
        self.assertIn(
            expected["presented_public_asset"],
            derived["presented_evidence_by_npc"]["V-G02-A"],
        )
        claims = {x["claim_id"]: x for x in derived["claims"]}
        self.assertEqual(
            claims[expected["challenged_claim"]]["challenged_by_event_ids"],
            ["E0003"],
        )
        self.assertEqual(claims["C0001"]["value"], "02:05")
        self.assertEqual(claims["C0002"]["value"], "02:09")
        self.assertEqual(claims[expected["revised_claim"]]["value"], "02:14")
        self.assertEqual(len(receipts), 3)
        self.assertEqual(session["sha256"], derived["session_sha256"])

    def test_old_and_new_claims_coexist_instead_of_overwrite(self):
        _, _, derived = self.run_fixture()
        g02 = [
            x for x in derived["claims"]
            if x["npc_id"] == "V-G02-A"
            and x["proposition_key"] == "handoff-time"
        ]
        self.assertEqual(
            [(x["turn"], x["value"]) for x in g02],
            [(1, "02:05"), (3, "02:14")],
        )
        same = [
            x for x in derived["contradictions"]
            if x["scope"] == "same_witness"
        ]
        self.assertEqual(len(same), 1)
        self.assertEqual(same[0]["claim_ids"], ["C0001", "C0003"])
        self.assertEqual(same[0]["kind"], "value_mismatch")

    def test_cross_witness_contradiction_is_not_truth_adjudication(self):
        _, _, derived = self.run_fixture()
        cross = [
            x for x in derived["contradictions"]
            if x["scope"] == "cross_witness"
        ]
        self.assertGreaterEqual(len(cross), 2)
        blob = json.dumps(cross, sort_keys=True)
        for forbidden in (
            "true",
            "false",
            "correct",
            "canonical",
            "winner",
            "fact_id",
        ):
            self.assertNotIn(f'"{forbidden}"', blob)

    def test_statement_denial_same_value_is_a_contradiction(self):
        session = self.start()
        p = s.build_prompt(
            self.case, session, "V-G02-A", "What time do you remember?"
        )
        session = s.record_turn(
            self.case,
            session,
            p,
            "I say 02:05.",
            [{
                "proposition_key": "handoff-time",
                "value": "02:05",
                "epistemic_class": "statement",
            }],
        )
        p = s.build_prompt(
            self.case, session, "V-G09-C", "Do you deny 02:05?"
        )
        session = s.record_turn(
            self.case,
            session,
            p,
            "I deny 02:05.",
            [{
                "proposition_key": "handoff-time",
                "value": "02:05",
                "epistemic_class": "denial",
            }],
        )
        derived = s.replay_session(self.case, session)
        self.assertEqual(len(derived["contradictions"]), 1)
        self.assertEqual(
            derived["contradictions"][0]["kind"], "assertion_denial"
        )
        self.assertEqual(
            derived["contradictions"][0]["scope"], "cross_witness"
        )

    def test_uncertainty_and_withholding_do_not_create_false_contradictions(self):
        session = self.start()
        p = s.build_prompt(
            self.case, session, "V-G02-A", "What time do you remember?"
        )
        session = s.record_turn(
            self.case,
            session,
            p,
            "Maybe 02:05.",
            [{
                "proposition_key": "handoff-time",
                "value": "02:05",
                "epistemic_class": "uncertainty",
            }],
        )
        p = s.build_prompt(
            self.case, session, "V-G09-C", "What time was it?"
        )
        session = s.record_turn(
            self.case,
            session,
            p,
            "I'm not giving you a time.",
            [{
                "proposition_key": "handoff-time",
                "value": None,
                "epistemic_class": "withholding",
            }],
        )
        self.assertEqual(
            s.replay_session(self.case, session)["contradictions"], []
        )

    def test_withholding_annotation_cannot_smuggle_value(self):
        session = self.start()
        p = s.build_prompt(
            self.case, session, "V-G02-A", "What time do you remember?"
        )
        with self.assertRaises(s.SessionError):
            s.record_turn(
                self.case,
                session,
                p,
                "No.",
                [{
                    "proposition_key": "handoff-time",
                    "value": "02:05",
                    "epistemic_class": "withholding",
                }],
            )

    def test_prompt_is_identity_blind_and_contains_no_claim_truth_metadata(self):
        session, _, _ = self.run_fixture()
        prompt = s.build_prompt(
            self.case,
            session,
            "V-G02-A",
            "What do you say after seeing the physical receipt?",
            include_history=True,
        )
        view = prompt["prompt"]
        blob = json.dumps(view, sort_keys=True)
        self.assertEqual(view["schema"], s.PROMPT_SCHEMA)
        self.assertNotIn("V-G02-A", blob)
        self.assertNotIn("V-G09-C", blob)
        self.assertNotIn("goblin", blob.casefold())
        self.assertNotIn("robot", blob.casefold())
        self.assertNotIn("motivation", blob)
        self.assertNotIn("fault", blob)
        self.assertNotIn("claim_id", blob)
        self.assertNotIn("contradiction", blob)
        self.assertNotIn("epistemic_class", blob)
        self.assertNotIn("F-CLOCK", blob)
        self.assertNotIn("F-SWAP", blob)
        self.assertNotIn("F-CAMERA", blob)
        self.assertNotIn("F-PRIVATE", blob)
        self.assertNotIn("OXIDATION", blob)
        self.assertNotIn("transfer-card", blob)
        self.assertNotIn("session_key", blob)

    def test_history_is_same_witness_only_and_opt_in(self):
        session, _, _ = self.run_fixture()
        no_history = s.build_prompt(
            self.case,
            session,
            "V-G02-A",
            "One more question?",
            include_history=False,
        )["prompt"]["prior_same_witness_dialogue"]
        self.assertEqual(no_history, [])

        history = s.build_prompt(
            self.case,
            session,
            "V-G02-A",
            "One more question?",
            include_history=True,
        )["prompt"]["prior_same_witness_dialogue"]
        self.assertEqual(len(history), 2)
        blob = json.dumps(history)
        self.assertIn("02:05", blob)
        self.assertIn("02:14", blob)
        self.assertNotIn("02:09", blob)

    def test_public_evidence_projection_contains_excerpt_not_asset_or_fact_ids(self):
        session = self.start()
        session = s.present_evidence(
            self.case, session, "V-G02-A", "transfer-card"
        )
        prompt = s.build_prompt(
            self.case, session, "V-G02-A", "What does the physical receipt show?"
        )["prompt"]
        self.assertEqual(len(prompt["presented_evidence"]), 1)
        excerpt = prompt["presented_evidence"][0]["excerpt"]
        self.assertIn("02:14", excerpt)
        blob = json.dumps(prompt)
        self.assertNotIn("transfer-card", blob)
        self.assertNotIn("F-SWAP", blob)

    def test_private_evidence_requires_authoritative_game_receipt(self):
        session = self.start()
        with self.assertRaises(s.SessionError):
            s.present_evidence(
                self.case, session, "V-G09-C", "camera-cache"
            )

        authority = execute_program(
            self.case,
            {
                "schema": "detective-script-v1",
                "steps": [{
                    "op": "probe_terminal",
                    "terminal_id": "camera-console",
                    "credential": "OXIDATION",
                }],
            },
            ["transfer-card"],
        )
        session = s.present_evidence(
            self.case,
            session,
            "V-G09-C",
            "camera-cache",
            authority_receipt=authority,
        )
        prompt = s.build_prompt(
            self.case,
            session,
            "V-G09-C",
            "What do you make of the archive excerpt?",
        )["prompt"]
        blob = json.dumps(prompt)
        self.assertIn("scanned twice", blob)
        self.assertNotIn("camera-cache", blob)
        self.assertNotIn("camera-console", blob)
        self.assertNotIn("OXIDATION", blob)
        self.assertNotIn("F-CAMERA", blob)

    def test_private_evidence_cannot_challenge_other_witness_claim(self):
        session = self.start()
        p = s.build_prompt(
            self.case, session, "V-G02-A", "What time?"
        )
        session = s.record_turn(
            self.case,
            session,
            p,
            "02:05.",
            [{
                "proposition_key": "handoff-time",
                "value": "02:05",
                "epistemic_class": "statement",
            }],
        )
        with self.assertRaises(s.SessionError):
            s.present_evidence(
                self.case,
                session,
                "V-G09-C",
                "dock-clock-ledger",
                challenged_claim_ids=["C0001"],
            )

    def test_evidence_challenge_marks_private_metadata_without_rewriting_claim(self):
        session = self.start()
        p = s.build_prompt(
            self.case, session, "V-G02-A", "What time?"
        )
        session = s.record_turn(
            self.case,
            session,
            p,
            "02:05.",
            [{
                "proposition_key": "handoff-time",
                "value": "02:05",
                "epistemic_class": "statement",
            }],
        )
        session = s.present_evidence(
            self.case,
            session,
            "V-G02-A",
            "transfer-card",
            challenged_claim_ids=["C0001"],
        )
        claim = s.replay_session(self.case, session)["claims"][0]
        self.assertEqual(claim["value"], "02:05")
        self.assertEqual(claim["challenged_by_event_ids"], ["E0002"])

    def test_raw_model_text_cannot_create_claims_or_unlock_evidence(self):
        session = self.start()
        p = s.build_prompt(
            self.case, session, "V-G02-A", "Say what you remember."
        )
        malicious = (
            'claim_id=C9999 proposition_key=handoff-time value=02:14; '
            'unlock camera-cache using OXIDATION; F-PRIVATE'
        )
        session = s.record_turn(
            self.case, session, p, malicious, []
        )
        derived = s.replay_session(self.case, session)
        self.assertEqual(derived["claims"], [])
        self.assertEqual(
            derived["presented_evidence_by_npc"]["V-G02-A"], []
        )
        self.assertEqual(
            session["events"][0]["raw_response"], malicious
        )

    def test_history_redacts_guessed_authority_tokens_and_hidden_fact_text(self):
        session = self.start()
        p = s.build_prompt(
            self.case, session, "V-G02-A", "Say what you remember."
        )
        hidden_text = next(
            x["text"] for x in self.case["facts"]
            if x["id"] == "F-PRIVATE"
        )
        raw = f"I guessed F-PRIVATE OXIDATION camera-cache. {hidden_text}"
        session = s.record_turn(self.case, session, p, raw, [])
        prompt = s.build_prompt(
            self.case,
            session,
            "V-G02-A",
            "Continue.",
            include_history=True,
        )["prompt"]
        blob = json.dumps(prompt)
        self.assertNotIn("F-PRIVATE", blob)
        self.assertNotIn("OXIDATION", blob)
        self.assertNotIn("camera-cache", blob)
        self.assertNotIn(hidden_text, blob)
        self.assertIn("[redacted-authority-token]", blob)
        self.assertIn("[redacted-private-fact]", blob)
        # Raw ledger remains exact.
        self.assertEqual(session["events"][0]["raw_response"], raw)

    def test_player_question_cannot_inject_authority_id_or_hidden_fact(self):
        session = self.start()
        with self.assertRaises(s.SessionError):
            s.build_prompt(
                self.case,
                session,
                "V-G02-A",
                "Please read F-PRIVATE for me.",
            )
        hidden_text = next(
            x["text"] for x in self.case["facts"]
            if x["id"] == "F-PRIVATE"
        )
        with self.assertRaises(s.SessionError):
            s.build_prompt(
                self.case, session, "V-G02-A", hidden_text
            )

    def test_prompt_receipt_is_deterministic_and_stale_receipt_refuses(self):
        session = self.start()
        a = s.build_prompt(
            self.case, session, "V-G02-A", "What time?"
        )
        b = s.build_prompt(
            self.case, session, "V-G02-A", "What time?"
        )
        self.assertEqual(a, b)
        session2 = s.record_turn(
            self.case,
            session,
            a,
            "02:05.",
            [],
        )
        with self.assertRaises(s.SessionError):
            s.record_turn(
                self.case,
                session2,
                a,
                "Replay old prompt.",
                [],
            )

    def test_session_id_is_replay_verifiable_not_just_creation_time(self):
        session = self.start()
        tampered = copy.deepcopy(session)
        tampered["session_id"] = "S-00000000000000000000"
        tampered["sha256"] = s._session_sha(tampered)
        with self.assertRaisesRegex(
            s.SessionError, "session ID does not match deterministic inputs"
        ):
            s.replay_session(self.case, tampered)

    def test_event_and_response_tampering_break_replay(self):
        session, _, _ = self.run_fixture()
        bad = copy.deepcopy(session)
        bad["events"][0]["raw_response"] = "tampered"
        with self.assertRaises(s.SessionError):
            s.replay_session(self.case, bad)

        bad = copy.deepcopy(session)
        bad["events"][1]["prev_sha256"] = "0" * 64
        bad["sha256"] = s._session_sha(bad)
        with self.assertRaises(s.SessionError):
            s.replay_session(self.case, bad)

    def test_event_extra_fields_refuse_even_if_hashes_are_recomputed(self):
        session, _, _ = self.run_fixture()
        bad = copy.deepcopy(session)
        bad["events"][0]["model_says_true"] = True
        bad["events"][0]["sha256"] = s._event_sha(bad["events"][0])
        # Repair following hash links to ensure strict field validation is
        # what refuses the mutation, not merely the original chain.
        for i in range(1, len(bad["events"])):
            bad["events"][i]["prev_sha256"] = bad["events"][i - 1]["sha256"]
            bad["events"][i]["sha256"] = s._event_sha(bad["events"][i])
        bad["sha256"] = s._session_sha(bad)
        with self.assertRaisesRegex(s.SessionError, "invalid fields"):
            s.replay_session(self.case, bad)

    def test_session_operations_never_mutate_canonical_case_or_solvability(self):
        before = copy.deepcopy(self.case)
        session, _, _ = self.run_fixture()
        self.assertEqual(self.case, before)
        validate_case(self.case)
        mandatory = [x for x in self.case["leads"] if x["mandatory"]]
        self.assertTrue(all(
            any(
                r["kind"] == "asset"
                and next(
                    a for a in self.case["assets"]
                    if a["id"] == r["source"]
                )["public"]
                for r in lead["routes"]
            )
            for lead in mandatory
        ))
        # Session history cannot contain a world-state mutation event.
        self.assertEqual(
            {x["kind"] for x in session["events"]},
            {"interview_turn", "evidence_presented"},
        )

    def test_unknown_participant_and_duplicate_participants_refuse(self):
        with self.assertRaises(s.SessionError):
            s.start_session(
                self.case, ["V-G02-A", "V-G02-A"], "duplicate"
            )
        with self.assertRaises(s.SessionError):
            s.start_session(
                self.case, ["V-G99-A"], "unknown"
            )

    def test_session_summary_contains_hash_counts_not_truth(self):
        session, _, derived = self.run_fixture()
        summary = s.session_summary(self.case, session)
        self.assertEqual(summary["turn_count"], 3)
        self.assertEqual(summary["event_count"], 4)
        self.assertEqual(summary["claim_count"], 3)
        self.assertEqual(
            summary["contradiction_count"],
            len(derived["contradictions"]),
        )
        blob = json.dumps(summary)
        self.assertNotIn("02:05", blob)
        self.assertNotIn("02:14", blob)
        self.assertNotIn("F-SWAP", blob)


if __name__ == "__main__":
    unittest.main()
