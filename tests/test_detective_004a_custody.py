"""Author-side secondary custody tests: inspect archive, never train or open finals."""
from __future__ import annotations

import copy
import json
import unittest

from scripts import detective_004a_custody_audit as audit
from scripts import detective_004a_protocol as protocol


class Detective004ACustodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.b3 = protocol.load_contract()
        cls.rows = protocol.build_training(cls.cfg, cls.b3)

    def test_all_36_exact_model_visible_views_and_receipts(self):
        for row in self.rows:
            with self.subTest(row=row["id"]):
                audit.audit_row(row, self.b3)

    def test_committed_archive_replay_without_final_or_training(self):
        result = audit.audit_archive(protocol.DATASET)
        self.assertEqual(result["rows_with_exact_model_visible_views"], 36)
        self.assertEqual(result["rows_with_replayed_assistant_char_spans"], 36)
        self.assertEqual(result["host_owned_result_continuations"], 6)
        self.assertFalse(result["real_tokenizer_labels_verified"])
        self.assertFalse(result["independent_protocol_approval"])
        self.assertEqual(result["optimizer_steps"], 0)

    def test_unexposed_view_change_refused_even_if_producer_accepted_it(self):
        row = copy.deepcopy(next(r for r in self.rows if r["kind"] == "tool_selection"))
        row["view"]["visible_assets"].append({
            "asset_id": "forged-out-of-band-asset", "excerpt": "not displayed to model"
        })
        # Old validator inspects view and assistant target, but not row view vs
        # the actual string inside the user turn. The secondary audit closes it.
        protocol.validate_example(row, self.b3)
        with self.assertRaisesRegex(protocol.ProtocolError, "not identical"):
            audit.audit_row(row, self.b3)

    def test_case_family_alias_refused(self):
        row = copy.deepcopy(self.rows[0])
        row["case_family"] = "cinder-span"
        with self.assertRaises(protocol.ProtocolError):
            audit.audit_row(row, self.b3)

    def test_host_receipt_is_not_the_user_turn(self):
        row = copy.deepcopy(next(r for r in self.rows if r["kind"] == "result_success"))
        row["view"]["player_request"] += " confirmed:amber-quay:5"
        with self.assertRaisesRegex(protocol.ProtocolError, "not identical"):
            audit.audit_row(row, self.b3)

    def test_host_result_tamper_rejected(self):
        row = copy.deepcopy(next(r for r in self.rows if r["kind"] == "result_denied"))
        row["turns"][3]["content"] = json.dumps(
            {"status": "ACCESS_DENIED", "evidence": "unadmitted record"}
        )
        with self.assertRaises(protocol.ProtocolError):
            audit.audit_row(row, self.b3)


if __name__ == "__main__":
    unittest.main()
