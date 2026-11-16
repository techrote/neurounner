"""Offline/adversarial readback/guide tests. No native/model dependencies."""
import io
import json
import struct
import unittest
from hashlib import sha256
from zipfile import ZipFile, ZIP_DEFLATED
from experiments.driv3r.stage_d.guidance import native_pair_shadow as s


class RetrospectiveGuardTests(unittest.TestCase):
    def _world_zip(self, *, changed: bool, wrong_world=False, bad_delta=False):
        before = bytearray(512 * 512 * 4)
        x, y = 12, 23
        off = 4 * (y * 512 + x)
        after = before[:]
        if changed:
            struct.pack_into('<i', after, off, 58)
        if bad_delta:
            struct.pack_into('<i', after, 4 * (y * 512 + x + 1), 58)
        name = 'readback-000005-after.world.i32le'
        resp = {'observation': {'world_material_path': name}}
        payload = io.BytesIO()
        with ZipFile(payload, 'w', compression=ZIP_DEFLATED) as z:
            z.writestr('execution/native/readback-000005-before.world.i32le', before)
            z.writestr('execution/native/' + name, after)
        payload.seek(0)
        return payload, {'request_id': 5}, resp, sha256(before).hexdigest(), (
            'f' * 64 if wrong_world else sha256(after).hexdigest())

    def test_real_one_cell_delta_and_brush_up_exact_bytes(self):
        for changed in (True, False):
            with self.subTest(changed=changed):
                payload, req, resp, b, a = self._world_zip(changed=changed)
                with ZipFile(payload) as z:
                    s._verify_world_readback(z, req, resp, b, a, [12, 23] if changed else None)

    def test_forbids_source_hash_substitution(self):
        payload, req, resp, b, a = self._world_zip(changed=True, wrong_world=True)
        with ZipFile(payload) as z, self.assertRaisesRegex(ValueError, 'source bytes mismatch'):
            s._verify_world_readback(z, req, resp, b, a, [12, 23])

    def test_forbids_second_unreceipted_material_edit(self):
        payload, req, resp, b, a = self._world_zip(changed=True, bad_delta=True)
        with ZipFile(payload) as z, self.assertRaisesRegex(ValueError, 'sole expected'):
            s._verify_world_readback(z, req, resp, b, a, [12, 23])

    def test_forbids_lifted_cursor_material_change(self):
        payload, req, resp, b, a = self._world_zip(changed=True)
        with ZipFile(payload) as z, self.assertRaisesRegex(ValueError, 'lifted cursor changed'):
            s._verify_world_readback(z, req, resp, b, a, None)

    def test_numerical_field_outside_is_neutral_not_permission(self):
        guide = {'frame': [4, 5, 2, 2]}
        data = struct.pack('<4H', 0, 20000, 40000, 65535)
        self.assertEqual(s._guide_at(data, guide, [4, 5]), 0)
        self.assertEqual(s._guide_at(data, guide, [5, 6]), 65535)
        self.assertEqual(s._guide_at(data, guide, [3, 5]), 0)
        self.assertEqual(s._guide_at(data, guide, [6, 5]), 0)

    def test_original_guide_sha_must_match_raw_u16_bytes(self):
        guide = {'protocol': 'cybersand.driv3r.d-common/1',
                 'encoding': 'u16le_higher_is_better_zero_is_neutral',
                 'native_material_authority': False, 'type': 'guidance',
                 'purpose': 'soft_positive_preference', 'frame': [4, 5, 2, 2],
                 'values': {'path': 'guide/positive.u16le', 'format': 'u16le',
                            'bytes': 8, 'sha256': 'a' * 64}}
        stream = io.BytesIO()
        with ZipFile(stream, 'w') as z:
            z.writestr('execution/original-guide.json', json.dumps(guide))
            z.writestr('execution/guide/positive.u16le', b'\x00' * 8)
        stream.seek(0)
        with ZipFile(stream) as z, self.assertRaisesRegex(ValueError, 'guide byte custody'):
            s._load_guide(z)

    def test_unauthorised_material_guide_rejected(self):
        guide = {'protocol': 'cybersand.driv3r.d-common/1',
                 'encoding': 'u16le_higher_is_better_zero_is_neutral',
                 'native_material_authority': True, 'type': 'guidance',
                 'purpose': 'soft_positive_preference', 'frame': [4, 5, 2, 2],
                 'values': {'path': 'guide/positive.u16le', 'format': 'u16le',
                            'bytes': 8, 'sha256': sha256(b'\x00' * 8).hexdigest()}}
        stream = io.BytesIO()
        with ZipFile(stream, 'w') as z:
            z.writestr('execution/original-guide.json', json.dumps(guide))
            z.writestr('execution/guide/positive.u16le', b'\x00' * 8)
        stream.seek(0)
        with ZipFile(stream) as z, self.assertRaisesRegex(ValueError, 'guide semantics'):
            s._load_guide(z)

    def test_rejects_invalid_arm_or_count(self):
        for mode, count in [('invalid', 36), ('free', 0), ('soft', 37), ('free', True)]:
            with self.subTest(mode=mode, count=count), self.assertRaisesRegex(ValueError, 'arm/count'):
                s.audit_arm('/not-found', mode, count)


class OriginalModelBindingTests(unittest.TestCase):
    def _turn(self):
        ids = ["MOVE_N", "MOVE_E", "STOP"]
        probs = {"MOVE_N": 0.7, "MOVE_E": 0.2, "STOP": 0.1}
        state = {"eligible_moves": ids[:],
                 "candidate_hints": {"MOVE_N": {}, "MOVE_E": {}}}
        questions = {"worm_move": {"type": "choice", "criteria": dict.fromkeys(ids, "x")}}
        turn = {"model_state": state, "questions": questions,
                "exact_candidate_ids_in_presented_order": ids[:],
                "chosen_candidate": "MOVE_N",
                "parsed_choice": {"candidate_id": "MOVE_N",
                                  "probabilities": probs.copy(), "confidence": 0.7},
                "original_response": {"status": "ok", "raw_response": {
                    "usage": {"input_tokens": 5, "output_tokens": 0},
                    "answers": {"worm_move": {"type": "choice", "choice": "MOVE_N",
                                "probabilities": probs.copy(), "confidence": 0.7}}}}}
        self._seal(turn)
        return turn

    def _seal(self, turn):
        ids = turn["exact_candidate_ids_in_presented_order"]
        turn["presented_order_sha256"] = s._canonical_digest(ids)
        turn["presentation_sha256"] = s._canonical_digest({
            "state": turn["model_state"], "questions": turn["questions"],
            "ordered_candidates": ids})

    def test_valid_original_raw_q8_binding(self):
        self.assertIsNone(s._verify_original_model_choice(self._turn()))

    def test_changed_logged_selected_action_rejected(self):
        t = self._turn()
        t["chosen_candidate"] = "MOVE_E"
        with self.assertRaisesRegex(ValueError, "differs from original Q8"):
            s._verify_original_model_choice(t)

    def test_missing_legal_move_hint_rejected_even_after_reseal(self):
        t = self._turn()
        del t["model_state"]["candidate_hints"]["MOVE_E"]
        self._seal(t)
        with self.assertRaisesRegex(ValueError, "missing or foreign"):
            s._verify_original_model_choice(t)

    def test_tampered_model_probability_distribution_rejected(self):
        t = self._turn()
        t["original_response"]["raw_response"]["answers"]["worm_move"][
            "probabilities"]["MOVE_E"] = 0.9
        with self.assertRaisesRegex(ValueError, "original Q8 response invalid"):
            s._verify_original_model_choice(t)



class OriginalHostRecoveryReplayTests(unittest.TestCase):
    """Original #611 host rules, not model-selected or native-authored authority."""

    @staticmethod
    def _state(history):
        return {"recent_positions": history[-7:],
                "already_inked_cells_this_run": len(history) - 1}

    def test_real_backtrack_host_menu_and_replayed_history(self):
        history = [[12, 24], [12, 25], [13, 25]]
        s._verify_recovery_catalog(self._state(history), ["BACKTRACK", "MOVE_N", "STOP"],
                                   history[-1], history, 2, 0)
        s._advance_replayed_history(history, [13, 25], "BACKTRACK", [12, 25])
        self.assertEqual(history, [[12, 24], [12, 25]])
        s._verify_recovery_catalog(self._state(history), ["MOVE_E", "STOP"],
                                   history[-1], history, 2, 1)

    def test_omitted_or_illegal_recovery_option_fails(self):
        history = [[12, 24], [12, 25], [13, 25]]
        with self.assertRaisesRegex(ValueError, "host recovery legality"):
            s._verify_recovery_catalog(self._state(history), ["MOVE_N", "STOP"],
                                       history[-1], history, 2, 0)
        short = [[12, 24]]
        with self.assertRaisesRegex(ValueError, "host recovery legality"):
            s._verify_recovery_catalog(self._state(short), ["BACKTRACK", "STOP"],
                                       short[-1], short, 0, 0)

    def test_resealed_fake_model_history_rejected(self):
        history = [[12, 24], [12, 25], [13, 25]]
        forged = self._state(history)
        forged["recent_positions"] = [[99, 99]]
        with self.assertRaisesRegex(ValueError, "history/model context"):
            s._verify_recovery_catalog(forged, ["BACKTRACK", "STOP"],
                                       history[-1], history, 2, 0)

    def test_forged_backtrack_target_rejected_without_history_mutation(self):
        history = [[12, 24], [12, 25], [13, 25]]
        with self.assertRaisesRegex(ValueError, "backtrack did not return"):
            s._advance_replayed_history(history, [13, 25], "BACKTRACK", [12, 24])
        self.assertEqual(history, [[12, 24], [12, 25], [13, 25]])

    def test_saved_birth_branch_relocation_and_cap(self):
        history = [[12, 24], [12, 25], [13, 25], [14, 25], [15, 25], [16, 25]]
        s._verify_recovery_catalog(self._state(history),
                                   ["BACKTRACK", "RELOCATE_START", "STOP"],
                                   history[-1], history, 5, 0)
        with self.assertRaisesRegex(ValueError, "original saved birth cursor"):
            s._advance_replayed_history(history, [16, 25], "RELOCATE_START", [13, 25])
        s._advance_replayed_history(history, [16, 25], "RELOCATE_START", [12, 24])
        self.assertEqual(history[-1], [12, 24])
        s._verify_recovery_catalog(self._state(history), ["STOP"],
                                   history[-1], history, 5, 8)

    def test_cardinal_move_appends_one_step_and_rejects_teleport(self):
        history = [[12, 24]]
        with self.assertRaisesRegex(ValueError, "cardinal control"):
            s._advance_replayed_history(history, [12, 24], "MOVE_E", [15, 24])
        self.assertEqual(history, [[12, 24]])
        s._advance_replayed_history(history, [12, 24], "MOVE_E", [13, 24])
        self.assertEqual(history, [[12, 24], [13, 24]])



if __name__ == '__main__':
    unittest.main()
