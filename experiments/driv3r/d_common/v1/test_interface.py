"""Synthetic conformance tests; never classified as native or inference evidence."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

from jsonschema.exceptions import ValidationError

from verify import ContractError, SCHEMA, records_from_examples, validate_bundle


class ContractV1Tests(unittest.TestCase):
    def setUp(self):
        self.rows = copy.deepcopy(records_from_examples())

    def reject(self, change):
        change(self.rows)
        with self.assertRaises((ContractError, ValidationError)):
            validate_bundle(self.rows)

    def test_five_synthetic_types_are_shape_valid(self):
        result = validate_bundle(self.rows)
        self.assertEqual(result["validated"], 5)
        self.assertEqual(result["types"], ["action", "guidance", "observation", "provenance", "reversal"])
        self.assertFalse(result["evidence_bytes_verified"])

    def test_unknown_field_rejected(self):
        self.reject(lambda rows: rows[1].__setitem__("guess_permission", True))

    def test_protocol_version_is_fenced(self):
        self.reject(lambda rows: rows[2].__setitem__("protocol", "cybersand.driv3r.d-common/2"))

    def test_guidance_never_grants_native_permissions(self):
        self.reject(lambda rows: rows[0].__setitem__("native_material_authority", True))

    def test_guidance_u16_extent_required(self):
        self.reject(lambda rows: rows[0]["values"].__setitem__("bytes", 127))

    def test_hysteresis_requires_reentry_above_exit(self):
        self.reject(lambda rows: rows[0]["hysteresis"].__setitem__("reenter_at_or_above", 12000))

    def test_stale_action_world_rejected(self):
        self.reject(lambda rows: rows[2]["world_precondition"].__setitem__("revision", 8))

    def test_action_cannot_claim_native_receipt_while_candidate(self):
        self.reject(lambda rows: rows[2].__setitem__("native_receipt_ref", "receipt-does-not-exist"))

    def test_guidance_ref_must_exist(self):
        self.reject(lambda rows: rows[1].__setitem__("guidance_ref", "guide-unknown"))

    def test_cursor_only_backtrack_cannot_erase(self):
        self.reject(lambda rows: rows[3].__setitem__("actual_material_cells_changed", 1))

    def test_cursor_only_reversal_preserves_hash(self):
        self.reject(lambda rows: rows[3]["after_world"].__setitem__("world_material_sha256", "9" * 64))

    def test_reversal_does_not_fake_material_undo(self):
        self.reject(lambda rows: rows[3].__setitem__("preserve_material_edits", False))

    def test_applied_true_undo_requires_receipt(self):
        def alter(rows):
            rows[3]["mechanism"] = "UNDO_MATERIAL_EDIT"
            rows[3]["preserve_material_edits"] = False
        self.reject(alter)

    def test_nonmonotonic_sequences_rejected(self):
        self.reject(lambda rows: rows[2].__setitem__("sequence", 0))

    def test_no_paint_action_must_lift_brush(self):
        self.reject(lambda rows: rows[2].__setitem__("brush_state", "erase"))

    def test_native_crop_material_extent_checked(self):
        self.reject(lambda rows: rows[1]["local_material"].__setitem__("bytes", 100))


    def test_variable_world_and_coordinates_above_511_supported(self):
        guidance, obs, action, reversal, _ = self.rows
        guidance["canvas_size"] = [2048, 1024]
        for world in (obs["world"], action["world_precondition"],
                      reversal["before_world"], reversal["after_world"]):
            world["width_cells"], world["height_cells"] = 2048, 1024
        obs["rules"]["focus"] = [0, 0, 800, 700]
        obs["cursor"] = [700, 600]
        obs["target"] = [710, 600]
        action["cursor_before"] = action["cursor_after"] = [700, 600]
        action["path"] = [[700, 600]]
        reversal["cursor_before"] = [700, 600]
        reversal["cursor_after"] = [699, 600]
        self.assertEqual(validate_bundle(self.rows)["validated"], 5)

    def test_non_square_world_supported(self):
        guidance, obs, action, reversal, _ = self.rows
        guidance["canvas_size"] = [640, 1024]
        for world in (obs["world"], action["world_precondition"],
                      reversal["before_world"], reversal["after_world"]):
            world["width_cells"], world["height_cells"] = 640, 1024
        self.assertEqual(validate_bundle(self.rows)["validated"], 5)

    def test_world_too_small_for_focus_refused(self):
        def change(rows):
            for world in (rows[1]["world"], rows[2]["world_precondition"],
                          rows[3]["before_world"], rows[3]["after_world"]):
                world["width_cells"] = 64
            rows[0]["canvas_size"][0] = 64
        self.reject(change)

    def test_guidance_canvas_must_equal_world(self):
        self.reject(lambda rows: rows[0].__setitem__("canvas_size", [640, 512]))

    def test_guidance_tile_must_fit_canvas_not_global_512(self):
        self.reject(lambda rows: rows[0].__setitem__("frame", [510, 0, 8, 8]))

    def test_cursor_must_remain_inside_authorized_focus(self):
        def change(rows):
            rows[1]["cursor"] = [100, 24]
            rows[2]["cursor_before"] = [100, 24]
            rows[2]["cursor_after"] = [100, 24]
            rows[2]["path"] = [[100, 24]]
        self.reject(change)

    def test_native_crop_within_dynamic_world(self):
        self.reject(lambda rows: rows[1].__setitem__("local_crop", [509, 0, 8, 8]))

    def test_unexecuted_candidate_cannot_claim_post_world(self):
        self.reject(lambda rows: rows[2].__setitem__("world_postcondition",
                                                   copy.deepcopy(rows[2]["world_precondition"])))

    def test_unexecuted_candidate_cannot_claim_hashed_receipt(self):
        self.reject(lambda rows: rows[2].__setitem__("native_receipt_artifact", {
            "path":"receipts/fake.json", "sha256":"a" * 64, "bytes":100, "format":"json"}))

    def test_synthetic_provenance_may_omit_native_identity(self):
        self.assertNotIn("source_commit", self.rows[-1])
        self.assertNotIn("native", self.rows[-1])
        self.assertEqual(validate_bundle(self.rows)["validated"], 5)

    def test_synthetic_cannot_claim_native_replay(self):
        self.reject(lambda rows: rows[-1].__setitem__("replay", "pass"))

    def test_applied_material_undo_needs_changed_native_world_and_receipt(self):
        def change(rows):
            rev = rows[3]
            rev["mechanism"] = "UNDO_MATERIAL_EDIT"
            rev["preserve_material_edits"] = False
            rev["native_undo_receipt_ref"] = "receipt-undo-001"
            rev["native_undo_receipt_artifact"] = {
                "path":"receipts/undo.json", "sha256":"b" * 64, "bytes":100, "format":"json"}
            rev["actual_material_cells_changed"] = 1
            # Same before/after hash is not a real native Undo.
        self.reject(change)

    def test_cursor_only_relocation_cannot_invent_native_receipt(self):
        self.reject(lambda rows: rows[3].__setitem__("native_undo_receipt_artifact", {
            "path":"receipts/fake.json", "sha256":"c" * 64, "bytes":100, "format":"json"}))

    def test_actual_material_mutation_requires_native_provenance(self):
        def change(rows):
            action = rows[2]
            action["mode"] = "add_material"
            action["brush_state"] = "add"
            action["operation_sha256"] = "a" * 64
            action["execution_status"] = "committed"
            action["native_receipt_ref"] = "receipt-001"
            action["native_receipt_artifact"] = {
                "path":"receipts/add.json", "sha256":"d" * 64, "bytes":100, "format":"json"}
            action["world_postcondition"] = copy.deepcopy(action["world_precondition"])
            action["world_postcondition"]["revision"] += 1
            action["world_postcondition"]["world_material_sha256"] = "e" * 64
            # Provenance remains 'synthetic', so this must refuse.
        self.reject(change)

    def test_native_claim_requires_verified_evidence_root(self):
        def change(rows):
            provenance=rows[-1]
            provenance.update({
                "evidence_class":"native", "source_commit":"a" * 40,
                "source_tree":"b" * 40, "base_commit":"c" * 40,
                "interface_schema_sha256":"d" * 64, "scenario_sha256":"e" * 64,
                "model":None, "native":{"path":"native/world.i32le",
                  "sha256":"f" * 64, "bytes":512*512*4, "format":"i32le"},
                "native_receipts":[], "final_world":copy.deepcopy(rows[1]["world"])
            })
        self.reject(change)

    def test_applied_material_undo_requires_post_world(self):
        self.reject(lambda rows: rows[3].__setitem__("after_world", None))


    def _constructed_native_byte_fixture(self, root: Path):
        """Byte-custody example only, NOT actual Godot execution evidence."""
        rows = copy.deepcopy(self.rows)
        def put(path, payload, fmt):
            location = root / path
            location.parent.mkdir(parents=True, exist_ok=True)
            location.write_bytes(payload)
            return {"path":path, "sha256":hashlib.sha256(payload).hexdigest(),
                    "bytes":len(payload), "format":fmt}

        def png():
            def chunk(kind, data):
                return (struct.pack(">I",len(data)) + kind + data +
                        struct.pack(">I",zlib.crc32(kind + data) & 0xFFFFFFFF))
            signature=b"\x89PNG\r\n\x1a\n"
            ihdr=struct.pack(">IIBBBBB",1,1,8,2,0,0,0)
            return signature + chunk(b"IHDR",ihdr) + chunk(b"IDAT",zlib.compress(b"\x00\x00\x00\x00")) + chunk(b"IEND",b"")
        rows[0]["values"] = put("guide/guide.u16le", b"\x00" * 128, "u16le")
        rows[0]["display"] = put("guide/speckles.png", png(), "png")
        rows[1]["local_material"] = put("native/crop.i32le", b"\x00" * 256, "i32le")
        full_world = put("native/full.i32le", b"\x00" * (512*512*4), "i32le")
        for world in (rows[1]["world"], rows[2]["world_precondition"],
                      rows[3]["before_world"], rows[3]["after_world"]):
            world["world_material_sha256"] = full_world["sha256"]
        rows[-1].update({
            "evidence_class":"native", "source_commit":"a" * 40,
            "source_tree":"b" * 40, "base_commit":"c" * 40,
            "interface_schema_sha256": hashlib.sha256(SCHEMA.read_bytes()).hexdigest(),
            "scenario_sha256":"d" * 64, "model":None,
            "native":full_world, "native_receipts":[],
            "final_world":copy.deepcopy(rows[1]["world"])
        })
        return rows

    def test_bounded_byte_custody_accepts_constructed_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            result = validate_bundle(self._constructed_native_byte_fixture(Path(folder)),
                                     evidence_root=Path(folder))
            self.assertTrue(result["evidence_bytes_verified"])
            # This demonstrates byte checking, not source-matched native qualification.

    def test_byte_custody_detects_corrupt_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            rows=self._constructed_native_byte_fixture(root)
            (root / rows[0]["values"]["path"]).write_bytes(b"\xff" * 128)
            with self.assertRaises(ContractError):
                validate_bundle(rows, evidence_root=root)

    def test_final_world_hash_binds_real_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            rows=self._constructed_native_byte_fixture(root)
            rows[-1]["final_world"]["world_material_sha256"]="0" * 64
            with self.assertRaises(ContractError):
                validate_bundle(rows, evidence_root=root)

    def test_native_source_schema_sha_must_match(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            rows=self._constructed_native_byte_fixture(root)
            rows[-1]["interface_schema_sha256"]="f" * 64
            with self.assertRaises(ContractError):
                validate_bundle(rows, evidence_root=root)

    def test_unverified_partial_material_write_rejected(self):
        def change(rows):
            action=rows[2]
            action["execution_status"]="partial"
            action["mode"]="add_material"
            action["brush_state"]="add"
            action["operation_sha256"]="a" * 64
            # No real receipt, no post-state, and synthetic provenance.
        self.reject(change)

    def test_reversal_requires_exact_source_action_post_state(self):
        def change(rows):
            action=rows[2]
            action["execution_status"]="committed"
            action["world_postcondition"]=copy.deepcopy(action["world_precondition"])
            rows[3]["before_world"]["revision"] += 1
            rows[3]["after_world"]["revision"] += 1
        self.reject(change)


if __name__ == "__main__":
    unittest.main()
