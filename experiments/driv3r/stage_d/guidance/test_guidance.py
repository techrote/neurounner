"""Cheap, isolated stdlib tests for experimental guidance; no native Worlds."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from experiments.driv3r.stage_d.guidance.field import (
    CONVERSION, GuidanceField, Hysteresis, decode_png8, export_bundle,
    import_png8, load_bundle, radial_airbrush, rgba_png, speckle_png,
)
from experiments.driv3r.stage_d.guidance.study import (
    CriticSnapshot, factorial_arms, matched_arms, shadow_verdict, stable_hash,
)


class FieldTests(unittest.TestCase):
    def test_u16le_order_and_bounds(self):
        f = GuidanceField(2, 2, (0, 1, 0x1234, 65535), 7, 9)
        self.assertEqual(f.raw(), bytes.fromhex("000001003412ffff"))
        self.assertEqual(f.at(8, 10), 65535)
        self.assertEqual(GuidanceField.from_raw(2, 2, f.raw(), 7, 9), f)
        with self.assertRaises(IndexError):
            f.at(6, 9)
        with self.assertRaises(ValueError):
            GuidanceField(1, 1, (65536,))
        with self.assertRaises(ValueError):
            GuidanceField.from_raw(2, 2, b"\0\0")

    def test_airbrush_gradient_and_determinism(self):
        f = radial_airbrush(11, 11, center_x=5.5, center_y=5.5,
                            radius_cells=5, hardness=0.2, gradient_length=0.9)
        self.assertEqual(f.at(5, 5), 65535)
        self.assertEqual(f.at(0, 0), 0)
        self.assertLess(f.at(1, 5), f.at(3, 5))
        self.assertEqual(f, radial_airbrush(11, 11, center_x=5.5, center_y=5.5,
                                            radius_cells=5, hardness=0.2, gradient_length=0.9))
        with self.assertRaises(ValueError):
            radial_airbrush(1, 1, center_x=0, center_y=0, radius_cells=0)

    def test_scalar_tuple_and_world_frame_cannot_alias_or_overflow(self):
        values = [0, 65535]
        with self.assertRaisesRegex(ValueError, "immutable tuple"):
            GuidanceField(2, 1, values)
        with self.assertRaisesRegex(ValueError, "frame exceeds"):
            GuidanceField(2, 1, (0, 65535), x=2**31 - 1)
        canonical = GuidanceField.from_raw(2, 1, bytes.fromhex("0000ffff"))
        self.assertEqual(canonical.values, (0, 65535))
        values[0] = 123
        self.assertEqual(canonical.values, (0, 65535))

    def test_speckles_not_numeric_truth(self):
        f = GuidanceField(8, 1, (65535,) * 8)
        raw_hash = f.sha256
        self.assertEqual(speckle_png(f, seed=42, centre_density=.5),
                         speckle_png(f, seed=42, centre_density=.5))
        self.assertNotEqual(speckle_png(f, seed=42, centre_density=.5),
                            speckle_png(f, seed=43, centre_density=.5))
        self.assertEqual(f.sha256, raw_hash)
        w, h, px = decode_png8(speckle_png(f, seed=42, centre_density=.5))
        self.assertEqual((w, h), (8, 1))
        self.assertTrue(all(alpha in (0, 255) for alpha in px[3::4]))

    def test_png_import_rgba_and_manifest(self):
        source = rgba_png(3, 1, bytes((255,255,255,255, 255,255,255,0, 0,0,0,255)))
        f = import_png8(source, x=3, y=4)
        self.assertEqual(f.values, (65535, 0, 0))
        self.assertEqual(f.at(3, 4), 65535)
        self.assertEqual(CONVERSION, "png8_rec709_luma_times_alpha_round_u16_v1")

    def test_png_crc_rejection(self):
        png = bytearray(rgba_png(1, 1, bytes((0, 1, 2, 3))))
        png[-5] ^= 4
        with self.assertRaises(ValueError):
            decode_png8(bytes(png))

    def test_bundle_roundtrip_and_tampering(self):
        f = GuidanceField(2, 2, (12, 450, 65000, 0))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "guide"
            export_bundle(p, f, seed=9, centre_density=.4)
            restored, manifest = load_bundle(p)
            self.assertEqual(restored, f)
            self.assertEqual(manifest["values"]["sha256"], f.sha256)
            self.assertFalse(manifest["native_material_authority"])
            self.assertEqual(manifest["source_kind"], "manual_scalar")
            with p.with_suffix(".u16le").open("ab") as target:
                target.write(b"z")
            with self.assertRaises(ValueError):
                load_bundle(p)

    def test_bundle_preview_tampering(self):
        f = GuidanceField(1, 1, (65535,))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "guide"
            export_bundle(p, f, seed=0, centre_density=1)
            document = json.loads(p.with_suffix(".json").read_text())
            document["seed"] = 123
            p.with_suffix(".json").write_text(json.dumps(document))
            # Note: at density=1 all seeds yield same image; change density.
            document["centre_density"] = 0
            p.with_suffix(".json").write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                load_bundle(p)

    def test_external_manifest_pin_rejects_metadata_drift(self):
        field = GuidanceField(2, 1, (100, 65535))
        with tempfile.TemporaryDirectory() as d:
            prefix = Path(d) / "guide"
            export_bundle(prefix, field, seed=17, centre_density=.6,
                          authoring_parameters={"radius_cells": 3})
            manifest_path = prefix.with_suffix(".json")
            pinned = sha256(manifest_path.read_bytes()).hexdigest()
            restored, _ = load_bundle(prefix, expected_manifest_sha256=pinned)
            self.assertEqual(restored, field)
            meta = json.loads(manifest_path.read_text())
            meta["authoring_parameters"]["radius_cells"] = 999
            manifest_path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "manifest SHA-256 mismatch"):
                load_bundle(prefix, expected_manifest_sha256=pinned)
            with self.assertRaisesRegex(ValueError, "manifest SHA-256 mismatch"):
                load_bundle(prefix, expected_manifest_sha256="not-a-sha")

    def test_hysteresis_rejects_truthy_nonboolean_state(self):
        hysteresis = Hysteresis(100, 200)
        for state in (1, 0, "False", None):
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, "boolean"):
                hysteresis.assess(99, state)

    def test_hysteresis_neutral_not_forbidden(self):
        h = Hysteresis(100, 200)
        self.assertEqual(h.assess(99, True), (False, "propose_recovery"))
        self.assertEqual(h.assess(150, False), (False, "no_transition"))
        self.assertEqual(h.assess(200, False), (True, "reentered"))
        self.assertEqual(h.assess(150, True), (True, "no_transition"))
        with self.assertRaises(ValueError):
            Hysteresis(200, 200)


class StudyTests(unittest.TestCase):
    def test_matched_sources_and_ablations(self):
        source = {"clearance": 5}
        views = matched_arms(snapshot_id="world:2:17", candidate_ids=["up", "down"],
                             geometry=source, guidance={"field_sha": "a"*64},
                             critic={"last": "KEEP"})
        self.assertEqual(len(views), 4)
        self.assertEqual(len({v["candidate_set_sha256"] for v in views.values()}), 1)
        self.assertEqual(views["geometry_only"]["input_hashes"],
                         {"geometry": stable_hash({"clearance": 5})})
        self.assertEqual(set(views["guidance_only"]["input_hashes"]), {"guidance"})
        self.assertEqual(set(views["critic_only"]["input_hashes"]), {"critic"})
        self.assertEqual(set(views["combined"]["input_hashes"]),
                         {"geometry", "guidance", "critic"})
        self.assertIsNone(views["geometry_only"]["guidance"])
        self.assertIsNone(views["guidance_only"]["geometry"])
        self.assertIsNone(views["critic_only"]["geometry"])
        source["clearance"] = -1
        self.assertEqual(views["combined"]["geometry"]["clearance"], 5)
        self.assertEqual(stable_hash({"a": 1, "b": 2}), stable_hash({"b": 2, "a": 1}))

    def test_factorial_geometry_baseline(self):
        views = factorial_arms(snapshot_id="world:g:2", candidate_ids=["a", "b"],
                               geometry={"clearance": 4}, guidance={"preference": 2000},
                               critic={"shadow": "not_actuated"})
        self.assertEqual(len(views), 4)
        self.assertTrue(all(v["geometry"] == {"clearance": 4} for v in views.values()))
        self.assertIsNone(views["A_geometry"]["guidance"])
        self.assertIsNone(views["B_geometry_guidance"]["critic"])
        self.assertIsNone(views["C_geometry_critic"]["guidance"])
        self.assertIsNotNone(views["D_geometry_guidance_critic"]["critic"])
        self.assertEqual(len({v["candidate_set_sha256"] for v in views.values()}), 1)

    def test_withheld_guidance_and_critic_cannot_leak_through_input_hashes(self):
        common = dict(snapshot_id="world:3", candidate_ids=["left", "right"],
                      geometry={"clearance": 3},
                      guidance={"secret": "original"}, critic={"verdict": "KEEP"})
        original = matched_arms(**common)
        guided = matched_arms(**{**common, "guidance": {"secret": "changed"}})
        critiqued = matched_arms(**{**common, "critic": {"verdict": "UNDO"}})
        self.assertEqual(original["geometry_only"], guided["geometry_only"])
        self.assertEqual(original["geometry_only"], critiqued["geometry_only"])
        self.assertEqual(original["critic_only"], guided["critic_only"])
        self.assertEqual(original["guidance_only"], critiqued["guidance_only"])
        self.assertNotEqual(original["combined"], guided["combined"])
        self.assertNotEqual(original["combined"], critiqued["combined"])
        self.assertEqual(set(original["combined"]["input_hashes"]),
                         {"geometry", "guidance", "critic"})
        reordered = matched_arms(**{**common, "candidate_ids": ["right", "left"]})
        self.assertEqual(original["combined"]["candidate_set_sha256"],
                         reordered["combined"]["candidate_set_sha256"])
        # Candidate display order itself is retained; only identity is canonical.
        self.assertNotEqual(original["combined"]["candidate_ids"],
                            reordered["combined"]["candidate_ids"])

    def test_factorial_hidden_fingerprints_do_not_contaminate_controls(self):
        common = dict(snapshot_id="world:4", candidate_ids=["a", "b"],
                      geometry={"clearance": 7}, guidance={"source": "g0"},
                      critic={"source": "c0"})
        original = factorial_arms(**common)
        alternate_guide = factorial_arms(**{**common, "guidance": {"source": "g1"}})
        alternate_critic = factorial_arms(**{**common, "critic": {"source": "c1"}})
        self.assertEqual(original["A_geometry"], alternate_guide["A_geometry"])
        self.assertEqual(original["A_geometry"], alternate_critic["A_geometry"])
        self.assertEqual(original["C_geometry_critic"],
                         alternate_guide["C_geometry_critic"])
        self.assertEqual(original["B_geometry_guidance"],
                         alternate_critic["B_geometry_guidance"])
        self.assertNotEqual(original["D_geometry_guidance_critic"],
                            alternate_guide["D_geometry_guidance_critic"])
        self.assertNotEqual(original["D_geometry_guidance_critic"],
                            alternate_critic["D_geometry_guidance_critic"])
        self.assertEqual(set(original["A_geometry"]["input_hashes"]), {"geometry"})
        self.assertEqual(set(original["B_geometry_guidance"]["input_hashes"]),
                         {"geometry", "guidance"})
        self.assertEqual(set(original["C_geometry_critic"]["input_hashes"]),
                         {"geometry", "critic"})

    def test_serialized_shadow_cases_and_hysteresis(self):
        data = json.loads((Path(__file__).parent / "fixtures.json").read_text())
        self.assertEqual(data["status"], "SYNTHETIC_ONLY_NOT_NATIVE_OBSERVATIONS")
        params = data["hysteresis"]
        hysteresis = Hysteresis(params["exit_below"], params["reenter_at_or_above"])
        engaged = params["initial_engaged"]
        for step in params["sequence"]:
            engaged, event = hysteresis.assess(step["scalar"], engaged)
            self.assertEqual((engaged, event), (step["engaged"], step["event"]))
        for case in data["shadow_cases"]:
            verdict = shadow_verdict(before=CriticSnapshot(**case["before"]),
                                     after=CriticSnapshot(**case["after"]),
                                     undo_eligible=case["undo_eligible"],
                                     verified_native_receipt=case["verified_native_receipt"])
            self.assertEqual(verdict["decision"], case["expected"], case["id"])
            self.assertFalse(verdict["native_material_changed_by_critic"])

    def test_shadow_rejects_truthy_control_flags_and_untyped_metrics(self):
        before = CriticSnapshot("before", 10, 0, 2)
        after = CriticSnapshot("after", 9, 1, 2)
        for flag in ("false", 1, None):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "boolean"):
                shadow_verdict(before=before, after=after,
                               undo_eligible=flag, verified_native_receipt=True)
            with self.assertRaisesRegex(ValueError, "boolean"):
                shadow_verdict(before=before, after=after,
                               undo_eligible=True, verified_native_receipt=flag)
        for score in (True, "1", float("nan"), float("inf")):
            with self.subTest(score=score), self.assertRaisesRegex(ValueError, "invalid shadow"):
                CriticSnapshot("bad", score, 0, 0)
        for snapshot_id in (123, "", "x" * 97):
            with self.subTest(snapshot_id=snapshot_id), self.assertRaisesRegex(ValueError, "invalid shadow"):
                CriticSnapshot(snapshot_id, 1.0, 0, 0)
        with self.assertRaisesRegex(ValueError, "typed before/after"):
            shadow_verdict(before={"snapshot_id": "before"}, after=after,
                           undo_eligible=True, verified_native_receipt=True)

    def test_shadow_keep_undo_stop_without_mutation(self):
        before = CriticSnapshot("before", 10.0, 0, 2)
        better = CriticSnapshot("after", 11.0, 0, 3)
        worse = CriticSnapshot("worse", 9.0, 1, 2)
        keep = shadow_verdict(before=before, after=better, undo_eligible=True, verified_native_receipt=True)
        undo = shadow_verdict(before=before, after=worse, undo_eligible=True, verified_native_receipt=True)
        stop = shadow_verdict(before=before, after=worse, undo_eligible=False, verified_native_receipt=True)
        self.assertEqual([keep["decision"], undo["decision"], stop["decision"]], ["KEEP", "UNDO", "STOP"])
        self.assertTrue(all(not v["native_material_changed_by_critic"] for v in (keep, undo, stop)))
        self.assertTrue(all(v["execution"].startswith("shadow_only") for v in (keep, undo, stop)))
        with self.assertRaises(ValueError):
            shadow_verdict(before=before, after=before, undo_eligible=True, verified_native_receipt=True)


if __name__ == "__main__":
    unittest.main()
