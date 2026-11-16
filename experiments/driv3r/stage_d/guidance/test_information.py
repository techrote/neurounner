"""Local synthetic information-accounting regression tests; no external runtime."""
import copy
import struct
import unittest

from experiments.driv3r.stage_d.guidance.information import Channel, make_shadow_arms


def channel(semantic="positive_scalar/v1", values=None, **overrides):
    kind = {"positive_scalar/v1": ("preference", "u16le"),
            "soft_attraction/v1": ("preference", "u16le"),
            "soft_avoidance/v1": ("preference", "u16le"),
            "signed_preference/v1": ("preference", "i16le"),
            "signed_distance/v1": ("geometry", "i16le"),
            "flow_vector_xy/v1": ("geometry", "vec2_i16le")}[semantic]
    if values is None:
        values = (0, 65535) if kind[1] == "u16le" else ((0, 0, 0, 0) if kind[1] == "vec2_i16le" else (0, 100))
    fmt = "H" if kind[1] == "u16le" else "h"
    attrs = dict(semantic_id=semantic, signal_class=kind[0], encoding=kind[1],
                 x=2, y=3, width=2, height=1,
                 samples=struct.pack("<" + fmt * len(values), *values),
                 producer_revision="a" * 40, config_sha256="b" * 64, seed=570)
    attrs.update(overrides)
    return Channel(**attrs)


def candidate(cid, x):
    return {"id": cid, "destination_xy": [x, 3],
            "swept_xy": [[x, 3]], "native_limit_cells": 4}


def scenario():
    return dict(snapshot_id="snapshot:frozen:1", world_sha256="e" * 64,
                source_revision="f" * 40,
                geometry={"clearance_cells": [6, 7], "selected_wall_id": "w1"},
                legal_candidates=[candidate("L", 2), candidate("R", 3)],
                channels=[channel(), channel("soft_attraction/v1", (100, 200)),
                          channel("signed_distance/v1", (-3, 4)),
                          channel("flow_vector_xy/v1", (2, -2, 0, 1))])


class InformationTests(unittest.TestCase):
    def test_true_unsigned_signed_and_vector_values(self):
        parts = scenario()["channels"]
        self.assertEqual(parts[0].descriptor(with_values=True)["values"], [0, 65535])
        self.assertEqual(parts[2].descriptor(with_values=True)["values"], [-3, 4])
        self.assertEqual(parts[3].descriptor(with_values=True)["values"], [[2, -2], [0, 1]])

    def test_matched_legal_geometry_and_pinned_world(self):
        output = make_shadow_arms(**scenario())
        views = [row["model_view"] for row in output.values()]
        self.assertEqual(len({v["legal_candidate_set_sha256"] for v in views}), 1)
        self.assertEqual(len({v["world_sha256"] for v in views}), 1)
        self.assertTrue(all(v["host_geometry"] == views[0]["host_geometry"] for v in views))
        self.assertTrue(all(v["evidence_class"] == "synthetic_pre_m1" for v in views))
        self.assertTrue(all(not row["audit_only"]["native_material_authority"] for row in output.values()))

    def test_strict_information_isolation_no_hidden_hash_leakage(self):
        output = make_shadow_arms(**scenario())
        self.assertEqual([c["semantic_id"] for c in output["A1"]["model_view"]["guidance"]],
                         ["positive_scalar/v1"])
        self.assertEqual([c["semantic_id"] for c in output["A2"]["model_view"]["guidance"]],
                         ["positive_scalar/v1", "soft_attraction/v1"])
        self.assertEqual([c["semantic_id"] for c in output["A3"]["model_view"]["guidance"]],
                         ["positive_scalar/v1", "signed_distance/v1", "flow_vector_xy/v1"])
        self.assertEqual(len(output["A1"]["audit_only"]["ignored_channels"]), 3)
        self.assertNotIn(parts := scenario()["channels"][1].descriptor(with_values=False)["sample_sha256"],
                         str(output["A1"]["model_view"]))
        self.assertIn(parts, str(output["A1"]["audit_only"]))

    def test_display_and_swept_order_do_not_change_host_catalog_identity(self):
        q = scenario()
        first = make_shadow_arms(**q)
        q["legal_candidates"].reverse()
        second = make_shadow_arms(**q)
        self.assertEqual(first["A1"]["audit_only"]["legal_candidate_set_sha256"],
                         second["A1"]["audit_only"]["legal_candidate_set_sha256"])
        self.assertNotEqual(first["A1"]["audit_only"]["model_view_sha256"],
                            second["A1"]["audit_only"]["model_view_sha256"])
        q["legal_candidates"][0]["destination_xy"][0] = 5
        third = make_shadow_arms(**q)
        self.assertNotEqual(first["A1"]["audit_only"]["legal_candidate_set_sha256"],
                            third["A1"]["audit_only"]["legal_candidate_set_sha256"])

    def test_input_isolation_and_hash_propagation(self):
        q = scenario()
        original = copy.deepcopy(q)
        out = make_shadow_arms(**q)
        self.assertEqual(original, q)
        q["geometry"]["clearance_cells"].append(-1)
        self.assertEqual(out["A1"]["model_view"]["host_geometry"]["clearance_cells"], [6, 7])
        q = scenario()
        result = make_shadow_arms(**q)
        q["channels"][0] = channel(values=(1, 65535))
        changed = make_shadow_arms(**q)
        self.assertNotEqual(result["A1"]["audit_only"]["model_view_sha256"],
                            changed["A1"]["audit_only"]["model_view_sha256"])

    def test_mutating_one_model_view_cannot_contaminate_other_frozen_arms(self):
        # Previously all three model views referenced the SAME catalog and
        # host_geometry dictionaries, defeating independent control custody.
        q = scenario()
        original = copy.deepcopy(q)
        output = make_shadow_arms(**q)
        a1 = output["A1"]["model_view"]
        a2 = output["A2"]["model_view"]
        a3 = output["A3"]["model_view"]
        before_a2 = copy.deepcopy(a2)
        before_a3 = copy.deepcopy(a3)
        a1["host_geometry"]["clearance_cells"][0] = 12345
        a1["legal_candidates"][0]["swept_xy"][0][0] = 9876
        a1["legal_candidates"][1]["destination_xy"][0] = 9876
        self.assertEqual(a2, before_a2)
        self.assertEqual(a3, before_a3)
        self.assertEqual(q, original)
        from experiments.driv3r.stage_d.guidance.information import _hashable
        self.assertEqual(output["A2"]["audit_only"]["model_view_sha256"], _hashable(a2))
        self.assertEqual(output["A3"]["audit_only"]["model_view_sha256"], _hashable(a3))
        self.assertNotEqual(output["A1"]["audit_only"]["model_view_sha256"], _hashable(a1))

    def test_reject_unregistered_and_mismatched_semantics(self):
        with self.assertRaisesRegex(ValueError, "unregistered"):
            channel(semantic_id="unregistered/new")
        with self.assertRaisesRegex(ValueError, "disagreement"):
            channel("signed_distance/v1", signal_class="preference")

    def test_reject_wrong_byte_length_and_bounds(self):
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            channel(samples=b"\x00")
        with self.assertRaisesRegex(ValueError, "bounded"):
            channel(width=4097, height=1)
        with self.assertRaisesRegex(ValueError, "origin"):
            channel(x=-1)
        with self.assertRaisesRegex(ValueError, "seed"):
            channel(seed=True)

    def test_reject_invalid_candidates_and_scores_in_legal_catalog(self):
        q = scenario()
        q["legal_candidates"][0]["guidance_score"] = 0.9
        with self.assertRaisesRegex(ValueError, "unsupported"):
            make_shadow_arms(**q)
        q = scenario()
        q["legal_candidates"][1]["id"] = "L"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            make_shadow_arms(**q)
        q = scenario()
        q["legal_candidates"][0]["swept_xy"] = [[2, 3]] * 2
        with self.assertRaisesRegex(ValueError, "repeated"):
            make_shadow_arms(**q)

    def test_reject_forged_frame_and_many_optional_channels(self):
        q = scenario()
        q["channels"][1] = channel("soft_attraction/v1", x=99)
        with self.assertRaisesRegex(ValueError, "identical tile"):
            make_shadow_arms(**q)
        q = scenario()
        q["channels"] = [channel("soft_attraction/v1"), channel()]
        with self.assertRaisesRegex(ValueError, "mandatory"):
            make_shadow_arms(**q)
        q = scenario()
        q["channels"].append(channel("soft_attraction/v1"))
        with self.assertRaisesRegex(ValueError, "unique"):
            make_shadow_arms(**q)

    def test_reject_unsafe_provenance_and_nan_geometry(self):
        q = scenario()
        q["source_revision"] = "unverified"
        with self.assertRaisesRegex(ValueError, "source_revision"):
            make_shadow_arms(**q)
        q = scenario()
        q["geometry"]["clearance_cells"] = [float("nan")]
        with self.assertRaises(ValueError):
            make_shadow_arms(**q)
        with self.assertRaisesRegex(ValueError, "config_sha256"):
            channel(config_sha256="invalid")


if __name__ == "__main__":
    unittest.main()
