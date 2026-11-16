"""M1 mapping tests only: synthetic source, no native execution or inference."""
import copy
import hashlib
import struct
import unittest

from experiments.driv3r.stage_d.guidance.m1_compat import (M1_COMMIT, M1_SCHEMA_SHA256, PRESETS, project_m1_guidance)
from experiments.driv3r.stage_d.guidance.information import Channel, make_shadow_arms


def desc(semantic, values):
    kind, fmt = (("preference", "i16le" if semantic == "signed_preference/v1" else "u16le") if semantic in
                 ("positive_scalar/v1", "soft_attraction/v1", "soft_avoidance/v1", "signed_preference/v1") else
                 ("geometry", "vec2_i16le" if semantic == "flow_vector_xy/v1" else "i16le"))
    flattened = [n for p in values for n in p] if fmt == "vec2_i16le" else values
    raw = struct.pack("<" + ("H" if fmt == "u16le" else "h") * len(flattened), *flattened)
    return {"semantic_id": semantic, "signal_class": kind, "encoding": fmt,
            "frame": [2, 3, 2, 1], "sample_sha256": hashlib.sha256(raw).hexdigest(),
            "sample_bytes": len(raw), "producer_revision": "a" * 40,
            "config_sha256": "b" * 64, "seed": 570, "values": values}


def inputs():
    all_channels = [desc("positive_scalar/v1", [0, 65535]),
                    desc("soft_attraction/v1", [100, 200]),
                    desc("soft_avoidance/v1", [800, 700]),
                    desc("signed_preference/v1", [-100, 20]),
                    desc("signed_distance/v1", [-3, 4]),
                    desc("flow_vector_xy/v1", [[1, -2], [0, 1]])]
    channels = []
    for d in all_channels:
        flattened = ([n for xy in d["values"] for n in xy]
                     if d["encoding"] == "vec2_i16le" else d["values"])
        fmt = "H" if d["encoding"] == "u16le" else "h"
        raw = struct.pack("<" + fmt * len(flattened), *flattened)
        channels.append(Channel(
            semantic_id=d["semantic_id"], signal_class=d["signal_class"],
            encoding=d["encoding"], x=d["frame"][0], y=d["frame"][1],
            width=d["frame"][2], height=d["frame"][3], samples=raw,
            producer_revision=d["producer_revision"],
            config_sha256=d["config_sha256"], seed=d["seed"]))
    # Use the REAL lane producer, including the immutable model-view audit SHA
    # and canonical host-candidate catalog, rather than a hand-fabricated stub.
    return make_shadow_arms(
        snapshot_id="s0", world_sha256="e" * 64, source_revision="a" * 40,
        legal_candidates=[
            {"id": "L", "destination_xy": [3, 3], "swept_xy": [[3, 3]], "native_limit_cells": 4},
            {"id": "R", "destination_xy": [4, 3], "swept_xy": [[4, 3]], "native_limit_cells": 4},
        ], geometry={"clearance_cells": 3}, channels=channels)


def build(arms=None, **changes):
    kw = dict(canvas_size=(32, 32), record_id="guidance1", episode_id="episode1",
              sequence=0, label="offline synthetic D1B", asset_root="signals/d1b",
              speckle={"seed": 570, "radius_cells": 5, "centre_density": .75,
                       "hardness": .3, "gradient_length": .8},
              exit_below=2000, reenter_at_or_above=5000,
              reference_ids={"signed_distance/v1": "boulder-edge-01"})
    kw.update(changes)
    return project_m1_guidance(inputs() if arms is None else arms, **kw)


class M1CompatTests(unittest.TestCase):
    def test_exact_frozen_identity_and_base(self):
        out = build()
        self.assertEqual(out["m1_source_commit"], M1_COMMIT)
        self.assertEqual(out["m1_schema_sha256"], M1_SCHEMA_SHA256)
        base = out["guidance"]
        self.assertEqual(base["protocol"], "cybersand.driv3r.d-common/1")
        self.assertEqual(base["values"]["bytes"], 4)
        self.assertIsNone(base["display"])
        self.assertFalse(base["native_material_authority"])
        for row in [base["values"]] + [c["values"] for c in base["optional_channels"]]:
            data = out["artifacts"][row["path"]]
            self.assertEqual(row["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(row["bytes"], len(data))

    def test_all_builtin_sign_units_and_rational_scale(self):
        channels = {c["semantic_id"]: c for c in build()["guidance"]["optional_channels"]}
        self.assertEqual(set(channels), set(PRESETS))
        for key, preset in PRESETS.items():
            row = channels[key]
            self.assertEqual(tuple(row[k] for k in ("signal_class", "encoding", "units")), preset[:3])
            self.assertEqual((row["scale"]["numerator"], row["scale"]["denominator"]), preset[3:5])
            self.assertEqual(tuple(row[k] for k in ("zero_meaning", "positive_meaning", "negative_meaning")), preset[5:8])
            self.assertFalse(row["native_material_authority"])
        self.assertEqual(channels["signed_distance/v1"]["reference_id"], "boulder-edge-01")

    def test_exact_information_omission_ledger(self):
        out = build()
        records = out["observation_channel_handling"]
        for arm, count in (("A1", 0), ("A2", 3), ("A3", 2)):
            h = records[arm]
            self.assertEqual(len(h["used_channel_ids"]), count)
            self.assertEqual(len(h["ignored_channels"]), 5 - count)
            self.assertEqual(h["unsupported_policy"], "ignore_with_audit")
            self.assertEqual(len(set(h["used_channel_ids"])), count)
            self.assertTrue(set(h["used_channel_ids"]).isdisjoint(
                r["channel_id"] for r in h["ignored_channels"]))
            self.assertEqual(set(h["used_channel_ids"]) |
                             {r["channel_id"] for r in h["ignored_channels"]},
                             {c["channel_id"] for c in out["guidance"]["optional_channels"]})

    def test_numeric_artifacts_are_canonical_not_stipple_png(self):
        out = build()
        self.assertEqual(struct.unpack("<hh", out["artifacts"][
            "signals/d1b/signed_distance_v1.i16le"]), (-3, 4))
        self.assertEqual(struct.unpack("<hhhh", out["artifacts"][
            "signals/d1b/flow_vector_xy_v1.vec2_i16le"]), (1, -2, 0, 1))
        self.assertEqual(len(out["artifacts"]), 6)

    def test_rejects_mismatched_frozen_state(self):
        arms = inputs()
        arms["A3"]["model_view"]["world_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "different frozen"):
            build(arms)

    def test_rejects_audit_leak_or_modified_samples(self):
        arms = inputs()
        arms["A1"]["model_view"]["guidance"].append(
            copy.deepcopy(arms["A2"]["model_view"]["guidance"][1]))
        with self.assertRaisesRegex(ValueError, "disagree"):
            build(arms)
        arms = inputs()
        arms["A2"]["model_view"]["guidance"][1]["values"] = [1, 2]
        with self.assertRaisesRegex(ValueError, "bytes/hash mismatch"):
            build(arms)

    def test_rejects_unaccounted_channel(self):
        arms = inputs()
        arms["A1"]["audit_only"]["ignored_channels"].pop()
        with self.assertRaisesRegex(ValueError, "missing, extra"):
            build(arms)

    def test_rejects_wrong_semantic_role_and_no_distance_reference(self):
        with self.assertRaisesRegex(ValueError, "named boundary"):
            build(reference_ids={})
        arms = inputs()
        for arm in arms:
            for part in ("used_channels", "ignored_channels"):
                for c in arms[arm]["audit_only"][part]:
                    if c["semantic_id"] == "signed_distance/v1":
                        c["signal_class"] = "preference"
            for c in arms[arm]["model_view"]["guidance"]:
                if c["semantic_id"] == "signed_distance/v1":
                    c["signal_class"] = "preference"
        with self.assertRaisesRegex(ValueError, "arm consumed wrong|type mismatch"):
            build(arms)

    def test_rejects_unsafe_artifact_paths_and_world_bounds(self):
        with self.assertRaisesRegex(ValueError, "unsafe"):
            build(asset_root="../escape")
        with self.assertRaisesRegex(ValueError, "outside accepted World"):
            build(canvas_size=(3, 4))

    def test_rejects_tampered_core_in_early_arm(self):
        arms = inputs()
        # A3 retains a good copy; do not let it conceal corruption of A1.
        arms["A1"]["model_view"]["guidance"][0]["values"] = [1, 65535]
        with self.assertRaisesRegex(ValueError, "bytes/hash mismatch"):
            build(arms)

    def test_rejects_tampered_core_in_middle_arm(self):
        arms = inputs()
        arms["A2"]["model_view"]["guidance"][0]["values"] = [4, 65535]
        with self.assertRaisesRegex(ValueError, "bytes/hash mismatch"):
            build(arms)

    def test_rejects_tampered_model_view_digest(self):
        arms = inputs()
        arms["A1"]["audit_only"]["model_view_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "model-view provenance hash mismatch"):
            build(arms)

    def test_rejects_changed_candidate_footprint_even_if_all_arms_match(self):
        arms = inputs()
        for pair in arms.values():
            pair["model_view"]["legal_candidates"][0]["destination_xy"] = [31, 31]
        with self.assertRaisesRegex(ValueError, "source legal candidate set hash mismatch"):
            build(arms)

    def test_rejects_candidate_ledger_drift(self):
        arms = inputs()
        arms["A2"]["audit_only"]["legal_candidate_set_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "audit legal candidate set hash mismatch"):
            build(arms)

    def test_rejects_undeclared_source_audit_semantics(self):
        arms = inputs()
        arms["A3"]["audit_only"]["unregistered_channels_used"] = ["unknown/v1"]
        with self.assertRaisesRegex(ValueError, "native/mismatched"):
            build(arms)

    def test_refuses_forged_native_claims(self):
        arms = inputs()
        arms["A2"]["audit_only"]["native_execution_verified"] = True
        with self.assertRaisesRegex(ValueError, "native/mismatched"):
            build(arms)


if __name__ == "__main__":
    unittest.main()
