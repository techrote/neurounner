"""A1/A2/A3 optional-channel contract tests; synthetic shapes, not actual model runs."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from jsonschema.exceptions import ValidationError

from verify import (ContractError, SCHEMA, compare_initial_matched_actions,
                    records_from_examples, validate_bundle)


ZERO = "0" * 64
CHANNELS = {
    "soft_attraction/v1": ("preference","u16le","normalized_0_1",1,65535,
                           "no_extra_attraction","greater_attraction",None,None),
    "soft_avoidance/v1": ("preference","u16le","normalized_0_1",1,65535,
                          "no_soft_avoidance","greater_soft_avoidance",None,None),
    "signed_preference/v1": ("preference","i16le","normalized_signed",1,32767,
                             "neutral","attraction","repulsion",None),
    "signed_distance/v1": ("geometry_metric","i16le","cells",1,1,
                           "on_reference_boundary","outside_reference","inside_reference","boulder-edge-01"),
    "flow_vector_xy/v1": ("geometry_metric","vec2_i16le","world_unit_vector",1,32767,
                           "no_direction","+x_right_+y_down","-x_left_-y_up",None),
}


def make_channel(semantic_id: str, channel_id: str) -> dict:
    kind, encoding, units, num, den, zero, pos, neg, reference = CHANNELS[semantic_id]
    return {
        "channel_id":channel_id,
        "signal_class":kind,
        "semantic_id":semantic_id,
        "encoding":encoding,
        "units":units,
        "scale":{"numerator":num,"denominator":den},
        "zero_meaning":zero,
        "positive_meaning":pos,
        "negative_meaning":neg,
        "reference_id":reference,
        "canvas_size":[512,512],
        "frame":[0,0,8,8],
        "values":{"path":"signals/" + channel_id + "." + encoding,
                  "sha256":hashlib.sha256(channel_id.encode()).hexdigest(),
                  "bytes":64*(4 if encoding=="vec2_i16le" else 2),
                  "format":encoding},
        "producer":{"kind":"derived_geometry" if kind=="geometry_metric" else "procedural",
                    "producer_id":"d-common-synthetic-generator",
                    "revision":"fixture-1",
                    "source_sha256":hashlib.sha256(("source:" + channel_id).encode()).hexdigest(),
                    "parameters_sha256":hashlib.sha256(("params:" + channel_id).encode()).hexdigest(),
                    "seed":17},
        "native_material_authority":False
    }


def arm(semantics: list[tuple[str, str]], *, candidate="CURVE_PAUSE") -> list[dict]:
    rows = copy.deepcopy(records_from_examples())
    channels = [make_channel(semantic, cid) for semantic, cid in semantics]
    rows[0]["optional_channels"] = channels
    rows[1]["guidance_channel_handling"] = {
        "unsupported_policy":"reject_unsupported",
        "used_channel_ids":[x["channel_id"] for x in channels],
        "ignored_channels":[]
    }
    rows[2]["candidate_id"] = candidate
    return rows


class OptionalGuidanceContractTests(unittest.TestCase):
    def reject(self, rows, alter):
        alter(rows)
        with self.assertRaises((ValidationError, ContractError)):
            validate_bundle(rows)

    def test_a1_mandatory_positive_core_without_extras(self):
        rows = arm([])
        self.assertEqual(rows[0]["encoding"], "u16le_higher_is_better_zero_is_neutral")
        self.assertEqual(rows[0]["values"]["bytes"], 128)
        self.assertEqual(validate_bundle(rows)["validated"], 5)

    def test_a2_independent_preference_and_flow_channels(self):
        rows = arm([("soft_attraction/v1","extra-attraction"),
                    ("soft_avoidance/v1","avoidance"),
                    ("flow_vector_xy/v1","steering")])
        self.assertEqual(validate_bundle(rows)["validated"], 5)

    def test_a3_soft_preference_plus_separate_geometric_metrics(self):
        rows = arm([("signed_distance/v1","boulder-distance"),
                    ("flow_vector_xy/v1","contour-flow")])
        self.assertEqual(validate_bundle(rows)["validated"], 5)
        self.assertEqual(set(x["signal_class"] for x in rows[0]["optional_channels"]),
                         {"geometry_metric"})

    def test_signed_preference_is_explicitly_typed(self):
        self.assertEqual(validate_bundle(arm([("signed_preference/v1","signed-score")]))["validated"], 5)

    def test_missing_channel_usage_rejected(self):
        rows = arm([("soft_avoidance/v1","avoidance")])
        self.reject(rows, lambda x: x[1]["guidance_channel_handling"]["used_channel_ids"].clear())

    def test_duplicate_channel_identifier_rejected(self):
        rows = arm([("soft_avoidance/v1","same"),("soft_attraction/v1","other")])
        self.reject(rows, lambda x: x[0]["optional_channels"][1].__setitem__("channel_id","same"))

    def test_duplicate_used_and_ignored_rejected(self):
        rows = arm([("soft_avoidance/v1","avoidance")])
        def bad(x):
            x[1]["guidance_channel_handling"]["unsupported_policy"]="ignore_with_audit"
            x[1]["guidance_channel_handling"]["ignored_channels"]=[{
                "channel_id":"avoidance","semantic_id":"soft_avoidance/v1",
                "values_sha256":x[0]["optional_channels"][0]["values"]["sha256"],
                "reason":"explicit_information_ablation"}]
        self.reject(rows,bad)

    def test_optional_artifact_encoding_must_match(self):
        rows=arm([("flow_vector_xy/v1","flow")])
        self.reject(rows,lambda x: x[0]["optional_channels"][0]["values"].__setitem__("format","i16le"))

    def test_vector_bytes_must_be_four_per_cell(self):
        rows=arm([("flow_vector_xy/v1","flow")])
        self.reject(rows,lambda x: x[0]["optional_channels"][0]["values"].__setitem__("bytes",128))

    def test_signed_distance_requires_named_reference(self):
        rows=arm([("signed_distance/v1","rock")])
        self.reject(rows,lambda x: x[0]["optional_channels"][0].__setitem__("reference_id",None))

    def test_semantic_role_must_match_builtin(self):
        rows=arm([("soft_avoidance/v1","avoidance")])
        self.reject(rows,lambda x: x[0]["optional_channels"][0].__setitem__("signal_class","geometry_metric"))

    def test_semantic_scale_must_match_builtin(self):
        rows=arm([("signed_distance/v1","distance")])
        self.reject(rows,lambda x: x[0]["optional_channels"][0]["scale"].__setitem__("denominator",32767))

    def test_semantic_sign_must_match_builtin(self):
        rows=arm([("signed_distance/v1","distance")])
        self.reject(rows,lambda x: x[0]["optional_channels"][0].__setitem__("positive_meaning","inside_reference"))

    def test_source_provenance_is_required_per_channel(self):
        rows=arm([("soft_avoidance/v1","avoidance")])
        self.reject(rows,lambda x: x[0]["optional_channels"][0].pop("producer"))

    def test_channel_data_does_not_authorize_native_material(self):
        rows=arm([("soft_avoidance/v1","avoidance")])
        self.reject(rows,lambda x: x[0]["optional_channels"][0].__setitem__("native_material_authority",True))

    def test_channel_world_dimensions_must_match_base(self):
        rows=arm([("soft_avoidance/v1","avoidance")])
        self.reject(rows,lambda x: x[0]["optional_channels"][0].__setitem__("canvas_size",[640,512]))

    def test_channel_fits_its_own_canvas(self):
        rows=arm([("soft_avoidance/v1","avoidance")])
        self.reject(rows,lambda x: x[0]["optional_channels"][0].__setitem__("frame",[510,0,8,8]))

    def test_unknown_semantics_not_silently_consumed(self):
        rows=arm([("soft_avoidance/v1","experimental-signal")])
        rows[0]["optional_channels"][0]["semantic_id"]="experimental.unregistered/v1"
        with self.assertRaises(ContractError):
            validate_bundle(rows)

    def test_unknown_semantics_can_be_explicitly_ignored_with_receipt(self):
        rows=arm([("soft_avoidance/v1","experimental-signal")])
        signal=rows[0]["optional_channels"][0]
        signal["semantic_id"]="experimental.unregistered/v1"
        rows[1]["guidance_channel_handling"]={
            "unsupported_policy":"ignore_with_audit","used_channel_ids":[],
            "ignored_channels":[{"channel_id":signal["channel_id"],
                                 "semantic_id":signal["semantic_id"],
                                 "values_sha256":signal["values"]["sha256"],
                                 "reason":"unsupported_semantics"}]
        }
        self.assertEqual(validate_bundle(rows)["validated"], 5)

    def test_ignored_hash_cannot_differ_from_signal(self):
        rows=arm([("soft_avoidance/v1","avoidance")])
        rows[1]["guidance_channel_handling"]={
            "unsupported_policy":"ignore_with_audit","used_channel_ids":[],
            "ignored_channels":[{"channel_id":"avoidance",
                                 "semantic_id":"soft_avoidance/v1",
                                 "values_sha256":ZERO,
                                 "reason":"explicit_information_ablation"}]}
        with self.assertRaises(ContractError):
            validate_bundle(rows)

    def test_ignore_requires_explicit_policy(self):
        rows=arm([("soft_avoidance/v1","avoidance")])
        signal=rows[0]["optional_channels"][0]
        rows[1]["guidance_channel_handling"]={
            "unsupported_policy":"reject_unsupported","used_channel_ids":[],
            "ignored_channels":[{"channel_id":"avoidance","semantic_id":signal["semantic_id"],
                                 "values_sha256":signal["values"]["sha256"],
                                 "reason":"explicit_information_ablation"}]}
        with self.assertRaises(ContractError):
            validate_bundle(rows)

    def test_registered_experimental_semantic_is_admitted(self):
        rows=arm([("soft_avoidance/v1","experimental-signal")])
        rows[0]["optional_channels"][0]["semantic_id"]="experimental.new_steering/v1"
        allowed=set(CHANNELS)|{"experimental.new_steering/v1"}
        self.assertEqual(validate_bundle(rows,supported_semantics=allowed)["validated"], 5)

    def test_extra_channels_maintain_a1_core_semantics(self):
        for rows in (arm([]),arm([("soft_avoidance/v1","avoidance")]),
                     arm([("signed_distance/v1","distance")])):
            self.assertEqual(rows[0]["purpose"],"soft_positive_preference")
            self.assertEqual(rows[0]["encoding"],"u16le_higher_is_better_zero_is_neutral")
            self.assertEqual(rows[0]["values"]["format"],"u16le")
            self.assertEqual(validate_bundle(rows)["validated"],5)

    def test_matched_arms_same_host_legal_candidates_at_initial_state(self):
        arms={
            "A1":[arm([])[2]],
            "A2":[arm([("soft_avoidance/v1","avoidance")],candidate="RIGHT")[2]],
            "A3":[arm([("signed_distance/v1","distance")],candidate="LEFT")[2]]
        }
        result=compare_initial_matched_actions(arms)
        self.assertEqual(result["first_choice_divergence_index"],0)
        self.assertTrue(result["legal_candidate_catalog_equal_until_divergence"])
        self.assertTrue(result["requires_native_effects_to_identify_consequential_divergence"])

    def test_different_prompt_presentation_hash_is_not_legal_set_divergence(self):
        left=arm([])[2]
        right=arm([("signed_distance/v1","distance")])[2]
        right["candidate_set_sha256"]="f"*64
        result=compare_initial_matched_actions({"A1":[left],"A3":[right]})
        self.assertIsNone(result["first_choice_divergence_index"])

    def test_different_native_legal_candidate_set_rejected(self):
        left=arm([])[2]
        right=arm([("signed_distance/v1","distance")])[2]
        right["legal_candidate_set_sha256"]="f"*64
        with self.assertRaises(ContractError):
            compare_initial_matched_actions({"A1":[left],"A3":[right]})

    def test_independent_world_ids_are_not_false_divergence(self):
        left=arm([])[2]
        right=arm([("signed_distance/v1","distance")])[2]
        right["world_precondition"]["id"]="independent-other-native-session"
        result=compare_initial_matched_actions({"A1":[left],"A3":[right]})
        self.assertIsNone(result["first_choice_divergence_index"])

    def test_diverged_native_material_cannot_be_treated_as_matched(self):
        left=arm([])[2]
        right=arm([("soft_avoidance/v1","avoidance")])[2]
        right["world_precondition"]["world_material_sha256"]="f"*64
        with self.assertRaises(ContractError):
            compare_initial_matched_actions({"A1":[left],"A2":[right]})

    def test_optional_channel_bytes_are_in_native_custody_inventory(self):
        from test_interface import ContractV1Tests
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            base_fixture = ContractV1Tests()
            base_fixture.setUp()
            rows=base_fixture._constructed_native_byte_fixture(root)
            signal=make_channel("signed_distance/v1","distance")
            blob=signal["values"]
            raw=b"\x00"*blob["bytes"]
            file=root/blob["path"]
            file.parent.mkdir(parents=True,exist_ok=True)
            file.write_bytes(raw)
            blob["sha256"]=hashlib.sha256(raw).hexdigest()
            rows[0]["optional_channels"]=[signal]
            rows[1]["guidance_channel_handling"]["used_channel_ids"]=["distance"]
            self.assertTrue(validate_bundle(rows,evidence_root=root)["evidence_bytes_verified"])
            file.write_bytes(b"\xff"*len(raw))
            with self.assertRaises(ContractError):
                validate_bundle(rows,evidence_root=root)


if __name__ == "__main__":
    unittest.main()
