"""Adversarial synthetic D-1B host-geometry value-shape tests (stdlib only)."""
import copy
import unittest

from experiments.driv3r.stage_d.guidance.geometry_guard import validate_host_geometry


class GeometryGuardTests(unittest.TestCase):
    def test_legitimate_frozen_geometry_and_copy(self):
        supplied = {"clearance_cells": [6, 7], "selected_wall_id": "wall:1",
                    "target_xy": [42, 7], "cursor_xy": [2, 3],
                    "signed_distance_cells": -4, "obstacle_count": 5,
                    "target_progress_cells": -1,
                    "blocked_mask_sha256": "a" * 64,
                    "visited_mask_sha256": "b" * 64}
        original = copy.deepcopy(supplied)
        validated = validate_host_geometry(supplied)
        self.assertEqual(validated, original)
        supplied["clearance_cells"][0] = 999
        self.assertEqual(validated["clearance_cells"], [6, 7])
        self.assertEqual(validate_host_geometry({"clearance_cells": 3}), {"clearance_cells": 3})
        self.assertEqual(validate_host_geometry({}), {})

    def test_rejects_hidden_signal_in_whitelisted_clearance(self):
        for injection in ("KEEP", {"shadow_verdict": "UNDO"},
                          [1, "secret"], [1, {"guidance": [3]}],
                          list(range(33)), [], [True],
                          [float("nan")], [-1], [2**31]):
            with self.subTest(injection=str(injection)[:60]):
                with self.assertRaisesRegex(ValueError, "clearance_cells"):
                    validate_host_geometry({"clearance_cells": injection})

    def test_rejects_boolean_and_out_of_bounds_geometry(self):
        for key, invalid in (("obstacle_count", True),
                             ("obstacle_count", -1),
                             ("signed_distance_cells", True),
                             ("target_progress_cells", float("inf")),
                             ("target_xy", [1, False]),
                             ("cursor_xy", [-1, 2]),
                             ("cursor_xy", [2, 3, 4])):
            with self.subTest(key=key, invalid=invalid):
                with self.assertRaisesRegex(ValueError, "invalid"):
                    validate_host_geometry({key: invalid})

    def test_rejects_covert_payload_as_identifier_or_hash(self):
        for key, invalid in (("selected_wall_id", "wall 1: shadow=UNDO"),
                             ("selected_wall_id", "x" * 97),
                             ("blocked_mask_sha256", "a" * 63),
                             ("visited_mask_sha256", "A" * 64),
                             ("blocked_mask_sha256", {"digest": "a" * 64})):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "invalid"):
                    validate_host_geometry({key: invalid})

    def test_rejects_unregistered_key_and_nonmapping(self):
        for bad in ({"critic_hint": "KEEP"}, {42: 4}, [], None):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "unregistered"):
                    validate_host_geometry(bad)


if __name__ == "__main__":
    unittest.main()
