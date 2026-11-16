"""Adversarial literal-composition controls for the prospective SMOL cohort."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curriculum_witness import inspect
from semantic_audit import graph_metrics, PIPE_REQUIREMENTS

CASES = {x["family"]: x for x in json.loads(
    Path(__file__).with_name("curriculum_proposals_v2.json").read_text())["proposals"]}


class CompositionWitnessTests(unittest.TestCase):
    def invalid(self, family, mutate):
        example = copy.deepcopy(CASES[family])
        mutate(example["program"])
        with self.assertRaisesRegex(ValueError, "composition witness"):
            inspect(example)

    def test_originals(self):
        for family in ("greebles", "pipework", "architecture"):
            with self.subTest(family=family):
                result = inspect(CASES[family])
                self.assertEqual(result["family"], family)

    def test_greeble_missing_module_gap(self):
        self.invalid("greebles", lambda p: p["components"][2]["box"].__setitem__(0, 54))

    def test_greeble_diagonal_not_perspective(self):
        self.invalid("greebles", lambda p: p["components"][2]["box"].__setitem__(1, 70))

    def test_greeble_repeated_modules(self):
        self.invalid("greebles", lambda p: p["components"][2].update(detail="vent"))

    def test_pipe_topology_rejects_three_endpoint_tee(self):
        self.invalid("pipework", lambda p: p["components"][0]["edges"].pop())

    def test_pipe_upward_split_rejects_inverted_tree(self):
        def invert(p):
            for point in p["components"][0]["nodes"]:
                point[1] = 95 - point[1]
        self.invalid("pipework", invert)

    def test_pipe_rejects_diameter_style_only_change(self):
        self.invalid("pipework", lambda p: p["components"][0].update(diameter=5))

    def test_p01_is_not_heldout_three_upward_riser_manifold(self):
        """P01 is a legitimate nested fan, not a mislabeled DEVELOPMENT example.

        The five endpoints and diameter agree, but an upper left/right/up fan
        is NOT three distinct risers rising from the lower horizontal trunk.
        Keep this explicit distinction when proposing future TRAIN labels.
        """
        observed = graph_metrics(CASES["pipework"]["program"])
        heldout = PIPE_REQUIREMENTS["pipework/three_branch_manifold/v1"]
        self.assertEqual(observed["endpoints"], 5)
        self.assertEqual(observed["diameter"], 3)
        self.assertEqual(observed["cycles"], 0)
        self.assertEqual(observed["connected_components"], 1)
        self.assertEqual(
            (observed["tees"], observed["crosses"], observed["upward_branches"]),
            (1, 1, 1),
        )
        self.assertEqual(
            (heldout["tees"], heldout["crosses"], heldout["upward_branches"]),
            (3, 0, 3),
        )
        self.assertFalse(all(observed[k] == v for k, v in heldout.items()))

    def test_architecture_door_alignment(self):
        self.invalid("architecture",
                     lambda p: p["components"][0]["openings"][2].update(offset=24))

    def test_architecture_roof_attachment(self):
        self.invalid("architecture", lambda p: p["components"][1]["box"].__setitem__(1, 11))

    def test_architecture_incorrect_route_clearance(self):
        self.invalid("architecture",
                     lambda p: p["requirements"]["route"][1].__setitem__(1, 40))


if __name__ == "__main__":
    unittest.main()
