#!/usr/bin/env python3
"""Adversarial, model-free controls for original split graph-neighbour review."""
import copy
import unittest

from visible_split_neighbours import canonical_tree, compare


def pipe(nodes, edges, *, diameter=3, fittings=False):
    return {"version": 2, "family": "pipework", "variation": 0,
            "components": [{"kind": "pipe_network", "nodes": nodes,
                            "edges": edges, "diameter": diameter,
                            "fittings": fittings}]}


class TopologyTests(unittest.TestCase):
    def setUp(self):
        self.nodes = [[12, 64], [50, 64], [82, 64], [50, 36],
                      [26, 36], [74, 36], [50, 14]]
        self.edges = [[0, 1], [1, 2], [1, 3], [3, 4], [3, 5], [3, 6]]
        self.original = pipe(self.nodes, self.edges)

    def test_label_permutation_preserves_topology(self):
        perm = [3, 6, 1, 4, 0, 5, 2]
        inv = {old: new for new, old in enumerate(perm)}
        alt = pipe([self.nodes[i] for i in perm],
                   [[inv[a], inv[b]] for a, b in reversed(self.edges)])
        self.assertEqual(canonical_tree(self.original), canonical_tree(alt))

    def test_geometry_seed_diameter_and_fittings_not_topology(self):
        alt = pipe([[x + 2, y - 2] for x, y in self.nodes],
                   list(reversed(self.edges)), diameter=5, fittings=True)
        self.assertEqual(canonical_tree(self.original), canonical_tree(alt))

    def test_different_degree_sequence_does_not_alias(self):
        # Relocate one edge so node3 ceases being degree4 and node2 becomes degree2.
        edges = [edge.copy() for edge in self.edges]
        edges[-1] = [2, 6]
        self.assertNotEqual(canonical_tree(self.original),
                            canonical_tree(pipe(self.nodes, edges)))

    def test_cycle_does_not_misclassify_as_tree(self):
        edges = self.edges + [[4, 5]]
        self.assertIsNone(canonical_tree(pipe(self.nodes, edges)))

    def test_disconnected_not_reported_as_tree(self):
        edges = [[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [5, 6]]
        self.assertIsNone(canonical_tree(pipe(self.nodes, edges)))

    def test_duplicate_edge_fails_closed(self):
        edges = self.edges[:]
        edges[0] = [1, 3]
        with self.assertRaises(ValueError):
            canonical_tree(pipe(self.nodes, edges))

    def test_bad_indices_fail_closed(self):
        with self.assertRaises(ValueError):
            canonical_tree(pipe(self.nodes, [[0, 7]] + self.edges[1:]))

    def test_non_pipe_family_unclassified(self):
        self.assertIsNone(canonical_tree({"family": "architecture", "components": []}))

    def test_flags_original_development_unlabeled_graph_alias(self):
        changed = copy.deepcopy(self.original)
        changed["components"][0]["diameter"] = 5
        result = compare([{"id": "P01", "family": "pipework", "program": self.original}],
                         {"TRAIN": [], "DEVELOPMENT": [{"program": changed}]})
        item = result["candidates"][0]
        self.assertTrue(item["same_unlabeled_tree_as_original_visible_split"]
                        ["DEVELOPMENT"])
        self.assertEqual(item["disposition"],
                         "QUARANTINE_POTENTIAL_VISIBLE_SPLIT_ALIAS")
        self.assertEqual(result["admitted_train_structures"], 0)
        self.assertFalse(result["final_target_or_image_opened"])

    def test_unmatched_shapes_require_review_not_admission(self):
        short = pipe(self.nodes[:3], [[0, 1], [1, 2]])
        result = compare([{"id": "P01", "family": "pipework", "program": self.original},
                          {"id": "G06", "family": "greebles",
                           "program": {"family": "greebles", "components": []}}],
                         {"TRAIN": [{"program": short}], "DEVELOPMENT": []})
        self.assertEqual([r["disposition"] for r in result["candidates"]],
                         ["REVIEW_ONLY_NO_AUTOMATIC_ADMISSION"] * 2)
        self.assertEqual(result["native_views"], 0)
        self.assertEqual(result["optimizer_steps"], 0)


if __name__ == "__main__":
    unittest.main()
