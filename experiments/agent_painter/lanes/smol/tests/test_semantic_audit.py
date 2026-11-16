"""SMOL model-free topology audit and source-custody regressions."""
from pathlib import Path
import hashlib
import importlib.util
import json
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "semantic_audit.py"
spec = importlib.util.spec_from_file_location("smol_semantic_audit", SCRIPT)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def network(nodes, edges, diameter=3):
    return {"version": 2, "family": "pipework",
            "components": [{"kind": "pipe_network", "nodes": nodes,
                            "edges": edges, "diameter": diameter,
                            "fittings": True}],
            "requirements": {"connected": True, "cycles": len(edges)-len(nodes)+1},
            "variation": 0}


class SemanticAuditTests(unittest.TestCase):
    def test_genuine_manifold_passes_request_topology(self):
        p = network([[10,72],[28,72],[48,72],[68,72],[86,72],[28,18],[48,32],[68,18]],
                    [[0,1],[1,2],[2,3],[3,4],[1,5],[2,6],[3,7]])
        observed = subject.graph_metrics(p)
        required = subject.PIPE_REQUIREMENTS["pipework/three_branch_manifold/v1"]
        self.assertTrue(all(observed[k] == value for k, value in required.items()))

    def test_training_tee_compiles_but_fails_manifold(self):
        p = network([[12,48],[48,48],[84,48],[48,14]], [[0,1],[1,2],[1,3]], 5)
        observed = subject.graph_metrics(p)
        required = subject.PIPE_REQUIREMENTS["pipework/three_branch_manifold/v1"]
        for name in ("endpoints", "tees", "upward_branches", "diameter"):
            self.assertNotEqual(observed[name], required[name])
        self.assertEqual(observed["cycles"], 0)

    def test_one_loop_and_one_tap(self):
        p = network([[20,24],[72,24],[72,72],[20,72],[20,48],[7,48]],
                    [[0,1],[1,2],[2,3],[3,4],[4,0],[4,5]])
        observed = subject.graph_metrics(p)
        required = subject.PIPE_REQUIREMENTS["pipework/closed_loop_with_tap/v1"]
        self.assertTrue(all(observed[k] == value for k, value in required.items()))

    def test_duplicate_json_refused(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            subject.strict_json('{"family":"pipework","family":"pipework"}')

    def test_compiler_success_is_not_semantic_success_and_final_never_opened(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            groups = ["pipework/three_branch_manifold/v1",
                      "pipework/closed_loop_with_tap/v1",
                      "greebles/a/v1", "greebles/b/v1",
                      "architecture/a/v1", "architecture/b/v1"]
            bad_tee = json.dumps(network([[12,48],[48,48],[84,48],[48,14]],
                                         [[0,1],[1,2],[1,3]], 5))
            for arm in subject.ARMS:
                for i, group in enumerate(groups):
                    for seed in (104729, 130363):
                        ident = f"case{i}-s{seed}"
                        directory = root / arm / ident
                        directory.mkdir(parents=True)
                        output = bad_tee if i == 0 and arm == "trained-candidate" else "unparseable"
                        (directory / "completion.txt").write_text(output)
                        record = {"id": ident, "structural_group": group,
                                  "host_seed": seed,
                                  "prompt_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
                                  "completion_sha256": hashlib.sha256(output.encode()).hexdigest(),
                                  "compiler_accepted": i == 0 and arm == "trained-candidate",
                                  "native_executed": False}
                        (directory / "receipt.json").write_text(json.dumps(record))
            manifest = {"training_source_sha": "train", "evaluation_source_sha": "eval",
                        "corpus_manifest_sha256": "0"*64, "max_new_tokens": 512,
                        "development_seed_cases": 12, "development_unique_prompt_count": 6,
                        "development_structural_groups": 6, "native_execution": False,
                        "final_targets_accessed": False}
            (root / "summary.json").write_text(json.dumps(manifest))
            audit = subject.audit(root, "train", "eval", "0"*64, 512)
            candidate = audit["arms"]["trained-candidate"]
            self.assertEqual(candidate["compiler_accepted_views"], 2)
            self.assertEqual(candidate["compiler_accepted_requests"], 1)
            self.assertEqual(candidate["pipe_semantic_conforming_requests"], 0)
            self.assertIsNone(candidate["all_family_semantic_score"])
            self.assertFalse(audit["final_opened"])
            with self.assertRaisesRegex(ValueError, "source/cohort"):
                subject.audit(root, "WRONG", "eval", "0"*64, 512)


if __name__ == "__main__":
    unittest.main()
