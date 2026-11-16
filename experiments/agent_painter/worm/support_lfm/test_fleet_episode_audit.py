#!/usr/bin/env python3
"""TEST-ONLY fabricated model metadata around REAL W0 replay/PNG test fixtures.

Never publish this test fixture as an LFM model, model resident, native paint,
Actions producer or optimizer receipt. It only tests strict source parsing.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import worm
from fleet_episode_audit import audit, SCHEMA, W0_SHA, MODEL_REV


def canonical(x):
    return (json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n").encode()


class FleetAuditFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = {"schema": SCHEMA, "campaign_id": "TEST-ONLY-FABRICATED-NOT-RESIDENT",
                         "decision_budget": 2, "episodes": []}
        for i in range(4):
            self.manifest["episodes"].append(self.fixture(i))

    def tearDown(self):
        self.temp.cleanup()

    def write(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def fixture(self, i):
        seed = 51001+i
        ident = "fixture-" + str(i)
        trace = worm.run(seed=seed, mode="replay", family="greebles",
                         steps=2, raw_actions=["STRAIGHT", "STRAIGHT"])
        inf = [{"tick": n, "model_forward": True, "raw_choice": "STRAIGHT",
                "producer_run_id": 10001+i, "inference_mode": "raw_first_pass",
                "model_receipt_sha256": "a"*64, "latency_ms": 10.0} for n in range(2)]
        logbytes = b"".join(canonical(row) for row in inf)
        pulses = []
        for slot, count, t in (("a", 1, 10000000), ("b", 2, 10020000)):
            pulses.append({"episode_id": ident, "runner_id": "test-runner-" + str(i//2),
                           "pid": 6000+i, "pid_start_ticks": 7000+i,
                           "sample_slot": slot, "state": "running",
                           "observed_at_ms": t, "observed_actions": count,
                           "observed_log_prefix_sha256": hashlib.sha256(
                               b"".join(canonical(row) for row in inf[:count])).hexdigest(),
                           "attestation": "linux-proc-pid-observed", "proc_alive": True,
                           "vcpus": 4, "rss_kib": 200000})
        episode = {"episode_id": ident, "lease_id": "test-lease-"+str(i),
                   "kind": "trained_lfm", "status": "completed", "runner_id": "test-runner-"+str(i//2),
                   "workflow_run_id": 10001+i, "pid": 6000+i, "pid_start_ticks": 7000+i,
                   "seed": seed, "family": "greebles", "model_id": "LiquidAI/LFM2.5-VL-450M",
                   "model_revision": MODEL_REV, "w0_source_sha": W0_SHA,
                   "source_sha": "b"*40, "checkpoint_sha256": "c"*64,
                   "original_producer_zip_sha256": "d"*64,
                   "native_execution": False, "optimizer_updates": 0}
        for name, extension, data in (
            ("trace", "json", canonical(trace)),
            ("png", "png", worm.png_bytes(trace)),
            ("inference", "jsonl", logbytes),
            ("pulses", "jsonl", b"".join(canonical(row) for row in pulses))):
            episode[name] = ident + "/" + name + "." + extension
            episode[name+"_sha256"] = self.write(episode[name], data)
        return episode

    def mutate_pulses(self, index, change):
        ep = self.manifest["episodes"][index]
        rows = [json.loads(x) for x in (self.root / ep["pulses"]).read_text().splitlines()]
        change(rows)
        ep["pulses_sha256"] = self.write(
            ep["pulses"], b"".join(canonical(row) for row in rows))

    def test_test_only_four_process_structure(self):
        report = audit(self.root, self.manifest, worm)
        self.assertTrue(report["preflight_pass"])
        self.assertEqual(report["resident_processes_by_runner"],
                         {"test-runner-0": 2, "test-runner-1": 2})
        self.assertFalse(report["model_weights_independently_executed"])

    def test_queue_does_not_count(self):
        self.manifest["episodes"][0]["status"] = "queued"
        with self.assertRaisesRegex(ValueError, "queued/failed"):
            audit(self.root, self.manifest, worm)

    def test_synthetic_label_does_not_count(self):
        self.manifest["episodes"][0]["kind"] = "synthetic_w0"
        with self.assertRaisesRegex(ValueError, "not trained LFM"):
            audit(self.root, self.manifest, worm)

    def test_reused_seed_is_not_four_worlds(self):
        self.manifest["episodes"][1]["seed"] = self.manifest["episodes"][0]["seed"]
        with self.assertRaisesRegex(ValueError, "duplicate obstacle world"):
            audit(self.root, self.manifest, worm)

    def test_original_png_mutation_rejected(self):
        ep = self.manifest["episodes"][0]
        (self.root / ep["png"]).write_bytes(b"spoofed png")
        with self.assertRaisesRegex(ValueError, "evidence digest mismatch"):
            audit(self.root, self.manifest, worm)

    def test_fake_model_forward_rejected(self):
        ep = self.manifest["episodes"][0]
        rows = [json.loads(x) for x in (self.root / ep["inference"]).read_text().splitlines()]
        rows[0]["model_forward"] = False
        ep["inference_sha256"] = self.write(
            ep["inference"], b"".join(canonical(row) for row in rows))
        with self.assertRaisesRegex(ValueError, "ORIGINAL LFM choice"):
            audit(self.root, self.manifest, worm)

    def test_duplicated_pid_not_four_processes(self):
        a, b = self.manifest["episodes"][:2]
        b["pid"], b["pid_start_ticks"], b["runner_id"] = (
            a["pid"], a["pid_start_ticks"], a["runner_id"])
        with self.assertRaisesRegex(ValueError, "duplicate process"):
            audit(self.root, self.manifest, worm)

    def test_log_prefix_must_match_current_action_progress(self):
        self.mutate_pulses(0, lambda rows: rows[0].update({
            "observed_log_prefix_sha256": "f"*64}))
        with self.assertRaisesRegex(ValueError, "live log prefix"):
            audit(self.root, self.manifest, worm)

    def test_simultaneity_not_inferred_from_individual_success(self):
        self.mutate_pulses(0, lambda rows: [
            row.update({"observed_at_ms": row["observed_at_ms"] + 50000})
            for row in rows])
        report = audit(self.root, self.manifest, worm)
        self.assertFalse(report["preflight_pass"])

    def test_path_traversal_refused(self):
        self.manifest["episodes"][0]["trace"] = "../not-source.json"
        with self.assertRaisesRegex(ValueError, "unsafe evidence path"):
            audit(self.root, self.manifest, worm)


if __name__ == "__main__":
    unittest.main()
