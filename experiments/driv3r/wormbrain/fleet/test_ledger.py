"""Adversarial source-only fixtures; mock PIDs NEVER prove resident inference."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from ledger import REQUEST, EPISODE, audit, canonical, digest, plan

SRC = "a"*64
HOST = "b"*64
MODEL = "c"*64
CHECK = "d"*64


def req():
    return {"schema": REQUEST, "campaign": "campaign1", "wrangler": "wrangler1",
            "source_sha256": SRC, "host_source_sha256": HOST, "training": False,
            "paid_api": False, "automatically_expand": False,
            "native_permissions_remain_host_owned": True,
            "max_runners": 2, "models_per_runner": 2,
            "episodes": [{"id": "worm"+str(i), "seed": 7200+i, "runner_slot": i//2,
                         "model_id": "dummy/test-no-real-model", "model_revision": "pinned-test",
                         "model_sha256": MODEL, "checkpoint_sha256": CHECK,
                         "evidence_class": "model_scored_w0",
                         "decision_budget": 24, "material_budget": 24,
                         "allow_model_training": False} for i in range(4)]}


def sample(root, r):
    p = plan(r)
    for i, e in enumerate(r["episodes"]):
        d = root / e["id"]
        d.mkdir()
        originals = {}
        for k, (name, value) in {
            "png": ("canvas.png", b"\x89PNG\r\n\x1a\n"+bytes([i])),
            "raw_decisions_jsonl": ("raw.jsonl", b'{"raw":"LEFT"}\n'),
            "receipt_jsonl": ("receipts.jsonl", b'{"legal":true}\n')}.items():
            (d / name).write_bytes(value)
            originals[k] = {"path": name, "sha256": digest(value)}
        lease = next(l["lease_id"] for j in p["runner_jobs"]
                     for l in j["leases"] if l["id"] == e["id"])
        x = {**e, "schema": EPISODE, "episode_id": e["id"], "campaign": r["campaign"],
             "wrangler": r["wrangler"], "request_sha256": p["request_sha256"],
             "source_sha256": SRC, "host_source_sha256": HOST,
             "lease_id": lease, "job_id": "job"+str(e["runner_slot"]), "run_id": "run1",
             "pid": 1000+i, "status": "completed", "optimizer_steps": 0,
             "forward_start_ns": (10+i)*10**9, "forward_end_ns": 60*10**9,
             "model_forward_count": 24, "decisions_applied": 24,
             "native_material_edits": 12, "recoveries": 1,
             "wall_ms": 30000, "rss_peak_kib": 300000,
             "raw_model_responses_retained": True, "legal_action_receipts_checked": True,
             "native_execution": False, "source_faithful_w0": True,
             "originals": originals}
        (d / "episode.json").write_bytes(canonical(x))
    return p


class TestFleet(unittest.TestCase):
    def test_four_unique_deterministic_leases(self):
        p = plan(req())
        self.assertEqual(p, plan(copy.deepcopy(req())))
        self.assertEqual(len({l["lease_id"] for j in p["runner_jobs"] for l in j["leases"]}), 4)
        self.assertFalse(p["dispatch_permitted"])

    def test_denies_training_paid_expansion(self):
        for field, value in [("training", True), ("paid_api", True),
                             ("automatically_expand", True),
                             ("native_permissions_remain_host_owned", False),
                             ("models_per_runner", 3)]:
            r = req()
            r[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                plan(r)

    def test_native_128_rejected(self):
        r = req()
        r["episodes"][0]["evidence_class"] = "native_model"
        r["episodes"][0]["decision_budget"] = 128
        with self.assertRaises(ValueError):
            plan(r)

    def test_duplicate_seed_and_runner_load_refused(self):
        for name in ("seed", "runner_slot", "checkpoint_sha256"):
            r = req()
            r["episodes"][2][name] = {"seed": 7200, "runner_slot": 0,
                                       "checkpoint_sha256": ""}[name]
            with self.subTest(name=name), self.assertRaises(ValueError):
                plan(r)

    def test_mock_four_overlap_does_not_claim_live_capacity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            r = req()
            p = sample(root, r)
            a = audit(r, p, root)
            self.assertEqual(a["original_episodes_checked"], 4)
            self.assertEqual(a["decision_count"], 96)
            self.assertEqual(len(a["claimed_same_runner_two_resident_overlap_seconds_by_job"]), 2)
            self.assertTrue(all(v > 1 for v in a["claimed_same_runner_two_resident_overlap_seconds_by_job"].values()))
            self.assertIsNone(a["claimed_four_forward_overlap_seconds"])
            self.assertFalse(a["cross_runner_wall_clock_overlap_verified"])
            self.assertEqual(a["independently_verified_current_active_models"], 0)

    def test_different_runner_monotonic_epochs_do_not_establish_four_way_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            r = req()
            p = sample(root, r)
            # Runner 1 has a completely unrelated monotonic epoch: this must
            # not create a false four-way elapsed-time comparison.
            for i in (2, 3):
                f = root / ("worm"+str(i)) / "episode.json"
                x = json.loads(f.read_text())
                x["forward_start_ns"] += 100000000000000
                x["forward_end_ns"] += 100000000000000
                f.write_bytes(canonical(x))
            a = audit(r, p, root)
            self.assertEqual(len(a["claimed_same_runner_two_resident_overlap_seconds_by_job"]), 2)
            self.assertIsNone(a["claimed_four_forward_overlap_seconds"])
            self.assertFalse(a["cross_runner_wall_clock_overlap_verified"])

    def test_tampering_fail_closed(self):
        for defect in ("png", "lease", "source", "pid", "optimizer", "state", "overlap", "label"):
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                r = req()
                p = sample(root, r)
                file = root / "worm0" / "episode.json"
                x = json.loads(file.read_text())
                if defect == "png":
                    (root/"worm0"/"canvas.png").write_bytes(b"FAKE")
                elif defect == "lease":
                    x["lease_id"] = "faked"
                elif defect == "source":
                    x["source_sha256"] = "e"*64
                elif defect == "pid":
                    x["pid"] = 1001
                elif defect == "optimizer":
                    x["optimizer_steps"] = 1
                elif defect == "state":
                    x["status"] = "queued"
                elif defect == "overlap":
                    x["forward_end_ns"] = 11*10**9
                elif defect == "label":
                    x["native_execution"] = True
                file.write_bytes(canonical(x))
                with self.subTest(defect=defect), self.assertRaises(ValueError):
                    audit(r, p, root)


if __name__ == "__main__":
    unittest.main()
