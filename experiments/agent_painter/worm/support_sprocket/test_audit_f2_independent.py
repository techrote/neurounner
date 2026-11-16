#!/usr/bin/env python3
"""Fail-closed WORM F2 audit tests. Original artifact is supplied by CI."""
import io
import json
import os
import unittest
import zipfile
from pathlib import Path

import audit_f2_independent as audit


def patch_json(data, target, edit):
    with zipfile.ZipFile(io.BytesIO(data)) as original:
        members = {p: original.read(p) for p in original.namelist()}
    changed = json.loads(members[target])
    edit(changed)
    members[target] = json.dumps(changed).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for name, payload in members.items():
            z.writestr(name, payload)
    return output.getvalue()


class F2OriginalAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Path(os.environ["SPROCKET_F2_ORIGINAL_ZIP"]).read_bytes()

    def test_original_four_attested(self):
        report = audit.check(self.data)
        self.assertEqual(len(report["streams"]), 4)
        self.assertEqual(report["host_attested_real_model_forwards"], 96)
        self.assertEqual(report["source_faithful_original_png_json_pairs"], 16)
        self.assertGreater(report["four_way_model_episode_overlap_seconds"], 160)
        self.assertFalse(report["individual_simultaneous_forward_calls_proven"])
        self.assertFalse(report["independent_weight_inference_repeated"])
        self.assertFalse(report["native"])

    def test_outer_zip_digest_fail_closed(self):
        changed = self.data[:-1] + bytes([self.data[-1] ^ 1])
        with self.assertRaisesRegex(ValueError, "ZIP digest"):
            audit.check(changed)

    def test_master_source_tampering_fail_closed(self):
        def edit(doc):
            doc["source_sha"] = "0" * 40
        changed = patch_json(self.data, "four-way-audit.json", edit)
        with self.assertRaisesRegex(ValueError, "master source"):
            audit.check(changed, expected=audit.digest(changed))

    def test_slot_receipt_tampering_fail_closed(self):
        def edit(doc):
            doc["decision_bound"] = 128
        changed = patch_json(self.data, "fleet-0/results/fleet.json", edit)
        with self.assertRaisesRegex(ValueError, "member SHA"):
            audit.check(changed, expected=audit.digest(changed))

    def test_duplicate_master_witness_fail_closed(self):
        def edit(doc):
            doc["original_worm_processes"][-1] = doc["original_worm_processes"][-2]
        changed = patch_json(self.data, "four-way-audit.json", edit)
        with self.assertRaisesRegex(ValueError, "master process witness"):
            audit.check(changed, expected=audit.digest(changed))


if __name__ == "__main__":
    unittest.main()
