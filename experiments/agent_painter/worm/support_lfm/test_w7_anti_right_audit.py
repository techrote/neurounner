#!/usr/bin/env python3
"""Immutable W7 source/counterfactual custody regressions."""
from __future__ import annotations
import copy
from pathlib import Path
import tempfile
import unittest
from w7_anti_right_audit import make_probe, verified_zip, canonical, digest


class AntiRightTests(unittest.TestCase):
    def sample(self):
        obs = {"family":"greebles", "heading":0, "local_5x5":["....."]*5,
               "legal":["STRAIGHT","LEFT","RIGHT","BRANCH","RETRACE","STOP"],
               "last_actions":["RIGHT"], "tick":12}
        text = canonical({k:obs[k] for k in
                          ("family","heading","local_5x5","legal","last_actions","tick")})
        return {"id":"greebles-wall-3101:observed", "kind":"observed",
                "observation_origin":"actual-W0", "observation":obs,
                "source_seed":3101, "source_mode":"wall", "source_tick":12,
                "prompt":text, "prompt_sha256":digest(text.encode())}

    def test_synthetic_mask_preserves_source(self):
        source = self.sample()
        unchanged = copy.deepcopy(source)
        p = make_probe(source)
        self.assertEqual(source, unchanged)
        self.assertNotIn("RIGHT", p["observation"]["legal"])
        self.assertIn("STOP", p["observation"]["legal"])
        self.assertIsNone(p["target_action"])
        self.assertEqual(p["observation_origin"], "synthetic-not-W0-executed")
        self.assertNotEqual(p["prompt_sha256"], source["prompt_sha256"])

    def test_tampering_fails_closed(self):
        original = self.sample()
        for alteration in ("body","hash","field","origin","no-right","no-stop","duplicate"):
            with self.subTest(alteration=alteration):
                x = copy.deepcopy(original)
                if alteration == "body":x["prompt"] += " "
                elif alteration == "hash":x["prompt_sha256"] = "0"*64
                elif alteration == "field":del x["observation"]["legal"]
                elif alteration == "origin":x["observation_origin"] = "synthetic"
                elif alteration == "no-right":x["observation"]["legal"].remove("RIGHT")
                elif alteration == "no-stop":x["observation"]["legal"].remove("STOP")
                elif alteration == "duplicate":x["observation"]["legal"].append("RIGHT")
                with self.assertRaises((ValueError, KeyError)):
                    make_probe(x)

    def test_reproducible_bytes(self):
        x = self.sample()
        a = make_probe(x)
        b = make_probe(copy.deepcopy(x))
        self.assertEqual(canonical(a), canonical(b))
        self.assertEqual(digest(a["prompt"].encode()), a["prompt_sha256"])

    def test_wrong_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "wrong.zip"
            f.write_bytes(b"not the original artifact")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verified_zip(f, "f"*64, support=True)


if __name__ == "__main__":
    unittest.main()
