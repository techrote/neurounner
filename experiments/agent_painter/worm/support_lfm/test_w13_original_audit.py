#!/usr/bin/env python3
"""Original W13 evidence regression and adversarial copy-only corruption tests."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import audit_w13_original as audit

ALPHA = None
BETA = None

def modified(src, suffix, mutation, parent):
    out=Path(parent)/"mutation.zip"
    with zipfile.ZipFile(src) as old, zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as new:
        found=False
        for item in old.infolist():
            data=old.read(item.filename)
            if item.filename.endswith(suffix):
                data=mutation(data)
                found=True
            new.writestr(item.filename,data)
        assert found, "fixture path unavailable"
    return out

class OriginalReceipts(unittest.TestCase):
    def test_frozen_original_four_128_turn_models(self):
        r=audit.audit(ALPHA,BETA)
        self.assertEqual(r["genuine_original_model_score_forwards"],512)
        self.assertEqual(r["original_STOP_legal_but_masked_choices"],512)
        self.assertEqual(r["same_state_counterfactual_STOP_choices_NOT_rollouts"],49)
        self.assertEqual(r["same_state_flat_choices_different_NOT_rollouts"],93)
        self.assertFalse(r["STOP_autonomy_qualified"])
        self.assertFalse(r["independent_pinned_W0_png_byte_replay"])

    def rejects(self,suffix,mutate):
        with tempfile.TemporaryDirectory() as d:
            fake=modified(ALPHA,suffix,mutate,d)
            with self.assertRaises(ValueError):
                audit.audit_shard(fake,"alpha",check_zip_hash=False)

    def test_modified_original_W0_png_rejected(self):
        self.rejects("original-w0.png",lambda b:b[:-7]+b"changed")

    def test_modified_applied_model_choice_rejected(self):
        def mutate(b):
            j=json.loads(b)
            j[0]["raw_choice_selected_by_host_sampler"]="STOP"
            return json.dumps(j).encode()
        self.rejects("model-decisions.json",mutate)

    def test_forged_observation_mask_rejected(self):
        def mutate(b):
            j=json.loads(b)
            j["events"][0]["observation"]["legal"]=["STOP"]
            return json.dumps(j).encode()
        self.rejects("original-w0.json",mutate)

    def test_changed_STOP_censorship_rejected(self):
        def mutate(b):
            j=json.loads(b)
            j[0]["sampling_receipt"]["sampled_legal"].append("STOP")
            return json.dumps(j).encode()
        self.rejects("model-decisions.json",mutate)

    def test_forged_inference_pulse_rejected(self):
        def mutate(b):
            rows=[json.loads(x) for x in b.splitlines()]
            rows[17]["real_score_forward"]=False
            return ("\n".join(json.dumps(x) for x in rows)+"\n").encode()
        self.rejects("forward-pulse.jsonl",mutate)

    def test_duplicate_resident_lease_rejected(self):
        def mutate(b):
            j=json.loads(b)
            j["distinct_lease_ids"][1]=j["distinct_lease_ids"][0]
            return json.dumps(j).encode()
        self.rejects("summary.json",mutate)

    def test_model_probability_is_not_flat(self):
        scores={a:float(i) for i,a in enumerate(audit.ACTIONS)}
        soft,mixed=audit.mixture(scores,list(audit.ACTIONS))
        self.assertAlmostEqual(sum(soft.values()),1)
        self.assertAlmostEqual(sum(mixed.values()),1)
        self.assertGreater(mixed["STOP"],mixed["STRAIGHT"])
        self.assertNotEqual(audit.sample(mixed,list(audit.ACTIONS),.12),
                            audit.sample({a:1/6 for a in audit.ACTIONS},list(audit.ACTIONS),.12))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--alpha",required=True,type=Path)
    ap.add_argument("--beta",required=True,type=Path)
    args=ap.parse_args()
    ALPHA,BETA=args.alpha,args.beta
    unittest.main(argv=["test_w13_original_audit.py"],verbosity=2)
