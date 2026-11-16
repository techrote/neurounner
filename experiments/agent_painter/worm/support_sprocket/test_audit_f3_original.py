#!/usr/bin/env python3
"""Adversarial original F3 audit tests. Requires pinned original immutable F3 ZIP."""
import io
import json
import os
import unittest
import zipfile
from pathlib import Path
import audit_f3_original as f3

def resign_member(raw,name,mutation):
    with zipfile.ZipFile(io.BytesIO(raw)) as zip_in:
        files={n:zip_in.read(n) for n in zip_in.namelist()}
    previous=files[name]
    if name.endswith(".json"):
        obj=json.loads(previous)
        mutation(obj)
        files[name]=json.dumps(obj,sort_keys=True).encode()
    else:
        files[name]=mutation(previous)
    slot=name.split("/")[0]
    checksum=f"{slot}/SHA256SUMS.txt"
    line_suffix="  ./"+name[len(slot)+1:]
    entries=files[checksum].decode().splitlines()
    assert sum(line.endswith(line_suffix) for line in entries)==1
    files[checksum]=("\n".join(
        (f3.digest(files[name])+line_suffix) if line.endswith(line_suffix) else line
        for line in entries)+"\n").encode()
    buff=io.BytesIO()
    with zipfile.ZipFile(buff,"w",zipfile.ZIP_DEFLATED) as output:
        for n,body in files.items():output.writestr(n,body)
    return buff.getvalue()

class OriginalF3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=Path(os.environ["SPROCKET_F3_ORIGINAL_ZIP"]).read_bytes()
        with zipfile.ZipFile(io.BytesIO(cls.raw)) as z:
            cls.trained=next(n for n in z.namelist() if
                             "original-png-and-json/" in n and
                             n.endswith(".json") and "sprocket-f1-" in n)

    def test_original_complete_and_negative_boundaries(self):
        r=f3.audit(self.raw)
        self.assertEqual(r["producer_attested_six_logit_model_forwards"],512)
        self.assertEqual(r["source_replayed_original_png_json_pairs"],16)
        self.assertEqual(r["model_vs_flat_changed_actions"],149)
        self.assertEqual(r["mean_trained_unique_cells"],71)
        self.assertFalse(r["new_model_inference_performed"])
        self.assertFalse(r["native"])
        self.assertFalse(r["simultaneous_individual_model_forward_calls_proven"])

    def test_modified_outer_archive_rejected(self):
        changed=self.raw[:-1]+bytes([self.raw[-1]^1])
        with self.assertRaisesRegex(ValueError,"outer original ZIP digest"):
            f3.audit(changed)

    def test_master_source_spoof_rejected_even_if_outer_digest_adjusted(self):
        with zipfile.ZipFile(io.BytesIO(self.raw)) as z:
            entries={n:z.read(n) for n in z.namelist()}
        doc=json.loads(entries["four-way-audit.json"])
        doc["source_sha"]="0"*40
        entries["four-way-audit.json"]=json.dumps(doc).encode()
        output=io.BytesIO()
        with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED) as z:
            for name,payload in entries.items():z.writestr(name,payload)
        with self.assertRaisesRegex(ValueError,"master provenance"):
            f3.audit(output.getvalue(),expected=f3.digest(output.getvalue()))

    def test_re_signed_fake_W0_position_rejected(self):
        changed=resign_member(self.raw,self.trained,
                              lambda d:d["events"][0].update({"position_after":[95,95]}))
        with self.assertRaisesRegex(ValueError,"W0 state/event exact replay"):
            f3.audit(changed,expected=f3.digest(changed))

    def test_re_signed_host_draw_spoof_rejected(self):
        changed=resign_member(self.raw,self.trained,
                              lambda d:d["score_receipts"][0].update({"host_uniform_draw":0.99999}))
        with self.assertRaisesRegex(ValueError,"seeded host RNG"):
            f3.audit(changed,expected=f3.digest(changed))

    def test_re_signed_bad_PNG_rejected(self):
        image=self.trained[:-5]+".png"
        changed=resign_member(self.raw,image,
                              lambda b:b[:-1]+bytes([b[-1]^1]))
        with self.assertRaisesRegex(ValueError,"original W0 pixel-exact PNG"):
            f3.audit(changed,expected=f3.digest(changed))

if __name__=="__main__":
    unittest.main()
