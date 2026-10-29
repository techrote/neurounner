"""Offline exact compressed-container and fail-closed negative archive checks."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import ncr_001_sprocket_original_zip_custody as c


class OriginalSprocketZipTests(unittest.TestCase):
    def test_original_three_containers_six_members_closed_gates(self):
        out=c.audit()
        self.assertEqual(out["raw_members"],6)
        self.assertEqual(set(out["lane_archives"]),{"source","chat","instruct"})
        self.assertFalse(out["train_authorized"])
        self.assertFalse(out["full_72_conformance_pass"])
        self.assertFalse(out["new_model_execution"])

    def mutate(self, filename, f):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            shutil.copy2(c.EVIDENCE/"manifest.json",d/"manifest.json")
            for lane in c.ORIGINALS:
                shutil.copy2(c.EVIDENCE/c.b64_path(lane),d/c.b64_path(lane))
            p=d/filename
            p.write_bytes(f(p.read_bytes()))
            with self.assertRaises((ValueError,KeyError)):
                c.audit(d)

    def test_b64_whitespace_mutation_refused_even_if_decoding_equivalent(self):
        self.mutate(c.b64_path("source"),lambda b:b+b"\n")

    def test_chat_archive_byte_flip_refused(self):
        self.mutate(c.b64_path("chat"),lambda b:b[:15]+(b"A" if b[15:16]!=b"A" else b"B")+b[16:])

    def test_instruct_archive_truncated_refused(self):
        self.mutate(c.b64_path("instruct"),lambda b:b[:-8])

    def test_false_model_scientific_admission_refused(self):
        def flip(b):
            m=json.loads(b)
            m["full_72_conformance_pass"]=True
            return json.dumps(m).encode()
        self.mutate("manifest.json",flip)

    def test_source_artifact_id_substitution_refused(self):
        def flip(b):
            m=json.loads(b)
            m["original_actions_artifacts"]["source"]["id"]=12826246889
            return json.dumps(m).encode()
        self.mutate("manifest.json",flip)

    def test_cross_variant_archive_substitution_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            shutil.copy2(c.EVIDENCE/"manifest.json",d/"manifest.json")
            for lane in c.ORIGINALS:
                shutil.copy2(c.EVIDENCE/c.b64_path(lane),d/c.b64_path(lane))
            (d/c.b64_path("chat")).write_bytes((d/c.b64_path("instruct")).read_bytes())
            with self.assertRaises(ValueError):c.audit(d)

    def test_modified_decoded_zip_rejected_by_whole_container_hash(self):
        raw=(c.EVIDENCE/c.b64_path("source")).read_bytes()
        import base64
        zipraw=base64.b64decode(raw)
        with self.assertRaises(ValueError): c.check_zip("source",zipraw+b"\x00")


if __name__=="__main__":unittest.main()
