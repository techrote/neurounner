"""Negative/tamper tests for original BYTE-EOS independent review archive."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))

from byte_eos_001_independent_archive_audit import (  # noqa:E402
    ARCHIVE, load_originals, verify_originals
)


class ByteEOSArchiveAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files,cls.p=load_originals()

    def test_original_review_archive_and_narrow_decision(self):
        verify_originals(copy.deepcopy(self.files),copy.deepcopy(self.p))

    def test_corrupt_full_actions_zip_base64_refused(self):
        raw=ARCHIVE.read_text(encoding="ascii")
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/"bad.b64"
            p.write_text(("A" if raw[0]!="A" else "B")+raw[1:])
            with self.assertRaises((ValueError,RuntimeError)):
                load_originals(p)
            p.write_text("not-base64")
            with self.assertRaises((ValueError,RuntimeError)):
                load_originals(p)

    def test_false_training_integration_or_historical_claim_refused(self):
        for key,value in (
            ("model_weights_loaded",True),("training_authorized",True),
            ("historical_train002b_modified",True),
            ("historical_train002b_reinterpreted",True),
            ("heldout_evaluation",True),("optimizer_steps",1),
            ("integration_requires_void72_and_explicit_training_authorization",False),
        ):
            p=copy.deepcopy(self.p)
            p["boundaries"][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                verify_originals(self.files,p)

    def test_candidate_tokenizer_prereg_source_identity_mutations_refused(self):
        for section,key,value in (
            ("reviewed_candidate","candidate_git_blob","0"*40),
            ("reviewed_candidate","candidate_file_sha256","0"*64),
            ("source","tokenizer_source_sha256","0"*64),
            ("source","max_units",768),
            ("source","preregistration_commit","wrong"),
            ("original_actions","artifact_zip_sha256","0"*64),
            ("original_actions","branch_head","wrong"),
        ):
            p=copy.deepcopy(self.p);p[section][key]=value
            with self.subTest(section=section,key=key),self.assertRaises(RuntimeError):
                verify_originals(self.files,p)

    def test_original_review_json_false_oracle_or_boundary_relabel_refused(self):
        for mode in ("oracle","terminal","overflow","impact","integration","historic"):
            files=dict(self.files)
            j=json.loads(files["byte-eos-001-independent-review.json"])
            if mode=="oracle":j["cases"][0]["exact_oracle_match"]=False
            elif mode=="terminal":j["cases"][0]["terminal_id"]=257
            elif mode=="overflow":j["overflow_1025_refused"]=False
            elif mode=="impact":
                j["decision"]["historical_generation_effect_size"]="FIXED_CAUSE"
            elif mode=="integration":
                j["decision"]["integration_into_void_training_accepted"]=True
            else:
                j["historical_identity"]["legacy_suffix_ids"]=[258]
            files["byte-eos-001-independent-review.json"]=json.dumps(j).encode()
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):
                verify_originals(files,self.p)

    def test_original_train002b_receipt_and_provenance_cannot_be_relabelled(self):
        p=copy.deepcopy(self.p)
        p["reviewed_candidate"]["original_byte_eos_receipt_sha256"]="0"*64
        with self.assertRaises(RuntimeError):
            verify_originals(self.files,p)
        p=copy.deepcopy(self.p)
        p["reviewed_candidate"]["original_byte_eos_provenance_sha256"]="0"*64
        with self.assertRaises(RuntimeError):
            verify_originals(self.files,p)


if __name__=="__main__":
    unittest.main()
