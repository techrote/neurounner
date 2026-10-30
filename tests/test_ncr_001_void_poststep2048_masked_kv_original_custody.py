"""Adversarial no-model tests for permanent masked-KV original custody."""
from __future__ import annotations
from contextlib import contextmanager
import copy, io, json, shutil, sys, tempfile, unittest, zipfile
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import ncr_001_void_poststep2048_masked_kv_original_custody as custody
import ncr_001_void_poststep2048_masked_kv_bootstrap_custody as acquisition

class Custody(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
    cls.report,cls.originals=custody.read_and_check()
    with zipfile.ZipFile(io.BytesIO(cls.originals["host-original.zip"])) as z:
      cls.producer=json.loads(z.read("ncr001-void-poststep2048/producer.json"))
      cls.consumer=json.loads(z.read("ncr001-void-poststep2048-masked-kv/consumer.json"))
 @contextmanager
 def copied(self):
    with tempfile.TemporaryDirectory() as td:
      e=Path(td)/"e";shutil.copytree(custody.EVIDENCE,e)
      with mock.patch.object(custody,"EVIDENCE",e),mock.patch.object(custody,"MANIFEST",e/"manifest.json"):
        yield e
 def test_exact_original_result(self):
    r=self.report;p=r["physical_result"]
    self.assertEqual(r["original_run"],41136709967)
    self.assertEqual(r["original_zips_and_logs"],4)
    self.assertEqual(r["original_zip_members"],10)
    self.assertTrue(r["permanent_git_custody"])
    self.assertEqual(p["nca_only_optimizer_steps"],1)
    for side in ("left","right"):
      x=p["sides"][side]
      self.assertEqual(x["genuine_B2_incremental_steps"],64)
      self.assertEqual(x["tail_real_incremental_units_per_row"],64)
      self.assertAlmostEqual(x["full_vs_cached_max_abs"],1.6927719116210938e-05)
      self.assertLessEqual(x["batch_vs_single_cached_max_abs"],9.5367431640625e-06)
      self.assertEqual(x["row_swap_cached_max_abs"],0.0)
 def test_literal_git_objects(self):
    custody.entry(custody.MANIFEST,custody.MANIFEST_BLOB)
    for n,(_,_,b) in custody.FILES.items(): custody.entry(custody.EVIDENCE/n,b)
 def test_mutated_or_missing_streams_refused(self):
    for n in custody.FILES:
      with self.subTest(name=n),self.copied() as e:
        p=e/n;p.write_bytes(p.read_bytes()+b"X")
        with self.assertRaisesRegex(RuntimeError,"REFUSED"):custody.read_and_check(verify_git=False)
      with self.subTest(missing=n),self.copied() as e:
        (e/n).unlink()
        with self.assertRaisesRegex(RuntimeError,"REFUSED"):custody.read_and_check(verify_git=False)
 def test_symlink_refused(self):
    with self.copied() as e:
      p=e/"host-original.zip";q=e/"saved";p.rename(q);p.symlink_to(q.name)
      with self.assertRaisesRegex(RuntimeError,"REFUSED"):custody.read_and_check(verify_git=False)
 def test_false_science_manifest_refused(self):
    for key in custody.BOUNDARY:
      with self.subTest(key=key),self.copied() as e:
        p=e/"manifest.json";d=json.loads(p.read_text());d["scientific_boundary"][key]=True
        p.write_text(json.dumps(d))
        with self.assertRaisesRegex(RuntimeError,"REFUSED"):custody.read_and_check(verify_git=False)
 def test_semantic_mask_cache_forgery_refused(self):
    muts=(
      lambda c:c["masked_incremental_kv"]["left"]["full_vs_cached"].__setitem__("max_abs",.1),
      lambda c:c["masked_incremental_kv"]["right"].__setitem__("genuine_B2_incremental_steps",4),
      lambda c:c["masked_incremental_kv"]["left"].__setitem__("batch_vs_single_cached_max_abs",.1),
      lambda c:c["masked_incremental_kv"]["right"]["row_swap_cached"].__setitem__("max_abs",.1),
      lambda c:c["masked_incremental_kv"]["right"].__setitem__("right_layout_is_synthetic_mask_gap",False))
    for m in muts:
      p,c=copy.deepcopy(self.producer),copy.deepcopy(self.consumer);m(c)
      with self.assertRaisesRegex(RuntimeError,"REFUSED"):acquisition.semantics(p,c)
 def test_no_adapter_in_original_archive(self):
    with zipfile.ZipFile(io.BytesIO(self.originals["host-original.zip"])) as z:
      self.assertEqual(len(z.namelist()),8)
      self.assertFalse(any(n.endswith((".pt",".safetensors","metadata.json")) for n in z.namelist()))
 def test_source_only_gate_flags(self):
    r=custody.audit()
    self.assertFalse(r["fresh_physical_execution"])
    self.assertFalse(r["train_authorized"])
    self.assertFalse(r["per_model_gate"])
    self.assertFalse(r["full_72_conformance_pass"])
    self.assertFalse(r["actual_compute_budget_qualified"])

if __name__=="__main__":unittest.main()
