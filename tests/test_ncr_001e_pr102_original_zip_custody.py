"""Original PR102 whole artifact and source snapshot fail-closed custody tests; no model."""
from pathlib import Path
import sys, shutil, tempfile, unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import ncr_001e_pr102_original_zip_custody as c
class OriginalZipTests(unittest.TestCase):
    def test_all_originals_exact(self):
        d=c.audit()
        self.assertEqual(d["original_members"],8)
        self.assertFalse(d["train_authorized"])
    def mutate(self,name,fn):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            for src in [c.MANIFEST,c.ORIGINAL_ZIP,c.ORIGINAL_TAR,c.ORIGINAL_LOG]:
                shutil.copy2(src,p/src.name)
            target=p/name
            target.write_bytes(fn(target.read_bytes()))
            with self.assertRaises((ValueError,FileNotFoundError)):
                c.audit(p)
    def test_original_zip_append_refused(self):
        self.mutate(c.ORIGINAL_ZIP.name,lambda b:b+b"\\n")
    def test_source_tar_append_refused(self):
        self.mutate(c.ORIGINAL_TAR.name,lambda b:b+b"\\n")
    def test_job_log_append_refused(self):
        self.mutate(c.ORIGINAL_LOG.name,lambda b:b+b"\\nFORGED")
    def test_manifest_extra_whitespace_refused(self):
        self.mutate(c.MANIFEST.name,lambda b:b+b"\\n")
    def test_missing_archive_refused(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            for src in [c.MANIFEST,c.ORIGINAL_TAR,c.ORIGINAL_LOG]:
                shutil.copy2(src,p/src.name)
            with self.assertRaises(FileNotFoundError):
                c.audit(p)
    def test_fake_zip_refused(self):
        with self.assertRaises(ValueError):
            c.check_zip(b"PK"+b"fabricated")
if __name__=="__main__":
    unittest.main()
