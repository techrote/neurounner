#!/usr/bin/env python3
"""Stdlib-only fail-closed archive and provenance tests for DG020 H22.

These tests DO NOT replay the unpublished H22 scientific evidence and must
never be reported as checkpoint/causal-selectivity CI qualification.
"""
import hashlib
from pathlib import Path
import stat
import tempfile
import unittest
import warnings
import zipfile

from h22_bundle_reproduce import (
    OUTER_SHA, PARENT_SHA, SOURCE_SHA, digest, unpack_safely,
)


class H22BundleContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='h22-contract-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.archive = self.root / 'case.zip'
        self.dest = self.root / 'unpacked'
        self.dest.mkdir()

    def make_zip(self, entries):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(self.archive, 'w') as z:
                for name, payload in entries:
                    z.writestr(name, payload)
        return self.archive

    def test_pinned_h22_source_parent_and_outer_identities(self):
        self.assertEqual(OUTER_SHA, '5a60ca1119a22916bda18ed46c0c6acecf77b4d891f5b1b841854a81533d1ca0')
        self.assertEqual(PARENT_SHA, '9a7895ea8c149e755a875fb5b765d9a81559e7ff63b62bc7de565a08234c9c3a')
        self.assertEqual(SOURCE_SHA, 'a032a2b9057c150beb918e83e17ff8ad7e73a764fefcb4cf11cfd3049e1ef35d')

    def test_digest_matches_unmodified_bytes(self):
        f = self.root / 'bytes.bin'
        f.write_bytes(b'DG020 H22 exact-byte identity\x00\xff')
        self.assertEqual(digest(f), hashlib.sha256(f.read_bytes()).hexdigest())

    def test_valid_nested_archive_extracts(self):
        self.make_zip([('folder/record.json', b'{"original": true}')])
        unpack_safely(self.archive, self.dest)
        self.assertEqual((self.dest / 'folder/record.json').read_bytes(), b'{"original": true}')

    def test_valid_overlay_prefix_extracts_without_prefix(self):
        self.make_zip([('H22-overlay/source/record.json', b'locked')])
        unpack_safely(self.archive, self.dest, prefix='H22-overlay')
        self.assertEqual((self.dest / 'source/record.json').read_bytes(), b'locked')
        self.assertFalse((self.dest / 'H22-overlay').exists())

    def test_rejects_path_traversal_absolute_and_windows_separator(self):
        for name in ('../outside', 'foo/../../outside', '/tmp/outside', r'foo\outside'):
            with self.subTest(name=name):
                self.make_zip([(name, b'unsafe')])
                with self.assertRaises(RuntimeError):
                    unpack_safely(self.archive, self.dest)

    def test_rejects_duplicate_records(self):
        self.make_zip([('a/record', b'original'), ('a/record', b'replaced')])
        with self.assertRaisesRegex(RuntimeError, 'duplicate ZIP path'):
            unpack_safely(self.archive, self.dest)

    def test_rejects_symlinks(self):
        info = zipfile.ZipInfo('symbolic-link')
        info.create_system = 3
        info.external_attr = ((stat.S_IFLNK | 0o777) << 16)
        self.make_zip([(info, b'../outside')])
        with self.assertRaisesRegex(RuntimeError, 'symlink prohibited'):
            unpack_safely(self.archive, self.dest)

    def test_rejects_wrong_overlay_prefix(self):
        self.make_zip([('unexpected/record', b'unsafe')])
        with self.assertRaisesRegex(RuntimeError, 'unexpected overlay'):
            unpack_safely(self.archive, self.dest, prefix='H22-overlay')

    def test_modified_outer_bundle_cannot_match_pinned_hash(self):
        self.make_zip([('REPRO_H22.md', b'not the sealed original')])
        self.assertNotEqual(digest(self.archive), OUTER_SHA)


if __name__ == '__main__':
    unittest.main()
