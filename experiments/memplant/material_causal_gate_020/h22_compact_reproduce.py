#!/usr/bin/env python3
"""Fail-closed DG020 H22 compact genuine-neural replay; NO new acceptance data.

Use with the 2026-11-18 immutable standalone owner handoff ZIP. The archive is
NOT yet durably published on GitHub. This script proves nothing without it.
CPU only, writes solely into a temporary directory; no network or training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tempfile
import zipfile

EXPECTED_ARCHIVE_SHA256 = '47546540b96ae0800697e8a729c169e1191f111cec8c09f704c99a8843568667'
EXPECTED_H22_SOURCE_SHA256 = 'a032a2b9057c150beb918e83e17ff8ad7e73a764fefcb4cf11cfd3049e1ef35d'
EXPECTED_ORIGINAL_SEAL_ENTRIES = 31
EXPECTED_FILES = 52
MAX_UNCOMPRESSED_BYTES = 32_000_000
SEEDS = ('26011639', '26011640', '26011641')


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_archive(z: zipfile.ZipFile) -> dict[str, str]:
    members = z.infolist()
    assert len(members) == EXPECTED_FILES, 'archive member count changed'
    names = [i.filename for i in members]
    assert len(set(names)) == len(names), 'duplicate archive entries'
    total = 0
    for entry in members:
        path = PurePosixPath(entry.filename)
        assert not path.is_absolute() and '..' not in path.parts, 'unsafe path'
        assert str(path) == entry.filename and not entry.filename.startswith('.'), 'noncanonical path'
        assert not entry.is_dir(), 'unexpected directory entry'
        mode = entry.external_attr >> 16
        assert not stat.S_ISLNK(mode), 'symlink forbidden'
        assert entry.file_size <= 1_000_000, 'oversized individual entry'
        total += entry.file_size
        assert total <= MAX_UNCOMPRESSED_BYTES, 'oversized expanded archive'
    assert 'MANIFEST.sha256' in names, 'manifest missing'
    entries = {}
    for line in z.read('MANIFEST.sha256').decode().splitlines():
        h, sep, name = line.partition('  ')
        assert sep and len(h) == 64 and name not in entries, 'malformed manifest'
        entries[name] = h
    assert set(entries) == set(names) - {'MANIFEST.sha256'}, 'manifest coverage'
    for name, h in entries.items():
        assert digest(z.read(name)) == h, f'file digest mismatch: {name}'
    assert entries['source/h22_wet_coordinate_only_confirmation.py'] == EXPECTED_H22_SOURCE_SHA256
    receipts = [p for p in entries if p.startswith('source/h22-evidence/receipts-') and p.endswith('.jsonl')]
    assert len(receipts) == 30, 'must preserve 30 complete H22 cell receipts'
    for seed in SEEDS:
        assert f'source/h19-evidence/receiver-quadrature-{seed}.pt' in entries
        assert f'source/checkpoints/{seed}/receiver-source-blind.pt' in entries
    for src in (29002117, 29002119):
        for law in (0, 1):
            for rep in (0, 1):
                assert f'source/h21-evidence/fresh-donor-{src}-law{law}-rep{rep}.pt' in entries
    return entries


def reproduce(path: Path) -> dict:
    assert path.is_file(), f'missing archive: {path}'
    actual = digest(path.read_bytes())
    assert actual == EXPECTED_ARCHIVE_SHA256, f'archive SHA-256 mismatch: {actual}'
    with tempfile.TemporaryDirectory(prefix='dg020-h22-replay-') as scratch:
        with zipfile.ZipFile(path) as z:
            entries = check_archive(z)
            z.extractall(scratch)
        cwd = Path(scratch)
        env = {**os.environ, 'OMP_NUM_THREADS': '2', 'PYTHONHASHSEED': '0'}
        proc = subprocess.run(
            [sys.executable, 'h22_independent_receipt_replay.py'],
            cwd=cwd, env=env, capture_output=True, text=True, timeout=240, check=False,
        )
        if proc.returncode:
            raise RuntimeError(f'neural checkpoint replay failed ({proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}')
        report = json.loads((cwd / 'H22_INDEPENDENT_AUDIT.json').read_text())
        count = report['counts']
        assert report['independent_audit'] == report['gate_verdict'] == 'PASS'
        assert report['cells_passed'] == 30 and report['receipt_pairs'] == 4800
        assert report['manifest_hashes'] == EXPECTED_ORIGINAL_SEAL_ENTRIES
        assert count['divergent_both_correct'] == count['divergent_required_flips'] == 1920
        assert count['invariant_both_correct'] == 2877 and count['invariant_wrong_flips'] == 3
        assert count['foreign_source_changes'] == count['dry_only_changes'] == 0
        assert count['shuffled_divergent_both_correct'] == 496
        assert report['max_abs_neural_logit_drift'] <= 0.00005
        return {
            'archive_sha256': actual,
            'capsule_files_verified': len(entries),
            'H22_original_evidence_hashes': report['manifest_hashes'],
            'replayed_counterfactual_pairs': report['receipt_pairs'],
            'acceptance_cells_passed': report['cells_passed'],
            'wrong_invariant_flips_retained': count['invariant_wrong_flips'],
            'neural_logit_max_abs_drift': report['max_abs_neural_logit_drift'],
            'scientific_verdict_original_replayed': report['gate_verdict'],
            'status': 'PASS: exact old evidence replay, not new acceptance',
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path, help='H22 immutable compact ZIP; MUST match built-in SHA-256')
    args = parser.parse_args()
    print(json.dumps(reproduce(args.archive), indent=2))


if __name__ == '__main__':
    main()
