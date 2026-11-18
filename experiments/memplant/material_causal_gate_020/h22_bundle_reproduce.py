#!/usr/bin/env python3
"""DG020 H22 read-only, fail-closed reproduction from original owner bundle.

Usage:
  python h22_bundle_reproduce.py /path/to/DG020-H22-wet-only-causal-selectivity-PASS-20261117.zip --contract-auditor PATH

Original archives and H22 scientific receipts remain immutable. Extract to a
fresh temporary directory. This is replay of an EXPOSED H22 cohort, NOT an
independent prospective confirmation.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import zipfile

OUTER_SHA = '5a60ca1119a22916bda18ed46c0c6acecf77b4d891f5b1b841854a81533d1ca0'
PARENT_SHA = '9a7895ea8c149e755a875fb5b765d9a81559e7ff63b62bc7de565a08234c9c3a'
SOURCE_SHA = 'a032a2b9057c150beb918e83e17ff8ad7e73a764fefcb4cf11cfd3049e1ef35d'
PARENT_NAME = 'DG020-H21R-fresh-donor-seed-confirmation-evidence-20261117.zip'
EXPECTED = {'receipt_pairs': 4800, 'cells_passed': 30, 'manifest_hashes': 31}
EXPECTED_COUNTS = {
    'divergent_both_correct': 1920,
    'divergent_required_flips': 1920,
    'invariant_both_correct': 2877,
    'invariant_wrong_flips': 3,
    'foreign_source_changes': 0,
    'dry_only_changes': 0,
    'shuffled_divergent_both_correct': 496,
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def demand(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def unpack_safely(zfile, dest, prefix=''):
    """Reject paths, links and duplicate entries before any extraction."""
    dest = Path(dest).resolve()
    with zipfile.ZipFile(zfile) as z:
        seen = set()
        for item in z.infolist():
            name = PurePosixPath(item.filename)
            demand(not name.is_absolute() and '..' not in name.parts and '\\' not in item.filename,
                   'unsafe ZIP path')
            mode = (item.external_attr >> 16) & 0o170000
            demand(mode != 0o120000, 'archive symlink prohibited')
            demand(name.parts and name.parts[0] not in ('', '.'), 'empty ZIP path')
            if prefix:
                demand(name.parts[0] == prefix, 'unexpected overlay file')
                name = PurePosixPath(*name.parts[1:])
                if not name.parts:
                    continue
            rel = str(name)
            demand(rel not in seen, 'duplicate ZIP path: ' + rel)
            seen.add(rel)
            target = (dest / rel).resolve()
            demand(target == dest or dest in target.parents, 'path escaped scratch directory')
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(item) as inp, target.open('wb') as out:
                    shutil.copyfileobj(inp, out)


def run(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    demand(result.returncode == 0, f'{cmd[1]} failed ({result.returncode}):\n'
           + result.stdout[-3500:] + '\n' + result.stderr[-3500:])
    return result.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('bundle', type=Path)
    ap.add_argument('--contract-auditor', type=Path, required=True,
                    help='Repository h22_receipt_contract_audit.py')
    args = ap.parse_args()
    bundle = args.bundle.resolve()
    auditor = args.contract_auditor.resolve()
    demand(bundle.is_file() and auditor.is_file(), 'bundle and auditor required')
    demand(digest(bundle) == OUTER_SHA, 'outer bundle SHA-256 mismatch')

    with tempfile.TemporaryDirectory(prefix='dg020-h22-verified-') as t:
        scratch = Path(t)
        outer = scratch / 'outer'
        outer.mkdir()
        unpack_safely(bundle, outer)
        parent = outer / PARENT_NAME
        demand(parent.is_file() and digest(parent) == PARENT_SHA,
               'nested H21R archive missing or SHA-256 mismatch')
        extracted = scratch / 'replay'
        extracted.mkdir()
        unpack_safely(parent, extracted)
        overlay_zip = scratch / 'h22-overlay.zip'
        with zipfile.ZipFile(overlay_zip, 'w') as z:
            for f in sorted((outer / 'H22-overlay').rglob('*')):
                if f.is_file():
                    z.write(f, f.relative_to(outer).as_posix())
        unpack_safely(overlay_zip, extracted, prefix='H22-overlay')
        evidence = extracted / 'source' / 'h22-evidence'
        demand(digest(extracted / 'source' / 'h22_wet_coordinate_only_confirmation.py')
               == SOURCE_SHA, 'preregistered source SHA-256 mismatch')
        demand(len(list(evidence.glob('receipts-*.jsonl'))) == 30,
               'H22 must have exactly 30 sealed receipt cells')

        receipt_output = run([sys.executable, str(auditor), str(evidence)], extracted)
        demand('PASS' in receipt_output, 'independent receipt contract did not PASS')
        checkpoint_output = run([sys.executable, 'h22_independent_receipt_replay.py'], extracted)
        report = json.loads((extracted / 'H22_INDEPENDENT_AUDIT.json').read_text())
        demand(report['independent_audit'] == 'PASS' and report['gate_verdict'] == 'PASS',
               'checkpoint replay failed')
        for key, expected in EXPECTED.items():
            demand(report[key] == expected, f'{key} != {expected}')
        for key, expected in EXPECTED_COUNTS.items():
            demand(report['counts'][key] == expected, f'{key} != {expected}')
        demand(report['max_abs_neural_logit_drift'] == 0.0,
               'replayed logits differ from original CPU archive')
        demand(len(report['wrong_or_sensitive_cases']) == 3,
               'original negatives missing or changed')
        print(json.dumps({
            'reproduction': 'PASS', 'cohort': 'H22 previously exposed',
            'source_sha256': SOURCE_SHA,
            'checkpoint_replay': report,
            'receipt_audit_excerpt': receipt_output[-500:],
            'checkpoint_output_excerpt': checkpoint_output[-300:]
        }, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('H22 BUNDLE REPRODUCTION FAIL:', exc, file=sys.stderr)
        sys.exit(1)
