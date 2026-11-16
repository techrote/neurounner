"""D-2 post-freeze held-out robustness comparison; offline, synthetic, stdlib.

Input source and manifest were frozen before this comparison was run. This
compares the exact committed reinforcement policy against two controls at equal
actual cell count, and never claims host/native paint or artistic usefulness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import zlib

from holdout_fixtures import (BUDGET, RADIUS, SPEC, canonical, digest_guide,
                              digest_points, preregister, shape)
from fragility_study import INK, EMPTY, BLOCKED, components, encode, measure
from reinforcement_study import articulations, trial

VERSION = 'd2.post-freeze-heldout-robustness/1'
INPUT_FREEZE_COMMIT = '65d4b8e477cccedfe8b9a25f0617d20ead0c820d'
FROZEN_MANIFEST_SHA256 = '7160002b2cee9f9d97210acff17afa11326f8dcb6fdaa8bfe78cd13d8c0d109d'
EXACT_SOURCE_GIT_BLOBS = {
    'holdout_fixtures.py': 'd2a3dec9df4984c2cd139e0020150d76bca4a449',
    'reinforcement_study.py': 'fb4a1f206c3b44a73ce3f6b084373d912b9eeb38',
    'fragility_study.py': 'cb1fdf1c5e4fdea1763351aa081de03f31e89cba',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git_blob(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


def verify_sources(directory):
    for name, expected in EXACT_SOURCE_GIT_BLOBS.items():
        if git_blob((directory / name).read_bytes()) != expected:
            raise ValueError('Git blob mismatch: ' + name)


def png(material, permitted):
    """Original PNG view, identical deterministic RGBA palette for all arms."""
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    pixels = bytearray()
    for y in range(32):
        pixels.append(0)
        for x in range(32):
            p = (x, y)
            pixels.extend(INK if p in material else (EMPTY if p in permitted else BLOCKED))
    header = struct.pack('>IIBBBBB', 32, 32, 8, 6, 0, 0, 0)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header) + chunk(b'IDAT', zlib.compress(bytes(pixels), 9)) + chunk(b'IEND', b'')


def run(manifest, directory, out=None):
    verify_sources(directory)
    raw = Path(manifest).read_bytes()
    if sha(raw) != FROZEN_MANIFEST_SHA256:
        raise ValueError('frozen input manifest byte mismatch')
    frozen = json.loads(raw)
    if canonical(preregister()) != raw:
        raise ValueError('fixture generator / preregistration mismatch')
    if frozen['budget_max_new_cells'] != BUDGET or frozen['radius'] != RADIUS:
        raise ValueError('unregistered budget change')
    rows = []
    for registered in frozen['cases']:
        cid, family, seed = registered['id'], registered['family'], registered['seed']
        initial, permitted, guide = shape(family, seed)
        if registered['initial_sha256'] != digest_points(initial) or \
           registered['permission_sha256'] != digest_points(permitted) or \
           registered['guidance_u16le_sha256'] != digest_guide(guide):
            raise ValueError('geometry/hash drift ' + cid)
        observed = trial(initial, initial, permitted, guide, seed, BUDGET, RADIUS)
        n = observed['policies']['reinforcement']['added_after_baseline']
        assert all(v['added_after_baseline'] == n for v in observed['policies'].values())
        assert all(all(tuple(p) in permitted for p in v['added_pixels'])
                   for v in observed['policies'].values())
        v = {'id': cid, 'family':family,'seed':seed,'initial_components':len(components(initial)),
             'initial_articulations':len(articulations(initial)),
             'initial_material_sha256':registered['initial_sha256'],
             'new_cell_budget_limit':BUDGET,'actual_new_cells_per_arm':n,
             'terminal':observed['terminal'],
             'synthetic_plans':observed['plans'],
             'arms':observed['policies']}
        if out:
            base = Path(out) / 'views' / cid
            base.mkdir(parents=True, exist_ok=True)
            original_png = png(initial, permitted)
            (base / 'original.png').write_bytes(original_png)
            v['original_png_sha256'] = sha(original_png)
            for arm, arm_result in v['arms'].items():
                final = initial | {tuple(p) for p in arm_result['added_pixels']}
                image = png(final, permitted)
                (base / (arm+'.png')).write_bytes(image)
                arm_result['png_sha256'] = sha(image)
                assert len(articulations(final)) == arm_result['articulations_after']
                assert measure(initial, final)['final_components'] == arm_result['final_components']
        rows.append(v)
    return {'version':VERSION,'qualification':'POST_FREEZE_EXPLORATORY_SYNTHETIC_ONLY',
            'input_freeze_commit':INPUT_FREEZE_COMMIT,
            'prereg_manifest_sha256':FROZEN_MANIFEST_SHA256,
            'sources_exact_git_blobs':EXACT_SOURCE_GIT_BLOBS,
            'model_inference':False,'native_material_receipts':0,
            'no_native_brush_transport_verified':True,
            'no_art_or_physical_strength_acceptance':True,
            'cases':rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    result = run(a.manifest, Path(__file__).parent, a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    data = encode(result)
    (a.out / 'holdout-results.json').write_bytes(data)
    print(json.dumps({'cases':len(result['cases']), 'sha256':sha(data),
                      'qualification':result['qualification']}))

if __name__ == '__main__': main()
