#!/usr/bin/env python3
"""SPROCKET support: original-pixel W0 family-sensitivity counterbaseline.

Reads fixed source-authenticated observed-v2 artifact, never trains a model or
modifies shared W0, primary SPROCKET, native host, FINAL or frozen SMOL.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import zlib
import zipfile

SOURCE_ZIP_SHA256 = 'db1372116a21201d1b9e95ecebae4dce15fd26b018efa2dca4abfaeec1716e83'
SOURCE_RUN = 43715197825
SOURCE_ARTIFACT = 14162755501
FAMILIES = ('greebles', 'pipework', 'architecture')
POLICIES = ('random', 'wall', 'fidget', 'groove', 'contour', 'corridor')
SEEDS = (211, 223)


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def png_pixels(blob):
    """Strict stdlib decoder for original W0 384x384 RGB, filter-0 PNG only."""
    require(blob[:8] == b'\x89PNG\r\n\x1a\n', 'PNG magic')
    offset, parts, ihdr, chunks = 8, [], None, []
    while offset < len(blob):
        require(offset + 12 <= len(blob), 'PNG truncated header')
        n = struct.unpack('>I', blob[offset:offset+4])[0]
        kind = blob[offset+4:offset+8]
        require(n <= 8_000_000 and offset+12+n <= len(blob), 'PNG oversize/truncated chunk')
        data = blob[offset+8:offset+8+n]
        expected = struct.unpack('>I', blob[offset+8+n:offset+12+n])[0]
        require(zlib.crc32(kind+data) & 0xffffffff == expected, 'PNG CRC')
        chunks.append(kind)
        if kind == b'IHDR':
            ihdr = struct.unpack('>IIBBBBB', data)
        elif kind == b'IDAT':
            parts.append(data)
        else:
            require(kind == b'IEND' and n == 0, 'unexpected PNG chunk')
        offset += 12+n
    require(offset == len(blob) and chunks == [b'IHDR', b'IDAT', b'IEND'], 'PNG chunks')
    require(ihdr == (384, 384, 8, 2, 0, 0, 0), 'PNG W0 image mode/size')
    raw = zlib.decompress(b''.join(parts))
    stride = 384*3
    require(len(raw) == 384*(stride+1), 'PNG decompressed size')
    rows = []
    for y in range(384):
        start = y*(stride+1)
        require(raw[start] == 0, 'PNG unsupported filter')
        rows.append(raw[start+1:start+1+stride])
    return rows


def png_chunk(kind, data):
    return (struct.pack('>I', len(data)) + kind + data +
            struct.pack('>I', zlib.crc32(kind+data) & 0xffffffff))


def sheet(originals):
    """Horizontal triple of exact original RGB raster pixels, no interpolation."""
    rowsets = [png_pixels(p) for p in originals]
    w, h = 384*3, 384
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for rowset in rowsets:
            raw.extend(rowset[y])
    return (b'\x89PNG\r\n\x1a\n' +
            png_chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) +
            png_chunk(b'IDAT', zlib.compress(raw, 6)) + png_chunk(b'IEND', b''))


def extent(trajectory):
    xs, ys = zip(*trajectory)
    return {'bbox_width': max(xs)-min(xs)+1, 'bbox_height': max(ys)-min(ys)+1,
            'max_manhattan_from_start': max(abs(x-xs[0])+abs(y-ys[0]) for x,y in trajectory)}


def compare(a, b):
    path_a = [tuple(p) for p in a['trajectory']]
    path_b = [tuple(p) for p in b['trajectory']]
    cells_a, cells_b = set(path_a), set(path_b)
    acts_a = [e['raw_choice'] for e in a['events']]
    acts_b = [e['raw_choice'] for e in b['events']]
    n = min(len(acts_a), len(acts_b))
    differences = sum(x != y for x,y in zip(acts_a, acts_b)) + abs(len(acts_a)-len(acts_b))
    first = next((i for i in range(n) if acts_a[i] != acts_b[i]), None)
    if first is None and len(acts_a) != len(acts_b):
        first = n
    return {'footprint_iou': round(len(cells_a & cells_b)/len(cells_a | cells_b), 6),
            'different_raw_actions': differences, 'compared_action_slots': max(len(acts_a), len(acts_b)),
            'first_action_divergence_tick': first,
            'same_raw_action_sequence': acts_a == acts_b,
            'same_trajectory': path_a == path_b,
            'unique_cells': [len(cells_a),len(cells_b)],
            'extents': [extent(path_a),extent(path_b)]}


def analyze(archive, output):
    archive_bytes = Path(archive).read_bytes()
    require(sha(archive_bytes) == SOURCE_ZIP_SHA256, 'immutable original gallery archive SHA-256')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None, 'archive ZIP CRC')
        checksum_text = z.read('SHA256SUMS.txt').decode()
        for row in checksum_text.splitlines():
            h, name = row.split(None, 1)
            name = name.lstrip('*').lstrip('./')
            require(h == sha(z.read(name)), 'internal checksum mismatch: '+name)
        manifest_bytes = z.read('gallery/manifest.json')
        manifest = json.loads(manifest_bytes)
        require(len(manifest['entries']) == 36, 'expected 36 source W0 gallery entries')
        data = {}
        pngs = {}
        for e in manifest['entries']:
            ident = e['id']
            key = (e['world_seed'],e['policy'],e['family'])
            require(e['world_seed'] in SEEDS and e['policy'] in POLICIES and e['family'] in FAMILIES,
                    'unexpected world/policy/family')
            require(key not in data and ident == f'{key[2]}-{key[1]}-{key[0]}', 'duplicate identity')
            j, p = z.read('gallery/'+ident+'.json'), z.read('gallery/'+ident+'.png')
            require(sha(j) == e['sha256_json'] and sha(p) == e['sha256_png'], 'original member SHA')
            record = json.loads(j)
            require(record['seed'] == key[0] and record['family'] == key[2] and record['size'] == 96
                    and record['step_budget'] == 96 and len(record['events']) == record['steps_executed'],
                    'record world/step identity')
            require(record['metrics']['invalid_raw_choices'] == 0, 'unexpected host substitution')
            png_pixels(p)
            data[key], pngs[key] = record, p
    require(len(data) == 36, 'incomplete exact cohort')
    comparisons, contacts = [], []
    for seed in SEEDS:
        for policy in POLICIES:
            raw = [pngs[(seed,policy,family)] for family in FAMILIES]
            contact_name = f'{policy}-{seed}-greebles-pipework-architecture.png'
            contact = sheet(raw)
            (output/contact_name).write_bytes(contact)
            contacts.append({'name':contact_name,'sha256':sha(contact)})
            for a_idx in range(len(FAMILIES)):
                for b_idx in range(a_idx+1,len(FAMILIES)):
                    aa, bb = FAMILIES[a_idx], FAMILIES[b_idx]
                    result = compare(data[(seed,policy,aa)],data[(seed,policy,bb)])
                    comparisons.append({'seed':seed,'policy':policy,'families':[aa,bb],**result})
                    if policy in ('random','wall'):
                        require(result['same_raw_action_sequence'] and result['same_trajectory']
                                and raw[a_idx] == raw[b_idx], 'family-invariant W0 negative control changed')
    summary = {}
    for policy in POLICIES:
        rows = [x for x in comparisons if x['policy'] == policy]
        summary[policy] = {
            'comparisons':len(rows),
            'same_trajectory_family_pairs':sum(x['same_trajectory'] for x in rows),
            'mean_footprint_iou':round(sum(x['footprint_iou'] for x in rows)/len(rows),6),
            'mean_raw_action_disagreement_fraction':round(sum(
                x['different_raw_actions']/x['compared_action_slots'] for x in rows)/len(rows),6),
            'first_divergence_ticks':[x['first_action_divergence_tick'] for x in rows],
        }
    report = {'schema':'cybersand.worm.support-sprocket.family-sensitivity/1',
              'kind':'original W0 Python nonmodel gallery comparison, not native/model scores',
              'source_run':SOURCE_RUN,'source_artifact':SOURCE_ARTIFACT,
              'source_zip_sha256':SOURCE_ZIP_SHA256,'source_gallery_manifest_sha256':sha(manifest_bytes),
              'independent_obstacle_worlds':list(SEEDS),'families':list(FAMILIES),
              'policies':list(POLICIES),'trials':36,'paired_family_comparisons':len(comparisons),
              'original_pixel_contacts':contacts,'summary':summary,'pair_details':comparisons,
              'interpretation':'Family-labelled views of the same obstacle seed are correlated; raw decisions, footprint overlap and span cannot by themselves establish artistic usefulness. Random and wall are strict theme-invariant controls.',
              'model_optimizer_updates':0,'native_execution':False,'sealed_final_opened':False}
    (output/'family-sensitivity-report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report


def self_test():
    tiny_a = {'trajectory':[[0,0],[1,0],[1,1]], 'events':[{'raw_choice':'STRAIGHT'},{'raw_choice':'RIGHT'}]}
    tiny_b = {'trajectory':[[0,0],[1,0],[2,0]], 'events':[{'raw_choice':'STRAIGHT'},{'raw_choice':'STRAIGHT'}]}
    assert compare(tiny_a,tiny_a)['footprint_iou'] == 1
    assert compare(tiny_a,tiny_b)['footprint_iou'] == .5
    assert compare(tiny_a,tiny_b)['first_action_divergence_tick'] == 1
    assert compare(tiny_a,tiny_b)['different_raw_actions'] == 1


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    self_test()
    report=analyze(args.archive,args.output)
    print('PASS',report['trials'],'original W0 paired JSON/PNG receipts;',
          len(report['original_pixel_contacts']),'unmodified-pixel family comparison PNGs;',
          report['paired_family_comparisons'],'theme pairs')
    for policy,summary in report['summary'].items():
        print(policy,json.dumps(summary,sort_keys=True))


if __name__=='__main__':
    main()
