"""Independent D-2 synthetic material-fragility analysis (offline, stdlib).

Consumes archived synthetic confirmation PNGs and their original hash manifest.
Never upgrades unverified policy runs to qualified confirmation or asserts native art.
No imports from, or modifications to, the frozen D-2 policy/fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import zipfile
import zlib

VERSION = 'd2.synthetic-fragility/1'
INK = (232, 222, 199, 255)
EMPTY = (24, 33, 43, 255)
BLOCKED = (108, 65, 71, 255)
COLORS = {INK, EMPTY, BLOCKED}
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode() + b'\n'


def decode_synthetic_rgba_png(raw: bytes):
    """Reject non-RGBA8/non-filter-0 and corrupt/trailing PNG bytes, not general PNG."""
    if not raw.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('PNG signature absent')
    offset, idats, dimensions, ended = 8, [], None, False
    while offset < len(raw):
        if offset + 12 > len(raw):
            raise ValueError('truncated PNG chunk')
        size = struct.unpack_from('>I', raw, offset)[0]
        kind = raw[offset + 4:offset + 8]
        finish = offset + 12 + size
        if finish > len(raw):
            raise ValueError('invalid PNG chunk length')
        payload = raw[offset + 8:offset + 8 + size]
        expected_crc = struct.unpack_from('>I', raw, offset + 8 + size)[0]
        if zlib.crc32(kind + payload) & 0xffffffff != expected_crc:
            raise ValueError('PNG CRC mismatch')
        if kind == b'IHDR':
            if dimensions is not None or size != 13:
                raise ValueError('duplicate or malformed header')
            width, height, depth, color, compress, filter_method, interlace = struct.unpack('>IIBBBBB', payload)
            if (width, height) != (32, 32) or (depth, color, compress, filter_method, interlace) != (8, 6, 0, 0, 0):
                raise ValueError('not frozen synthetic 32x32 RGBA8 layout')
            dimensions = (width, height)
        elif kind == b'IDAT':
            idats.append(payload)
        elif kind == b'IEND':
            if size != 0 or finish != len(raw):
                raise ValueError('nonterminal IEND or trailing bytes')
            ended = True
        else:
            raise ValueError('unsupported PNG chunk')
        offset = finish
    if not (dimensions and idats and ended):
        raise ValueError('incomplete synthetic PNG')
    width, height = dimensions
    inflate = zlib.decompressobj()
    pixels = inflate.decompress(b''.join(idats), (width * 4 + 1) * height + 1)
    if inflate.unconsumed_tail or inflate.unused_data or not inflate.eof:
        raise ValueError('oversized or incomplete compressed pixels')
    pixels += inflate.flush()
    if len(pixels) != (width * 4 + 1) * height:
        raise ValueError('wrong raw PNG pixel count')
    ink, permitted = set(), set()
    stride = width * 4 + 1
    for y in range(height):
        row = pixels[y * stride:(y + 1) * stride]
        if row[0] != 0:
            raise ValueError('nonzero PNG row filter')
        for x in range(width):
            color = tuple(row[1 + 4 * x:5 + 4 * x])
            if color not in COLORS:
                raise ValueError('unknown synthetic pixel')
            if color != BLOCKED:
                permitted.add((x, y))
            if color == INK:
                ink.add((x, y))
    return ink, permitted


def components(material):
    pending = set(material)
    groups = []
    while pending:
        stack = [min(pending)]
        group = set(stack)
        pending.remove(stack[0])
        while stack:
            x, y = stack.pop()
            for dx, dy in DIRECTIONS:
                p = (x + dx, y + dy)
                if p in pending:
                    pending.remove(p)
                    group.add(p)
                    stack.append(p)
        groups.append(group)
    return groups


def measure(initial, final):
    if not initial <= final:
        raise ValueError('deposition analysis cannot silently erase initial cells')
    additions = final - initial
    start_components = len(components(initial))
    end_components = len(components(final))
    # A new cell whose removal fractures the final graph is fragile even if
    # the final artwork appears fully connected to a simple component metric.
    articulation_new = sorted(p for p in additions
                              if len(components(final - {p})) > end_components)
    leaves_new = sorted(p for p in additions if sum((p[0] + dx, p[1] + dy) in final
                              for dx, dy in DIRECTIONS) == 1)
    # 2x2 occupancy is a transparent robustness proxy, not walkability or art.
    solid_squares = sum(all((x + dx, y + dy) in final for dx in (0, 1) for dy in (0, 1))
                        for x in range(31) for y in range(31))
    return {'initial_components':start_components,'final_components':end_components,
            'component_reduction':start_components-end_components,
            'new_cells':len(additions),
            'new_articulation_cells':len(articulation_new),
            'new_articulation_fraction':(len(articulation_new) / len(additions) if additions else 0),
            'new_leaf_cells':len(leaves_new),'solid_2x2_squares':solid_squares,
            'articulation_points_new':[[x,y] for x,y in articulation_new]}


def run_archive(path: Path):
    """Read actual original 32x32 PNGs and matched policy/control views."""
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        raw = archive.read('confirmation-results.json')
        report = json.loads(raw)
        if report.get('version') != 'd2.synthetic-confirmation/1':
            raise ValueError('unrecognized experimental source report')
        rows = []
        for case in report['cases']:
            case_id = case['id']
            original_name = case_id + '/original.png'
            if original_name not in names:
                raise ValueError('missing original')
            original_bytes = archive.read(original_name)
            if sha(original_bytes) != case['original_png_sha256']:
                raise ValueError('original raster digest mismatch')
            initial, permitted = decode_synthetic_rgba_png(original_bytes)
            if len(initial) != case['registered']['initial_count'] or len(permitted) != case['registered']['permitted_count']:
                raise ValueError('registered initial raster disagreement')
            arms = {}
            for name, arm in sorted(case['policies'].items()):
                results = {}
                for suffix, key in (('.png','view_sha256'),('-control.png','control_view_sha256')):
                    filename = case_id + '/' + name + suffix
                    if filename not in names:
                        raise ValueError('missing trial raster')
                    contents = archive.read(filename)
                    if sha(contents) != arm[key]:
                        raise ValueError('trial raster digest mismatch')
                    ink, permission = decode_synthetic_rgba_png(contents)
                    if permission != permitted:
                        raise ValueError('trial changed hard permission raster')
                    result = measure(initial, ink)
                    expected = arm if suffix == '.png' else arm['matched_control']
                    if result['final_components'] != expected['components'] or result['new_cells'] != expected['new_cells']:
                        raise ValueError('stored material summary disagrees with pixels')
                    results['policy' if suffix == '.png' else 'matched_control'] = result
                arms[name] = results
            rows.append({'id':case_id,'arms':arms})
    return {'version':VERSION,'input_archive_sha256':sha(Path(path).read_bytes()),
            'source_qualification':report['qualification'],
            'source_policy_verified':report.get('policy_source_verified',False),
            'qualification':'INDEPENDENT_ARCHIVED_SYNTHETIC_MORPHOLOGY_ONLY',
            'not_native_or_visual_quality':True,'cases':rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    options = parser.parse_args()
    data = encode(run_archive(options.archive))
    options.out.parent.mkdir(parents=True,exist_ok=True)
    options.out.write_bytes(data)
    print(json.dumps({'cases':len(json.loads(data)['cases']),'sha256':sha(data),
                      'qualification':'INDEPENDENT_ARCHIVED_SYNTHETIC_MORPHOLOGY_ONLY'},sort_keys=True))


if __name__ == '__main__':
    main()
