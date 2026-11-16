"""D-2 independent post-freeze material-robustness INPUTS ONLY.

Deterministic, offline 32x32 synthetic shapes; these fixtures and preregistered
limits are committed before policy evaluation. They do not modify D-COMMON,
model prompts or native World materials, and are not Stage A/B/C evidence.
"""
from __future__ import annotations

import hashlib
import json
import random
import struct

VERSION = 'd2.reinforcement-holdout-inputs/1'
WIDTH = HEIGHT = 32
BUDGET = 20
RADIUS = 3
SPEC = (
    ('crossbars-a', 'crossbars', 21001),
    ('crossbars-b', 'crossbars', 21002),
    ('staggered-decks-a', 'staggered_decks', 21003),
    ('staggered-decks-b', 'staggered_decks', 21004),
    ('broadcast-tree-a', 'broadcast_tree', 21005),
    ('broadcast-tree-b', 'broadcast_tree', 21006),
    ('staircase-a', 'staircase', 21007),
    ('staircase-b', 'staircase', 21008),
    ('gated-choke', 'gated_choke', 21009),
    ('perimeter-a', 'perimeter', 21010),
    ('perimeter-b', 'perimeter', 21011),
    ('disjoint-blocks', 'disjoint_blocks', 21012),
)


def line(material, a, b):
    """Draw two-component orthographic cardinal polyline (x first)."""
    x, y = a
    xx, yy = b
    for px in range(min(x, xx), max(x, xx) + 1):
        material.add((px, y))
    for py in range(min(y, yy), max(y, yy) + 1):
        material.add((xx, py))


def rect(material, x0, y0, x1, y1, fill=False):
    if fill:
        material.update((x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1))
    else:
        for a, b in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                     ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]:
            line(material, a, b)


def shape(family, seed):
    rng = random.Random(seed)
    ink = set()
    # Distinct deliberate structural families, not permutations of one grid.
    if family == 'crossbars':
        y = 14 + (seed % 3) - 1
        line(ink, (3, y), (28, y))
        for x in (7, 12, 18, 24):
            s = -1 if (x + seed) % 2 else 1
            line(ink, (x, y), (x, y + 6 * s))
            line(ink, (x - 2, y + 6 * s), (x + 2, y + 6 * s))
    elif family == 'staggered_decks':
        offset = seed % 3
        for i, (a, b) in enumerate(((3, 25), (7, 28), (3, 25), (7, 28))):
            y = 5 + i * 7
            line(ink, (a + offset, y), (b - offset, y))
        for i, x in enumerate((11, 23, 12)):
            line(ink, (x + offset % 2, 5 + 7*i), (x + offset % 2, 12 + 7*i))
    elif family == 'broadcast_tree':
        axis = 15 + (seed % 3) - 1
        line(ink, (axis, 3), (axis, 28))
        for j, y in enumerate((6, 11, 16, 21, 26)):
            side = 1 if (j + seed) % 2 else -1
            length = 7 + rng.randrange(5)
            line(ink, (axis, y), (axis + side * length, y))
            tip = axis + side * length
            line(ink, (tip, y - 2), (tip, y + 2))
    elif family == 'staircase':
        cursor = (3, 4 + seed % 3)
        for i in range(6):
            target = (min(28, cursor[0] + 4), cursor[1])
            line(ink, cursor, target)
            nxt = (target[0], min(28, target[1] + 3 + (i + seed) % 2))
            line(ink, target, nxt)
            cursor = nxt
        for x in (9, 18):
            for y in (10 + seed % 2, 20 + seed % 2):
                ink.add((x, y))
    elif family == 'gated_choke':
        line(ink, (3, 15), (29, 15))
        line(ink, (7, 8), (7, 23))
        line(ink, (24, 7), (24, 25))
        line(ink, (4, 8), (11, 8))
        line(ink, (21, 25), (28, 25))
    elif family == 'perimeter':
        offset = seed % 2
        rect(ink, 4 + offset, 4, 27 - offset, 27)
        rect(ink, 10 + offset, 10, 21 - offset, 21)
        line(ink, (15, 4), (15, 10))
        line(ink, (21 - offset, 16), (27 - offset, 16))
        for p in ((14, 4), (15, 10), (21 - offset, 16)):
            ink.discard(p)
    elif family == 'disjoint_blocks':
        for x, y in ((3, 3), (23, 3), (3, 23), (23, 23), (13, 13)):
            rect(ink, x, y, x + 3, y + 3, True)
    else:
        raise ValueError('unknown structural family')

    permitted = {(x, y) for y in range(HEIGHT) for x in range(WIDTH)}
    if family == 'gated_choke':
        permitted -= {(16, y) for y in range(HEIGHT) if y != 15}
        permitted -= {(x, y) for x in range(12, 20) for y in (9, 21)}
        permitted |= ink
    elif family == 'staggered_decks':
        # A few hard bulkheads that can affect local bypass planning.
        permitted -= {(x, y) for x in (5, 26) for y in range(10, 23)} - ink
    elif family == 'disjoint_blocks':
        # Fully isolated permitted footprints: legal early STOP negative control.
        permitted = set(ink)
    else:
        holes = {(x, y) for x in range(12, 20) for y in range(29, 31)}
        permitted -= holes - ink

    # 16-bit positive scalar, with seed-specific anisotropy. High values are
    # soft preferences; the material and permission mask define legality.
    guidance = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            phase = ((x * (7 + seed % 5) + y * (11 + seed % 7)) % 23)
            noise = random.Random(seed * 1009 + x * 37 + y).randrange(0, 101)
            guide = 1 + (phase * 2743 + noise * 19 + ((x * y + seed) % 13)) % 65535
            guidance.append(guide)
    assert len(guidance) == 1024 and min(guidance) > 0 and max(guidance) <= 65535
    assert ink <= permitted and all(0 <= x < WIDTH and 0 <= y < HEIGHT for x, y in permitted)
    return ink, permitted, guidance


def digest_points(points):
    return hashlib.sha256(json.dumps(sorted(points), separators=(',', ':')).encode()).hexdigest()


def digest_guide(guide):
    return hashlib.sha256(struct.pack('<1024H', *guide)).hexdigest()


def preregister():
    entries = []
    for case_id, family, seed in SPEC:
        initial, permitted, guide = shape(family, seed)
        entries.append(dict(id=case_id, family=family, seed=seed,
                            initial_cells=len(initial), permitted_cells=len(permitted),
                            initial_sha256=digest_points(initial),
                            permission_sha256=digest_points(permitted),
                            guidance_u16le_sha256=digest_guide(guide)))
    return dict(version=VERSION, cases=entries,
                guidance='positive u16le, host-soft, not a legality constraint',
                policies=('reinforcement','guidance_first','seeded_random_frontier'),
                metric='four-neighbour articulation count, all-material graph',
                secondary=('same-actual-cell-budget','components','2x2 solidity',
                           'below-soft-threshold','single-cell vulnerability'),
                budget_max_new_cells=BUDGET, radius=RADIUS, source='fresh synthetic shapes',
                native_receipts_required=False,
                evidence_class='POST_FREEZE_EXPLORATORY_SYNTHETIC_ONLY')


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


if __name__ == '__main__':
    import sys
    sys.stdout.buffer.write(canonical(preregister()))
