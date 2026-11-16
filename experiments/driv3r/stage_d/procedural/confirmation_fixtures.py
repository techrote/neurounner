"""PRE-REGISTERED synthetic geometry cohort, not native or model evidence.

D-2 confirmation fixtures are defined before evaluating any procedural policies.
No import of pareto_study, no outcome computation, no learned decisions.
Stable integer recurrence avoids ambient random-state or library-version drift.
"""
from __future__ import annotations

import hashlib
import json
import struct

WIDTH = HEIGHT = 32
VERSION = 'd2.synthetic-confirmation-fixtures/1'
CASE_SEEDS = (
    ('diagonal-fence', 1401),
    ('nested-courtyard', 1402),
    ('paired-vaults', 1403),
    ('staggered-shelves', 1404),
    ('misleading-scalar', 1405),
    ('spiral-islands', 1406),
    ('one-cell-choke', 1407),
    ('hard-separator', 1408),
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def cells_sha(cells):
    return digest(encode([list(p) for p in sorted(cells)]))


def draw(cells, x0, y0, x1, y1):
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            cells.add((x, y))


def randwords(seed):
    while True:
        seed = (1664525 * seed + 1013904223) & 0xffffffff
        yield seed


def fixture(name, seed):
    """Return (hard-allowed, initial material, positive u16le scalar).

    First-class hard mask and soft preference remain semantically disjoint.
    Every case is structurally different from the seven DEVELOPMENT cases.
    """
    if (name, seed) not in CASE_SEEDS:
        raise ValueError('unregistered case and seed: no search by outcomes')
    rng = randwords(seed)
    permitted = {(x, y) for y in range(32) for x in range(32)}
    material = set()
    favored = set()
    if name == 'diagonal-fence':
        draw(material, 3, 14, 5, 16)
        draw(material, 26, 14, 28, 16)
        # Staircase-shaped inclined forbidden barrier with a bottom aperture.
        permitted -= {(x, y) for y in range(3, 28) for x in (y-1, y, y+1) if y != 19}
        favored = {(x, y) for x in range(32) for y in range(32) if abs(x-y) <= 4}
    elif name == 'nested-courtyard':
        draw(material, 3, 13, 5, 17)
        draw(material, 23, 13, 25, 17)
        # Ring, not the old solid central obstacle. Only one gate in upper wall.
        permitted -= {(x, y) for x in range(9, 23) for y in range(8, 24)
                      if (x in (9, 22) or y in (8, 23)) and (x,y) != (16,8)}
        favored = {(x,y) for x in range(32) for y in range(32) if 7 <= y <= 10}
    elif name == 'paired-vaults':
        draw(material, 4, 7, 6, 9)
        draw(material, 24, 23, 26, 25)
        # Segmented vault boundaries with staggered gates and alternate route.
        permitted -= {(15, y) for y in range(1, 31) if y not in (5, 27)}
        permitted -= {(x, 16) for x in range(1, 31) if x not in (7, 24)}
        favored = {(x,y) for x in range(32) for y in range(32) if x >= 18 and y <= 11}
    elif name == 'staggered-shelves':
        draw(material, 2, 3, 4, 5)
        draw(material, 26, 26, 28, 28)
        for y in (9, 16, 23):
            permitted -= {(x, y) for x in range(32) if x not in (3 if y == 16 else 28,)}
        favored = {(x,y) for x in range(32) for y in range(32) if x >= 21}
    elif name == 'misleading-scalar':
        draw(material, 4, 11, 6, 13)
        draw(material, 24, 17, 26, 19)
        # Soft guide is concentrated inside a hard-forbidden strip.
        permitted -= {(x, y) for x in range(11, 21) for y in range(9, 23)
                      if (x,y) != (15,15)}
        favored = {(x,y) for x in range(11,21) for y in range(9,23)}
    elif name == 'spiral-islands':
        for x,y in ((4,4),(25,5),(25,25),(4,26),(14,15)):
            draw(material, x, y, x+1, y+1)
        # Discontinuous C-shaped corridor obstructions, not entire walls.
        permitted -= {(x, y) for x in range(9, 24) for y in (9, 22)}
        permitted -= {(9, y) for y in range(9,23) if y != 16}
        favored = {(x,y) for x in range(32) for y in range(32)
                   if min(abs(x-14),abs(y-15)) <= 2}
    elif name == 'one-cell-choke':
        draw(material, 4, 13, 6, 15)
        draw(material, 25, 16, 27, 18)
        permitted -= {(16, y) for y in range(32) if y != 3}
        favored = {(x,y) for x in range(32) for y in range(32) if y >= 11}
    elif name == 'hard-separator':
        draw(material, 4, 5, 6, 8)
        draw(material, 25, 23, 27, 26)
        permitted -= {(15, y) for y in range(32)}
        favored = {(x,y) for x in range(32) for y in range(32) if 11 <= y <= 20}
    assert material <= permitted
    # Seeded tiny scalar perturbations preserve the frozen guide topology.
    scalar = {(x, y): (55000 if (x,y) in favored else 8000) + next(rng) % 257
              for y in range(32) for x in range(32)}
    assert all(0 <= value <= 65535 for value in scalar.values())
    return permitted, material, scalar


def manifest():
    cases = []
    for name, seed in CASE_SEEDS:
        allowed, material, scalar = fixture(name, seed)
        u16 = struct.pack('<1024H', *(scalar[(x,y)] for y in range(32) for x in range(32)))
        cases.append({'id': name, 'seed': seed, 'width': 32, 'height': 32,
                      'hard_permitted_sha256': cells_sha(allowed),
                      'initial_material_sha256': cells_sha(material),
                      'guidance_u16le_sha256': digest(u16),
                      'permitted_count': len(allowed),
                      'initial_count': len(material)})
    return {'version': VERSION, 'qualification': 'PREDECLARED_SYNTHETIC_NON_NATIVE',
            'outcomes_unobserved_at_freeze': True,
            'policy_code': 'pareto_study.py @ Git blob 7fe2f1c49f765730354077703c1aae76c40a41b9',
            'budget_new_material_cells': 44, 'max_brush_up_hops': 120,
            'policy_arm_names': ['weighted-0', 'weighted-2', 'weighted-8', 'weighted-32', 'guidance-strict'],
            'baseline': 'equal actual edits adjacent high-guidance frontier',
            'primary_metric': '4-neighbour connected component count (lower better)',
            'secondaries': ['sub-threshold edits', 'actual new edits', 'brush-up route hops',
                            'failures', '4-neighbour material connectivity'],
            'negative_control': 'hard-separator MUST remain disconnected',
            'interpretation': 'no model/native/art claims; report every case and every failure',
            'cases': cases}


if __name__ == '__main__':
    print(encode(manifest()).decode())
