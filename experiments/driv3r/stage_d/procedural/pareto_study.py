"""D-2 disjoint-topology guided connectivity sweep. SYNTHETIC DEVELOPMENT ONLY.

No native execution, material history, learned decisions, inference or permission
inference from guidance. All counterfactual arms share immutable source geometry.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import heapq
import json
from pathlib import Path
import struct
import zlib

VERSION = 'd2.synthetic-topology-pareto/1'
WIDTH = HEIGHT = 32
THRESHOLD = 48000
DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
INK, EMPTY, BLOCKED = (232, 222, 199, 255), (24, 33, 43, 255), (108, 65, 71, 255)
POLICIES = ('weighted-0', 'weighted-2', 'weighted-8', 'weighted-32', 'guidance-strict')
CASE_NAMES = ('strait', 'offset-gate', 'archipelago', 'central-obstacle', 'sealed-halves', 'river-gate', 'guide-detour')


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def cell_sha(cells):
    return sha(encoded([list(p) for p in sorted(cells)]))


def components(cells):
    unseen = set(cells)
    result = []
    while unseen:
        start = min(unseen)
        component, stack = {start}, [start]
        unseen.remove(start)
        while stack:
            x, y = stack.pop()
            for dx, dy in DIRECTIONS:
                p = (x + dx, y + dy)
                if p in unseen:
                    unseen.remove(p)
                    component.add(p)
                    stack.append(p)
        result.append(component)
    return sorted(result, key=lambda c: (-len(c), min(c)))


def rectangle(cells, x1, y1, x2, y2):
    for y in range(y1, y2 + 1):
        for x in range(x1, x2 + 1):
            cells.add((x, y))


def fixture(name):
    """Six predeclared structural families; not translations/seeds of Moore.

    Scalar is a *soft preference* deliberately distinct from hard allowed mask.
    """
    if name not in CASE_NAMES:
        raise ValueError('unregistered geometry family')
    allowed = {(x, y) for y in range(HEIGHT) for x in range(WIDTH)}
    material = set()
    if name == 'strait':
        rectangle(material, 4, 13, 5, 17)
        rectangle(material, 24, 13, 25, 17)
    elif name == 'offset-gate':
        rectangle(material, 3, 5, 5, 9)
        rectangle(material, 25, 23, 27, 26)
        allowed -= {(x, 15) for x in range(4, WIDTH)}
    elif name == 'archipelago':
        for x, y in ((4, 4), (25, 6), (6, 24), (25, 25)):
            rectangle(material, x, y, x + 1, y + 1)
    elif name == 'central-obstacle':
        for x, y in ((3, 15), (25, 14), (13, 3), (14, 25)):
            rectangle(material, x, y, x + 2, y + 2)
        allowed -= {(x, y) for y in range(11, 22) for x in range(11, 22)}
    elif name == 'sealed-halves':
        rectangle(material, 5, 12, 7, 17)
        rectangle(material, 25, 12, 27, 17)
        allowed -= {(15, y) for y in range(HEIGHT)}
    elif name == 'river-gate':
        rectangle(material, 3, 5, 6, 9)
        rectangle(material, 24, 5, 27, 9)
        allowed -= {(15 + (y // 5) % 3, y) for y in range(HEIGHT) if y != 26}
    elif name == 'guide-detour':
        rectangle(material, 5, 14, 6, 16)
        rectangle(material, 25, 14, 26, 16)
    assert material <= allowed
    assert len(components(material)) > 1
    scalar = {}
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if name == 'strait':
                wanted = abs(y - 15) <= 1
            elif name == 'offset-gate':
                wanted = abs(y - (7 + (x * 18 // 31))) <= 2
            elif name == 'archipelago':
                wanted = (x < 10 or x > 22) and (y < 11 or y > 21)
            elif name == 'central-obstacle':
                wanted = abs(x - y) <= 2 or abs(x + y - 31) <= 2
            elif name == 'sealed-halves':
                wanted = abs(y - 15) <= 2
            elif name == 'guide-detour':
                # The longer upper detour is preferred; the short direct line is low.
                wanted = y == 6 or ((x in (5, 6, 25, 26)) and 6 <= y <= 16)
            else:
                wanted = abs(y - 7) <= 2
            scalar[(x, y)] = 57000 if wanted else 17000 + (x * 37 + y * 13) % 2000
    return allowed, material, scalar


def brush_up_route(start, end, permitted, max_hops):
    if start not in permitted or end not in permitted or max_hops < 0:
        return None
    pending = deque([start]); parent = {start: None}; depth = {start: 0}
    while pending:
        point = pending.popleft()
        if point == end:
            path = [point]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return path[::-1]
        if depth[point] == max_hops:
            continue
        for dx, dy in DIRECTIONS:
            nxt = (point[0] + dx, point[1] + dy)
            if nxt in permitted and nxt not in parent:
                parent[nxt], depth[nxt] = point, depth[point] + 1
                pending.append(nxt)
    return None


def weighted_link(material, permitted, scalar, budget, weight, strict=False):
    """Deterministic multi-source Dijkstra; new cells pay 8 + 8*weight if low.

    Existing material costs zero, guide values never alter native legality.
    Dijkstra state includes edit cost to avoid pruning feasible bounded routes.
    """
    if budget < 0 or weight < 0:
        raise ValueError('negative budget/weight')
    groups = components(material)
    if len(groups) <= 1:
        return None
    best = None
    for source in groups:
        other = material - source
        frontier = []
        parents, costs = {}, {}
        for point in sorted(source):
            key = (point, 0)
            costs[key] = (0, 0, 0)
            parents[key] = None
            heapq.heappush(frontier, (0, 0, 0, point, 0))
        while frontier:
            score, low, hops, point, edits = heapq.heappop(frontier)
            key = (point, edits)
            if costs.get(key) != (score, low, hops):
                continue
            if best is not None and (score, edits, low, hops) > best[0][:4]:
                continue
            if point in other:
                route = [point]
                while parents[key] is not None:
                    key = parents[key]
                    route.append(key[0])
                route.reverse()
                rank = (score, edits, low, hops, tuple(route))
                if best is None or rank < best[0]:
                    best = (rank, route)
                break
            for dx, dy in DIRECTIONS:
                nxt = (point[0] + dx, point[1] + dy)
                if nxt not in permitted:
                    continue
                fresh = nxt not in material
                below = fresh and scalar[nxt] < THRESHOLD
                if strict and below:
                    continue
                new_edits = edits + int(fresh)
                if new_edits > budget:
                    continue
                next_score = score + (8 * (1 + weight * int(below)) if fresh else 0)
                next_cost = (next_score, low + int(below), hops + 1)
                next_key = (nxt, new_edits)
                if next_key not in costs or next_cost < costs[next_key]:
                    costs[next_key], parents[next_key] = next_cost, key
                    heapq.heappush(frontier, (*next_cost, nxt, new_edits))
    return None if best is None else best[1]


def run_bridge(allowed, initial, scalar, budget=44, weight=0, strict=False, max_hops=120):
    assert initial <= allowed and budget >= 0
    material = set(initial)
    cursor = min(initial)
    records = []; used = 0; travel = 0
    terminal = 'ALREADY_CONNECTED' if len(components(material)) == 1 else 'CONNECTED'
    while len(components(material)) > 1:
        route = weighted_link(material, allowed, scalar, budget - used, weight, strict)
        if route is None:
            terminal = 'NO_BUDGETED_PERMITTED_LINK'
            break
        move = brush_up_route(cursor, route[0], allowed, max_hops)
        if move is None:
            terminal = 'NO_BRUSH_UP_ROUTE'
            break
        before = cell_sha(material)
        records.append({'kind': 'BRUSH_UP_ROUTE', 'from': list(cursor), 'to': list(route[0]),
                        'path': [list(p) for p in move], 'before': before, 'after': before,
                        'native_receipt': None, 'synthetic_only': True})
        travel += len(move) - 1
        records.append({'kind': 'BRIDGE_PATH_SYNTHETIC', 'path': [list(p) for p in route],
                        'before': before, 'after': before, 'native_receipt': None})
        for point in route:
            if point not in material:
                assert point in allowed and used < budget
                previous = cell_sha(material)
                material.add(point); used += 1
                records.append({'kind': 'SYNTHETIC_BRIDGE_PAINT', 'point': list(point),
                                'below_soft_threshold': scalar[point] < THRESHOLD,
                                'before': previous, 'after': cell_sha(material),
                                'native_receipt': None})
        cursor = route[-1]
    sizes = [len(c) for c in components(material)]
    low = sum(scalar[tuple(e['point'])] < THRESHOLD for e in records if e['kind'] == 'SYNTHETIC_BRIDGE_PAINT')
    return {'terminal': terminal, 'material_cells': len(material), 'new_cells': used,
            'components': len(sizes), 'largest_component_fraction': round(max(sizes) / len(material), 6),
            'below_soft_threshold_cells': low, 'brush_up_hops': travel,
            'material_sha256': cell_sha(material), 'events_sha256': sha(encoded(records)),
            'native_replay_verified': False, 'native_receipts': 0,
            'events': records}, material


def guidance_greedy(allowed, initial, scalar, add_count):
    """Strong non-bridging baseline: best-guidance adjacent frontier, equal edits."""
    material = set(initial)
    for _ in range(add_count):
        candidates = [( -scalar[p], p) for p in sorted(allowed - material)
                      if any((p[0] + dx, p[1] + dy) in material for dx, dy in DIRECTIONS)]
        if not candidates:
            break
        material.add(min(candidates)[1])
    added = material - initial
    return {'material_cells': len(material), 'new_cells': len(added),
            'components': len(components(material)),
            'below_soft_threshold_cells': sum(scalar[p] < THRESHOLD for p in added),
            'material_sha256': cell_sha(material)}, material


def png(cells, allowed):
    rows = []
    for y in range(HEIGHT):
        rows.append(b'\x00' + bytes(v for x in range(WIDTH) for v in
                                (INK if (x, y) in cells else EMPTY if (x, y) in allowed else BLOCKED)))
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', WIDTH, HEIGHT, 8, 6, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(b''.join(rows), 9)) + chunk(b'IEND', b''))


def dominates(a, b):
    # Minimise components, soft-guidance threshold violations, *and* new edit count.
    keys = ('components', 'below_soft_threshold_cells', 'new_cells')
    return all(a[k] <= b[k] for k in keys) and any(a[k] < b[k] for k in keys)


def study(output, budget=44):
    if budget < 1 or budget > 128:
        raise ValueError('bounded budget 1..128 required')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    cases = []
    for name in CASE_NAMES:
        allowed, initial, scalar = fixture(name)
        folder = output / name; folder.mkdir(exist_ok=True)
        guide = struct.pack('<' + 'H' * (WIDTH * HEIGHT), *(scalar[(x, y)] for y in range(HEIGHT) for x in range(WIDTH)))
        (folder / 'guidance.u16le').write_bytes(guide)
        baseline = png(initial, allowed); (folder / 'original.png').write_bytes(baseline)
        policies = {}
        for policy in POLICIES:
            strict = policy == 'guidance-strict'
            weight = 0 if strict else int(policy.split('-')[1])
            result, cells = run_bridge(allowed, initial, scalar, budget, weight, strict)
            raw = png(cells, allowed); (folder / (policy + '.png')).write_bytes(raw)
            controls, ctrl_cells = guidance_greedy(allowed, initial, scalar, result['new_cells'])
            ctrl_raw = png(ctrl_cells, allowed)
            (folder / (policy + '-control.png')).write_bytes(ctrl_raw)
            policies[policy] = {'bridge': result, 'equal_actual_cells_guidance_greedy': controls,
                                'bridge_view_sha256': sha(raw), 'control_view_sha256': sha(ctrl_raw),
                                'equal_actual_cell_count': result['material_cells'] == controls['material_cells']}
        # Non-dominated policies are interesting candidate trade-offs, not a quality verdict.
        objectives = lambda p: tuple(policies[p]['bridge'][key] for key in (
            'components', 'below_soft_threshold_cells', 'new_cells'))
        # Deduplicate identical objective triples; do not report aliases as distinct wins.
        frontier = []
        for policy in POLICIES:
            if any(objectives(policy) == objectives(previous) for previous in frontier):
                continue
            if not any(dominates(policies[other]['bridge'], policies[policy]['bridge'])
                       for other in POLICIES if other != policy):
                frontier.append(policy)
        cases.append({'id': name, 'split': 'development_exploratory_no_final_claim',
                      'world': [WIDTH, HEIGHT], 'budget_new_material_cells': budget,
                      'initial_cells': len(initial), 'initial_components': len(components(initial)),
                      'initial_sha256': cell_sha(initial), 'permitted_sha256': cell_sha(allowed),
                      'guidance_u16le_sha256': sha(guide), 'original_png_sha256': sha(baseline),
                      'policies': policies, 'nondominated_policies': frontier})
    report = {'version': VERSION, 'qualification': 'DEVELOPMENT_SYNTHETIC_ONLY',
              'source_family_catalog': list(CASE_NAMES),
              'not_structural_holdout_confirmation': True, 'threshold': THRESHOLD,
              'model_inference': False, 'native_execution': False,
              'material_undo': 'UNAVAILABLE_WITHOUT_HOST_NATIVE_HISTORY_AND_RECEIPT',
              'reversal_semantics': {'BACKTRACK_CURSOR': 'brush-up no material change',
                                     'RELOCATE_TO_BRANCH': 'logical no material change',
                                     'UNDO_MATERIAL_EDIT': 'native history/receipt required'},
              'cases': cases}
    raw = encoded(report) + b'\n'; (output / 'pareto-study.json').write_bytes(raw)
    return report, sha(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--budget', type=int, default=44)
    opts = parser.parse_args()
    result, checksum = study(opts.out, opts.budget)
    print(encoded({'cases': len(result['cases']), 'sha256': checksum, 'qualification': result['qualification']}).decode())


if __name__ == '__main__':
    main()
