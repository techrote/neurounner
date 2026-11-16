"""D-2 post-freeze, synthetic-only, bridge robustness experiment.

Consumes the *provisional* eight-case confirmation archive without changing the
frozen baseline policy or fixtures. Builds explicit material-preserving edit
plans, never asserts native paint receipts or physical brush transport.
Only Python stdlib. No external model/network/GPU use.
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import json
from pathlib import Path
import random
import struct
import zipfile

from fragility_study import (DIRECTIONS, decode_synthetic_rgba_png,
                             encode, measure, sha)

VERSION = 'd2.post-freeze-reinforcement/1'
SOFT_THRESHOLD = 32768  # Diagnostic only; not a native legal-action limit.


def neighbors(point):
    x, y = point
    return ((x + dx, y + dy) for dx, dy in DIRECTIONS)


def articulations(material):
    """Iterative Tarjan; supports fully occupied 32x32 without recursion limits."""
    discovery, low, parent, children, result = {}, {}, {}, {}, set()
    clock = 0
    for root in sorted(material):
        if root in discovery:
            continue
        clock += 1
        discovery[root] = low[root] = clock
        parent[root] = None
        children[root] = 0
        stack = [(root, iter(sorted(p for p in neighbors(root) if p in material)))]
        while stack:
            node, adjacent = stack[-1]
            try:
                nxt = next(adjacent)
            except StopIteration:
                stack.pop()
                previous = parent[node]
                if previous is None:
                    if children[node] > 1:
                        result.add(node)
                else:
                    low[previous] = min(low[previous], low[node])
                    if parent[previous] is not None and low[node] >= discovery[previous]:
                        result.add(previous)
                continue
            if nxt not in discovery:
                parent[nxt] = node
                children[node] += 1
                children[nxt] = 0
                clock += 1
                discovery[nxt] = low[nxt] = clock
                stack.append((nxt, iter(sorted(p for p in neighbors(nxt) if p in material))))
            elif nxt != parent[node]:
                low[node] = min(low[node], discovery[nxt])
    return result


def bounded_bypass(material, permitted, pivot, start, goal, radius=3,
                   maximum_edits=8):
    """Find lexicographically deterministic min-edit loop around a critical cell.

    Every step is four-neighbour and permitted, with an exact cardinal route.
    Existing material is free to traverse, but the pivot is excluded.  Routes
    are a *synthetic edit-planning trace*, not a brush-up native action.
    """
    if pivot not in material or start not in material or goal not in material:
        raise ValueError('bypass endpoints/pivot must be painted')
    if start == goal:
        return None
    px, py = pivot
    allowed = {p for p in permitted if abs(p[0] - px) <= radius and
               abs(p[1] - py) <= radius and p != pivot}
    if start not in allowed or goal not in allowed:
        return None
    queue = [(0, 0, (start,), start)]
    best = {start: (0, 0, (start,))}
    while queue:
        edit_count, hops, path, cursor = heapq.heappop(queue)
        if (edit_count, hops, path) != best.get(cursor):
            continue
        if cursor == goal:
            if not edit_count:
                return None
            return tuple(p for p in path if p not in material), path
        for nxt in sorted(n for n in neighbors(cursor) if n in allowed):
            edits = edit_count + int(nxt not in material)
            if edits > maximum_edits:
                continue
            candidate = (edits, hops + 1, path + (nxt,))
            if nxt not in best or candidate < best[nxt]:
                best[nxt] = candidate
                heapq.heappush(queue, (*candidate, nxt))
    return None


def best_reinforcement(material, permitted, budget, radius=3):
    """Select a material bridge bypass with a **proven** articulation reduction.

    No zero-gain speculative strokes; explicit STOP when no useful local loop.
    Evaluation uses the final *whole-material* connectivity graph. Structural
    gains are measured after applying *all* cells in one atomic synthetic plan.
    """
    before = articulations(material)
    plans = []
    for pivot in sorted(before):
        exits = sorted(p for p in neighbors(pivot) if p in material)
        for start, goal in itertools.combinations(exits, 2):
            found = bounded_bypass(material, permitted, pivot, start, goal,
                                   radius=radius, maximum_edits=min(budget, 8))
            if found is None:
                continue
            additions, route = found
            if not additions or len(set(additions)) != len(additions):
                continue
            after = material | set(additions)
            new_articulations = articulations(after)
            improvement = len(before) - len(new_articulations)
            if improvement <= 0:
                continue
            # Improvement per new edit, then absolute improvement, then fewer
            # new edits, with coordinate/route tie breaks. All deterministic.
            plans.append(((-improvement / len(additions), -improvement,
                           len(additions), pivot, start, goal, route),
                          {'pivot': pivot, 'start': start, 'goal': goal,
                           'route': route, 'additions': additions,
                           'articulation_reduction': improvement}))
    if not plans:
        return None
    return min(plans, key=lambda item: item[0])[1]


def reinforce(initial, permitted, budget=24, radius=3):
    if budget < 0 or radius < 1:
        raise ValueError('negative budget or invalid radius')
    if not initial <= permitted:
        raise ValueError('material outside permitted mask')
    material = set(initial)
    history = []
    while len(material - initial) < budget:
        plan = best_reinforcement(material, permitted,
                                  budget - len(material - initial), radius)
        if plan is None:
            break
        before = sha(encode(sorted(material)))
        for pixel in plan['additions']:
            if pixel not in permitted or pixel in material:
                raise AssertionError('invalid synthetic edit')
            material.add(pixel)
        history.append({'pivot': list(plan['pivot']),
                        'route': [list(p) for p in plan['route']],
                        'edits': [list(p) for p in plan['additions']],
                        'articulation_reduction': plan['articulation_reduction'],
                        'before_material_sha256': before,
                        'after_material_sha256': sha(encode(sorted(material)))})
    return material, history, ('EDIT_BUDGET_EXHAUSTED' if
                               len(material - initial) == budget else
                               'NO_BUDGETED_ARTICULATION_BYPASS')


def equal_edit_control(material, permitted, guidance, n, seed, kind):
    """Exactly n new cells, using the *same policy baseline* and permission mask.

    Controls intentionally do not optimize articulation count; their fixed
    allocation is a negative control, not proof that guidance is inadequate.
    """
    material = set(material)
    rng = random.Random(seed)
    additions = []
    while len(additions) < n:
        frontier = {p for painted in material for p in neighbors(painted)
                    if p in permitted and p not in material}
        if not frontier:
            # Preserve the denominator even with unreachable empty space.
            frontier = permitted - material
        if not frontier:
            raise ValueError('insufficient permitted cells for equal-budget control')
        if kind == 'guidance_first':
            # Higher scalar is soft-preferred. Fixed deterministic ties.
            choice = min(frontier, key=lambda p: (-guidance[p[1] * 32 + p[0]], p))
        elif kind == 'seeded_random_frontier':
            choice = rng.choice(sorted(frontier))
        else:
            raise ValueError('unknown control')
        material.add(choice)
        additions.append(choice)
    return material, additions


def trial(initial, baseline, permitted, guidance, seed, budget=24, radius=3):
    if not initial <= baseline <= permitted or len(guidance) != 32 * 32:
        raise ValueError('invalid baseline geometry or guidance size')
    material, plans, terminal = reinforce(baseline, permitted, budget, radius)
    count = len(material - baseline)
    results = {}
    for name, final in [('reinforcement', material)] + [
        (name, equal_edit_control(baseline, permitted, guidance, count, seed, name)[0])
        for name in ('guidance_first', 'seeded_random_frontier')]:
        delta = final - baseline
        if len(delta) != count or not baseline <= final <= permitted:
            raise AssertionError('mismatched edit budgets or permission violation')
        result = measure(initial, final)
        result.update({'added_after_baseline': count,
                       'articulations_after': len(articulations(final)),
                       'articulations_before': len(articulations(baseline)),
                       'below_soft_threshold': sum(guidance[y * 32 + x] < SOFT_THRESHOLD
                                                   for x,y in delta),
                       'final_material_sha256': sha(encode(sorted(final))),
                       'added_pixels': [list(p) for p in sorted(delta)]})
        results[name] = result
    return {'terminal':terminal,'plans':plans,'policies':results}


def run_archive(path: Path, budget=24, radius=3):
    """Independent intervention on provisional evidence; never promotion."""
    if budget > 64 or budget < 0 or radius > 8 or radius < 1:
        raise ValueError('synthetic resource bounds exceeded')
    archive_bytes = path.read_bytes()
    with zipfile.ZipFile(path) as archive:
        report = json.loads(archive.read('confirmation-results.json'))
        if report.get('version') != 'd2.synthetic-confirmation/1':
            raise ValueError('unrecognized archive source')
        if len(report['cases']) != 8:
            raise ValueError('expected frozen eight-case cohort')
        cases = []
        for case in report['cases']:
            cid = case['id']
            original_bytes = archive.read(cid + '/original.png')
            baseline_bytes = archive.read(cid + '/weighted-0.png')
            guidance_bytes = archive.read(cid + '/guidance.u16le')
            if sha(original_bytes) != case['original_png_sha256']:
                raise ValueError('corrupted original PNG')
            if sha(baseline_bytes) != case['policies']['weighted-0']['view_sha256']:
                raise ValueError('corrupted baseline PNG')
            if sha(guidance_bytes) != case['registered']['guidance_u16le_sha256']:
                raise ValueError('corrupted guidance bytes')
            if len(guidance_bytes) != 2048:
                raise ValueError('invalid scalar dimensions')
            original, permitted = decode_synthetic_rgba_png(original_bytes)
            baseline, baseline_permitted = decode_synthetic_rgba_png(baseline_bytes)
            if permitted != baseline_permitted:
                raise ValueError('permission mutation')
            if not original <= baseline:
                raise ValueError('baseline material erasure')
            if len(original) != case['registered']['initial_count'] or \
                    len(baseline) - len(original) != case['policies']['weighted-0']['new_cells']:
                raise ValueError('source raster disagrees with archive')
            guidance_flat = struct.unpack('<1024H', guidance_bytes)
            result = trial(original, baseline, permitted, guidance_flat,
                           case['seed'], budget, radius)
            cases.append({'id':cid,'seed':case['seed'],
                          'initial_material_sha256':sha(encode(sorted(original))),
                          'baseline_material_sha256':sha(encode(sorted(baseline))),
                          'hard_permitted_count':len(permitted),
                          'baseline_articulations':len(articulations(baseline)),
                          **result})
    return {'version':VERSION,
            'input_archive_sha256':sha(archive_bytes),
            'input_policy_source_verified':report.get('policy_source_verified',False),
            'input_qualification':report.get('qualification'),
            'qualification':'POST_FREEZE_EXPLORATORY_PROVISIONAL_ONLY',
            'native_material_receipts':0,'model_inference':False,
            'not_native_or_art_accepted':True,
            'budget_max_new_edits_per_case':budget,
            'route_radius':radius,'cases':cases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--budget',type=int,default=24)
    parser.add_argument('--radius',type=int,default=3)
    args = parser.parse_args()
    result = run_archive(args.archive,args.budget,args.radius)
    data = encode(result)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_bytes(data)
    print(json.dumps({'cases':len(result['cases']), 'report_sha256':sha(data),
                      'qualification':result['qualification']},sort_keys=True))


if __name__ == '__main__':
    main()
