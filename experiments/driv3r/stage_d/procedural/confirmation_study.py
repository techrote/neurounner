"""D-2 pre-registered off-line confirmation consumer. No model/native execution.

Fixture source+manifest frozen independently at commit 8b2fe242666ba8ee4c6459a43e39de0ef0f582a6
BEFORE this module or its outcomes existed. Do not tune fixture/policy on results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import confirmation_fixtures as ff
import pareto_study as policy

FROZEN_COMMIT = '8b2fe242666ba8ee4c6459a43e39de0ef0f582a6'
FROZEN_FIXTURE_SHA256 = 'de9ca09eca3ec6a67d106c40e8fb2084b53c719a18259552518bfc4184bfe14c'
FROZEN_MANIFEST_SHA256 = 'fcdf3103ffba3da992d46ceee61ce21829d5a185a75a2fece673e431d079b9fb'
POLICY_GIT_BLOB = '7fe2f1c49f765730354077703c1aae76c40a41b9'
VERSION = 'd2.synthetic-confirmation/1'


def verify_frozen_input(root: Path) -> dict:
    original = (root / 'corpus' / 'confirmation-prereg-v1.json').read_bytes()
    expected = ff.encode(ff.manifest()) + b'\n'
    if ff.digest(original) != FROZEN_MANIFEST_SHA256 or original != expected:
        raise ValueError('predeclared manifest changed; do not silently regenerate holdout')
    if ff.digest((root / 'confirmation_fixtures.py').read_bytes()) != FROZEN_FIXTURE_SHA256:
        raise ValueError('fixture source changed; refuse post-freeze edits')
    return json.loads(original)


def permitted_reachability(permitted, initial):
    """Hard-mask reachability ignores paint budget, guidance, and controller.

    If material islands are in separate 4-connected *permitted* domains, no
    authorized bridge can connect them, irrespective of algorithm or budget.
    """
    remaining = set(permitted)
    reach = []
    while remaining:
        p = min(remaining); pending = [p]; remaining.remove(p); group = {p}
        while pending:
            x, y = pending.pop()
            for dx,dy in ((0,1),(1,0),(0,-1),(-1,0)):
                q = (x+dx,y+dy)
                if q in remaining:
                    remaining.remove(q); group.add(q); pending.append(q)
        reach.append(group)
    occupied_domains = sum(bool(group & initial) for group in reach)
    return {'permitted_domains':len(reach),
            'initial_occupied_permitted_domains':occupied_domains,
            'all_initial_islands_globally_reachable':occupied_domains == 1}


def audit_events(initial, allowed, result, final, scalar=None, max_hops=120):
    """Independently reconstruct edits AND verify each synthetic brush-up/plan.

    This is synthetic material/event replay, NOT host/native route admission.
    The live logical cursor may travel brush-up only; a plan may only paint
    exactly its ordered, previously empty, 4-neighbour, permission-safe cells.
    """
    material = set(initial)
    cursor = min(initial)
    awaiting_plan = False
    pending_new = None
    steps = low = transport_hops = 0
    for index, event in enumerate(result['events']):
        kind = event['kind']
        if event['before'] != policy.cell_sha(material):
            raise AssertionError(f'event {index}: synthetic material before-hash mismatch')
        if event.get('native_receipt') is not None:
            raise AssertionError('synthetic event cannot assert native receipt')
        if kind == 'BRUSH_UP_ROUTE':
            if awaiting_plan or pending_new is not None:
                raise AssertionError('unfinished bridge before brush-up')
            route = [tuple(p) for p in event['path']]
            if not route or len(route) - 1 > max_hops or not all(p in allowed for p in route):
                raise AssertionError('brush-up route missing, over budget or forbidden')
            if (route[0] != cursor or route[0] != tuple(event['from'])
                    or route[-1] != tuple(event['to'])):
                raise AssertionError('synthetic cursor teleport or false route endpoints')
            if not all(abs(p[0]-q[0]) + abs(p[1]-q[1]) == 1 for p,q in zip(route,route[1:])):
                raise AssertionError('non-cardinal synthetic brush-up')
            if event['before'] != event['after'] or event.get('synthetic_only') is not True:
                raise AssertionError('brush-up material change or false native status')
            cursor = route[-1]
            transport_hops += len(route) - 1
            awaiting_plan = True
        elif kind == 'BRIDGE_PATH_SYNTHETIC':
            if not awaiting_plan or pending_new is not None:
                raise AssertionError('bridge plan without immediately preceding brush-up')
            route = [tuple(p) for p in event['path']]
            if (not route or route[0] != cursor or route[0] not in material
                    or route[-1] not in material or not all(p in allowed for p in route)):
                raise AssertionError('bridge path not anchored or permitted')
            if not all(abs(p[0]-q[0]) + abs(p[1]-q[1]) == 1 for p,q in zip(route,route[1:])):
                raise AssertionError('non-cardinal synthetic bridge plan')
            if event['before'] != event['after']:
                raise AssertionError('bridge planning modified material')
            new = [point for point in route if point not in material]
            if not new or len(set(new)) != len(new):
                raise AssertionError('bridge plan must contain unique new material cells')
            pending_new = new
            cursor = route[-1]
            awaiting_plan = False
        elif kind == 'SYNTHETIC_BRIDGE_PAINT':
            point = tuple(event['point'])
            if pending_new is None or not pending_new or point != pending_new.pop(0):
                raise AssertionError('paint missing or differs from recorded bridge plan')
            if point not in allowed or point in material:
                raise AssertionError('duplicate or prohibited new material')
            if scalar is not None:
                below = scalar[point] < policy.THRESHOLD
                if event['below_soft_threshold'] is not below:
                    raise AssertionError('synthetic guidance threshold label mismatch')
                low += int(below)
            material.add(point)
            steps += 1
            if event['after'] != policy.cell_sha(material):
                raise AssertionError('synthetic material after-hash mismatch')
            if not pending_new:
                pending_new = None
        else:
            raise AssertionError('unknown event kind')
    if awaiting_plan or pending_new is not None:
        raise AssertionError('unfinished synthetic brush/plan sequence')
    if material != final or steps != result['new_cells'] or result['material_sha256'] != policy.cell_sha(final):
        raise AssertionError('final material or edit count mismatch')
    if result['brush_up_hops'] != transport_hops:
        raise AssertionError('reported brush-up count differs from executed paths')
    if scalar is not None and result['below_soft_threshold_cells'] != low:
        raise AssertionError('reported guidance violations differ from edit events')
    if result['events_sha256'] != policy.sha(policy.encoded(result['events'])):
        raise AssertionError('event hash mismatch')
    if result['components'] != len(policy.components(final)):
        raise AssertionError('material connectivity summary mismatch')

def run(out: Path, names=None, *, allow_unqualified_shadow=False):
    root = Path(__file__).resolve().parent
    frozen = verify_frozen_input(root)
    if policy.VERSION != 'd2.synthetic-topology-pareto/1' or tuple(policy.POLICIES) != tuple(frozen['policy_arm_names']):
        raise ValueError('policy interface changed since preregistration')
    if policy.WIDTH != 32 or policy.HEIGHT != 32 or policy.THRESHOLD != 48000:
        raise ValueError('policy geometry/threshold changed')
    # Actual Git blob of the imported policy, not merely an advertised source ID.
    policy_bytes = (root / 'pareto_study.py').read_bytes()
    actual_policy_blob = hashlib.sha1(b'blob ' + str(len(policy_bytes)).encode() + b'\0' + policy_bytes).hexdigest()
    exact_policy_source = actual_policy_blob == POLICY_GIT_BLOB
    if not exact_policy_source and not allow_unqualified_shadow:
        raise ValueError('unqualified policy source; explicit allow_unqualified_shadow required')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    registered_ids = tuple(c['id'] for c in frozen['cases'])
    registered_set = set(registered_ids)
    if len(registered_set) != len(registered_ids):
        raise ValueError('duplicate IDs in frozen preregistration')
    if names is None:
        requested = registered_set
    else:
        supplied = tuple(names)
        if not supplied or len(supplied) != len(set(supplied)):
            raise ValueError('empty or duplicate requested case IDs')
        requested = set(supplied)
        if not requested <= registered_set:
            raise ValueError('not predeclared')
    full_cohort = requested == registered_set
    cases = []
    for registered in frozen['cases']:
        name = registered['id']
        if name not in requested:
            continue
        seed = registered['seed']; allowed, initial, scalar = ff.fixture(name, seed)
        guide_expected = struct.pack('<1024H', *(scalar[(x,y)] for y in range(32) for x in range(32)))
        if (ff.cells_sha(allowed) != registered['hard_permitted_sha256']
                or ff.cells_sha(initial) != registered['initial_material_sha256']
                or ff.digest(guide_expected) != registered['guidance_u16le_sha256']):
            raise ValueError('frozen per-case permission/material/guidance hash mismatch')
        case_dir = out / name; case_dir.mkdir(exist_ok=True)
        original_view = policy.png(initial, allowed)
        guide_bytes = struct.pack('<1024H',*(scalar[(x,y)] for y in range(32) for x in range(32)))
        (case_dir / 'original.png').write_bytes(original_view)
        (case_dir / 'guidance.u16le').write_bytes(guide_bytes)
        reach = permitted_reachability(allowed, initial)
        policies = {}
        for policy_name in frozen['policy_arm_names']:
            strict = policy_name == 'guidance-strict'
            weight = 0 if strict else int(policy_name.split('-')[1])
            result, after = policy.run_bridge(allowed, initial, scalar, budget=44,
                                              weight=weight, strict=strict, max_hops=120)
            audit_events(initial, allowed, result, after, scalar)
            control, ctrl_after = policy.guidance_greedy(allowed,initial,scalar,result['new_cells'])
            if len(ctrl_after - initial) != result['new_cells']:
                raise AssertionError('unequal actual paint budget')
            after_view = policy.png(after,allowed)
            control_view = policy.png(ctrl_after,allowed)
            events_raw = ff.encode(result['events']) + b'\n'
            (case_dir / (policy_name + '.png')).write_bytes(after_view)
            (case_dir / (policy_name + '-control.png')).write_bytes(control_view)
            (case_dir / (policy_name + '-events.json')).write_bytes(events_raw)
            summary = {k:v for k,v in result.items() if k != 'events'}
            summary.update({'view_sha256':ff.digest(after_view),
                            'events_file_sha256':ff.digest(events_raw),
                            'control_view_sha256':ff.digest(control_view),
                            'matched_control':control,
                            'control_components_delta':result['components'] - control['components']})
            policies[policy_name] = summary
        if name == 'hard-separator' and any(p['components']==1 for p in policies.values()):
            raise AssertionError('hard-separator negative control was crossed')
        cases.append({'id':name, 'seed':seed, 'registered':registered,
                      'reachability':reach,'original_png_sha256':ff.digest(original_view),
                      'policies':policies})
    qualification = ('PRE_REGISTERED_SYNTHETIC_OFFLINE_ONLY'
                     if exact_policy_source and full_cohort else
                     'PRE_REGISTERED_PARTIAL_SYNTHETIC_DIAGNOSTIC'
                     if exact_policy_source else
                     'UNQUALIFIED_LOCAL_POLICY_REIMPLEMENTATION')
    report = {'version':VERSION,'qualification':qualification,
              'selection_scope':('FULL_PREREGISTERED_COHORT' if full_cohort
                                 else 'PARTIAL_SUBSET_DIAGNOSTIC'),
              'included_case_ids':[name for name in registered_ids if name in requested],
              'omitted_case_ids':[name for name in registered_ids if name not in requested],
              'complete_source_qualified_cohort':bool(full_cohort and exact_policy_source),
              'freeze_commit':FROZEN_COMMIT,
              'frozen_fixture_sha256':FROZEN_FIXTURE_SHA256,
              'frozen_manifest_sha256':FROZEN_MANIFEST_SHA256,
              'policy_git_blob_sha1':POLICY_GIT_BLOB,
              'actual_policy_git_blob_sha1':actual_policy_blob,
              'policy_source_verified':exact_policy_source,
              'native_executed':False, 'native_receipts':0,
              'model_inference':False,'model_training':False,
              'material_undo':'HOST_NATIVE_RECEIPT_REQUIRED_NOT_TESTED',
              'visual_acceptance':'NOT_PERFORMED',
              'no_policy_or_fixture_tuning_after_freeze':True,
              'all_registered_cases_included':bool(full_cohort and len(cases)==len(frozen['cases'])),
              'cases':cases}
    raw = ff.encode(report) + b'\n'
    (out / 'confirmation-results.json').write_bytes(raw)
    return report, ff.digest(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--allow-unqualified-shadow', action='store_true',
                        help='explicitly permit labelled, non-source-qualified local debugging')
    args=parser.parse_args()
    result, h = run(args.out, allow_unqualified_shadow=args.allow_unqualified_shadow)
    print(ff.encode({'cases':len(result['cases']),'sha256':h,
                     'qualification':result['qualification']}).decode())


if __name__ == '__main__':main()
