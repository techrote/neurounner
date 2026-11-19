"""Read-only post-hoc H22 control-power audit. Not a scientific acceptance cohort.
Usage: python h22_control_power_audit_compact.py H22-CAPSULE.zip
Standard library only; original trained-checkpoint replay is separate.
"""
import collections
import hashlib
import json
import struct
import sys
import zipfile

CAPSULE_SHA = '47546540b96ae0800697e8a729c169e1191f111cec8c09f704c99a8843568667'
SOURCE_SHA = 'a032a2b9057c150beb918e83e17ff8ad7e73a764fefcb4cf11cfd3049e1ef35d'
DIR = 'source/h22-evidence/'
sha = lambda b: hashlib.sha256(b).hexdigest()


def audit(archive):
    with open(archive, 'rb') as f:
        assert sha(f.read()) == CAPSULE_SHA, 'capsule tampered'
    with zipfile.ZipFile(archive) as z:
        assert sha(z.read('source/h22_wet_coordinate_only_confirmation.py')) == SOURCE_SHA
        report = json.loads(z.read(DIR + 'report.json'))
        assert report['original_source_sha256'] == SOURCE_SHA
        manifest = json.loads(z.read(DIR + 'sha256.json'))
        assert len(manifest) == 31
        for name, expected in manifest.items():
            assert sha(z.read(DIR + name)) == expected, name
        receipt_paths = sorted(n for n in z.namelist() if n.startswith(DIR + 'receipts-') and n.endswith('.jsonl'))
        assert len(receipt_paths) == 30
        rows = [json.loads(s) for n in receipt_paths for s in z.read(n).splitlines()]
    assert len(rows) == 4800
    groups, hashes, wrong = collections.Counter(), set(), []
    counts = collections.Counter()
    for r in rows:
        groups[(r['seed'], r['family'], r['bucket'])] += 1
        obs = r['observation']
        packed = b''.join(struct.pack('<f', x) for route in obs for tile in route for x in tile)
        digest = sha(packed)
        assert digest == r['observation_sha256'] and digest not in hashes
        hashes.add(digest)
        oracle = []
        for law in (.35, 2.30):
            cost = [sum((law if t[0] > .5 else 1.) + t[1] for t in route) for route in obs]
            oracle.append(int(cost[1] < cost[0]))
        assert oracle == r['oracle_actions']
        assert r['both_correct'] == (r['actions'] == oracle)
        assert r['foreign_sham_changed'] == (r['actions'] != r['foreign_actions'])
        assert r['dry_placebo_changed'] == [a != b for a, b in zip(r['actions'], r['dry_placebo_actions'])]
        picks = r['shuffled_donor_selection']
        assert r['shuffled_actions'] == [r['actions'][picks[0]], r['actions'][picks[1]]]
        counts['shuffle_derivable'] += 1
        if r['bucket'].endswith('divergent'):
            assert oracle[0] != oracle[1]
            counts['divergent'] += 1
            counts['divergent_correct'] += r['both_correct']
            counts['shuffled_correct'] += (r['shuffled_actions'] == oracle)
            counts['shuffled_random_0_1'] += (picks == [0, 1])
        else:
            assert oracle[0] == oracle[1]
            counts['invariant'] += 1
            counts['invariant_correct'] += r['both_correct']
        if not r['both_correct']:
            wrong.append((r['seed'], r['family'], r['bucket'], r['index'], digest))
        counts['foreign_changed'] += r['foreign_sham_changed']
        counts['dry_changed'] += any(r['dry_placebo_changed'])
    assert len(groups) == 30 and set(groups.values()) == {160}
    assert dict(counts)['divergent_correct'] == 1920
    assert dict(counts)['invariant_correct'] == 2877
    assert counts['shuffled_correct'] == counts['shuffled_random_0_1'] == 496
    assert counts['shuffle_derivable'] == 4800
    assert counts['foreign_changed'] == counts['dry_changed'] == 0
    assert len(wrong) == 3
    a, b = report['learned_material_values']['29002117'], report['learned_material_values']['29002119']
    law_delta = abs(a['1'][1] - a['0'][1])
    max_sham = max(abs(a['0'][0] - b['0'][0]), *(abs(a[str(i)][1] - b[str(i)][1]) for i in (0, 1)))
    return {'verdict': 'POST-HOC METHODS CHECK PASS; original restricted H22 scientific PASS unchanged',
            'capsule_sha256': CAPSULE_SHA, 'pairs': len(rows), 'cells': len(groups),
            'counts': dict(counts), 'same_law_sham_to_wet_law_ratio': max_sham / law_delta,
            'negative_cases': sorted(wrong),
            'warning': 'Shuffled score follows binary coin flips; zero/source-blind both-correct on divergent pairs is tautologically impossible. This is NOT external review.'}


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Usage: python h22_control_power_audit_compact.py capsule.zip')
    print(json.dumps(audit(sys.argv[1]), indent=2, sort_keys=True))
