#!/usr/bin/env python3
"""H64: compare original frozen H63 neural receipts with a fresh unchanged-source replay.

Run from the extracted H63 evidence root after restoring nested H62/H56 dependency
files and running h63_family_disjoint_selectivity.py evaluate --out h64_local_output.
This is a RETROSPECTIVE numerical replay, never a new scientific holdout.
"""
import collections, hashlib, json, pathlib, platform

ROOT = pathlib.Path(__file__).resolve().parent
ORIGINAL = ROOT / 'h63_output'
REPLAY = ROOT / 'h64_local_output'
ORIGINAL_SHA = '48170223daf0b906b5a3d4c44cf70c05398aedafd9a9ac4ba45ba549b5ee6c48'
SOURCE_SHA = 'e2ed326dc96e2cc4f08c6513f0011ca20f578ec833d8ffe802dc68be33c8f0c0'

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def records(p):
    with p.open() as f:
        return {(r['receiver_seed'],r['family'],r['bucket'],r['index']):r
                for r in map(json.loads,f)}
def flat(pairs): return [x for pair in pairs for x in pair]

def main():
    import torch, numpy
    assert digest(ROOT/'h63_family_disjoint_selectivity.py') == SOURCE_SHA
    name='h56_world_profile_neural_receipts.jsonl'
    a,b=ORIGINAL/name,REPLAY/name
    assert digest(a)==ORIGINAL_SHA
    ra,rb=records(a),records(b)
    assert ra.keys()==rb.keys() and len(ra)==7200
    pa,pb=(json.loads((folder/'h56_result.json').read_text()) for folder in (ORIGINAL,REPLAY))
    assert pa['donor_endpoints']==pb['donor_endpoints']
    assert pa['passing_cells']==pb['passing_cells']==45
    assert pa['verdict']==pb['verdict']=='PASS'
    assert pa['raw_receipt_sha256']==digest(a) and pb['raw_receipt_sha256']==digest(b)
    diffs=collections.Counter(); changed_actions=changed_margins=0; max_drift=0.0
    smallest_signed=1e9; inside_old_h48=0
    for k in sorted(ra):
        x,y=ra[k],rb[k]
        assert x.keys()==y.keys()
        for field in x:
            if field != 'neural_route_margins': assert x[field]==y[field],(k,field)
        old,new=flat(x['neural_route_margins']),flat(y['neural_route_margins'])
        action_old,action_new=flat(x['neural_actions']),flat(y['neural_actions'])
        for m,n,ac,bc in zip(old,new,action_old,action_new):
            assert int(m>0)==ac and int(n>0)==bc
            d=abs(m-n)
            changed_margins+=int(d!=0);changed_actions+=int(ac!=bc)
            diffs[k[0]]+=int(d!=0)
            max_drift=max(max_drift,d)
            smallest_signed=min(smallest_signed,(2*bc-1)*n)
            inside_old_h48+=int(abs(m)<=0.0000762939453125)
    i,j=(json.loads((folder/'h62_continuous_invariant_result.json').read_text()) for folder in (ORIGINAL,REPLAY))
    assert i['source_intervals']==j['source_intervals']==17280
    assert i['wrong_source_intervals']==j['wrong_source_intervals']==0
    report={'experiment':'DG020_H64','classification':'RETROSPECTIVE_UNCHANGED_H63_REPLAY',
      'verdict':'PASS' if changed_actions==0 else 'FAIL','worlds':2400,'receipts':7200,
      'neural_decisions':57600,'changed_actions':changed_actions,
      'changed_float32_margins':changed_margins,'max_absolute_margin_drift':max_drift,
      'minimum_correct_replay_margin':smallest_signed,'prior_h48_window_original_decisions':inside_old_h48,
      'changed_margin_values_by_receiver':dict(diffs),'original_receipt_sha256':digest(a),
      'replayed_receipt_sha256':digest(b),'continuous_invariant_source_intervals':17280,
      'continuous_invariant_wrong_intervals':0,'all_45_cells_preserved':True,
      'host':{'cpu_model':next((line.split(':',1)[1].strip() for line in pathlib.Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown'),
        'platform':platform.platform(),'python':platform.python_version(),
        'torch':torch.__version__,'numpy':numpy.__version__},
      'physical_host_separation':'UNQUALIFIED: original H63 physical CPU fingerprint absent',
      'scientific_gate':'H63 restricted 45/45 PASS unchanged; no new holdout'}
    (ROOT/'h64_comparison_compact.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,sort_keys=True,indent=2))

if __name__=='__main__': main()
