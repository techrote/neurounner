"""Independent H11 receipt-level oracle, law-invariance, shuffle and median audit."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
OUT=BASE/'h11-evidence'
EPS=2.0e-5

def audit():
    report=json.loads((OUT/'report.json').read_text())
    sha=json.loads((OUT/'sha256.json').read_text())
    for name,digest in sha.items():
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest,name
    files=sorted(OUT.glob('receipts-*.jsonl'))
    assert len(files)==30,len(files)
    grouped=defaultdict(lambda:dict(n=0,both=0,flip=0,law0=0,law1=0,sham=0,
                                    shuffled_preinference_both=0,comparator_both=0,
                                    neural_comparator_disagreements=0,zero_both=0,source_blind_both=0,
                                    illegal=0,max_donor_mse=0.))
    hashes=set();div=0;inv=0
    for file in files:
        rows=[json.loads(s) for s in file.read_text().splitlines()]
        assert len(rows)==160
        for row in rows:
            k=(str(row['receiver_seed']),row['family'],row['bucket'])
            c=grouped[k];c['n']+=1
            ob=np.asarray(row['observation'],dtype='f4')
            assert ob.shape==(2,12,4)
            h=hashlib.sha256(ob.tobytes()).hexdigest()
            assert h==row['observation_sha256'] and h not in hashes
            hashes.add(h)
            assert row['legal_mask']==[1,1]
            wet=ob[...,0]>.5;fee=ob[...,1]
            assert np.all((fee>=0)&(fee<=.3))
            assert (int(wet.sum(axis=1).max()-wet.sum(axis=1).min())==
                    (0 if k[2]=='equal_wet_invariant' else 2))
            oracle=[]
            for law,price in enumerate((.35,2.30)):
                costs=(np.where(wet,price,1.)+fee).sum(axis=1,dtype='f8')
                assert np.allclose(costs,row['oracle_route_costs'][law],atol=EPS)
                oracle.append(int(costs.argmin()))
                raw=np.asarray(row['raw_donor_forecasts'][law]);conditioned=np.asarray(row['conditioned_donor_forecasts'][law])
                assert raw.shape==(2,12)
                nocost=(raw-fee).ravel();mask=wet.ravel()
                def lower_median(arr):
                    v=np.sort(arr);return v[(len(v)-1)//2]
                want=np.where(wet,lower_median(nocost[mask]),lower_median(nocost[~mask]))+fee
                assert np.allclose(want,conditioned,atol=EPS),(k,law,abs(want-conditioned).max())
                mse=float(np.mean((raw-(np.where(wet,price,1.)+fee))**2))
                shamraw=np.asarray(row['sham_raw_donor_forecasts'][law])
                sham_mse=float(np.mean((shamraw-(np.where(wet,price,1.)+fee))**2))
                c['max_donor_mse']=max(c['max_donor_mse'],mse,sham_mse)
            assert oracle==row['oracle_actions']
            act=[int(np.argmax(x)) for x in row['logits']]
            assert act==row['actions']
            sham=[int(np.argmax(x)) for x in row['same_law_replica_logits']]
            assert sham==row['same_law_replica_actions']
            simple=[int(np.asarray(f).sum(axis=1).argmin()) for f in row['conditioned_donor_forecasts']]
            assert simple==row['cost_comparator_actions']
            pick=row['preinference_donor_selection']
            shuffled=[act[pick[0]],act[pick[1]]]
            assert shuffled==row['preinference_shuffled_actions'],(k,shuffled,row['preinference_shuffled_actions'])
            both=int(act==oracle)
            c['both']+=both;c['flip']+=int(act[0]!=act[1]);c['law0']+=int(act[0]==oracle[0]);c['law1']+=int(act[1]==oracle[1]);c['sham']+=int(sham!=act)
            c['shuffled_preinference_both']+=int(shuffled==oracle)
            c['comparator_both']+=int(simple==oracle)
            c['neural_comparator_disagreements']+=int(simple!=act)
            c['zero_both']+=int(row['zero_action']==oracle[0]==oracle[1])
            c['source_blind_both']+=int(row['source_blind_action']==oracle[0]==oracle[1])
            c['illegal']+=int(any(a not in (0,1) for a in act))
            div+=k[2].endswith('divergent');inv+=not k[2].endswith('divergent')
    assert div==1920 and inv==2880
    assert len(hashes)==4800
    for k,c in grouped.items():
        expected=report['results'][k[0]][k[1]][k[2]]['counts']
        for metric,value in expected.items():
            if metric=='max_donor_mse':
                assert abs(value-c[metric])<1e-5,(k,metric,value,c[metric])
            else:assert value==c[metric],(k,metric,value,c[metric])
    result={'source':'independent receipt-only audit with no imported H11 evaluator code',
            'receipts_checked':len(hashes),'cells_checked':len(grouped),'divergent_pairs':div,
            'invariant_pairs':inv,'all_reported_metrics_agree':True,
            'manifest_files_verified':len(sha),
            'H11_source_sha256':hashlib.sha256((BASE/'h11_tight_margin_stress.py').read_bytes()).hexdigest(),
            'verified_scientific_verdict':report['status']}
    (OUT/'receipt-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':audit()
