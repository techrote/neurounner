"""Independent, fail-closed M14 original checkpoint/state/image auditor."""
from __future__ import annotations
import argparse, hashlib, json, random, tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from . import m14_dagger as m
from .core import ACTIONS, raster

def audit(root:Path):
    manifest=json.loads((root/'manifest.json').read_text())
    actual={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and p.name!='manifest.json'}
    if manifest!=actual:raise AssertionError('Missing/additional/tampered manifest entries')
    report=json.loads((root/'report.json').read_text())
    if report['protocol']!=m.PROTOCOL or report['rollouts']!=189:raise AssertionError('Protocol or rollout mismatch')
    heads={}
    for name in ('bc-mlp','dagger-mlp','bc-gru','dagger-gru'):
        chk=root/'checkpoints'/f'{name}.pt'
        if report['training'][name]['checkpoint_sha256']!=m.sha(chk.read_bytes()):raise AssertionError('Checkpoint digest mismatch')
        payload=torch.load(chk,map_location='cpu',weights_only=True)
        if payload['protocol']!=m.PROTOCOL or payload['name']!=name or payload['updates']!=160:raise AssertionError('Checkpoint identity/update mismatch')
        head=m.CTORS[name.split('-')[-1]]().eval()
        head.load_state_dict(payload['state_dict'],strict=True);heads[name]=head
    checked=0;actions=0
    for seed in m.EVAL:
        for fam in m.FAMILIES:
            for pol in m.POLICIES:
                stem=f'{seed}-{fam}-{pol}';rec=json.loads((root/'rollouts'/f'{stem}.json').read_text())
                if (rec['protocol'],rec['seed'],rec['family'],rec['policy'])!=(m.PROTOCOL,seed,fam,pol):raise AssertionError('Metadata mismatch')
                w=m.world(seed,fam,'eval');goals=m.reachable(w)
                if rec['goals']!=[list(g) for g in goals]:raise AssertionError('Goal mismatch')
                idx=0;hist=[];rng=random.Random(seed*103+m.FAMILIES.index(fam))
                for row in rec['events']:
                    idx=m.advance(w,goals,idx)
                    feature=m.obs(w,goals,idx);hist.append(feature)
                    if w.state_sha256()!=row['state'] or m.sha(feature.tobytes())!=row['observation']:
                        raise AssertionError('Host observation/state mismatch')
                    if [ACTIONS[a] for a in w.legal()]!=row['legal']:raise AssertionError('Legal mask mismatch')
                    predicted=m.choose(w,goals,idx,pol,heads,hist,rng)
                    if ACTIONS[predicted]!=row['action']:raise AssertionError('Checkpoint action mismatch')
                    receipt=w.apply(int(predicted))
                    if receipt!={k:row[k] for k in receipt}:raise AssertionError('Host transition mismatch')
                    if w.blocked[w.y,w.x] or w.protected[w.y,w.x]:raise AssertionError('Protected cell breach')
                    actions+=1
                if rec['state']!=w.state_sha256() or rec['cells']!=int(w.visited.sum()) or rec['branches']!=w.branch_nodes() or rec['retraces']!=w.revisit_landings or rec['completed_goals']!=m.advance(w,goals,idx):
                    raise AssertionError('Terminal statistics mismatch')
                if rec['geometry']!=m.sha(w.blocked.tobytes()+w.protected.tobytes()):raise AssertionError('Geometry digest mismatch')
                with Image.open(root/'rollouts'/f'{stem}.png') as original:
                    if not np.array_equal(np.array(original.convert('RGB')),np.array(raster(w,2))):raise AssertionError('PNG pixels mismatch')
                checked+=1
    return {'protocol':m.PROTOCOL,'original_replays':checked,'actions':actions,'checkpoint_loads':len(heads),'manifest_entries':len(manifest),'all_passed':True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True,type=Path);args=ap.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(audit(args.input),indent=2))
if __name__=='__main__':main()
