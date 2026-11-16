"""Independent disk-based M11 audit: source of truth is serialized checkpoints and receipts."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from .core import ACTIONS, FAMILIES, raster
from .m11_memory_policy import (PROTOCOL, EVAL, POLICIES, world, sha, replay,
                               MLP, GRU, TCN, CONTEXT, nearest_frontier, novelty)


def audit(root):
    metadata=json.loads((root/'report.json').read_text())
    assert metadata['protocol']==PROTOCOL and metadata['synthetic_only'] is True
    listed=json.loads((root/'manifest.json').read_text())
    actual={p.relative_to(root).as_posix():sha(p.read_bytes()) for p in root.rglob('*') if p.is_file() and p.name!='manifest.json'}
    assert listed==actual,'Missing/altered evidence file'
    heads={};checks={}
    for name,ctor in (('mlp',MLP),('gru',GRU),('tcn',TCN)):
        p=root/'checkpoints'/f'{name}.pt'
        assert sha(p.read_bytes())==metadata['training']['training'][name]['checkpoint_sha256']
        ckpt=torch.load(p,map_location='cpu',weights_only=True)
        assert ckpt['protocol']==PROTOCOL and ckpt['architecture']==name
        assert ckpt['train_identity']==metadata['training']['train']['identity_sha256']
        assert ckpt['dev_identity']==metadata['training']['dev']['identity_sha256']
        assert ckpt['optimizer_updates']==metadata['training']['training'][name]['optimizer_updates']
        model=ctor();model.load_state_dict(ckpt['state_dict'],strict=True);model.eval();heads[name]=model
        checks[name]=0
    assert len(list((root/'rollouts').glob('*.json')))==len(EVAL)*len(FAMILIES)*len(POLICIES)
    assert len(list((root/'rollouts').glob('*.png')))==len(EVAL)*len(FAMILIES)*len(POLICIES)
    visited=0; decisions=0
    for seed in EVAL:
        for family in FAMILIES:
            for policy in POLICIES:
                stem=f'{seed}-{family}-{policy}'
                r=json.loads((root/'rollouts'/f'{stem}.json').read_text())
                assert (r['seed'],r['family'],r['policy'])==(seed,family,policy)
                w=replay(r)
                # Image parity compares raw source pixels, not PNG file metadata.
                from PIL import Image
                got=np.asarray(Image.open(root/'rollouts'/f'{stem}.png').convert('RGB'))
                assert np.array_equal(got,np.asarray(raster(w,2))),f'{stem}: raster mismatch'
                if policy in heads:
                    ghost=world(seed,family,'eval');hist=[]
                    for e in r['events']:
                        hist.append(ghost.observe())
                        x=torch.from_numpy(np.stack(hist[-CONTEXT:]))[None,:,:]
                        with torch.inference_mode(): logits=heads[policy](x)[0,-1]
                        a=max(ghost.legal(),key=lambda a:(float(logits[a]),-a))
                        assert ACTIONS[a]==e['action'],f'{stem}: checkpoint action mismatch'
                        ghost.apply(int(a));checks[policy]+=1
                    assert ghost.state_sha256()==w.state_sha256()
                elif policy=='planner':
                    ghost=world(seed,family,'eval')
                    for e in r['events']:
                        a=nearest_frontier(ghost);assert ACTIONS[a]==e['action'];ghost.apply(a)
                elif policy=='novel':
                    ghost=world(seed,family,'eval')
                    for e in r['events']:
                        a=novelty(ghost);assert ACTIONS[a]==e['action'];ghost.apply(a)
                visited+=1;decisions+=len(r['events'])
    return {'protocol':PROTOCOL,'matched_manifest_entries':len(actual),'replayed_trajectories':visited,
            'host_legal_applied_actions':decisions,'trained_checkpoint_actions_recomputed':checks,
            'status':'PASS'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    print(json.dumps(audit(args.input),indent=2,sort_keys=True))
if __name__=='__main__':main()
