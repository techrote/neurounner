"""Fail-closed independent M7 original checkpoint/action/state/PNG evidence audit."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from .core import FAMILIES,raster
from .m5_topology_train import EVAL_SEEDS
from .m7_ablate import PROTOCOL,ARCH,MODES,SHA256,load_heads,evaluate,replay

def audit(root,checkpoints):
    manifest=json.loads((root/'manifest.json').read_text())
    members={str(f.relative_to(root)) for f in root.rglob('*') if f.is_file() and f.name!='manifest.json'}
    if set(manifest)!=members:raise ValueError('Missing/extra M7 archive member')
    for path,digest in manifest.items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest:
            raise ValueError('M7 manifest mismatch: '+path)
    report=json.loads((root/'report.json').read_text())
    policies=('novel','random')+tuple(a+':'+m for a in ARCH for m in MODES)
    if (report['protocol']!=PROTOCOL or report['rollouts']!=len(EVAL_SEEDS)*len(FAMILIES)*len(policies)
            or report['checkpoint_sha256']!=SHA256 or report['applied_illegal_actions']!=0
            or report['optimizer_updates']!=0 or report['synthetic_only'] is not True
            or tuple(report['policies'])!=policies):raise ValueError('M7 report provenance mismatch')
    torch,nets=load_heads(checkpoints)
    count=steps=0
    for seed in EVAL_SEEDS:
        for family in FAMILIES:
            for policy in policies:
                name=f'{seed}-{family}-{policy.replace(":","_")}'
                rec=json.loads((root/'rollouts'/(name+'.json')).read_text())
                if (rec['seed'],rec['family'],rec['policy'])!=(seed,family,policy):
                    raise ValueError('M7 rollout identity mismatch')
                reconstructed=replay(rec)
                expected,w=evaluate(seed,family,policy,torch,nets)
                if rec!=expected or reconstructed.state_sha256()!=w.state_sha256():
                    raise ValueError('Original model forward/action trace mismatch')
                original=np.asarray(Image.open(root/'rollouts'/(name+'.png')).convert('RGB'))
                if not np.array_equal(original,np.asarray(raster(w,2))):
                    raise ValueError('Original PNG not reproducible')
                steps+=len(rec['actions']);count+=1
    return {'status':'PASS','rollouts':count,'host_actions_recomputed':steps,
            'manifest_members':len(manifest),'model_checkpoints':len(ARCH),'illegal_actions':0}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--checkpoints',required=True,type=Path)
    args=p.parse_args();print(json.dumps(audit(args.root,args.checkpoints),indent=2))
if __name__=='__main__':main()
