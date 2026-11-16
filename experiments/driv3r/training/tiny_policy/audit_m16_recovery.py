"""Independent disk auditor for M16 research receipts and checkpoint identity."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from .m16_recovery import World, MLP, GRU, ACTIONS, PROTOCOL, EVAL, FAMILIES, POLICIES

def sha(b):return hashlib.sha256(b).hexdigest()
def audit(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    actual={p.relative_to(root).as_posix():sha(p.read_bytes()) for p in root.rglob('*') if p.is_file() and p.name!='manifest.json'}
    assert actual==manifest,'manifest mismatch or untracked file'
    report=json.loads((root/'report.json').read_text())
    assert report['protocol']==PROTOCOL and report['rollouts']==len(EVAL)*len(FAMILIES)*len(POLICIES)
    assert set(report['splits']['train']).isdisjoint(report['splits']['dev']+report['splits']['eval'])
    models={}
    for name,ctor in (('bc-mlp',MLP),('recovery-mlp',MLP),('bc-gru',GRU),('recovery-gru',GRU)):
        path=root/'checkpoints'/f'{name}.pt'
        assert report['training'][name]['checkpoint_sha256']==sha(path.read_bytes())
        saved=torch.load(path,map_location='cpu',weights_only=True)
        assert saved['name']==name and saved['protocol']==PROTOCOL and saved['steps']==report['training'][name]['steps']
        net=ctor();net.load_state_dict(saved['state'],strict=True);net.eval();models[name]=net
    total=0;decisions=0
    for seed in EVAL:
        for family in FAMILIES:
            for policy in POLICIES:
                stem=f'{seed}-{family}-{policy}';r=json.loads((root/'rollouts'/f'{stem}.json').read_text())
                assert r['seed']==seed and r['family']==family and r['policy']==policy
                w=World(seed,family,'eval')
                assert r['world_sha256']==sha(w.block.tobytes()+w.protect.tobytes())
                for e in r['events']:
                    w.progress();obs=w.observe()
                    assert e['obs_sha256']==sha(obs.tobytes())
                    assert e['legal']==[ACTIONS[a] for a in w.legal()]
                    action=ACTIONS.index(e['action']);assert action in w.legal()
                    before=w.state();assert before==e['before_sha256']
                    got=w.apply(action)
                    assert got=={k:e[k] for k in got}
                    decisions+=1
                assert w.state()==r['final_sha256']
                assert int(w.visited.sum())==r['cells'] and w.goal_idx==r['waypoints']
                assert w.retraces==r['retraces'] and w.branches()==r['branches']
                # Independently reconstruct raster pixels (do not call original render).
                from PIL import Image
                pixels=np.full((32,32,3),(238,235,226),np.uint8)
                pixels[w.block]=(66,66,68);pixels[w.protect]=(190,72,82)
                pixels[w.visited]={'Greebles':(36,133,112),'Pipework':(41,101,195),'Architecture':(184,113,36)}[family]
                pixels[w.y,w.x]=(255,220,70)
                pixels=np.repeat(np.repeat(pixels,4,axis=0),4,axis=1)
                with Image.open(root/'rollouts'/f'{stem}.png') as im:
                    assert np.array_equal(np.asarray(im.convert('RGB')),pixels)
                total+=1
    return {'passed':True,'hashes':len(manifest),'checkpoints':len(models),'rollouts':total,'legal_decisions':decisions}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    print(json.dumps(audit(parser.parse_args().out),sort_keys=True))
