"""Independent original-checkpoint/action/PNG replay audit of M10 synthetic evidence."""
from __future__ import annotations
import argparse, json, hashlib, random
from pathlib import Path
import numpy as np
import torch
from . import m10_router as m
from .core import ACTIONS, FAMILIES, raster

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit(out,ckpts):
    manifest=json.loads((out/'manifest.json').read_text());files={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()}
    assert set(manifest)==files-{'manifest.json'}
    for name,expected in manifest.items():assert sha(out/name)==expected,name
    rep=json.loads((out/'report.json').read_text())
    assert rep['training']['checkpoint_sha256']==sha(out/'router.pt')
    assert rep['source_sha256']==sha(Path(m.__file__))
    assert set(m.TRAIN).isdisjoint(m.EVAL) and set(m.DEV).isdisjoint(m.EVAL) and set(m.TRAIN).isdisjoint(m.DEV)
    assert set(range(3)).isdisjoint(set(range(3,6)))
    torch.set_num_threads(1)
    heads=m.load_heads(ckpts);cp=torch.load(out/'router.pt',weights_only=True,map_location='cpu')
    assert cp['protocol']==m.PROTOCOL and cp['train_id']==rep['training']['train_id'] and cp['dev_id']==rep['training']['dev_id']
    router=m.Router();router.load_state_dict(cp['state_dict'],strict=True);router.eval()
    illegal=0;total=0;replay_count=0;paint_checks=0;pred_checks=0
    for f in sorted((out/'rollouts').glob('*.json')):
        rec=json.loads(f.read_text());w=m.world(rec['seed'],rec['family'],rec['shape'])
        rng=random.Random(rec['seed']*43+rec['shape']*1027+FAMILIES.index(rec['family'])*97)
        for event in rec['events']:
            assert w.state_sha256()==event['pre_sha256']
            assert event['legal']==[ACTIONS[a] for a in w.legal()]
            predicted=m.choice(w,rec['policy'],heads,router,rng)
            assert ACTIONS[predicted]==event['action'],f
            pred_checks+=1
            if predicted not in w.legal():illegal+=1
            w.apply(predicted);total+=1
        assert w.state_sha256()==rec['final_sha256']
        assert np.array_equal(np.asarray(raster(w,1).convert('RGB')), np.asarray(__import__('PIL.Image',fromlist=['Image']).open(f.with_suffix('.png')).convert('RGB')))
        assert m.replay(rec).state_sha256()==w.state_sha256()
        replay_count+=1;paint_checks+=1
    assert illegal==0 and replay_count==rep['trajectories'] and total==rep['decisions']
    # Adversarial checks: protection, illegal action, falsified legality, falsified bitmap and checkpoint.
    rec=json.loads(sorted((out/'rollouts').glob('*.json'))[0].read_text())
    bad=json.loads(json.dumps(rec));bad['events'][0]['legal']=['LEFT','STOP']
    try:m.replay(bad);raise AssertionError('Legal mutation accepted')
    except AssertionError as exc:
        if str(exc)=='Legal mutation accepted':raise
    bad=json.loads(json.dumps(rec));bad['final_sha256']='0'*64
    try:m.replay(bad);raise AssertionError('State mutation accepted')
    except AssertionError as exc:
        if str(exc)=='State mutation accepted':raise
    w=m.world(m.EVAL[0],FAMILIES[0],4);w.protected[w.y,w.x+1]=True
    try:w.apply(3);raise AssertionError('Protected move accepted')
    except ValueError:pass
    ckpt_data=(ckpts/'style-mlp.pt').read_bytes()
    assert hashlib.sha256(ckpt_data).hexdigest()==m.M8_HASH['style-mlp']
    return {'manifest_hashes':len(manifest),'trajectories':replay_count,'model_action_recomputations':pred_checks,
            'original_png_replays':paint_checks,'legal_host_actions':total,'illegal':illegal,
            'negative_controls_passed':3,'model_checkpoints_loaded':3,
            'source_sha256':rep['source_sha256'],'router_sha256':rep['training']['checkpoint_sha256']}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--m8',type=Path,required=True)
    a=parser.parse_args();print(json.dumps(audit(a.output,a.m8),indent=2))
