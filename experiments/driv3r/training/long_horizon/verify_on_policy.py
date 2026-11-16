"""Independent exact-source, original trajectory, checkpoint and gallery verifier.

Excludes HELDOUT and rejects rehashed-but-falsified policy probabilities or
future-lookahead. PNGs must reproduce from replayed synthetic state.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .on_policy_adaptation import ARMS, verify_episode, VERSION
from .policy import Ranker
from .train import _png, sha_file, source_pins


def verify_run(root: Path):
    report=json.loads((root/'report.json').read_text())
    if (report['version']!=VERSION or report['heldout_read'] or report['native_executed'] or
        report['slm_loaded'] or report['paid_external_gpu_or_api'] or
        report['native_64_logical_move_ceiling']!='unchanged'):
        raise ValueError('Forbidden source/model/native execution claim')
    if report['source_sha256']!=source_pins():
        raise ValueError('Training source pins differ')
    for name,hashval in report['additional_source_sha256'].items():
        if name!='on_policy_adaptation.py' or sha_file(Path(__file__).parent/name)!=hashval:
            raise ValueError('On-policy source identity differs')
    expected=json.loads((root/'SHA256SUMS.json').read_text())
    actual={str(p.relative_to(root)):sha_file(p) for p in root.rglob('*')
            if p.is_file() and p.name!='SHA256SUMS.json'}
    if actual!=expected:
        raise ValueError('Artifact file or source manifest tampered')
    models={}
    for arm in ARMS:
        info=report['training']['extra'][arm]
        cp=root/info['checkpoint']
        if sha_file(cp)!=info['checkpoint_sha256']:
            raise ValueError('Checkpoint provenance differs')
        models[arm]=Ranker.restore(cp)
        if models[arm].updates!=4*report['updates_per_stage']:
            raise ValueError('Adam update count differs')
    count=0
    for path in sorted((root/'trajectories').rglob('*.json')):
        episode=json.loads(path.read_text())
        model=models[episode['arm']]
        if episode['checkpoint_sha256']!=report['training']['extra'][episode['arm']]['checkpoint_sha256']:
            raise ValueError('Episode/weight identity mismatch')
        world=verify_episode(episode,model)
        image=path.with_suffix('.png')
        if not image.is_file():
            raise ValueError('Original gallery missing')
        with TemporaryDirectory() as td:
            local=Path(td)/'replay.png'
            _png(local,world)
            if image.read_bytes()!=local.read_bytes():
                raise ValueError('PNG not from source-matched synthetic trajectory')
        count+=1
    if count!=report['episode_count']:
        raise ValueError('Episode count mismatch')
    if set(report['training']['extra'])!=set(ARMS):
        raise ValueError('Missing or extra checkpoint arm')
    return {'status':'PASS','source_replay_episodes':count,
            'original_artifact_files':len(actual),'checkpoints':len(models),
            'heldout_read':False,'native_receipts':0}


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    print(json.dumps(verify_run(p.parse_args().root),sort_keys=True))

if __name__=='__main__':main()
