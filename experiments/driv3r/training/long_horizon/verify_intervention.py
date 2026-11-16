"""Independent byte/source/menu/probability audit for original DEV-only recovery stress.

Refuse forged model probabilities or slipped host action, even after manifest rehash.
Reads only manifest / original trajectories and train-owned source, never held-out.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import random
from pathlib import Path
import tempfile

from .intervention_lab import TRAIN_ARMS, PROBE_TURNS, forced_choice
from .policy import Ranker, teacher, random_legal_persistent
from .train import HORIZONS, MAX_RECOVERIES, sha_file, source_pins, _png
from .world import World, digest, specs


def verify_episode(obj:dict, checkpoint:Ranker|None):
    if obj['split']!='development' or obj['horizon'] not in HORIZONS:
        raise ValueError('Held-out or unknown synthetic horizon in stress evidence')
    sources=specs('development',2)
    sources += [replace(s,challenge='trap_corridor') for s in sources if s.seed%2==0]
    matches=[s for s in sources if s.ident==obj['world_id']]
    if len(matches)!=1:
        raise ValueError('World identity is not one of predeclared DEVELOPMENT specs')
    spec=matches[0]
    if (spec.family,spec.seed,spec.mode,spec.challenge)!=(obj['family'],obj['seed'],obj['mode'],obj['challenge']):
        raise ValueError('World identity fields tampered')
    world=World.create(spec)
    if world.base_sha()!=obj['base_sha256']:
        raise ValueError('World source is inconsistent')
    events=obj['events']
    if digest(events)!=obj['trajectory_sha256']:
        raise ValueError('Trajectory body SHA mismatch')
    if any(e['index']!=i for i,e in enumerate(events)):
        raise ValueError('Noncontiguous original decisions')
    arm=obj['arm']; seed=obj['policy_seed']; horizon=obj['horizon']
    if (checkpoint is None)!=(arm not in TRAIN_ARMS):
        raise ValueError('Missing learned weight linkage')
    rng=random.Random(seed*137+spec.seed*17+horizon)
    recorded_probs={v['decision']:v for v in obj['decision_probabilities']}
    attempts=[]; applied=[]; actions=Counter()
    for i,e in enumerate(events):
        if arm=='teacher_mixture':
            chosen=teacher(world,horizon,rng)[0];probs=None
        elif arm=='random_legal_persistent':
            chosen=random_legal_persistent(world,rng);probs=None
        else:
            chosen,probs=checkpoint.choose(world,horizon)
            if i not in recorded_probs:
                raise ValueError('Missing original neural probability at decision')
            original=recorded_probs[i]
            if original['chosen']!=e['chosen'] or set(original['probabilities'])!=set(probs):
                raise ValueError('Original candidate probability/menu mismatch')
            if any(abs(float(original['probabilities'][k])-v)>2.1e-8
                   for k,v in probs.items()) or abs(sum(original['probabilities'].values())-1)>1e-6:
                raise ValueError('Forged decision probability rejected')
        option=chosen
        if obj['perturbed'] and i in PROBE_TURNS:
            alt=forced_choice(world,chosen,horizon)
            attempts.append({'decision':i,'preferred':chosen,'selected_alternative':alt,
                             'world_pre_sha256':world.state_sha(),
                             'reason':'no_legal_alternative' if alt is None else 'lowest_local_teacher_score_other_legal_move'})
            if alt is not None:
                option=alt
                applied.append((i,chosen,option))
        if option!=e['chosen']:
            raise ValueError('Model/teacher/random action differs from original transcript')
        if world.apply(option,MAX_RECOVERIES)!=e:
            raise ValueError('Original host action/material/cursor receipt differs')
        actions[option]+=1
    if len(recorded_probs)!=(len(events) if arm in TRAIN_ARMS else 0):
        raise ValueError('Unexpected or missing model probability records')
    if attempts!=obj['attempts']:
        raise ValueError('Forced intervention schedule or provenance tampered')
    if [(r['decision'],r['preferred'],r['applied']) for r in obj['actual_interventions']]!=applied:
        raise ValueError('Actual intervention count/identity differs')
    for x in obj['actual_interventions']:
        t=x['decision']
        if x['pre_sha256']!=events[t]['pre_sha256'] or x['ink_before']!=(events[t-1]['ink_count'] if t else 0):
            raise ValueError('Forced intervention prior World state differs')
        if x['new_ink_next_16_decisions']!=sum(len(z['painted']) for z in events[t+1:t+17]):
            raise ValueError('Forced intervention progress forged')
        count=sum(z['logical_moves']-(events[j-1]['logical_moves'] if j else 0)
                  for j,z in enumerate(events[t+1:t+17],start=t+1))
        if x['cursor_moves_next_16_decisions']!=count:
            raise ValueError('Forced intervention logical moves forged')
    if (world.state_sha()!=obj['final_state_sha256'] or
        world.metrics(horizon)!=obj['metrics'] or
        dict(sorted(actions.items()))!=obj['action_distribution']):
        raise ValueError('Forged final state, metrics or action distribution')
    if (obj['native_executed'] or obj['native_receipts'] or
        world.ink.intersection(world.materials) or world.ink.intersection(world.protected)):
        raise ValueError('Synthetic claim or illegal edit')
    return world


def verify_run(root:Path):
    report=json.loads((root/'report.json').read_text())
    if (report['heldout_read'] or report['native_executed'] or report['slm_models_loaded'] or
        report['paid_gpu_or_api'] or report['native_64_logical_move_ceiling']!='unchanged'):
        raise ValueError('Run changed protected source/operation boundaries')
    if any(sha_file(Path(__file__).parent/name)!=sha for name,sha in
           report['source_sha256'].items() if name!='intervention_lab.py'):
        raise ValueError('Source replay lacks original Python hash')
    if sha_file(Path(__file__).parent/'intervention_lab.py')!=report['source_sha256']['intervention_lab.py']:
        raise ValueError('Training source changed')
    digest_map=json.loads((root/'SHA256SUMS.json').read_text())
    actual={str(p.relative_to(root)):sha_file(p) for p in root.rglob('*')
            if p.is_file() and p.name!='SHA256SUMS.json'}
    if digest_map!=actual:
        raise ValueError('Original evidence SHA256 manifest mismatch')
    models={r['arm']:Ranker.restore(root/r['checkpoint']) for r in report['checkpoint_records']}
    for entry in report['checkpoint_records']:
        if sha_file(root/entry['checkpoint'])!=entry['sha256']:
            raise ValueError('Actual checkpoint SHA mismatch')
    checked=0
    for p in sorted((root/'trajectories').rglob('*.json')):
        obj=json.loads(p.read_text())
        model=models.get(obj['arm'])
        reconstructed=verify_episode(obj,model)
        img=p.with_suffix('.png')
        if not img.is_file():raise ValueError('Missing original geometry PNG')
        with tempfile.TemporaryDirectory() as td:
            copy=Path(td)/'replayed.png'
            _png(copy,reconstructed)
            if copy.read_bytes()!=img.read_bytes():
                raise ValueError('Original geometry PNG does not match replay')
        checked+=1
    if checked!=report['episode_count']:
        raise ValueError('Episode count mismatch')
    return {'status':'PASS','verified_episodes':checked,'checkpoints':len(models),
            'original_artifact_bytes_verified':len(actual),
            'heldout_read':False,'native_receipts':0}


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    print(json.dumps(verify_run(p.parse_args().root),sort_keys=True))

if __name__=='__main__':main()
