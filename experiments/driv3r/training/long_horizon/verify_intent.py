"""Independently check the original intent-ablation source, checkpoints and decisions.

Manifest SHA-256 binds original bytes but is NOT a cryptographic signature.
No model inference is counted as a CyberSand native receipt.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .intent_lab import ARMS, choose, VERSION
from .policy import Ranker
from .train import MAX_RECOVERIES, sha_file, source_pins
from .world import World, specs, digest


def verify(root: Path):
    report=json.loads((root/'report.json').read_text())
    hashes=json.loads((root/'SHA256SUMS.json').read_text())
    actual={str(f.relative_to(root)):sha_file(f)
            for f in sorted(root.rglob('*')) if f.is_file() and f.name!='SHA256SUMS.json'}
    if actual!=hashes:raise ValueError('Artifact manifest mismatch')
    if report['version']!=VERSION or report['heldout_read'] or report['native_executed'] or report['native_receipts']:
        raise ValueError('Unapproved experiment scope or provenance')
    if report['source_sha256'] != {**source_pins(), 'intent_lab.py':sha_file(Path(__file__).with_name('intent_lab.py'))}:
        raise ValueError('Exact model source mismatch')
    records={(r['seed'],r['arm']):r for r in report['checkpoint_records']}
    models={}
    for key,row in records.items():
        checkpoint=root/row['filename']
        if sha_file(checkpoint)!=row['sha256']:raise ValueError('Original checkpoint mismatch')
        model=Ranker.restore(checkpoint)
        if model.updates!=row['updates'] or model.parameter_count!=row['parameters']:
            raise ValueError('Fake optimizer evidence')
        models[key]=model
    worlds={s.ident:s for s in specs('development',report['dev_worlds_per_family'])}
    checked=0;counts=Counter();teacher_match=Counter();teacher_total=Counter()
    for path in sorted((root/'traces').rglob('*.json')):
        row=json.loads(path.read_text())
        if row.get('version')!=VERSION or row.get('split')!='development' or row.get('spec_id') not in worlds:
            raise ValueError('Heldout/invalid world in original traces')
        spec=worlds[row['spec_id']]
        w=World.create(spec)
        if w.base_sha()!=row['source_base_sha256']:
            raise ValueError('Wrong World source')
        arm=row['arm'];key=(row['model_seed'],arm)
        if key not in models: raise ValueError('Missing trained model identity')
        events=row['events'];commands=row['commands']
        if len(events)!=len(commands) or row['metrics']['decisions']!=len(events):
            raise ValueError('Missing decisions')
        for i,(event,cmd) in enumerate(zip(events,commands)):
            intent = row['initial_model_intent'] if row['intent_switch_turn'] is None or i<row['intent_switch_turn'] else row['post_switch_intent']
            if cmd['turn']!=i or cmd['model_only_requested_mode']!=intent:
                raise ValueError('Untraceable intent scheduling')
            selected,probs,teacher_choice=choose(w,models[key],row['horizon'],intent,
                arm=='intent_blind',arm=='intent_aware_cross_features')
            if (cmd['chosen']!=selected or cmd['probabilities']!=probs or
                cmd['teacher_argmax']!=teacher_choice or cmd['pre_sha256']!=w.state_sha() or
                cmd['host_legal_menu']!=w.menu()):
                raise ValueError('Policy logits, host legal menu, or state receipt falsified')
            if w.apply(selected,MAX_RECOVERIES)!=event:
                raise ValueError('Original step/material replay mismatch')
            teacher_match[arm]+=int(selected==teacher_choice)
            teacher_total[arm]+=1
            counts[(arm,selected)]+=1
        if (w.state_sha()!=row['final_state_sha256'] or digest(events)!=row['trajectory_sha256']
            or row['ink_cells']!=sorted([*p] for p in w.ink) or w.metrics(row['horizon'])!=row['metrics']):
            raise ValueError('Original geometry/metrics falsified')
        checked+=1
    if checked!=report['episodes']:raise ValueError('Episode count mismatch')
    for arm in ARMS:
        a=report['by_arm'][arm]
        if a['episodes']!=sum(1 for p in (root/'traces').rglob('*.json') if json.loads(p.read_text())['arm']==arm):
            raise ValueError('Episode summary mismatch')
        if a['action_distribution']!={act:n for (name,act),n in counts.items() if name==arm}:
            raise ValueError('Action distribution mismatch')
        if abs(a['teacher_argmax_agreement']-teacher_match[arm]/teacher_total[arm])>1e-6:
            raise ValueError('Agreement mismatch')
    return {'verified_original_episodes':checked,'checkpoint_count':len(models),
            'verified_sequential_decisions':sum(counts.values()),
            'trained_weights_restored':True,'original_checkpoint_hashes_verified':True,
            'neural_probabilities_recomputed_from_checkpoint':True,
            'source_and_manifest_match':True,'native_receipts':0}


def main():
    p=argparse.ArgumentParser();p.add_argument('artifact_directory',type=Path)
    print(json.dumps(verify(p.parse_args().artifact_directory),sort_keys=True))

if __name__=='__main__':main()
