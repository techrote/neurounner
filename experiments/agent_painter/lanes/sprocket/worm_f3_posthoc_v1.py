#!/usr/bin/env python3
"""F4: source-pinned retrospective of F3's REAL model-score receipts.

No model forward, optimizer or native action here. The flat and frozen-score
choices are OFF-POLICY on observed trained-path states, not full W0 rollouts.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import zipfile

ACTIONS=('STRAIGHT','LEFT','RIGHT','BRANCH','RETRACE','STOP')
ZIP_SHA='5eb64e5f7d5335809f631aa11caf9edbfcf1a67f546601d5d14e795cc61f7af6'
CHECKPOINT='7328159322adb831f2a1b758a70740fbd39ee17be79592c660e78d48c36df98a'
SEEDS={8111:'greebles',8123:'pipework',8147:'architecture',8161:'greebles'}

def sha(b):
    return hashlib.sha256(b).hexdigest()

def sample(logits,legal,draw,temp=2.25):
    if set(logits)!=set(ACTIONS) or any(not math.isfinite(float(v)) for v in logits.values()):
        raise ValueError('bad model logits')
    if not legal or len(legal)!=len(set(legal)) or set(legal)-set(ACTIONS):
        raise ValueError('bad host legality')
    if not math.isfinite(temp) or temp<=0 or not 0<=draw<1:
        raise ValueError('bad temperature/draw')
    actions=[a for a in ACTIONS if a in legal]
    scores=[logits[a]/temp for a in actions]
    m=max(scores)
    weights=[math.exp(v-m) for v in scores]
    threshold=0
    for a,v in zip(actions,weights):
        threshold+=v/sum(weights)
        if draw<threshold:return a
    return actions[-1]

def inspect(archive,expected_sha=ZIP_SHA):
    archive=Path(archive)
    if sha(archive.read_bytes())!=expected_sha:raise ValueError('original outer SHA mismatch')
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise ValueError('original ZIP CRC mismatch')
        names=set(z.namelist())
        hashes=0
        for slot in (0,1):
            for line in z.read(f'fleet-{slot}/SHA256SUMS.txt').decode().splitlines():
                expected,relative=line.split(None,1)
                member=f'fleet-{slot}/'+relative.removeprefix('./')
                if member not in names or sha(z.read(member))!=expected:
                    raise ValueError('original member SHA mismatch: '+member)
                hashes+=1
        fleet=json.loads(z.read('four-way-audit.json'))
        ids=fleet['original_worm_processes']
        if len(ids)!=4 or len({(slot,w['pid'],w['lease'],w['seed']) for slot,w in ids})!=4:
            raise ValueError('four unique physical worker identities not proven')
        if not fleet['not_native'] or fleet['optimizer_steps_this_run']!=0 or any(
                w['checkpoint_inventory_sha256']!=CHECKPOINT or
                w['real_trained_six_score_forwards']!=128 for slot,w in ids):
            raise ValueError('F3 model/host evidence drift')
        worlds=[]
        ticks=[]
        for seed,family in SEEDS.items():
            matching=[name for name in names if '/original-png-and-json/' in name
                      and '-sprocket-f1-' in name and name.endswith(f'-{seed}.json')]
            if len(matching)!=1:raise ValueError('missing original learned W0 trace')
            path=matching[0]
            trace=json.loads(z.read(path))
            if trace['family']!=family or trace['seed']!=seed or trace['steps_executed']!=128:
                raise ValueError('world identity drift')
            events=trace['events']
            receipts=trace['score_receipts']
            coords=trace['trajectory']
            if len(events)!=128 or len(receipts)!=128 or len(coords)!=129:
                raise ValueError('length drift')
            frozen=receipts[0]['full_six_token_logits']
            counts=Counter()
            first_difference=None
            for tick,(event,r) in enumerate(zip(events,receipts)):
                if r['tick']!=tick or event['observation']['tick']!=tick:
                    raise ValueError('tick drift')
                if [event['observation']['x'],event['observation']['y']]!=coords[tick] or event['position_after']!=coords[tick+1]:
                    raise ValueError('W0 application/trajectory drift')
                legal=r['host_policy_legal']
                if set(legal)-set(event['observation']['legal']):raise ValueError('illegal host mask')
                scores=r['full_six_token_logits']
                draw=r['host_uniform_draw']
                temp=r['temperature']
                learned=sample(scores,legal,draw,temp)
                flat=sample(dict.fromkeys(ACTIONS,0.),legal,draw,temp)
                constant=sample(frozen,legal,draw,temp)
                if learned!=event['applied'] or learned!=r['actual_W0_applied'] or learned!=r['host_selected']:
                    raise ValueError('original learned choice cannot be replayed')
                if flat!=r['same_observation_flat'] or (flat!=learned)!=r['differs_from_flat']:
                    raise ValueError('original paired-flat choice cannot be replayed')
                if not r['model_forward'] or not r['trained_adapter_active'] or r['is_eos5_completion'] or not event['raw_legal']:
                    raise ValueError('model-score provenance corrupted')
                if r['raw_unmasked_top_action']!=max(ACTIONS,key=lambda a:scores[a]):
                    raise ValueError('raw score top mismatch')
                if r['host_stop_deferred']!=(tick<126):raise ValueError('host STOP policy drift')
                if first_difference is None and learned!=constant:first_difference=tick
                counts['learned_differs_from_flat']+=learned!=flat
                counts['learned_differs_from_frozen_first']+=learned!=constant
                counts['raw_top_BRANCH']+=r['raw_unmasked_top_action']=='BRANCH'
                counts['raw_top_STRAIGHT']+=r['raw_unmasked_top_action']=='STRAIGHT'
                counts['STOP_deferred']+=r['host_stop_deferred']
                counts['STOP_selected']+=learned=='STOP'
                counts['BRANCH_selected']+=learned=='BRANCH'
                counts['RETRACE_selected']+=learned=='RETRACE'
                counts['blocked_STRAIGHT_observed']+='STRAIGHT' not in event['observation']['legal']
                ticks.append({'seed':seed,'family':family,'tick':tick,'actual_learned':learned,
                              'flat_same_observation_only':flat,'frozen_first_same_observation_only':constant,
                              'local_score_change_switched_action':learned!=constant})
            worlds.append({'seed':seed,'family':family,'first_local_difference_tick':first_difference,
                           'original_unique_cells':trace['metrics']['unique_trail_cells'],
                           'counts':dict(counts)})
        total=Counter()
        for world in worlds:total.update(world['counts'])
        return {'category':'off-policy same-observation F3 first-logit diagnostic (NO full alternative W0 trails)',
                'source_commit':'e56e0aac49b310cc1e00aafe3ffeddb90eca6e38',
                'original_run':43769025429,'original_artifact':14183924474,
                'zip_sha256':expected_sha,'crc_members':len(names),'verified_original_member_sha256':hashes,
                'checkpoint_inventory_sha256':CHECKPOINT,'original_model_forwards':512,
                'no_new_forwards':True,'no_optimizer_steps':True,'not_native':True,
                'totals':dict(total),'worlds':worlds,'same_observation_off_policy_ticks':ticks}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('original_f3_zip',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=inspect(args.original_f3_zip)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'totals':result['totals'],'sha256':sha(args.output.read_bytes())},sort_keys=True))
if __name__=='__main__':main()
