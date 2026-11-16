"""GAL-002 original-material replay: never re-run neural/linear policy inference.

Replays host-offered action IDs, source/lease/material tokens and every original
synthetic receipt against the same versioned external host. This establishes
record self-consistency, not source-independent native/game authentication.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from lab import World, TinyPolicy, FEATURES, ARMS, sha
import random
from matched_eval import file_sha, DECISION_MARKS, INK_TARGETS


class AuditRefusal(ValueError):
    pass


def audit(directory):
    out=Path(directory)
    original=json.loads((out/'matched-episodes.json').read_text())
    traces=json.loads((out/'matched-traces.json').read_text())
    report=json.loads((out/'metrics.json').read_text())
    if report.get('original_heldout_rows_sha256') != file_sha(out/'matched-episodes.json'):
        raise AuditRefusal('metrics/rows hash mismatch')
    if report.get('original_heldout_traces_sha256') != file_sha(out/'matched-traces.json'):
        raise AuditRefusal('metrics/traces hash mismatch')
    if sorted(original)!=sorted(traces):raise AuditRefusal('panels differ')
    policies={}
    for arm in ARMS:
        path=out/'checkpoints'/(arm+'.json')
        if file_sha(path)!=report['training'][arm]['checkpoint_sha256']:
            raise AuditRefusal('checkpoint hash mismatch')
        ck=json.loads(path.read_text())
        if ck['features']!=list(FEATURES) or ck['arm']!=arm or len(ck['weights'])!=len(FEATURES):
            raise AuditRefusal('checkpoint schema mismatch')
        model=TinyPolicy();model.weights=ck['weights'];policies[arm]=model
    for filename,digest in report['source_files_sha256'].items():
        if file_sha(Path(__file__).with_name(filename))!=digest:
            raise AuditRefusal('source byte drift')
    count=0;turns=0;sampled_model_choices=0
    for panel,rows in original.items():
        if len(rows)!=len(traces[panel]):raise AuditRefusal('panel count mismatch')
        for row,events in zip(rows,traces[panel]):
            arm=row['arm'];w=World(row['seed'],row['family'],arm,lease=row['lease'])
            if sha(w.original)!=row['initial_material_sha256']:
                raise AuditRefusal('original World material mismatch')
            if row.get('panel')!=panel or row.get('policy_checkpoint_arm') not in ARMS:
                raise AuditRefusal('panel/model identity mismatch')
            policy=policies[row['policy_checkpoint_arm']]
            rng=random.Random(f'gal002:{row["seed"]}:same-coins')
            first={str(k):None for k in INK_TARGETS}
            cumulative=[]
            for tick,receipt in enumerate(events):
                if receipt.get('turn')!=tick or receipt.get('lease')!=row['lease']:
                    raise AuditRefusal('turn / lease mismatch')
                # Host reproduces offered action set and token from actual prior state.
                menu,token=w.offer()
                action=receipt.get('action_id')
                sampled_u=rng.random()  # one exact draw on EVERY tick, including unsampled
                if tick < 3 or tick % 16 == 0:
                    names=list(menu)
                    vecs=[w.features(n,menu[n]) for n in names]
                    probs=policy.probabilities(vecs)
                    cumulative_prob=0;expected=names[-1]
                    for n,prob in zip(names,probs):
                        cumulative_prob+=prob
                        if sampled_u<=cumulative_prob:
                            expected=n;break
                    if action!=expected:
                        raise AuditRefusal('model-selected action mismatch')
                    sampled_model_choices+=1
                if receipt.get('original_response')!=f'tiny-linear-policy:{action}':
                    raise AuditRefusal('model response mismatch')
                if action not in menu or receipt.get('offered_sha256')!=token:
                    raise AuditRefusal('action not in exact offered catalog')
                actual=w.apply(action,token,original_response=receipt.get('original_response'))
                if json.loads(json.dumps(actual,sort_keys=True)) != receipt:
                    raise AuditRefusal('per-action material/receipt mismatch')
                cumulative.append(len(w.own))
                for k in INK_TARGETS:
                    if first[str(k)] is None and len(w.own)>=k:
                        first[str(k)]=w.decisions
            if len(events)!=w.decisions or w.decisions!=row['decisions']:
                raise AuditRefusal('action count mismatch')
            if sha(w.trace)!=row['host_receipts_sha256']:
                raise AuditRefusal('trace digest mismatch')
            if sha(w.material)!=row['final_material_sha256']:
                raise AuditRefusal('material hash mismatch')
            if w.logical_moves!=row['logical_moves'] or len(w.own)!=row['new_ink']:
                raise AuditRefusal('move/ink mismatch')
            if w.stopped!=row['stopped'] or first!=row['decisions_to_ink_target']:
                raise AuditRefusal('stop/target mismatch')
            by={str(k):cumulative[min(k,len(cumulative))-1] if cumulative else 0
                for k in DECISION_MARKS}
            reached={str(k):len(events)>=k for k in DECISION_MARKS}
            if by!=row['ink_by_decision_budget'] or reached!=row['actually_reached_decision']:
                raise AuditRefusal('decision budget claims mismatch')
            count+=1;turns+=w.decisions
    return {'schema':'gal-002-original-replay/1', 'qualification':'same-host-version synthetics',
            'replayed_episodes':count,'replayed_actions':turns,'model_selection_sampled':sampled_model_choices,'native_receipts':0,
            'resident_model_processes':0,'source_sha256':report['source_files_sha256'],
            'original_heldout_traces_sha256':report['original_heldout_traces_sha256']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args()
    print(json.dumps(audit(args.directory),sort_keys=True,indent=2))
