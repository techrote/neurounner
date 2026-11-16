"""GAL-002: matched-world, equal-decision/equal-ink action-language ablations.

Research-only; outputs are external *synthetic* 48x48 materials, not Godot,
WORM-001 W0, neural SLM inference, or game/native receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

from lab import (ARMS, DECISION_CAP, FEATURES, INK, START, TinyPolicy, World,
                 png, sha, train)

FAMILIES = ('empty', 'pillars', 'gates', 'scaffold', 'scatter')
DECISION_MARKS = (16, 32, 64)
INK_TARGETS = (32, 64)
SCHEMA = 'gal-002-matched-external-synthetic/1'


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(policy, arm, seed, family, *, noise_label='same-coins', episode_label='default'):
    """One genuine tiny-policy rollout; STOP remains offered and terminal.

    Deterministic random stream is independent of action-arm; identical uniform
    samples are used for matched policy comparisons, not identical actions.
    Material changes stay host-owned. No native per-cell receipt is claimed.
    """
    w = World(seed, family, arm, lease=f'gal002-{episode_label}-{seed}-{family}-{arm}')
    rng = random.Random(f'gal002:{seed}:{noise_label}')
    source_digest = sha(w.original)
    cumulative_ink = []
    first_targets = {str(t): None for t in INK_TARGETS}
    while not w.stopped and w.decisions < DECISION_CAP:
        before_ink = len(w.own)
        option, token, _, _, _, _ = policy.choose(w, rng)
        receipt = w.apply(option, token, original_response=f'tiny-linear-policy:{option}')
        assert len(w.own) - before_ink == receipt['new_ink'], 'material delta mismatch'
        assert all(w.material[w.idx(p)] == w.original[w.idx(p)]
                   for p in w.protected), 'protected material modified'
        assert all(w.material[w.idx(p)] == INK for p in w.own), 'own material mismatch'
        cumulative_ink.append(len(w.own))
        for t in INK_TARGETS:
            if first_targets[str(t)] is None and len(w.own) >= t:
                first_targets[str(t)] = w.decisions
    assert len(cumulative_ink) == w.decisions == len(w.trace)
    # Ink by budget counts terminal STOP as terminal material held for remaining
    # wall-clock decision opportunities; DOES NOT pretend that decisions occurred.
    by_budget = {str(k): cumulative_ink[min(k, len(cumulative_ink))-1]
                 if cumulative_ink else 0 for k in DECISION_MARKS}
    reached_marks = {str(k): w.decisions >= k for k in DECISION_MARKS}
    return w, {
        'seed': seed, 'family': family, 'arm': arm, 'initial_material_sha256': source_digest,
        'lease': w.lease, 'evidence_class': 'external_synthetic_trained_tiny_policy',
        'native_receipts': 0, 'resident_slm_processes': 0,
        'decisions': w.decisions, 'new_ink': len(w.own),
        'logical_moves': w.logical_moves, 'stopped': w.stopped,
        'ink_by_decision_budget': by_budget, 'actually_reached_decision': reached_marks,
        'decisions_to_ink_target': first_targets,
        'host_receipts_sha256': sha(w.trace), 'final_material_sha256': sha(w.material),
    }


def summarise(rows):
    n = len(rows)
    assert n and len({(r['family'], r['seed']) for r in rows}) == n
    mean = lambda field: round(statistics.mean(r[field] for r in rows), 4)
    v = {
        'worlds': n, 'mean_new_ink': mean('new_ink'),
        'mean_decisions': mean('decisions'), 'mean_logical_moves': mean('logical_moves'),
        'pooled_ink_per_decision': round(sum(r['new_ink'] for r in rows) /
                                         max(1, sum(r['decisions'] for r in rows)), 4),
        'stop_fraction': round(sum(r['stopped'] for r in rows)/n,4),
        'ink_at_decision_budget_mean': {},
        'actually_reached_decision_fraction': {},
        'target_reached_fraction': {}, 'mean_calls_if_target_reached': {},
        'record_digest': sha([r['host_receipts_sha256'] for r in rows]),
    }
    for k in DECISION_MARKS:
        v['ink_at_decision_budget_mean'][str(k)] = round(statistics.mean(
            r['ink_by_decision_budget'][str(k)] for r in rows),4)
        v['actually_reached_decision_fraction'][str(k)] = round(sum(
            r['actually_reached_decision'][str(k)] for r in rows)/n,4)
    for k in INK_TARGETS:
        got=[r['decisions_to_ink_target'][str(k)] for r in rows
             if r['decisions_to_ink_target'][str(k)] is not None]
        v['target_reached_fraction'][str(k)]=round(len(got)/n,4)
        v['mean_calls_if_target_reached'][str(k)]=round(statistics.mean(got),4) if got else None
    return v


def paired_target(a,b,target):
    """Strict matched-success comparison; failures remain in denominator."""
    ka = {(r['seed'],r['family']):r for r in a}
    kb = {(r['seed'],r['family']):r for r in b}
    if ka.keys()!=kb.keys(): raise ValueError('paired worlds differ')
    both=[]; a_only=0; b_only=0; neither=0
    for key in sorted(ka):
        x=ka[key]['decisions_to_ink_target'][str(target)]
        y=kb[key]['decisions_to_ink_target'][str(target)]
        if x is not None and y is not None: both.append(x-y)
        elif x is not None: a_only+=1
        elif y is not None: b_only+=1
        else:neither+=1
    return {'target_ink':target, 'worlds_total':len(ka),
            'both_reached':len(both),'first_only':a_only,'second_only':b_only,
            'neither_reached':neither,
            'mean_call_saving_second_vs_first_if_both_reached':
                 round(statistics.mean(both),4) if both else None,
            'paired_differences_first_minus_second':both}


def evaluate_panel(policies, *, count=30, seed_base=97000,
                   gallery_dir=None, gallery_per_arm=2, all_traces=None):
    if not 5 <= count <= 200: raise ValueError('bounded heldout count')
    if not 95000 <= seed_base <= 1000000 or seed_base+count >= 1000000:
        raise ValueError('heldout custody seed range')
    if gallery_per_arm < 0 or gallery_per_arm > 5:
        raise ValueError('bounded gallery')
    trials=[(seed_base+i,FAMILIES[i%len(FAMILIES)]) for i in range(count)]
    arms=[('cell_trained','cell','cell'),('sweep_trained','sweep','sweep'),
          ('sweep_policy_cell','cell','sweep'),('cell_policy_sweep','sweep','cell')]
    arms += [(a+'_trained',a,a) for a in ARMS if a not in ('cell','sweep')]
    rows_by={}
    for label,arm,policy_name in arms:
        rows=[]
        for i,(seed,family) in enumerate(trials):
            # use same noise-label for all panels, controlling the random coins
            w,row=measure(policies[policy_name],arm,seed,family,episode_label=label)
            row['panel']=label;row['policy_checkpoint_arm']=policy_name
            rows.append(row)
            if all_traces is not None:
                all_traces.setdefault(label, []).append(w.trace)
            if gallery_dir and i<gallery_per_arm:
                gallery_dir=Path(gallery_dir)
                gallery_dir.mkdir(parents=True,exist_ok=True)
                stem=f'{label}-{family}-{seed}'
                png(gallery_dir/(stem+'.png'),w)
                (gallery_dir/(stem+'.json')).write_text(json.dumps(w.trace,sort_keys=True,indent=2)+'\n')
        rows_by[label]=rows
    comparisons={
        'adapted_cell_vs_adapted_sweep': {},
        'same_sweep_policy_cell_vs_sweep': {},
        'same_cell_policy_cell_vs_sweep': {},
    }
    pairs=(('adapted_cell_vs_adapted_sweep','cell_trained','sweep_trained'),
           ('same_sweep_policy_cell_vs_sweep','sweep_policy_cell','sweep_trained'),
           ('same_cell_policy_cell_vs_sweep','cell_trained','cell_policy_sweep'))
    for label,first,second in pairs:
        comparisons[label]={str(k):paired_target(rows_by[first],rows_by[second],k)
                            for k in INK_TARGETS}
    return rows_by, { 'panel_summary':{label:summarise(rows) for label,rows in rows_by.items()},
                      'paired_target_comparisons':comparisons }


def write_run(output,*,training_worlds=60, heldout_worlds=30, gallery_per_arm=2):
    if not 1 <= training_worlds <= 250: raise ValueError('bounded training count')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    checkpoints=out/'checkpoints';checkpoints.mkdir(exist_ok=True)
    policy_by={}; trainings={}
    for arm in ARMS:
        policy,updates=train(arm,training_worlds,base_seed=20000)
        policy_by[arm]=policy
        d={'schema':'gal-002-plain-linear/1', 'arm':arm,
           'features':list(FEATURES),'weights':policy.weights,
           'updates':updates,'training_worlds':training_worlds,'training_seed_base':20000}
        file=checkpoints/(arm+'.json')
        file.write_text(json.dumps(d,sort_keys=True,indent=2)+'\n')
        trainings[arm]={'updates':updates,'checkpoint_sha256':file_sha(file)}
    all_traces={}
    rows,stats=evaluate_panel(policy_by,count=heldout_worlds,
                              gallery_dir=out/'gallery',gallery_per_arm=gallery_per_arm,
                              all_traces=all_traces)
    trials=out/'matched-episodes.json'
    trials.write_text(json.dumps(rows,sort_keys=True,indent=2)+'\n')
    traces=out/'matched-traces.json'
    traces.write_text(json.dumps(all_traces,sort_keys=True,separators=(',',':'))+'\n')
    report={'schema':SCHEMA, 'truth':{'synthetic_only':True,
        'not_native_godot':True,'not_worm_001_w0':True,
        'not_resident_slm':True,'not_native_receipts':True},
        'train_seed_range':[20000,20000+training_worlds-1],
        'heldout_seed_range':[97000,97000+heldout_worlds-1],
        'train_heldout_disjoint': True,
        'training':trainings, 'heldout_worlds':heldout_worlds,
        'matched_source_material_per_seed':True,
        'stop_unmasked_in_all_arms':True,
        'panel_random_coins_seeded_identically':True,
        'censored_targets_not_imputed':True,
        'source_files_sha256':{f:file_sha(Path(__file__).with_name(f)) for f in
           ('lab.py','matched_eval.py','audit_original.py')},
        'original_heldout_rows_sha256':file_sha(trials),
        'original_heldout_traces_sha256':file_sha(traces),
        **stats}
    (out/'metrics.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    # Manifest not self-referential and source weights remain only in output ZIP.
    items=sorted(p for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    (out/'SHA256SUMS').write_text(''.join(file_sha(p)+'  '+p.relative_to(out).as_posix()+'\n'
                                      for p in items))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--train',type=int,default=60)
    parser.add_argument('--heldout',type=int,default=30)
    parser.add_argument('--gallery',type=int,default=2)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    report=write_run(args.output,training_worlds=args.train,
                     heldout_worlds=args.heldout,gallery_per_arm=args.gallery)
    print(json.dumps(report,sort_keys=True,indent=2))
