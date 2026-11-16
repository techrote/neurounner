#!/usr/bin/env python3
"""Size/world-count matched topology ablation using TRAIN-only host-applied episodes.

No CyberSand game changes. All three families share a world group, so world
success (not individual family episodes) is the independent evaluation unit.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import zipfile

from recovery_curriculum import make_world, teacher_data, fit, sha, distance_field
from domain_recovery_v3 import (collect_on_worlds, make_slalom_world,
    validate_partitions, world_digest)
from on_policy_recovery_v2 import (make_cross_world, dedup_corrections,
    evaluate, fit_student, export_corrections, verify_sources)

SCHEMA = 'cybersand/synthetic-recovery-topology-size-control-v4'
SOURCE = tuple(range(1000, 1024))
ORIGINAL_ONLY = tuple(range(2300, 2312))
MIXED_ORIGINAL = tuple(range(2300, 2306))
MIXED_BARRIER = tuple(range(7300, 7306))
EVAL_ORIGINAL = tuple(range(4300, 4308))
EVAL_BARRIER = tuple(range(6300, 6308))
EVAL_SLALOM = tuple(range(9300, 9308))
PINNED_PARENTS = {
    'recovery_curriculum.py': 'fcb1c1f5edfd5724e7cbe73fffdd9365e918b0d4',
    'on_policy_recovery_v2.py': '3804065af7432b57a52b524de4f4140c54f49e92',
    'domain_recovery_v3.py': 'c284f09de420f05f95746b141e5d990dddc22cc1',
}


def sample_world_balanced(rows, count):
    """Choose exact count with at least one row from every world group.

    Hash ranking is independent of source file order and all EVAL information.
    Never duplicate examples to inflate training size.
    """
    groups=defaultdict(list)
    seen=set()
    for row in rows:
        key=row['observable_sha256']
        if key in seen:raise ValueError('duplicate normalized observation')
        seen.add(key)
        if row['label'] not in row['observation']['legal']:
            raise ValueError('illegal TRAIN target')
        groups[row['world_group_id']].append(row)
    if not groups or count<len(groups) or count>len(rows):
        raise ValueError('sample size must cover all world groups without replacement')
    for k in groups:
        groups[k].sort(key=lambda r:(sha((r['observable_sha256']+'|topology-v4').encode()),r['observable_sha256']))
    order=sorted(groups)
    result=[]
    while len(result)<count:
        progress=False
        for k in order:
            if groups[k]:
                result.append(groups[k].pop(0));progress=True
                if len(result)==count:break
        if not progress:raise AssertionError('sample pool exhausted')
    return result


def _summarize(rows):
    by_domain=Counter(r['world_group_id'].split(':')[0] for r in rows)
    by_label=Counter(r['label'] for r in rows)
    by_epoch=Counter(min(3,r['observation']['tick']//40) for r in rows)
    return {'rows':len(rows),'world_groups':len(set(r['world_group_id'] for r in rows)),
            'domain_counts':dict(sorted(by_domain.items())),
            'teacher_actions':dict(sorted(by_label.items())),
            'history_quartiles':dict(sorted(by_epoch.items())),
            'student_disagreements':sum(r['was_student_teacher_disagreement'] for r in rows),
            'repeated_oriented_cells':sum(r['repeated_oriented_cell'] for r in rows)}


def verify_parents():
    import subprocess
    here=Path(__file__).parent
    for name,expected in PINNED_PARENTS.items():
        actual=subprocess.run(['git','hash-object',str(here/name)],
                     check=True,capture_output=True,text=True).stdout.strip()
        if actual!=expected:raise ValueError(f'parent source changed: {name}')
    verify_sources(here)


def run(out, budget=160, source=SOURCE, original_only=ORIGINAL_ONLY,
        mixed_original=MIXED_ORIGINAL, mixed_barrier=MIXED_BARRIER,
        original_eval=EVAL_ORIGINAL, barrier_eval=EVAL_BARRIER,
        slalom_eval=EVAL_SLALOM):
    if out.exists():raise FileExistsError('immutable output directory already exists')
    if budget<32 or budget>192:raise ValueError('bounded budget required')
    if len(original_only)!=len(mixed_original)+len(mixed_barrier):
        raise ValueError('training world-group counts must match')
    # Sharing the original 12 correction worlds between arms is intentional,
    # but no TRAIN world may occur in any evaluation set.
    training_ids=set(source)|set(original_only)|set(mixed_original)|set(mixed_barrier)
    evaluation_ids=list(original_eval)+list(barrier_eval)+list(slalom_eval)
    if len(evaluation_ids)!=len(set(evaluation_ids)) or training_ids.intersection(evaluation_ids):
        raise ValueError('TRAIN/EVAL seed overlap')
    if set(source)&(set(original_only)|set(mixed_original)|set(mixed_barrier)):
        raise ValueError('source/correction seed overlap')
    if len(set(original_only))!=len(original_only) or len(set(mixed_original))!=len(mixed_original) or len(set(mixed_barrier))!=len(mixed_barrier):
        raise ValueError('duplicate correction world IDs')
    verify_parents()
    worlds={'source_train':[make_world(s) for s in source],
            'original_only_corrections':[make_world(s) for s in original_only],
            'mixed_barrier_corrections':[make_cross_world(s) for s in mixed_barrier],
            'eval_original':[make_world(s) for s in original_eval],
            'eval_barrier':[make_cross_world(s) for s in barrier_eval],
            'eval_slalom':[make_slalom_world(s) for s in slalom_eval]}
    # Mixed-original worlds are intentionally the SAME original correction
    # worlds as the original-only arm, so they are not inserted twice here.
    if not set(mixed_original).issubset(original_only):
        raise ValueError('shared original correction world IDs must be subset')
    counts=validate_partitions(worlds)
    # Reject undersized evaluation budgets: in this domain, an otherwise
    # perfect shortest-path policy may need 136+ steps merely to reach goal.
    for partition in ('eval_original','eval_barrier','eval_slalom'):
        for w in worlds[partition]:
            minimum=distance_field(w).get((*w['start'],0))
            if minimum is None or minimum>budget:
                raise ValueError(f'evaluation goal infeasible under budget: {partition}:{w["seed"]} minimum={minimum} budget={budget}')
    rows,source_worlds,hard,labels,source_stats=teacher_data(source,budget)
    vec,baseline,baseline_fit=fit(rows)
    original_raw,original_worlds,original_stats=collect_on_worlds(
        original_only,make_world,'original_v1',vec,baseline,budget)
    barrier_raw,barrier_worlds,barrier_stats=collect_on_worlds(
        mixed_barrier,make_cross_world,'barrier_v2',vec,baseline,budget)
    original_clean,original_drops=dedup_corrections(original_raw,rows)
    mixed_shared=[r for r in original_raw if r['seed'] in set(mixed_original)]
    mixed_clean,mixed_drops=dedup_corrections(mixed_shared+barrier_raw,rows)
    matched_size=min(len(original_clean),len(mixed_clean))
    if matched_size<24:raise ValueError('insufficient unique corrections')
    original_matched=sample_world_balanced(original_clean,matched_size)
    mixed_matched=sample_world_balanced(mixed_clean,matched_size)
    ovec,omodel,ofit=fit_student(rows,original_matched)
    mvec,mmodel,mfit=fit_student(rows,mixed_matched)
    policies={'original_12_worlds':(ovec,omodel),'mixed_6_plus_6_worlds':(mvec,mmodel)}
    evaluation={};all_images=[]
    for label,seeds,maker in (
        ('original_generator',original_eval,make_world),
        ('barrier_generator',barrier_eval,make_cross_world),
        ('slalom_generator',slalom_eval,make_slalom_world)):
        summary,details,traces,images=evaluate(seeds,maker,policies,budget)
        evaluation[label]={'summary':summary,'worlds':details,'trajectories':traces,
                 'world_sha256':[world_digest(maker(s)) for s in seeds]}
        all_images.extend((label+'--'+n,p) for n,p in images)
    out.mkdir(parents=True);(out/'png').mkdir()
    export_o=export_corrections(original_matched,original_worlds,
                       out/'original_only_train.zip')
    export_m=export_corrections(mixed_matched,
             [w for w in original_worlds if w['seed'] in set(mixed_original)]+barrier_worlds,
             out/'mixed_topology_train.zip')
    png_manifest=[]
    for name,data in all_images:
        (out/'png'/name).write_bytes(data)
        png_manifest.append({'file':name,'sha256':sha(data)})
    report={'schema':SCHEMA,'source_blob_pins':PINNED_PARENTS,
        'evidence_class':'external-synthetic-trained-CPU-and-host-applied-evaluation',
        'warning':'TRAIN-internal comparison; not official DEV/heldout, primary SLMs, native gameplay, or artwork quality. Family episodes correlated.',
        'design':'12 correction world groups each, same 24 source worlds, same number of unique correction states, same optimizer settings/epochs; differing obstacle topology. History/difficulty distributions may still differ.',
        'train_world_counts':counts,'shared_original_correction_ids':list(mixed_original),
        'source_training':{'rows':len(rows),'hard_negatives':hard,'actions':labels,
                           'generator_stats':source_stats,'baseline_model':baseline_fit},
        'correction_pools':{'original_unique':len(original_clean),'mixed_unique':len(mixed_clean),
             'original_duplicates_rejected':original_drops,'mixed_duplicates_rejected':mixed_drops,
             'original_host_stats':original_stats,'barrier_host_stats':barrier_stats},
        'matched_corrections':{'size':matched_size,'original':_summarize(original_matched),
             'mixed':_summarize(mixed_matched),
             'original_training_fit':ofit,'mixed_training_fit':mfit,
             'original_export':export_o,'mixed_export':export_m},
        'evaluation':evaluation,'pngs':png_manifest,
        'evaluation_label_training_rows':0,'native_events':0,'paid_spend':0}
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    report=run(args.out)
    print(json.dumps({k:v['summary'] for k,v in report['evaluation'].items()},sort_keys=True,indent=2))
