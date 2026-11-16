"""GAL-007: fixed-checkpoint interaction knockouts with independent seed block.

External synthetic CyberSand-like host; this is NOT native PAINT, W0 or SLM.
Only the candidate scoring weights change. Legal menus, materials, STOP and
host-receipted action execution are inherited unmodified from GAL-006.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import statistics
import tempfile
from pathlib import Path

from lab import DECISION_CAP, png, sha
from candidate_features import FAMILIES
from interaction_ablation import (ARMS, BASE_SEED, TARGET, FEATURE_MAP,
    INTERACTIONS, InkTargetWorld, Policy, digest_file, feature_map, train)
from shape_metrics import morphology

SCHEMA = 'gal-007-fixed-checkpoint-knockouts/1'
TRAIN_WORLD_COUNT = 90
# These are frozen SHA256s of genuine GAL-006 90-world optimizer checkpoints.
EXPECTED = {'cell_linear':'fb1958ff1deb89273693b827730cbdfee62c76251c87b4785bd9328284851296',
            'cell_pairwise':'04217da6ae630b4d1f0a78b1d25329f1f768c08a90d9daf49880aecabd410e1c',
            'sweep_linear':'d7a05281f4c0d9ef69009ec2a11d7538ac6bab57076e468269c269c05c68a1b1',
            'sweep_pairwise':'273daf647740f494605c1695b169fe9b33fb8914da30d521826d6425b27f9a57'}
SEED_BLOCKS = {'replication':310000,'fresh_independent':410000}
PANEL_NAMES = ('full', 'all_off', 'near_group_off', 'clear_group_off') + tuple('off_'+x for x in INTERACTIONS) + ('separately_trained_linear',)
SOURCE = ('lab.py','candidate_features.py','shape_metrics.py','interaction_ablation.py','interaction_knockout.py')


def checkpoint(arm, nonlinear, episodes=TRAIN_WORLD_COUNT):
    model, updates = train(arm,nonlinear,episodes)
    name=f'{arm}_{"pairwise" if nonlinear else "linear"}'
    cp={'schema':'gal006-optimizer-checkpoint/1','arm':arm,'model':'pairwise' if nonlinear else 'linear',
        'features':list(model.names),'weights':model.weights,'updates':updates,
        'optimizer':'bounded_onpolicy_REINFORCE','train_seeds':[BASE_SEED,BASE_SEED+episodes-1]}
    raw=json.dumps(cp,sort_keys=True,indent=2)+'\n'
    digest=hashlib.sha256(raw.encode()).hexdigest()
    if episodes==TRAIN_WORLD_COUNT and digest!=EXPECTED[name]:
        raise ValueError(f'frozen GAL006 checkpoint drift {name}: {digest}')
    return model,raw,digest


def masked_model(original, panel):
    if panel=='separately_trained_linear':
        raise ValueError('separately trained model must be supplied explicitly')
    model=Policy(True)
    model.weights=original.weights.copy()
    if panel=='full': disabled=()
    elif panel=='all_off':disabled=INTERACTIONS
    elif panel=='near_group_off': disabled=tuple(x for x in INTERACTIONS if x.startswith('near_'))
    elif panel=='clear_group_off':disabled=tuple(x for x in INTERACTIONS if x.startswith('clear_'))
    elif panel.startswith('off_') and panel[4:] in INTERACTIONS:disabled=(panel[4:],)
    else:raise ValueError('invalid ablation panel')
    for feature in disabled:model.weights[FEATURE_MAP.index(feature)]=0.0
    return model


def episode(model,arm,seed,block,panel):
    family=FAMILIES[(seed-SEED_BLOCKS[block])%len(FAMILIES)]
    lease=f'gal007:{block}:{arm}:{panel}:{seed}'
    world=InkTargetWorld(seed,family,arm,'candidate',lease=lease)
    rng=random.Random(f'gal007-eval:{arm}:{block}:{seed}:paired-coins')
    while not world.stopped and world.decisions<DECISION_CAP and len(world.own)<TARGET:
        menu,token=world.offer()
        names=list(menu)
        vectors=[feature_map(world,a,menu[a],nonlinear=model.nonlinear) for a in names]
        probabilities=model.probabilities(vectors)
        u=rng.random(); accum=0.; idx=len(names)-1
        for i,p in enumerate(probabilities):
            accum+=p
            if u<=accum:idx=i;break
        action=names[idx]
        world.apply(action,token,original_response=f'gal007-model-output:{action}')
    if len(world.own)>TARGET or any(world.material[world.idx(p)]!=world.original[world.idx(p)] for p in world.protected):
        raise AssertionError('host material authority violation')
    row={'arm':arm,'panel':panel,'seed_block':block,'seed':seed,'family':family,
         'lease':lease,'world_sha256':sha(world.original),'material_sha256':sha(world.material),
         'receipts_sha256':sha(world.trace),'ink':len(world.own),'decisions':world.decisions,
         'logical_moves':world.logical_moves,'selected_stop':world.stopped,
         'reached_32':len(world.own)==TARGET,'decisions_to_32':world.decisions if len(world.own)==TARGET else None,
         'shape':morphology(world.own,world.trace),'evidence':'external_synthetic_only',
         'native_receipts':0,'resident_slms':0,'w0_executions':0}
    return world,row


def paired(full, comparator):
    if len(full)!=len(comparator):raise ValueError('unmatched panel lengths')
    full_only=other_only=both=neither=0
    ink_deltas=[]; conditional_decision_savings=[]
    for a,b in zip(full,comparator):
        if (a['seed'],a['family'],a['world_sha256'])!=(b['seed'],b['family'],b['world_sha256']):
            raise ValueError('unmatched seed or initial material')
        x,y=a['reached_32'],b['reached_32']
        if x and y:both+=1;conditional_decision_savings.append(b['decisions']-a['decisions'])
        elif x:full_only+=1
        elif y:other_only+=1
        else:neither+=1
        ink_deltas.append(a['ink']-b['ink'])
    discordant=full_only+other_only
    # Exact two-sided McNemar/binomial sign test on discordant completion outcomes.
    p_exact=min(1.0,2*sum(math.comb(discordant,k) for k in range(min(full_only,other_only)+1))/2**discordant) if discordant else 1.0
    return {'full_only_reached':full_only,'comparator_only_reached':other_only,
            'both_reached':both,'neither_reached':neither,
            'mean_full_minus_comparator_ink':round(statistics.mean(ink_deltas),4),
            'mean_decisions_saved_full_if_both_reach':round(statistics.mean(conditional_decision_savings),4) if both else None,
            'exact_two_sided_mcnemar_p_exploratory':round(p_exact,6)}


def summarize(rows):
    reached=[r for r in rows if r['reached_32']]
    return {'episodes':len(rows),'reached_32':len(reached),
            'pooled_ink_per_decision':round(sum(r['ink'] for r in rows)/max(1,sum(r['decisions'] for r in rows)),4),
            'mean_ink':round(statistics.mean(r['ink'] for r in rows),3),
            'decisions_to_32_success_only':round(statistics.mean(r['decisions'] for r in reached),3) if reached else None,
            'early_stop':sum(r['selected_stop'] and r['ink']<TARGET for r in rows),
            'mean_bends':round(statistics.mean(r['shape']['paint_path_bends'] for r in rows),3)}


def run(output,counts=(40,60),gallery_per_block=2):
    if len(counts)!=2 or any(type(x)!=int or x<5 or x>120 for x in counts) or not 0<=gallery_per_block<=3:
        raise ValueError('bounded evaluation only')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'checkpoints').mkdir(exist_ok=True)
    models={};checkpoints={}
    for arm in ARMS:
        for nonlinear in (False,True):
            model,raw,h=checkpoint(arm,nonlinear)
            key=f'{arm}_{"pairwise" if nonlinear else "linear"}'
            (out/'checkpoints'/f'{key}.json').write_text(raw)
            models[key]=model;checkpoints[key]=h
    rows={}; receipts={}; report={}; matched={}
    for block,n in zip(SEED_BLOCKS,counts):
        for arm in ARMS:
            for panel in PANEL_NAMES:
                key=f'{block}:{arm}:{panel}'
                model=(models[f'{arm}_linear'] if panel=='separately_trained_linear'
                       else masked_model(models[f'{arm}_pairwise'],panel))
                rows[key]=[];receipts[key]=[]
                for j in range(n):
                    seed=SEED_BLOCKS[block]+j
                    world,row=episode(model,arm,seed,block,panel)
                    rows[key].append(row);receipts[key].append(world.trace)
                    if j<gallery_per_block and panel in ('full','all_off','separately_trained_linear'):
                        p=out/'gallery';p.mkdir(exist_ok=True)
                        png(p/f'{block}-{arm}-{panel}-{seed}.png',world)
                report[key]=summarize(rows[key])
            baseline=rows[f'{block}:{arm}:full']
            for panel in PANEL_NAMES[1:]:
                matched[f'{block}:{arm}:{panel}']=paired(baseline,rows[f'{block}:{arm}:{panel}'])
    (out/'episodes.json').write_text(json.dumps(rows,sort_keys=True,separators=(',',':'))+'\n')
    (out/'receipts.json').write_text(json.dumps(receipts,sort_keys=True,separators=(',',':'))+'\n')
    metrics={'schema':SCHEMA,'evidence':'external_synthetic_only','gallery_per_block':gallery_per_block,'counts':dict(zip(SEED_BLOCKS,counts)),
             'seed_ranges':{b:[s,s+n-1] for (b,s),n in zip(SEED_BLOCKS.items(),counts)},
             'train_seeds':[BASE_SEED,BASE_SEED+TRAIN_WORLD_COUNT-1],
             'frozen_GAL006_checkpoint_sha256':checkpoints,
             'source_sha256':{f:digest_file(Path(__file__).with_name(f)) for f in SOURCE},
             'summary':report,'matched':matched,'episodes_sha256':digest_file(out/'episodes.json'),
             'receipts_sha256':digest_file(out/'receipts.json'),
             'qualification':'not_native_not_W0_not_resident_SLM'}
    (out/'metrics.json').write_text(json.dumps(metrics,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS').write_text(''.join(f'{digest_file(p)}  {p.relative_to(out).as_posix()}\n'
         for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    return metrics


class AuditRefusal(ValueError):pass


def audit(output):
    root=Path(output); report=json.loads((root/'metrics.json').read_text())
    if report.get('schema')!=SCHEMA or report.get('train_seeds')!=[BASE_SEED,BASE_SEED+TRAIN_WORLD_COUNT-1]:
        raise AuditRefusal('schema/training custody')
    if report.get('source_sha256')!={f:digest_file(Path(__file__).with_name(f)) for f in SOURCE}:
        raise AuditRefusal('source drift')
    observed={};
    for line in (root/'SHA256SUMS').read_text().splitlines():
        h,path=line.split('  ',1)
        if path in observed or Path(path).is_absolute() or '..' in Path(path).parts:
            raise AuditRefusal('unsafe manifest entry')
        observed[path]=h
        if not (root/path).is_file() or digest_file(root/path)!=h:raise AuditRefusal('artifact changed')
    files={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS'}
    if files!=set(observed):raise AuditRefusal('unlisted artifact')
    if digest_file(root/'episodes.json')!=report['episodes_sha256'] or digest_file(root/'receipts.json')!=report['receipts_sha256']:
        raise AuditRefusal('original evidence mismatch')
    rows=json.loads((root/'episodes.json').read_text());receipts=json.loads((root/'receipts.json').read_text())
    models={};checks={}
    for arm in ARMS:
        for nonlinear in (False,True):
            model,raw,h=checkpoint(arm,nonlinear)
            key=f'{arm}_{"pairwise" if nonlinear else "linear"}'
            if (root/'checkpoints'/f'{key}.json').read_text()!=raw:raise AuditRefusal('checkpoint contents changed')
            models[key]=model;checks[key]=h
    if checks!=report.get('frozen_GAL006_checkpoint_sha256'):raise AuditRefusal('checkpoint audit')
    gallery_count=report.get('gallery_per_block')
    if type(gallery_count)!=int or not 0<=gallery_count<=3:
        raise AuditRefusal('gallery budget')
    expected=set();n_episodes=n_decisions=0
    for block,start in SEED_BLOCKS.items():
        count=report['counts'].get(block)
        if type(count)!=int or not 5<=count<=120 or report['seed_ranges'].get(block)!=[start,start+count-1]:
            raise AuditRefusal('seed custody')
        for arm in ARMS:
            for panel in PANEL_NAMES:
                key=f'{block}:{arm}:{panel}';expected.add(key)
                model=models[f'{arm}_linear'] if panel=='separately_trained_linear' else masked_model(models[f'{arm}_pairwise'],panel)
                rs=rows.get(key);ts=receipts.get(key)
                if not isinstance(rs,list) or not isinstance(ts,list) or len(rs)!=count or len(ts)!=count:
                    raise AuditRefusal('episode coverage')
                for j in range(count):
                    w,r=episode(model,arm,start+j,block,panel)
                    if r!=rs[j] or sha(w.trace)!=sha(ts[j]):raise AuditRefusal(f'host/model replay mismatch {key} {j}')
                    if j<gallery_count and panel in ('full','all_off','separately_trained_linear'):
                        gallery_path=root/'gallery'/f'{block}-{arm}-{panel}-{start+j}.png'
                        with tempfile.TemporaryDirectory() as td:
                            regenerated=Path(td)/'from_replay.png'
                            png(regenerated,w)
                            if not gallery_path.is_file() or digest_file(gallery_path)!=digest_file(regenerated):
                                raise AuditRefusal(f'original gallery mismatch {key} {j}')
                    n_episodes+=1;n_decisions+=w.decisions
                if summarize(rs)!=report['summary'].get(key):raise AuditRefusal('summary mismatch')
            for panel in PANEL_NAMES[1:]:
                key=f'{block}:{arm}:{panel}'
                if paired(rows[f'{block}:{arm}:full'],rows[key])!=report['matched'].get(key):
                    raise AuditRefusal('paired evaluation mismatch')
    if set(rows)!=expected or set(receipts)!=expected or set(report['summary'])!=expected:
        raise AuditRefusal('extra/missing panels')
    return {'replayed_episodes':n_episodes,'replayed_model_decisions':n_decisions,
            'checkpoint_retraining_verified':4,'native_receipts':0,'resident_slms':0,'w0_executions':0}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--replication',type=int,default=40)
    parser.add_argument('--fresh',type=int,default=60)
    parser.add_argument('--gallery',type=int,default=2)
    args=parser.parse_args()
    m=run(args.output,(args.replication,args.fresh),args.gallery)
    print(json.dumps({'schema':m['schema'],'summary':m['summary'],'matched':m['matched']},indent=2))
    print('AUDIT:',json.dumps(audit(args.output)))


if __name__=='__main__':main()
