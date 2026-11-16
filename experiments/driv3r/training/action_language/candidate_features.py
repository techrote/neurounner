"""GAL-005: candidate-endpoint material features in an external CyberSand-like host.

Trains a small linear policy only. No Godot, W0, native receipts, resident SLM,
production provider, or paid API. Original World owns legal actions and material.
The legacy reward is held FIXED while only model-facing observations change.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import math
import random
import statistics

from lab import (World, TinyPolicy, FEATURES, DIRS, DECISION_CAP, INK, H, W,
                 HostRefusal, png, sha, reward)
from shape_metrics import morphology

SCHEMA='gal-005-candidate-observation/1'
FAMILIES=('empty','pillars','gates','scaffold','scatter')
ARMS=('cell','sweep')
TRAIN_SEED=41000
HELDOUT_SEED=240000
TARGET=32
VIEWS=('legacy','candidate','candidate_no_ink','candidate_no_open')
IDX_OPEN=FEATURES.index('open')
IDX_NEAR=FEATURES.index('near_ink')


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CandidateWorld(World):
    """Projection for model ONLY; does not alter offers, receipts or material.

    'near_ink': existing, already-painted own material within Manhattan radius
    three around proposed endpoint (no imaginary stroke material counted).
    'open': proportion of cardinal neighbours that remain host-clear after
    proposed stroke; proposed painted stroke cells are treated as occupied.
    For STOP and SAVE_ANCHOR no speculative endpoint is constructed.
    """
    def __init__(self,seed,family,arm,view,*,lease=None):
        if view not in VIEWS: raise ValueError('unknown feature projection')
        self.view=view
        super().__init__(seed,family,arm,lease=lease or f'gal005-{arm}-{view}-{seed}-{family}')

    def features(self,name,path):
        v=list(World.features(self,name,path))
        if self.view=='legacy' or name in ('STOP','SAVE_ANCHOR'):
            return tuple(v)
        if not path: return tuple(v)
        endpoint=path[-1]
        stroke=set(path) if name.startswith(('INK_','JOG_')) else set()
        # No reads from held-out labels, rewards, future actions or other worlds.
        if self.view!='candidate_no_ink':
            v[IDX_NEAR]=sum(abs(x-endpoint[0])+abs(y-endpoint[1])<=3
                            for x,y in self.own)/24.0
        else:
            v[IDX_NEAR]=0.0
        if self.view!='candidate_no_open':
            open_count=0
            for dx,dy in DIRS.values():
                q=(endpoint[0]+dx,endpoint[1]+dy)
                if q not in stroke and self.clear(q): open_count+=1
            v[IDX_OPEN]=open_count/4.0
        else:
            v[IDX_OPEN]=0.0
        return tuple(v)


def train_model(arm,view,episodes=60):
    if arm not in ARMS or view not in ('legacy','candidate') or not 1<=episodes<=160:
        raise ValueError('invalid bounded training request')
    policy=TinyPolicy();updates=0
    for i in range(episodes):
        seed=TRAIN_SEED+i;family=FAMILIES[i%len(FAMILIES)]
        world=CandidateWorld(seed,family,arm,view)
        rng=random.Random(f'gal005-train:{arm}:{seed}:shared-uniform-coins')
        baseline=.25
        while not world.stopped and world.decisions<DECISION_CAP:
            action,token,names,vecs,probs,pick=policy.choose(world,rng)
            # Fixed reward definition: original GAL-001 18-feature encoding,
            # intentionally NOT the experimental candidate-specific features.
            legacy_phi=World.features(world,action,world.legal()[action])
            receipt=world.apply(action,token,original_response=f'tiny-linear-policy:{action}')
            r=reward(world,receipt,legacy_phi)
            adv=max(-1,min(1,r-baseline));baseline=.95*baseline+.05*r
            expectation=[sum(p*v[j] for p,v in zip(probs,vecs)) for j in range(len(FEATURES))]
            for j in range(len(FEATURES)):
                entropy_grad=sum((1/len(probs)-p)*v[j] for p,v in zip(probs,vecs))
                policy.weights[j]+=0.06*(adv*(vecs[pick][j]-expectation[j])+.015*entropy_grad)
                policy.weights[j]=max(-6,min(6,policy.weights[j]))
            updates+=1
    return policy,updates


def run_world(policy,seed,family,arm,view,panel):
    world=CandidateWorld(seed,family,arm,view,lease=f'gal005-{panel}-{seed}-{family}')
    rng=random.Random(f'gal005-eval:{arm}:{seed}:shared-uniform-coins')
    first32=None
    ink_at_32_decisions=None
    while not world.stopped and world.decisions<DECISION_CAP:
        action,token,_,_,_,_=policy.choose(world,rng)
        receipt=world.apply(action,token,original_response=f'tiny-linear-policy:{action}')
        assert receipt['new_ink']>=0
        if first32 is None and len(world.own)>=TARGET: first32=world.decisions
        if world.decisions==32:ink_at_32_decisions=len(world.own)
        assert all(world.material[world.idx(p)]==world.original[world.idx(p)]
                   for p in world.protected)
    if ink_at_32_decisions is None:ink_at_32_decisions=len(world.own)
    shape=morphology(world.own,world.trace)
    row={'panel':panel,'seed':seed,'family':family,'arm':arm,'view':view,
         'lease':world.lease,'initial_material_sha256':sha(world.original),
         'final_material_sha256':sha(world.material),'trace_sha256':sha(world.trace),
         'action_sequence_sha256':sha([r['action_id'] for r in world.trace]),
         'decisions':world.decisions,'logical_moves':world.logical_moves,
         'painted':len(world.own),'first_32':first32,
         'ink_by_32_decision_opportunities':ink_at_32_decisions,
         'selected_STOP':world.stopped,
         'shape':shape,
         'evidence_class':'external_synthetic_tiny_linear_policy',
         'native_receipts':0,'resident_slm_processes':0,'W0_executions':0}
    return world,row


def summarize(rows):
    successes=[x['first_32'] for x in rows if x['first_32'] is not None]
    return {'worlds':len(rows),'reached_32':len(successes),
            'mean_decisions_to_32_success_only':round(statistics.mean(successes),3) if successes else None,
            'mean_ink_by_32_opportunities':round(statistics.mean(x['ink_by_32_decision_opportunities'] for x in rows),3),
            'pooled_ink_per_decision':round(sum(x['painted'] for x in rows)/max(1,sum(x['decisions'] for x in rows)),4),
            'selected_STOP':sum(x['selected_STOP'] for x in rows),
            'mean_painted_path_bends':round(statistics.mean(x['shape']['paint_path_bends'] for x in rows),3),
            'mean_cycle_rank':round(statistics.mean(x['shape']['graph_cycle_rank'] for x in rows),3),
            'mean_junctions':round(statistics.mean(x['shape']['junctions'] for x in rows),3)}


def paired(a,b):
    if len(a)!=len(b) or not a:raise ValueError('unpaired sets')
    counts={'both_reached':0,'first_only':0,'second_only':0,'neither':0}
    deltas=[];diverged=0
    for x,y in zip(a,b):
        if (x['seed'],x['family'],x['initial_material_sha256']) != (y['seed'],y['family'],y['initial_material_sha256']):
            raise ValueError('mismatched held-out custody')
        p,q=x['first_32'],y['first_32']
        if p is not None and q is not None:
            counts['both_reached']+=1;deltas.append(p-q)
        elif p is not None:counts['first_only']+=1
        elif q is not None:counts['second_only']+=1
        else:counts['neither']+=1
        if x['action_sequence_sha256']!=y['action_sequence_sha256']:diverged+=1
    return {**counts,'diverged_action_sequence_worlds':diverged,
            'mean_decision_saving_second_conditional_on_both_success':
                round(statistics.mean(deltas),3) if deltas else None}


def write_run(out,*,train_worlds=60,heldout_worlds=30,gallery_worlds=2):
    if not 1<=train_worlds<=160 or not 5<=heldout_worlds<=80 or not 0<=gallery_worlds<=5:
        raise ValueError('experiment budget refused')
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    (root/'checkpoints').mkdir(exist_ok=True)
    policies={};train_records={}
    for arm in ARMS:
        for view in ('legacy','candidate'):
            key=f'{arm}_{view}'
            policy,updates=train_model(arm,view,train_worlds)
            policies[key]=policy
            checkpoint={'schema':'gal005-tiny-linear/1','arm':arm,'view':view,
                        'optimizer':'bounded on-policy REINFORCE with legacy-reward lock',
                        'weights':policy.weights,'features':list(FEATURES),
                        'updates':updates,'train_seed_range':[TRAIN_SEED,TRAIN_SEED+train_worlds-1]}
            cp=root/'checkpoints'/f'{key}.json'
            cp.write_text(json.dumps(checkpoint,indent=2,sort_keys=True)+'\n')
            train_records[key]={'checkpoint_sha256':file_sha(cp),'updates':updates}
    rows={};traces={}
    for arm in ARMS:
        for suffix,view,policy_view in (
            ('legacy','legacy','legacy'),
            ('candidate','candidate','candidate'),
            ('ablate_ink','candidate_no_ink','candidate'),
            ('ablate_open','candidate_no_open','candidate'),
        ):
            panel=f'{arm}_{suffix}'
            rows[panel]=[];traces[panel]=[]
            for i in range(heldout_worlds):
                seed=HELDOUT_SEED+i;family=FAMILIES[i%len(FAMILIES)]
                world,row=run_world(policies[f'{arm}_{policy_view}'],seed,family,arm,view,panel)
                rows[panel].append(row);traces[panel].append(world.trace)
                if i<gallery_worlds:
                    d=root/'gallery';d.mkdir(exist_ok=True)
                    stem=f'{panel}-{seed}-{family}'
                    png(d/(stem+'.png'),world)
                    (d/(stem+'.json')).write_text(json.dumps(world.trace,indent=2,sort_keys=True)+'\n')
    (root/'episodes.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
    (root/'traces.json').write_text(json.dumps(traces,separators=(',',':'),sort_keys=True)+'\n')
    paths=('lab.py','shape_metrics.py','candidate_features.py')
    report={'schema':SCHEMA,'source_sha256':{p:file_sha(Path(__file__).with_name(p)) for p in paths},
            'evidence_class':'external_synthetic_tiny_linear_policy',
            'not_native':True,'not_W0':True,'not_SLM':True,
            'train_seed_range':[TRAIN_SEED,TRAIN_SEED+train_worlds-1],
            'heldout_seed_range':[HELDOUT_SEED,HELDOUT_SEED+heldout_worlds-1],
            'candidate_only_affects_model_view':True,'reward_held_fixed':True,
            'protected_and_legal_offers_host_owned':True,'STOP_unmasked':True,
            'training':train_records,'episodes_sha256':file_sha(root/'episodes.json'),
            'traces_sha256':file_sha(root/'traces.json'),
            'summary':{p:summarize(x) for p,x in rows.items()},
            'paired':{f'{arm}_{suffix}_vs_legacy':paired(rows[f'{arm}_legacy'],rows[f'{arm}_{suffix}'])
                      for arm in ARMS for suffix in ('candidate','ablate_ink','ablate_open')},
            'ablations':{f'{arm}_{suffix}_vs_full':paired(rows[f'{arm}_candidate'],rows[f'{arm}_{suffix}'])
                         for arm in ARMS for suffix in ('ablate_ink','ablate_open')},
            'qualification':'source-exact synthetic self-replay, never native'}
    (root/'metrics.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    (root/'SHA256SUMS').write_text(''.join(f'{file_sha(f)}  {f.relative_to(root).as_posix()}\n'
        for f in sorted(root.rglob('*')) if f.is_file() and f.name!='SHA256SUMS'))
    return report


class AuditRefusal(ValueError):pass


def audit(root):
    root=Path(root);report=json.loads((root/'metrics.json').read_text())
    if report.get('schema')!=SCHEMA:raise AuditRefusal('schema mismatch')
    if report.get('source_sha256')!={p:file_sha(Path(__file__).with_name(p))
                                  for p in ('lab.py','shape_metrics.py','candidate_features.py')}:
        raise AuditRefusal('source drift')
    inventory={}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        h,p=line.split('  ',1)
        if p in inventory or Path(p).is_absolute() or '..' in Path(p).parts:
            raise AuditRefusal('manifest path')
        inventory[p]=h
        if not (root/p).is_file() or file_sha(root/p)!=h:
            raise AuditRefusal('manifest bytes mismatch')
    files={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS'}
    if set(inventory)!=files:raise AuditRefusal('unlisted artifact')
    if file_sha(root/'episodes.json')!=report['episodes_sha256'] or file_sha(root/'traces.json')!=report['traces_sha256']:
        raise AuditRefusal('evidence SHA')
    n=report['heldout_seed_range'][1]-report['heldout_seed_range'][0]+1
    if not 5<=n<=80 or report['heldout_seed_range'][0]!=HELDOUT_SEED:
        raise AuditRefusal('seed custody')
    rows=json.loads((root/'episodes.json').read_text());traces=json.loads((root/'traces.json').read_text())
    keys={f'{arm}_{suffix}' for arm in ARMS for suffix in ('legacy','candidate','ablate_ink','ablate_open')}
    if set(rows)!=keys or set(traces)!=keys:raise AuditRefusal('panel mismatch')
    policies={}
    for arm in ARMS:
        for view in ('legacy','candidate'):
            key=f'{arm}_{view}';p=root/'checkpoints'/f'{key}.json'
            if file_sha(p)!=report['training'][key]['checkpoint_sha256']:
                raise AuditRefusal('checkpoint digest')
            cp=json.loads(p.read_text())
            if cp.get('schema')!='gal005-tiny-linear/1' or cp.get('arm')!=arm or cp.get('view')!=view \
                or cp.get('features')!=list(FEATURES) or len(cp.get('weights',[]))!=len(FEATURES) \
                or any(type(v) not in (int,float) or not math.isfinite(v) or not -6<=v<=6 for v in cp['weights']):
                raise AuditRefusal('checkpoint structure')
            m=TinyPolicy();m.weights=cp['weights'];policies[key]=m
    total=0;choices=0
    for arm in ARMS:
        for suffix,view,policy_view in (
            ('legacy','legacy','legacy'),('candidate','candidate','candidate'),
            ('ablate_ink','candidate_no_ink','candidate'),('ablate_open','candidate_no_open','candidate')):
            panel=f'{arm}_{suffix}'
            if len(rows[panel])!=n or len(traces[panel])!=n:raise AuditRefusal('panel length')
            for i,(record,original_receipts) in enumerate(zip(rows[panel],traces[panel])):
                seed=HELDOUT_SEED+i;family=FAMILIES[i%len(FAMILIES)]
                world,row=run_world(policies[f'{arm}_{policy_view}'],seed,family,arm,view,panel)
                if record!=row or json.loads(json.dumps(world.trace))!=original_receipts:
                    raise AuditRefusal('original decision/material/receipt replay')
                total+=1;choices+=len(world.trace)
    if report['summary']!={k:summarize(v) for k,v in rows.items()} or \
       report['paired']!={f'{arm}_{suffix}_vs_legacy':paired(rows[f'{arm}_legacy'],rows[f'{arm}_{suffix}'])
                           for arm in ARMS for suffix in ('candidate','ablate_ink','ablate_open')} or \
       report['ablations']!={f'{arm}_{suffix}_vs_full':paired(rows[f'{arm}_candidate'],rows[f'{arm}_{suffix}'])
                           for arm in ARMS for suffix in ('ablate_ink','ablate_open')}:
        raise AuditRefusal('summary changed')
    return {'episodes_replayed':total,'model_decisions_replayed':choices,
            'native_receipts':0,'resident_slm_processes':0,'source_sha256':report['source_sha256']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--train-worlds',type=int,default=60)
    p.add_argument('--heldout-worlds',type=int,default=30)
    p.add_argument('--gallery-worlds',type=int,default=2)
    p.add_argument('--audit-only',action='store_true')
    a=p.parse_args()
    result=audit(a.output) if a.audit_only else write_run(a.output,
        train_worlds=a.train_worlds,heldout_worlds=a.heldout_worlds,gallery_worlds=a.gallery_worlds)
    print(json.dumps(result,sort_keys=True,indent=2))

if __name__=='__main__':main()
