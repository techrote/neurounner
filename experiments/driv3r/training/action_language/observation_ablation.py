"""GAL-004: frozen trained tiny-policy observation ablation in an external host.

The host's material, protection, candidate list and receipts never depend on the
model-facing feature mask. This is not native Godot, WORM-001 W0 or SLM inference.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

from lab import ARMS, DECISION_CAP, FEATURES, TinyPolicy, World, png, sha, train

SCHEMA = 'gal-004-model-facing-observation-ablation/1'
FAMILIES = ('empty', 'pillars', 'gates', 'scaffold', 'scatter')
ARMS_TESTED = ('cell', 'sweep')
MODES = {
    'full': (),
    'without_local_ink': ('near_ink',),
    'without_visit_memory': ('novelty', 'already_visited'),
    'without_openness': ('open',),
}
TRAIN_BASE = 36000
EVAL_BASE = 160000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MaskedWorld(World):
    """Only model-view features are ablated, never host action permissions."""
    def __init__(self, seed, family, arm, mode):
        if mode not in MODES: raise ValueError('unknown observation mode')
        self.mode = mode
        super().__init__(seed, family, arm,
                         lease=f'gal004-{arm}-{mode}-{seed}-{family}')

    def features(self, name, path):
        original = super().features(name, path)
        drop = MODES[self.mode]
        return tuple(0.0 if feature in drop else value
                     for feature, value in zip(FEATURES, original))


def run_world(policy, seed, family, arm, mode):
    w = MaskedWorld(seed, family, arm, mode)
    rng = random.Random(f'gal004:{arm}:{seed}:shared-sampling-coins')
    first_32 = None
    while not w.stopped and w.decisions < DECISION_CAP:
        action, token, _, _, _, _ = policy.choose(w, rng)
        w.apply(action, token, original_response=f'tiny-linear-policy:{action}')
        if first_32 is None and len(w.own) >= 32:
            first_32 = w.decisions
    row = {
        'seed':seed,'family':family,'arm':arm,'mode':mode,'lease':w.lease,
        'initial_material_sha256':sha(w.original),
        'material_sha256':sha(w.material),'trace_sha256':sha(w.trace),
        'decisions':w.decisions,'new_ink':len(w.own),
        'logical_moves':w.logical_moves,'selected_STOP':w.stopped,
        'decision_to_32_ink':first_32,'action_ids':[x['action_id'] for x in w.trace],
        'evidence_class':'trained_tiny_linear_external_synthetic',
        'native_receipts':0,'resident_slm_processes':0,'w0_executions':0,
    }
    return w,row


def summary(rows):
    wins=[r['decision_to_32_ink'] for r in rows if r['decision_to_32_ink'] is not None]
    return {'worlds':len(rows),'reached_32_ink':len(wins),
            'mean_calls_to_32_on_success_only':round(statistics.mean(wins),4) if wins else None,
            'mean_new_ink_all':round(statistics.mean(r['new_ink'] for r in rows),4),
            'mean_decisions_all':round(statistics.mean(r['decisions'] for r in rows),4),
            'pooled_new_ink_per_decision':round(sum(r['new_ink'] for r in rows)/
                 max(1,sum(r['decisions'] for r in rows)),4),
            'model_selected_STOP':sum(r['selected_STOP'] for r in rows)}


def comparison(base,altered):
    a={(r['seed'],r['family']):r for r in base}
    b={(r['seed'],r['family']):r for r in altered}
    if set(a)!=set(b): raise ValueError('unmatched worlds')
    d={'both_reached_32':0,'full_only':0,'altered_only':0,'neither':0}
    delta=[];divergence=[]
    for key in sorted(a):
        x,y=a[key],b[key]
        if x['initial_material_sha256']!=y['initial_material_sha256']:
            raise ValueError('mismatched source material')
        first=next((i for i,(p,q) in enumerate(zip(x['action_ids'],y['action_ids']))
                    if p!=q),None)
        if first is None and len(x['action_ids'])!=len(y['action_ids']):
            first=min(len(x['action_ids']),len(y['action_ids']))
        if first is not None: divergence.append(first)
        s1=x['decision_to_32_ink'];s2=y['decision_to_32_ink']
        if s1 is not None and s2 is not None:
            d['both_reached_32']+=1;delta.append(s1-s2)
        elif s1 is not None:d['full_only']+=1
        elif s2 is not None:d['altered_only']+=1
        else:d['neither']+=1
    return {**d,'diverged_worlds':len(divergence),
            'mean_first_divergence_zero_based':round(statistics.mean(divergence),4)
             if divergence else None,
            'mean_calls_saved_by_altered_both_success':round(statistics.mean(delta),4)
             if delta else None,
            'warning':'Per-world first divergence is descriptive; trajectories differ afterward. '
                      'Call savings are conditioned on both reaching 32 ink.'}


def write_run(out, *, train_worlds=60, eval_worlds=30, gallery_worlds=2):
    if not 1<=train_worlds<=120 or not 5<=eval_worlds<=100 or not 0<=gallery_worlds<=5:
        raise ValueError('unbounded experiment')
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    (root/'checkpoints').mkdir(exist_ok=True)
    policies={};models={}
    for arm in ARMS_TESTED:
        policy,updates=train(arm,train_worlds,base_seed=TRAIN_BASE)
        policies[arm]=policy
        doc={'schema':'gal-004-trained-tiny-linear/1','arm':arm,'features':list(FEATURES),
             'weights':policy.weights,'updates':updates,
             'train_seed_range':[TRAIN_BASE,TRAIN_BASE+train_worlds-1],
             'optimizer':'REINFORCE from lab.py; local CPU'}
        file=root/'checkpoints'/(arm+'.json')
        file.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
        models[arm]={'sha256':digest(file),'updates':updates}
    rows={};traces={}
    for arm in ARMS_TESTED:
        for mode in MODES:
            panel=f'{arm}_{mode}';rows[panel]=[];traces[panel]=[]
            for i in range(eval_worlds):
                seed=EVAL_BASE+i;family=FAMILIES[i%len(FAMILIES)]
                w,row=run_world(policies[arm],seed,family,arm,mode)
                rows[panel].append(row);traces[panel].append(w.trace)
                if i<gallery_worlds:
                    gallery=root/'gallery';gallery.mkdir(exist_ok=True)
                    stem=f'{panel}-{family}-{seed}'
                    png(gallery/(stem+'.png'),w)
                    (gallery/(stem+'.json')).write_text(json.dumps(w.trace,indent=2,
                                                                sort_keys=True)+'\n')
    (root/'episodes.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
    (root/'traces.json').write_text(json.dumps(traces,separators=(',',':'),
                                               sort_keys=True)+'\n')
    report={'schema':SCHEMA,'source_sha256':{p:digest(Path(__file__).with_name(p))
                                           for p in ('lab.py','observation_ablation.py')},
            'evidence_class':'trained_tiny_linear_external_synthetic',
            'native_receipts':0,'resident_slm_processes':0,'w0_executions':0,
            'train_seed_range':[TRAIN_BASE,TRAIN_BASE+train_worlds-1],
            'eval_seed_range':[EVAL_BASE,EVAL_BASE+eval_worlds-1],
            'model_checkpoints':models,'stop_unmasked':True,
            'host_legal_menu_unchanged_across_masks':True,
            'episodes_sha256':digest(root/'episodes.json'),
            'traces_sha256':digest(root/'traces.json'),
            'summaries':{p:summary(rs) for p,rs in rows.items()},
            'comparisons':{f'{arm}_{mode}':comparison(rows[f'{arm}_full'],rows[f'{arm}_{mode}'])
                           for arm in ARMS_TESTED for mode in MODES if mode!='full'},
            'qualification_boundary':'same-source synthetic replay, not native or independent model execution'}
    (root/'metrics.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    (root/'SHA256SUMS').write_text(''.join(f'{digest(f)}  {f.relative_to(root).as_posix()}\n'
        for f in sorted(root.rglob('*')) if f.is_file() and f.name!='SHA256SUMS'))
    return report


class AuditRefusal(ValueError): pass


def audit(root):
    root=Path(root);report=json.loads((root/'metrics.json').read_text())
    if report['schema']!=SCHEMA:raise AuditRefusal('schema')
    for p,h in report['source_sha256'].items():
        if p not in ('lab.py','observation_ablation.py') or digest(Path(__file__).with_name(p))!=h:
            raise AuditRefusal('source drift')
    inventory={}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        h,path=line.split('  ',1)
        if path in inventory or Path(path).is_absolute() or '..' in Path(path).parts:
            raise AuditRefusal('invalid manifest path')
        inventory[path]=h
        if digest(root/path)!=h:raise AuditRefusal('manifest mismatch')
    if report['episodes_sha256']!=digest(root/'episodes.json') or \
       report['traces_sha256']!=digest(root/'traces.json'):
        raise AuditRefusal('trace/episode digests')
    rows=json.loads((root/'episodes.json').read_text())
    traces=json.loads((root/'traces.json').read_text())
    n=report['eval_seed_range'][1]-report['eval_seed_range'][0]+1
    if not 5<=n<=100 or report['eval_seed_range'][0]!=EVAL_BASE:
        raise AuditRefusal('evaluation custody')
    expected={f'{arm}_{mode}' for arm in ARMS_TESTED for mode in MODES}
    if set(rows)!=expected or set(traces)!=expected:raise AuditRefusal('panel schema')
    policies={}
    for arm in ARMS_TESTED:
        path=root/'checkpoints'/(arm+'.json')
        if digest(path)!=report['model_checkpoints'][arm]['sha256']:
            raise AuditRefusal('checkpoint SHA')
        ck=json.loads(path.read_text())
        if ck.get('schema')!='gal-004-trained-tiny-linear/1' or ck.get('arm')!=arm or \
           ck.get('features')!=list(FEATURES) or len(ck.get('weights',[]))!=len(FEATURES) or \
           any(type(v) not in (int,float) or not (-6<=v<=6) for v in ck['weights']):
            raise AuditRefusal('checkpoint identity')
        p=TinyPolicy();p.weights=ck['weights'];policies[arm]=p
    count=0;actions=0
    for arm in ARMS_TESTED:
        for mode in MODES:
            panel=f'{arm}_{mode}'
            if len(rows[panel])!=n or len(traces[panel])!=n:raise AuditRefusal('panel count')
            for i,(expected_row,expected_trace) in enumerate(zip(rows[panel],traces[panel])):
                seed=EVAL_BASE+i;family=FAMILIES[i%len(FAMILIES)]
                w,row=run_world(policies[arm],seed,family,arm,mode)
                if row!=expected_row or json.loads(json.dumps(w.trace))!=expected_trace:
                    raise AuditRefusal('host/model/material/receipt replay mismatch')
                count+=1;actions+=row['decisions']
    if report['summaries']!={p:summary(v) for p,v in rows.items()}:
        raise AuditRefusal('summary falsified')
    if report['comparisons']!={f'{arm}_{mode}':comparison(rows[f'{arm}_full'],rows[f'{arm}_{mode}'])
            for arm in ARMS_TESTED for mode in MODES if mode!='full'}:
        raise AuditRefusal('paired metrics falsified')
    return {'audited_synthetic_episodes':count,'replayed_model_selections':actions,
            'native_receipts':0,'resident_slms':0}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    ap.add_argument('--train-worlds',type=int,default=60)
    ap.add_argument('--eval-worlds',type=int,default=30)
    ap.add_argument('--gallery-worlds',type=int,default=2)
    ap.add_argument('--audit-only',action='store_true')
    args=ap.parse_args()
    result=audit(args.output) if args.audit_only else write_run(args.output,
            train_worlds=args.train_worlds,eval_worlds=args.eval_worlds,
            gallery_worlds=args.gallery_worlds)
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=='__main__':main()
