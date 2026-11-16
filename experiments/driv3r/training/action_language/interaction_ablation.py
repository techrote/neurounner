"""GAL-006: train/evaluate candidate-specific pairwise interaction terms.

External CyberSand-like synthetic host only. Material/legal actions/STOP/reward
remain host owned; an alternative policy feature map is the only model change.
No native Godot PAINT, W0, resident SLM, game write or paid inference.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from lab import World, FEATURES, DECISION_CAP, sha, png, reward, HostRefusal
from candidate_features import CandidateWorld, FAMILIES
from shape_metrics import morphology

SCHEMA = 'gal-006-pairwise-interactions/1'
BASE_SEED = 53000
TEST_SEED = 310000
TARGET = 32
ARMS = ('cell', 'sweep')
INTERACTIONS = (
    'near_x_span', 'clear_x_span', 'near_x_turn', 'clear_x_turn',
    'near_x_clear', 'near_x_visited', 'clear_x_persist', 'near_x_straight',
)
FEATURE_MAP = FEATURES + INTERACTIONS


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class InkTargetWorld(CandidateWorld):
    """Separate external host experiment: cap new material at 32 pixels.

    A completed 32-ink target ends the *experiment*, not the model's STOP.
    All strokes continue to use the original host path and token authority.
    """
    def legal(self):
        actions = super().legal()
        remaining = TARGET-len(self.own)
        return {name:path for name,path in actions.items()
                if not name.startswith(('INK_', 'JOG_')) or len(path)<=remaining}


def feature_map(world, action, path, *, nonlinear=True):
    base = tuple(world.features(action, path))
    if not nonlinear:
        return base
    d = dict(zip(FEATURES, base))
    # Do not fabricate geometry for STOP; action-specific terms only.
    eligible = float(action.startswith(('INK_', 'JOG_')))
    near, clear = d['near_ink'], d['open']
    turn = d['left']+d['right']
    interaction = (
        near*d['span'], clear*d['span'], near*turn, clear*turn,
        near*clear, near*d['already_visited'], clear*d['persist'], near*d['straight'],
    )
    return base + tuple(eligible*x for x in interaction)


class Policy:
    def __init__(self, nonlinear):
        self.nonlinear = bool(nonlinear)
        self.names = FEATURE_MAP if nonlinear else FEATURES
        self.weights = [0.0]*len(self.names)

    def probabilities(self, vectors):
        scores=[sum(w*x for w,x in zip(self.weights,vec)) for vec in vectors]
        mx=max(scores)
        z=[math.exp(max(-40.,min(0.,v-mx))) for v in scores]
        total=sum(z)
        return [a/total for a in z]

    def choose(self, world, rng):
        menu, token = world.offer()
        names=list(menu)
        vectors=[feature_map(world,n,menu[n],nonlinear=self.nonlinear) for n in names]
        probs=self.probabilities(vectors)
        u=rng.random()
        cumulative=0.
        index=len(names)-1
        for k,p in enumerate(probs):
            cumulative+=p
            if u<=cumulative:
                index=k
                break
        return names[index],token,names,vectors,probs,index


def train(arm, nonlinear, episodes):
    if arm not in ARMS or not 1<=episodes<=120:
        raise ValueError('bounded training only')
    model=Policy(nonlinear)
    steps=0
    for i in range(episodes):
        seed=BASE_SEED+i
        family=FAMILIES[i%len(FAMILIES)]
        world=InkTargetWorld(seed,family,arm,'candidate',lease=f'gal006-train:{arm}:{i}')
        rng=random.Random(f'gal006-train:{arm}:{seed}:shared-coins')
        baseline=.25
        while not world.stopped and world.decisions<DECISION_CAP and len(world.own)<TARGET:
            action,token,names,vecs,probs,index=model.choose(world,rng)
            # Lock reward to GAL-001 *legacy features*, not extra interactions.
            legacy_phi=World.features(world, action, world.legal()[action])
            receipt=world.apply(action,token,original_response=f'gal006-{int(nonlinear)}:{action}')
            r=reward(world,receipt,legacy_phi)
            adv=max(-1.,min(1.,r-baseline))
            baseline=.95*baseline+.05*r
            expected=[sum(p*v[j] for p,v in zip(probs,vecs)) for j in range(len(model.weights))]
            for j in range(len(model.weights)):
                entropy=sum((1./len(probs)-p)*v[j] for p,v in zip(probs,vecs))
                model.weights[j] = max(-6.,min(6.,model.weights[j]+.06*(adv*(vecs[index][j]-expected[j])+.015*entropy)))
            steps+=1
    return model,steps


def rollout(model,seed,family,arm,panel):
    world=InkTargetWorld(seed,family,arm,'candidate',lease=f'gal006-eval:{panel}:{seed}:{family}')
    rng=random.Random(f'gal006-eval:{arm}:{seed}:shared-coins')
    first_target=None
    while not world.stopped and world.decisions<DECISION_CAP and len(world.own)<TARGET:
        action,token,_,_,_,_=model.choose(world,rng)
        world.apply(action,token,original_response=f'gal006-{int(model.nonlinear)}:{action}')
        if len(world.own)==TARGET: first_target=world.decisions
    assert len(world.own)<=TARGET
    assert all(world.material[world.idx(p)]==world.original[world.idx(p)] for p in world.protected)
    shape=morphology(world.own,world.trace)
    row={'seed':seed,'family':family,'arm':arm,'panel':panel,'lease':world.lease,
         'world_sha256':sha(world.original),'final_material_sha256':sha(world.material),
         'receipts_sha256':sha(world.trace),'new_ink':len(world.own),
         'decisions':world.decisions,'logical_moves':world.logical_moves,
         'selected_stop':world.stopped,'reached_target':first_target is not None,
         'decisions_to_target':first_target, 'shape':shape,
         'evidence_class':'external_synthetic_trained_small_policy','native_receipts':0,
         'resident_slm_processes':0,'w0_executions':0}
    return world,row


def aggregate(rows):
    reached=[r['decisions_to_target'] for r in rows if r['reached_target']]
    return {'worlds':len(rows),'reached_exactly_32':len(reached),
            'decisions_to_target_success_only':round(statistics.mean(reached),3) if reached else None,
            'mean_ink':round(statistics.mean(r['new_ink'] for r in rows),3),
            'pooled_ink_per_decision':round(sum(r['new_ink'] for r in rows)/max(1,sum(r['decisions'] for r in rows)),4),
            'mean_bends':round(statistics.mean(r['shape']['paint_path_bends'] for r in rows),3),
            'mean_junctions':round(statistics.mean(r['shape']['junctions'] for r in rows),3),
            'early_model_stop':sum(r['selected_stop'] and r['new_ink']<TARGET for r in rows)}


def compare(linear,interaction):
    if len(linear)!=len(interaction): raise ValueError('unmatched length')
    both=[];improved=regressed=tied=0
    for a,b in zip(linear,interaction):
        if (a['seed'],a['family'],a['world_sha256'])!=(b['seed'],b['family'],b['world_sha256']):
            raise ValueError('unmatched world')
        if b['new_ink']>a['new_ink']:improved+=1
        elif b['new_ink']<a['new_ink']:regressed+=1
        else:tied+=1
        if a['reached_target'] and b['reached_target']:
            both.append(a['decisions_to_target']-b['decisions_to_target'])
    return {'nonlinear_more_ink':improved,'linear_more_ink':regressed,'equal_ink':tied,
            'both_target':len(both),
            'nonlinear_decision_saving_when_both_complete':round(statistics.mean(both),3) if both else None}


def run(out, train_worlds=75, eval_worlds=30, gallery_worlds=2):
    if not 1<=train_worlds<=120 or not 5<=eval_worlds<=60 or not 0<=gallery_worlds<=3:
        raise ValueError('bounded experiment only')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'checkpoints').mkdir(exist_ok=True)
    models={}; training={};rows={};traces={}
    for arm in ARMS:
        for nonlinear in (False,True):
            tag='pairwise' if nonlinear else 'linear'
            panel=f'{arm}_{tag}'
            model,updates=train(arm,nonlinear,train_worlds)
            models[panel]=model
            check={'schema':'gal006-optimizer-checkpoint/1','arm':arm,'model':tag,
                   'features':list(model.names),'weights':model.weights,'updates':updates,
                   'optimizer':'bounded_onpolicy_REINFORCE','train_seeds':[BASE_SEED,BASE_SEED+train_worlds-1]}
            fp=out/'checkpoints'/f'{panel}.json'
            fp.write_text(json.dumps(check,sort_keys=True,indent=2)+'\n')
            training[panel]={'updates':updates,'checkpoint_sha256':digest_file(fp)}
            rows[panel]=[];traces[panel]=[]
            for j in range(eval_worlds):
                seed=TEST_SEED+j;family=FAMILIES[j%len(FAMILIES)]
                world,row=rollout(model,seed,family,arm,panel)
                rows[panel].append(row);traces[panel].append(world.trace)
                if j<gallery_worlds:
                    p=out/'gallery';p.mkdir(exist_ok=True)
                    png(p/f'{panel}-{seed}-{family}.png',world)
    (out/'episodes.json').write_text(json.dumps(rows,sort_keys=True,indent=2)+'\n')
    (out/'receipts.json').write_text(json.dumps(traces,sort_keys=True,separators=(',',':'))+'\n')
    report={'schema':SCHEMA,'evidence':'synthetic_only_not_native_or_W0',
            'reward_is_unchanged':True,'host_offers_are_unchanged_except_32_ink_cap':True,
            'STOP_always_in_menu':True,'train_seeds':[BASE_SEED,BASE_SEED+train_worlds-1],
            'eval_seeds':[TEST_SEED,TEST_SEED+eval_worlds-1],
            'source_sha256':{p:digest_file(Path(__file__).with_name(p)) for p in
                ('lab.py','candidate_features.py','shape_metrics.py','interaction_ablation.py')},
            'training':training,'summary':{name:aggregate(r) for name,r in rows.items()},
            'matched':{arm:compare(rows[f'{arm}_linear'],rows[f'{arm}_pairwise']) for arm in ARMS},
            'episodes_sha256':digest_file(out/'episodes.json'),
            'receipts_sha256':digest_file(out/'receipts.json')}
    (out/'metrics.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS').write_text(''.join(f'{digest_file(f)}  {f.relative_to(out).as_posix()}\n'
            for f in sorted(out.rglob('*')) if f.is_file() and f.name!='SHA256SUMS'))
    return report


class AuditRefusal(ValueError): pass


def audit(path):
    """Independent re-execution of selections, host receipts, and held-out material."""
    root=Path(path)
    report=json.loads((root/'metrics.json').read_text())
    if report.get('schema')!=SCHEMA: raise AuditRefusal('wrong schema')
    ntrain=report['train_seeds'][1]-report['train_seeds'][0]+1
    neval=report['eval_seeds'][1]-report['eval_seeds'][0]+1
    if report['train_seeds'][0]!=BASE_SEED or not 1<=ntrain<=120 or report['eval_seeds'][0]!=TEST_SEED or not 5<=neval<=60:
        raise AuditRefusal('seed range/custody')
    sources={p:digest_file(Path(__file__).with_name(p)) for p in
             ('lab.py','candidate_features.py','shape_metrics.py','interaction_ablation.py')}
    if sources!=report.get('source_sha256'):raise AuditRefusal('source mismatch')
    inventory={}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        h,p=line.split('  ',1)
        if p in inventory or Path(p).is_absolute() or '..' in Path(p).parts:
            raise AuditRefusal('unsafe/duplicate manifest path')
        inventory[p]=h
        if not (root/p).is_file() or digest_file(root/p)!=h:
            raise AuditRefusal('artifact mismatch')
    included={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS'}
    if set(inventory)!=included:raise AuditRefusal('unlisted artifact')
    if digest_file(root/'episodes.json')!=report['episodes_sha256'] or digest_file(root/'receipts.json')!=report['receipts_sha256']:
        raise AuditRefusal('original evidence mismatch')
    rows=json.loads((root/'episodes.json').read_text())
    receipts=json.loads((root/'receipts.json').read_text())
    if set(rows)!={f'{arm}_{t}' for arm in ARMS for t in ('linear','pairwise')} or set(rows)!=set(receipts):
        raise AuditRefusal('panel identities')
    count=choices=0
    for panel in rows:
        arm,tag=panel.split('_')
        cp_path=root/'checkpoints'/f'{panel}.json'
        if digest_file(cp_path)!=report['training'][panel]['checkpoint_sha256']:
            raise AuditRefusal('checkpoint hash')
        cp=json.loads(cp_path.read_text())
        nonlinear=(tag=='pairwise')
        model=Policy(nonlinear)
        if cp['features']!=list(model.names) or cp['arm']!=arm or cp['model']!=tag or len(cp['weights'])!=len(model.weights):
            raise AuditRefusal('checkpoint schema')
        if any(type(v) not in (int,float) or not math.isfinite(v) or not -6<=v<=6 for v in cp['weights']):
            raise AuditRefusal('checkpoint values')
        model.weights=cp['weights']
        if len(rows[panel])!=neval or len(receipts[panel])!=neval:
            raise AuditRefusal('panel coverage')
        for j,(row,events) in enumerate(zip(rows[panel],receipts[panel])):
            if row['seed']!=TEST_SEED+j or row['family']!=FAMILIES[j%len(FAMILIES)] or row['panel']!=panel or row['lease']!=f'gal006-eval:{panel}:{row["seed"]}:{row["family"]}':
                raise AuditRefusal('seed custody')
            w=InkTargetWorld(row['seed'],row['family'],arm,'candidate',lease=row['lease'])
            if sha(w.original)!=row['world_sha256']:raise AuditRefusal('initial World')
            rng=random.Random(f'gal006-eval:{arm}:{row["seed"]}:shared-coins')
            for receipt in events:
                if w.stopped or w.decisions>=DECISION_CAP or len(w.own)>=TARGET:
                    raise AuditRefusal('continuation past end')
                action,token,_,_,_,_=model.choose(w,rng)
                if receipt['action_id']!=action:raise AuditRefusal('model-selected action mismatch')
                actual=w.apply(action,token,original_response=f'gal006-{int(nonlinear)}:{action}')
                if sha(actual)!=sha(receipt):raise AuditRefusal('original host receipt mismatch')
                choices+=1
            if (w.decisions!=row['decisions'] or len(w.own)!=row['new_ink'] or
                w.logical_moves!=row['logical_moves'] or w.stopped!=row['selected_stop'] or
                sha(w.trace)!=row['receipts_sha256'] or sha(w.material)!=row['final_material_sha256']):
                raise AuditRefusal('material/decision mismatch')
            if w.decisions<DECISION_CAP and not w.stopped and len(w.own)<TARGET:
                raise AuditRefusal('incomplete episode')
            count+=1
        if aggregate(rows[panel])!=report['summary'][panel]:
            raise AuditRefusal('metrics mismatch')
    if {arm:compare(rows[f'{arm}_linear'],rows[f'{arm}_pairwise']) for arm in ARMS}!=report['matched']:
        raise AuditRefusal('paired metrics mismatch')
    return {'replayed_episodes':count,'replayed_model_actions':choices,
            'native_receipts':0,'resident_slm_processes':0,'w0_executions':0,
            'qualification':'synthetic self-replay, not independent native validation'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--train',type=int,default=75)
    p.add_argument('--eval',type=int,default=30)
    p.add_argument('--gallery',type=int,default=2)
    a=p.parse_args()
    print(json.dumps(run(a.output,a.train,a.eval,a.gallery),indent=2,sort_keys=True))
    print(json.dumps(audit(a.output),indent=2,sort_keys=True))


if __name__=='__main__':main()
