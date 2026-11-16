"""D-STUDENT-003: rehearsal versus exploration repair; synthetic external game only.

The 144 archived parent forwards share seed 570. No new parent inference, no
native CyberSand change, and no faithful parent-model distillation is claimed.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .lab import STYLES, curriculum, load_original, sha, js
from .repair import CASES, VALID, TEST, episode, legacy_val, load

SCHEMA = 'cybersand.driv3r.distill.style-rehearsal/1'
FRACTIONS = (0.15, 0.35, 0.55)
TRAIN_SEEDS = range(1,65)
SELECT_SEEDS = range(1001,1009)
TEST_SEEDS = range(2001,2017)
BUDGET = 80

def legal_softmax(logits,legal):
    if logits.shape != legal.shape or logits.ndim != 2 or not np.all(legal.any(1)):
        raise ValueError('Broken raw-vs-legal feature contract')
    z = np.where(legal,logits,-1e5)
    z -= z.max(1,keepdims=True)
    p = np.exp(z)
    p /= p.sum(1,keepdims=True)
    return p.astype(np.float32)

def make_targets(features,correction,base,fraction):
    if features.ndim != 2 or features.shape[1] != 100 or correction.shape != (len(features),6):
        raise ValueError('Incompatible 100-feature/six-action contract')
    if not (0 <= fraction <= 1) or not np.isfinite(features).all() or not np.isfinite(correction).all():
        raise ValueError('Nonfinite input or invalid fraction')
    legal = features[:,75:81] > .5
    if not legal.any(1).all() or np.max(np.abs(correction.sum(1)-1)) > 1e-4:
        raise ValueError('Missing legal action or nonnormalised corrective distribution')
    if np.any(correction[~legal] > 1e-6):
        raise ValueError('Illegal corrective probability mass')
    rehearsal = legal_softmax(base.scores(features),legal)
    target = ((1.-fraction)*correction+fraction*rehearsal)*legal
    target /= target.sum(1,keepdims=True)
    return target.astype(np.float32),rehearsal

def evaluate(model,seeds):
    out = {}
    for case in CASES:
        rows = []
        for seed in seeds:
            for style in STYLES:
                _,r = episode(seed,style,case,'repaired',model,model,BUDGET)
                rows.append(r)
        out[case] = {k:float(np.mean([r[k] for r in rows])) for k in
          ('cells','branches','revisits','retraces','raw_preprojection_invalid',
           'deadend_exposures','deadend_recoveries','decisions','repetition')}
    return out

def select_score(candidate,base,repaired):
    cov = float(np.mean([candidate[c]['cells']/max(1.,repaired[c]['cells'])
                         for c in ('normal','trap')]))
    branches = float(np.mean([candidate[c]['branches']/max(1.,base[c]['branches'])
                              for c in CASES]))
    score = .65*min(cov,1.15)+.35*min(branches,1.3)
    if cov < .90:
        score -= 2.*(.90-cov)
    return {'score':float(score),'coverage_retained_ratio':cov,
       'original_branch_ratio':branches,'coverage_at_least_90pct':cov>=.90}

def heldout(base,repaired,restored):
    records=[]
    for case in CASES:
        for seed in TEST_SEEDS:
            for style in STYLES:
                for label,m in (('original_001',base),('repaired_002',repaired),
                                ('restored_003',restored)):
                    _,row=episode(seed,style,case,'repaired',base,m,BUDGET)
                    records.append({k:v for k,v in row.items() if k!='action_trace'}|{'controller':label})
                for label in ('flat_continue','wall'):
                    _,row=episode(seed,style,case,label,base,restored,BUDGET)
                    records.append({k:v for k,v in row.items() if k!='action_trace'})
    summary={}
    for case in CASES:
        for label in ('original_001','repaired_002','restored_003','flat_continue','wall'):
            rows=[r for r in records if r['case']==case and r['controller']==label]
            summary[case+'/'+label]={k:float(np.mean([r[k] for r in rows])) for k in
              ('cells','branches','revisits','retraces','raw_preprojection_invalid',
               'deadend_exposures','deadend_recoveries','decisions','repetition')}
    return records,summary

def gallery(base,repaired,restored,out):
    seeds=(2001,2005,2009,2013)
    controls=(('original_001',base),('repaired_002',repaired),
              ('restored_003',restored),('flat_continue',None))
    im=Image.new('RGB',(4*192,4*218),(236,238,242))
    d=ImageDraw.Draw(im)
    for iy,(seed,style) in enumerate(zip(seeds,STYLES)):
        for ix,(name,model) in enumerate(controls):
            policy='flat_continue' if model is None else 'repaired'
            game,_=episode(seed,style,'squeeze',policy,base,model or restored,BUDGET)
            im.paste(game.png(),(ix*192,iy*218+24))
            d.text((ix*192+3,iy*218+4),style+'/'+name,fill=(10,10,10))
    im.save(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--repair',type=Path,required=True)
    ap.add_argument('--corpus',type=Path,required=True)
    ap.add_argument('--repair-metrics',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=160)
    args=ap.parse_args()
    if not 1<=args.steps<=400:
        raise ValueError('Optimizer update count out of bounded range')
    args.out.mkdir(parents=True,exist_ok=True)
    meta=json.loads(args.repair_metrics.read_text())
    original=load_original()
    if sha(args.base.read_bytes())!=meta['base_sha256'] or sha(args.repair.read_bytes())!=meta['repair_sha256']:
        raise ValueError('D001/D002 checkpoint integrity failure')
    if sha(args.corpus.read_bytes())!=meta['train_corpus_sha256']:
        raise ValueError('D002 corpus integrity failure')
    base=load(args.base);repaired=load(args.repair)
    with np.load(args.corpus,allow_pickle=False) as z:
        if set(z.files)!={'features','soft_targets'}:
            raise ValueError('Unexpected corpus members')
        x=z['features'].astype(np.float32);y=z['soft_targets'].astype(np.float32)
    if len(x)!=meta['onpolicy_synthetic_examples'] or len(x)!=33661:
        raise ValueError('Synthetic corpus cardinality changed')
    if meta['parent_inference_new_seeds']!=0 or meta['parent_seed']!=570 or len(original)!=144:
        raise ValueError('Parent provenance changed: manual audit required')
    base_val=evaluate(base,SELECT_SEEDS)
    repair_val=evaluate(repaired,SELECT_SEEDS)
    candidates=[]
    for fraction in FRACTIONS:
        target,anchor=make_targets(x,y,base,fraction)
        model=load(args.repair)
        training=model.train(x,target,iterations=args.steps,lr=.004,seed=317)
        v=evaluate(model,SELECT_SEEDS)
        selected=select_score(v,base_val,repair_val)
        name=f'style_candidate_{fraction:.2f}.npz'
        model.save(args.out/name)
        candidates.append({'rehearsal_fraction':fraction,'optimizer':training,
         'validation':v,'selection':selected,'checkpoint':name,
         'checkpoint_sha256':sha((args.out/name).read_bytes()),
         'corrective_argmax_agreement':float(np.mean(model.scores(x).argmax(1)==y.argmax(1))),
         'original_student_argmax_agreement':float(np.mean(model.scores(x).argmax(1)==anchor.argmax(1)))})
    winner=max(candidates,key=lambda r:(r['selection']['score'],-r['rehearsal_fraction']))
    restored=load(args.out/winner['checkpoint'])
    restored.save(args.out/'style_restored_student.npz')
    records,summary=heldout(base,repaired,restored)
    (args.out/'style_rehearsal_rollouts.jsonl').write_text(''.join(js(r)+'\n' for r in records))
    gallery(base,repaired,restored,args.out/'style_rehearsal_gallery.png')
    oldval=curriculum(original,VALID.start,VALID.stop)
    results={'schema':SCHEMA,'research_kind':'external_synthetic_game_not_native',
     'new_parent_inference':0,'original_parent_receipts':144,'parent_seed':570,
     'aerohound_actual_parent_forwards':0,'training_source_rows':len(x),
     'source_corpus_sha256':sha(args.corpus.read_bytes()),
     'original_teacher_sha256':meta['original_corpus_sha256'],
     'base_001_sha256':sha(args.base.read_bytes()),
     'repair_002_sha256':sha(args.repair.read_bytes()),
     'restored_003_sha256':sha((args.out/'style_restored_student.npz').read_bytes()),
     'train_seeds':[TRAIN_SEEDS.start,TRAIN_SEEDS.stop-1],
     'validation_seeds':[SELECT_SEEDS.start,SELECT_SEEDS.stop-1],
     'test_seeds':[TEST_SEEDS.start,TEST_SEEDS.stop-1],
     'optimizer_updates_per_candidate':args.steps,'candidate_models':len(candidates),
     'validation_baselines':{'original_001':base_val,'repaired_002':repair_val},
     'candidate_experiments':candidates,'selected_fraction':winner['rehearsal_fraction'],
     'selected_validation_score':winner['selection'],
     'surrogate_teacher_validation':{'original_001':legacy_val(base,oldval),
       'repaired_002':legacy_val(repaired,oldval),'restored_003':legacy_val(restored,oldval)},
     'heldout_episode_count':len(records),'heldout_summary':summary,
     'gallery_sha256':sha((args.out/'style_rehearsal_gallery.png').read_bytes()),
     'rollouts_sha256':sha((args.out/'style_rehearsal_rollouts.jsonl').read_bytes()),
     'disclaimer':'Rehearsal against a synthetic teacher proxy, not genuine new Frankenbrain inference'}
    (args.out/'style_rehearsal_metrics.json').write_text(js(results)+'\n')
    print(js({'chosen':winner['rehearsal_fraction'],'selection':winner['selection'],
      'candidate_scores':[{'fraction':r['rehearsal_fraction'],**r['selection']} for r in candidates],
      'test_summary':summary,'surrogate_validation':results['surrogate_teacher_validation']}))

if __name__=='__main__':
    main()
