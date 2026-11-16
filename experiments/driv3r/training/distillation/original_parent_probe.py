"""D-STUDENT-013: small *actual source-parent* behavior probes, no new parent forwards.

Inputs: immutable 144-action Q8 original-teacher JSONL; NOT synthetic teacher logits.
Rigorous family-held-out plus duplicate-observation embargo, legal soft labels,
state-agnostic prior control, frozen source/corpus pins, synthetic OOD stress.
All checkpoints/evidence remain OUT of Git history. Research only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .lab import ACTIONS, STYLES, Game, feats, Student
from .repair import CASES, adverse, load

SOURCE_SHA = '140342c16a0b8c6c8d2c3bc6071767885b15096d5a53bbf304979f3015c86522'
SOURCE_ARCHIVES={14181164165:'e4bf6b63997967cb0ccc1b4a34e517c7abb62a79d5dc8069e93b9a02371ecff3',
                 14182441899:'0ca2cf83eca44e3462a29ba0a27ae41e62d342bb8a4224f677539a26b68bc181'}
PARENTS=['stock','greyhound','branchwitch']
FAMILIES=['greebles','pipework','architecture']
SEEDS=range(26001,26017)
MAX_DECISIONS=80

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def jd(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)
def softmax(z):
 z=np.asarray(z,dtype=np.float64);z=z-np.max(z,axis=-1,keepdims=True)
 e=np.exp(np.clip(z,-80,30));return e/e.sum(axis=-1,keepdims=True)

def observation_key(r):
 # An exact native-model input is allowed to appear in only one side of a fold.
 # Discard family (part of prompt) only from this conservative duplicate guard.
 o=r['observation']
 return jd({'teacher':r['teacher'],'heading':o['heading'],'last_actions':o['last_actions'],
  'legal':o['legal'],'local_5x5':o['local_5x5'],'tick':o['tick'],'x':o['x'],'y':o['y']})

def source_features(r):
 o=r['observation']
 # Canonical 100-feature synthetic student cannot read true source-parent prompts.
 # This is an explicit, LOSSY adapter. '+' in archived W0 is painted terrain.
 rows=o['local_5x5']
 if len(rows)!=5 or any(len(row)!=5 or set(row)-set('.+@#P*') for row in rows):raise ValueError('invalid parent local grid')
 assert r['teacher'] in STYLES
 translated={**o, 'local_5x5':[row.replace('+','*') for row in rows],
  'style':r['teacher'], 'visited_fraction':sum(row.count('+') for row in rows)/25.,
  'last_revisit':o['last_actions'][-1:] == ['RETRACE']}
 x=feats(translated)
 if x.shape!=(100,) or not np.isfinite(x).all():raise AssertionError('invalid bridge encoding')
 return x

def load_original(path):
 if sha(path)!=SOURCE_SHA:raise ValueError('real parent corpus SHA mismatch')
 records=[json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
 if len(records)!=144 or Counter((r['teacher'],r['family']) for r in records)!=Counter({(t,f):16 for t in PARENTS for f in FAMILIES}):
  raise AssertionError('real Q8 source counts changed')
 x=[];y=[];mask=[];labels=[];keys=[]
 for r in records:
  if r['schema']!='cybersand.driv3r.distill.original-teacher/1' or r['source_kind']!='original_model_inference' or r['source_seed']!=570:raise ValueError('not an original pinned Q8 record')
  if SOURCE_ARCHIVES.get(r['source_artifact_id'])!=r['source_archive_sha256']:raise ValueError('teacher artifact identity changed')
  if r['teacher'] not in PARENTS or r['family'] not in FAMILIES:raise ValueError('invalid original teacher')
  if set(r['observation']['legal'])-set(ACTIONS) or r['action'] not in r['observation']['legal']:raise ValueError('not host-legal')
  yy=np.asarray([r['probabilities'][a] for a in ACTIONS],dtype=np.float64)
  mm=np.asarray([a in r['observation']['legal'] for a in ACTIONS],bool)
  if (yy[~mm]>1e-8).any() or not np.isclose(yy.sum(),1.,atol=1e-6) or ACTIONS[int(np.argmax(yy))]!=r['action']:raise ValueError('teacher logit/action mismatch')
  x.append(source_features(r));y.append(yy);mask.append(mm);labels.append(r['action']);keys.append(observation_key(r))
 return records, np.array(x,dtype=np.float32),np.array(y,dtype=np.float32),np.array(mask),labels,keys

def train(x,y,mask,width,steps,seed,hard_weight=.28):
 if width not in (8,16,24) or not (1<=steps<=600):raise ValueError('bounded training only')
 mod=Student(nf=100,hidden=width,seed=seed)
 p=[mod.w1,mod.b1,mod.w2,mod.b2]
 m=[np.zeros_like(a) for a in p];v=[np.zeros_like(a) for a in p]
 rng=np.random.default_rng(seed); loss=[]
 hard=np.eye(6,dtype=np.float32)[np.argmax(y,axis=1)]
 target=(1-hard_weight)*y+hard_weight*hard
 for t in range(1,steps+1):
  ix=rng.integers(len(x),size=64);xx=x[ix];yy=target[ix];legal=mask[ix]
  h=np.tanh(xx@mod.w1+mod.b1);z=h@mod.w2+mod.b2
  probs=softmax(np.where(legal,z,-1e5))
  dz=np.asarray((probs-yy)/len(ix),dtype=np.float32)
  # Small explicit negative margin for illegal raw logits, without illegal targets.
  best=np.max(np.where(legal,z,-1e5),axis=1)
  margin=np.clip(z-best[:,None]+.5,-30,30);sig=1/(1+np.exp(-margin));sig*=~legal
  dz+=.10*sig/len(ix)
  legal_best=np.argmax(np.where(legal,z,-1e5),axis=1)
  dz[np.arange(len(ix)),legal_best]-=.10*sig.sum(axis=1)/len(ix)
  grads=[xx.T@((dz@mod.w2.T)*(1-h*h)), ((dz@mod.w2.T)*(1-h*h)).sum(0), h.T@dz,dz.sum(0)]
  for j,(param,grad) in enumerate(zip(p,grads)):
   m[j]=.9*m[j]+.1*grad;v[j]=.999*v[j]+.001*(grad*grad)
   param-=.008*(m[j]/(1-.9**t))/(np.sqrt(v[j]/(1-.999**t))+1e-8)
  loss.append(float(-np.mean(np.sum(yy*np.log(probs+1e-12),axis=1))))
 return mod,{'updates':steps,'params':sum(a.size for a in p),'loss_first20':float(np.mean(loss[:20])),
             'loss_last20':float(np.mean(loss[-20:])),'hard_weight':hard_weight}

def empirical_prior(y,mask,records,ids):
 pr={}
 for who in PARENTS:
  ix=[i for i in ids if records[i]['teacher']==who]
  if not ix:raise AssertionError('missing parent')
  pr[who]=np.mean(y[ix],axis=0)
 return pr

def evaluate(rows,records,x,y,mask,prior=None,student=None):
 ans=[]
 for i in rows:
  if prior is not None:
   logits=np.log(np.clip(prior[records[i]['teacher']],1e-8,1))
  else:logits=student.scores(x[i]);
  pred=softmax(np.where(mask[i],logits,-1e5))
  ans.append({'teacher':records[i]['teacher'],'family':records[i]['family'],'tick':int(records[i]['tick']),
      'ce':float(-np.sum(y[i]*np.log(pred+1e-12))),'kl':float(np.sum(y[i]*(np.log(y[i]+1e-12)-np.log(pred+1e-12)))),
      'top1_agreement':int(np.argmax(pred)==np.argmax(y[i])),'raw_illegal':int(np.argmax(logits) not in np.flatnonzero(mask[i])),
      'teacher_top':records[i]['action'],'student_top':ACTIONS[int(np.argmax(pred))]})
 return ans

def summary(a):
 return {key:float(np.mean([r[key] for r in a])) for key in ('ce','kl','top1_agreement','raw_illegal')}

def collapse_student_observations(records,x,y,mask,keys):
 # The parent was prompted with a *family*; the 100-feature student has NO family
 # input. Repeated family-specific parent calls are therefore one equivalent
 # student observation, but targets can genuinely differ between families.
 groups={}
 for i,k in enumerate(keys):groups.setdefault(k,[]).append(i)
 reps=[indices[0] for indices in groups.values()]
 gx=np.asarray([x[i] for i in reps]);gm=np.asarray([mask[i] for i in reps]);gy=np.asarray([np.mean(y[ids],axis=0) for ids in groups.values()],dtype=np.float32)
 alias=[];variation=[]
 for ids in groups.values():
  a=y[ids];mu=np.mean(a,axis=0)
  alias.append(len(ids))
  variation.extend([float(np.sum(z*(np.log(z+1e-12)-np.log(mu+1e-12)))) for z in a])
 return reps,gx,gy,gm,alias,variation

def original_cv(records,x,y,mask,keys,steps):
 reps,gx,gy,gm,aliases,variation=collapse_student_observations(records,x,y,mask,keys)
 # Unlike family-LOO, this split has NO repeated student-visible state across
 # train/test. Tick-modulo folds test unseen temporal states (still ONE source seed).
 folds=[];candidate={};n=len(reps)
 for width in (8,16,24):
  results=[];priors=[]
  for fno in range(4):
   test=[i for i,ridx in enumerate(reps) if int(records[ridx]['tick'])%4==fno]
   trainids=[i for i in range(n) if i not in set(test)]
   raw_test=[reps[i] for i in test]
   raw_train=[reps[i] for i in trainids]
   prior=empirical_prior(gy,gm,[records[i] for i in reps],trainids)
   baseline=evaluate(test,[records[i] for i in reps],gx,gy,gm,prior=prior)
   mod,fit=train(gx[trainids],gy[trainids],gm[trainids],width,steps,seed=13013+width+fno)
   res=evaluate(test,[records[i] for i in reps],gx,gy,gm,student=mod)
   folds.append({'width':width,'heldout_tick_mod4':fno,'train_unique_states':len(trainids),
      'test_unique_states':len(test),'training_visible_test_overlap':0,
      'baseline_prior':summary(baseline),'source_student':summary(res),'fit':fit})
   results.extend(res);priors.extend(baseline)
  candidate[width]={**summary(results),'prior':summary(priors)}
 best_width=min(candidate,key=lambda w:(candidate[w]['kl'],w))
 # Audit whether family-only heldout could have worked without identical
 # student inputs contaminating training. For GREYHOUND/BRANCHWITCH: no.
 family_dedup={}
 for family in FAMILIES:
  denied={keys[i] for i,r in enumerate(records) if r['family']==family}
  family_dedup[family]={who:sum(r['teacher']==who and r['family']!=family and keys[i] not in denied for i,r in enumerate(records)) for who in PARENTS}
 uniqueness={'source_inference_calls':len(records),'unique_student_visible_observations':n,
  'alias_group_sizes':dict(Counter(aliases)),'mean_family_blind_irreducible_kl':float(np.mean(variation)),
  'max_family_blind_irreducible_kl':float(np.max(variation)),
  'heldout_family_remaining_train_per_teacher_after_embargo':family_dedup,
  'interpretation':'An archived source prompt includes family, but the 100-feature student does not. Identical student-visible states with differing parent distributions cannot be exactly represented.'}
 return folds,candidate,best_width,uniqueness,(gx,gy,gm)

def synthetic_eval(models,draw):
 rows=[];receipts=[]
 if draw:draw.mkdir(parents=True,exist_ok=True)
 for seed in SEEDS:
  for style in STYLES:
   for case in CASES:
    for label,model in models.items():
     g=Game(seed,style);raw_invalid=0
     for tick in range(MAX_DECISIONS):
      adverse(g,case);obs=g.obs()
      raw=model.predict(obs,mask=False);raw_invalid+=int(raw not in obs['legal'])
      act=model.predict(obs)
      if act not in obs['legal']:raise AssertionError('host illegal')
      if seed in (26001,26005,26009,26013) and style=='branchwitch':
       receipts.append({'seed':seed,'style':style,'case':case,'model':label,'tick':tick,
         'action':act,'raw':raw,'legal':obs['legal'],'x':obs['x'],'y':obs['y']})
      g.step(act)
      if g.stopped:break
     rows.append({'seed':seed,'style':style,'case':case,'model':label,'cells':len(g.painted),
       'branches':g.branch_count,'revisits':g.revisits,'retraces':g.retraces,
       'decisions':g.tick,'raw_illegal':raw_invalid,'invalid':g.invalid,'stopped':int(g.stopped)})
     if draw and seed in (26001,26005,26009,26013) and style=='branchwitch':
      g.png().save(draw/f'{case}_{seed}_{style}_{label}.png')
 return rows,receipts

def galleries(out,labels):
 for case in CASES:
  im=Image.new('RGB',(192*len(labels),4*216),(240,241,243));pen=ImageDraw.Draw(im)
  for r,seed in enumerate((26001,26005,26009,26013)):
   for c,label in enumerate(labels):
    filename=out/'pngs'/f'{case}_{seed}_branchwitch_{label}.png'
    im.paste(Image.open(filename),(192*c,216*r+24));pen.text((192*c+3,216*r+5),f'{seed} / {label}',fill=(13,18,25))
  im.save(out/f'd013_{case}_gallery.png')

def main(argv=None):
 p=argparse.ArgumentParser()
 p.add_argument('--source',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--d007',type=Path,default=None)
 p.add_argument('--steps',type=int,default=300)
 args=p.parse_args(argv)
 out=args.out;out.mkdir(parents=True,exist_ok=True)
 records,x,y,mask,actions,keys=load_original(args.source)
 folds,candidates,winner,uniqueness,(gx,gy,gm)=original_cv(records,x,y,mask,keys,args.steps)
 final,fit=train(gx,gy,gm,winner,args.steps,seed=13131)
 finalpath=out/'original_parent_student.npz';final.save(finalpath)
 # Copy NO parent logits into a synthetic teacher slot; honest source coverage only.
 models={'d013_source':final}
 if args.d007 is not None:
  assert sha(args.d007)=='1a4a35f8f15f979b8460121d31021d562a457de78d86de9b9d2fedbcec05e38f','D007 pinned checkpoint mismatch'
  models={'d007':load(args.d007),**models}
 rows,receipts=synthetic_eval(models,out/'pngs')
 galleries(out,list(models))
 (out/'synthetic_rollouts.jsonl').write_text(''.join(jd(z)+'\n' for z in rows))
 (out/'action_receipts.jsonl').write_text(''.join(jd(z)+'\n' for z in receipts))
 outsum={}
 for case in CASES:
  for name in models:
   q=[r for r in rows if r['case']==case and r['model']==name]
   outsum[f'{case}/{name}']={k:float(np.mean([r[k] for r in q])) for k in ('cells','branches','revisits','retraces','decisions','raw_illegal','stopped','invalid')}
 # Block bootstrap by *world seed*, not 64 style-world observations as independent.
 paired={}
 if 'd007' in models:
  rng=np.random.default_rng(13013)
  for case in CASES:
   paired[case]={}
   for metric in ('cells','branches','revisits','retraces'):
    diff=np.asarray([np.mean([next(r[metric] for r in rows if (r['case'],r['seed'],r['style'],r['model'])==(case,s,st,'d013_source'))-
       next(r[metric] for r in rows if (r['case'],r['seed'],r['style'],r['model'])==(case,s,st,'d007')) for st in STYLES]) for s in SEEDS])
    boots=diff[rng.integers(len(diff),size=(4000,len(diff)))].mean(axis=1)
    paired[case][metric]={'delta':float(np.mean(diff)),'seed_block_bootstrap_95':[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))]}
 # Source-corpus is already immutable elsewhere. Only carry derived data/no model binaries in Git.
 np.savez_compressed(out/'original_parent_features.npz',x=x,y=y,legal=mask.astype(np.uint8),
   family=np.array([FAMILIES.index(r['family']) for r in records],dtype=np.uint8),
   teacher=np.array([PARENTS.index(r['teacher']) for r in records],dtype=np.uint8),
   tick=np.array([r['tick'] for r in records],dtype=np.uint8))
 report={'schema':'cybersand.driv3r.distill.original-parent-compression/1',
  'source_kind':'144 GENUINE archived Q8 decisions/soft distributions, NO new parent forwards',
  'teacher_model_identity_caveat':'All 3 archived teacher labels share an identical base model_sha256; Frankenlayer mutation is a separate source artifact identity',
  'data_sha256':sha(args.source),'source_archive_sha256':SOURCE_ARCHIVES,'original_parent_calls':len(records),
  'source_seed':570,'source_families':FAMILIES,'heldout_type':'leave-one-family-out, exact-observation duplicate embargo',
  'new_parent_forwards':0,'native_cybersand_runs':0,'feature_bridge':'lossy 100-feature synthetic host adapter, not native runtime contract',
  'folds':folds,'width_cv':candidates,'uniqueness_and_leakage_audit':uniqueness,'selected_width':winner,
  'optimizer_updates':(4*3+1)*args.steps,'final_fit':fit,
  'final_checkpoint_sha256':sha(finalpath),'feature_corpus_sha256':sha(out/'original_parent_features.npz'),
  'ood_synthetic_episodes':len(rows),'synthetic_seed_range':[SEEDS.start,SEEDS.stop-1],
  'synthetic_summary':outsum,'paired_vs_d007':paired,
  'synthetic_rollout_sha256':sha(out/'synthetic_rollouts.jsonl'),
  'action_receipts_sha256':sha(out/'action_receipts.jsonl'),
  'interpretation':'Original parent fit and CV are source-observation tests; external synthetic game is intentionally OOD. No faithful new-parent generalization or native gameplay acceptance.'}
 (out/'metrics.json').write_text(jd(report)+'\n')
 (out/'SHA256SUMS.txt').write_text(''.join(f'{sha(q)}  {q.relative_to(out).as_posix()}\n' for q in sorted(out.rglob('*')) if q.is_file() and q.name!='SHA256SUMS.txt'))
 print(jd({'selected_width':winner,'checkpoint':sha(finalpath),'original_unique_observation_cv':candidates,'unique_input_audit':uniqueness,'synthetic_summary':outsum,'updates':report['optimizer_updates']}),flush=True)
 return report

if __name__=='__main__':main()
