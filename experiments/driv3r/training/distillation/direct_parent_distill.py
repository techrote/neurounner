"""D-STUDENT-015: parent-IDENTITY conditioned tiny student, with strict late-tick gate.

Contrasts parent graph identity and layer-flow graph fingerprint, and a
variant-conditional constant prior. All parent choices are ORIGINAL frozen Q8
forwards at seed 570; this is NOT native CyberSand or new parent inference.
"""
from __future__ import annotations
from collections import Counter
import argparse
from pathlib import Path
import numpy as np
from .franken_source_distill import (
 extract,arrays,ALL,PINS,train,evaluate,smry,family_prior,sha,jd,softmax)

FULL=('stock',)+tuple(v for v in PINS if v!='branchwitch_restore12')

def embed(x,records):
 return np.concatenate([x,np.asarray([np.eye(len(ALL),dtype=np.float32)[ALL.index(r['variant'])] for r in records])],axis=1)

def score(m,records,x,y,mask,ids):
 return smry(evaluate(m,records,x,y,mask,ids))

def prior_eval(records,y,mask,ids,trainids):
 prior={k:v for (k,),v in family_prior(records,y,trainids,('variant',)).items()}
 return smry(evaluate(None,records,None,y,mask,ids,prior=prior))

def main(argv=None):
 p=argparse.ArgumentParser()
 p.add_argument('--sources',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--steps',type=int,default=480)
 args=p.parse_args(argv)
 records,_=extract(args.sources)
 x,y,mask=arrays(records)
 ident=embed(x,records)
 # All true late ticks are entirely excluded from optimization and model selection.
 tr=np.asarray([i for i,r in enumerate(records) if r['tick']<12],dtype=int)
 val=np.asarray([i for i,r in enumerate(records) if 12<=r['tick']<14],dtype=int)
 test=np.asarray([i for i,r in enumerate(records) if 14<=r['tick']<16],dtype=int)
 if len(set(tr)&set(val)) or len(set(tr)&set(test)) or len(set(val)&set(test)):raise AssertionError('split overlap')
 if len(val)!=len(test) or len(test)!=6*len(FULL):raise AssertionError('late split mismatch')
 candidates=[];models={}
 for cfg,width,hard in [('graph32',32,.18),('graph64',64,.18),('identity32',32,.18),('identity64',64,.18),('identity64_soft',64,0.)]:
  xx=ident if cfg.startswith('identity') else x
  model,fit=train(xx[tr],y[tr],mask[tr],width,args.steps,seed=15015+width+len(cfg),hard_weight=hard)
  path=args.out/f'{cfg}.npz';args.out.mkdir(parents=True,exist_ok=True);model.save(path)
  a=score(model,records,xx,y,mask,val)
  candidates.append({'config':cfg,'fit':fit,'validation':a,'sha256':sha(path)})
  models[cfg]=model
 chosen=max(candidates,key=lambda c:(c['validation']['action_agreement']-.25*c['validation']['kl'],-c['fit']['parameters']))
 config=chosen['config'];xx=ident if config.startswith('identity') else x
 predictions=evaluate(models[config],records,xx,y,mask,test)
 (args.out/'unseen_late_tick_receipts.jsonl').write_text(''.join(jd(r)+'\n' for r in predictions))
 per={v:smry([q for q in predictions if q['variant']==v]) for v in FULL}
 # Convert source to a compressed *student* (full corpus re-training, no heldout claims).
 full,fit=train(xx,y,mask,(64 if '64' in config else 32),args.steps,seed=15515,hard_weight=0. if config.endswith('soft') else .18)
 fullpath=args.out/'trained_all_parent_personas.npz';full.save(fullpath)
 report={'schema':'cybersand.driv3r.distill.parent-identity/1','source_seed':570,'source_call_count':len(records),
  'train_late_split':{'train_ticks':'0..11','valid_ticks':'12..13','test_ticks':'14..15'},
  'source_distinct_run_identity':'Q8 System One model with ten different executed layer flows',
  'actual_parent_calls_new':0,'native_cybersand_runs':0,'steps':args.steps,'optimizer_updates':args.steps*(len(candidates)+1),
  'test_prior':prior_eval(records,y,mask,test,tr),'validation_prior':prior_eval(records,y,mask,val,tr),
  'candidates':candidates,'selected':config,'late_test':smry(predictions),'late_test_per_variant':per,
  'trained_full_checkpoint_sha256':sha(fullpath),'full_training_fit':score(full,records,xx,y,mask,np.arange(len(records))),
  'interpretation':'Source-parent labels genuinely used in optimizer; tick holdout shares source seed 570 and has only 6 decisions per source graph. Do NOT claim seed-independent parent fidelity or native game usability.'}
 (args.out/'metrics.json').write_text(jd(report)+'\n')
 (args.out/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(args.out.glob('*')) if p.is_file() and p.name!='SHA256SUMS.txt'))
 print(jd({'selected':config,'validation':chosen['validation'],'test':report['late_test'],
     'prior_test':report['test_prior'],'per':per,'full_sha256':sha(fullpath),'updates':report['optimizer_updates']}),flush=True)
 return report
if __name__=='__main__':main()
