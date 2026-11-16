"""D020: branch-constrained compact student repair, on a source-pinned synthetic game.

Purpose: preserve D017's conditional BRANCH choices under standalone student
compression without sacrificing spatial coverage. This is *not* inference from
a Frankenbrain on the synthetic game: original Q8 is only an identity prior.

The hard branch and coverage constraints are assessed on disjoint *validation*
rollouts. Test/confirmation worlds never enter optimiser, calibration or model
selection. If no candidate satisfies all constraints, refuse promotion.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from .lab import ACTIONS, STYLES, Game, feats
from .repair import adverse
from PIL import Image, ImageDraw
from .repair import load
from .source_parent_branch_gate import Gate, Controller, parents, sha, js

D007_SHA='1a4a35f8f15f979b8460121d31021d562a457de78d86de9b9d2fedbcec05e38f'
D017_SHA='715dff35c08a4d19bb7f523925eace020f35bdcaa3bc7bfe44bf2aa67606f11f'
Q8_SHA='140342c16a0b8c6c8d2c3bc6071767885b15096d5a53bbf304979f3015c86522'
CASES=('normal','trap','squeeze')

def rollout(seeds,controllers,draw=None,receipt_seeds=()):
 rows=[];receipts=[]
 if draw:draw.mkdir(parents=True,exist_ok=True)
 for seed in seeds:
  for style in STYLES:
   for case in CASES:
    for name,model in controllers.items():
     g=Game(seed,style)
     for tick in range(80):
      adverse(g,case);o=g.obs();a=model.predict(o)
      if a not in o['legal']:raise AssertionError('illegal host action')
      if seed in receipt_seeds and style=='branchwitch':
       receipts.append({'seed':seed,'style':style,'case':case,'policy':name,'tick':tick,'action':a,
        'legal':o['legal'],'position':[o['x'],o['y']]})
      g.step(a)
      if g.stopped:break
     rows.append({'seed':seed,'style':style,'case':case,'model':name,'cells':len(g.painted),
      'branches':g.branch_count,'retraces':g.retraces,'revisits':g.revisits,'decisions':g.tick,
      'stopped':int(g.stopped),'illegal':g.invalid})
     if draw and seed in receipt_seeds and style=='branchwitch':
      g.png().save(draw/f'{case}_{seed}_{name}.png')
 return rows,receipts,[]

def summary(rows):
 return {f'{case}/{model}':{k:round(float(np.mean([r[k] for r in rows if r['case']==case and r['model']==model])),4)
   for k in ('cells','branches','retraces','revisits','decisions','stopped','illegal')}
   for case in CASES for model in sorted({r['model'] for r in rows})}

def bootstrap(rows,name,seeds):
 rng=np.random.default_rng(18018);res={}
 by={(r['seed'],r['style'],r['case'],r['model']):r for r in rows}
 for c in CASES:
  res[c]={}
  for key in ('cells','branches'):
   d=np.array([np.mean([by[(seed,st,c,name)][key]-by[(seed,st,c,'d017')][key] for st in STYLES]) for seed in seeds])
   b=d[rng.integers(len(d),size=(2000,len(d)))].mean(1)
   res[c][key]={'mean':round(float(d.mean()),4),'seed_block_95ci':[round(float(t),4) for t in np.quantile(b,[.025,.975])]}
 return res

def gallery(draw,output,names,seeds):
 for case in CASES:
  im=Image.new('RGB',(192*len(names),216*len(seeds)),'#e7e8eb');d=ImageDraw.Draw(im)
  for i,name in enumerate(names):
   for j,seed in enumerate(seeds):
    im.paste(Image.open(draw/f'{case}_{seed}_{name}.png'),(i*192,j*216+24))
    d.text((i*192+4,j*216+4),f'{seed} {name}',fill='#161a22')
  im.save(output/f'd018_{case}_gallery.png')

def dumpjsonl(path,rows):Path(path).write_text(''.join(js(r)+'\n' for r in rows))

SCHEMA='neurounner.driv3r.distill.branch-retention/1'
VALID=range(52001,52009)
TEST=range(53001,53017)
CONFIRM=range(54001,54017)
BRANCH=3

def admissibility(s,name,limits=None):
 """Hard, predeclared behavioural-retention + competence guard, no weighted bypass."""
 limits=limits or {'branch_normal':-1.,'branch_trap':-1.,'coverage_normal':-2.,
                   'coverage_trap':-2.,'coverage_squeeze':-1.5}
 delta={case:{k:s[f'{case}/{name}'][k]-s[f'{case}/d017'][k]
              for k in ('cells','branches')} for case in CASES}
 good=(all(s[f'{case}/{name}']['illegal']==0 for case in CASES)
       and delta['normal']['branches']>=limits['branch_normal']
       and delta['trap']['branches']>=limits['branch_trap']
       and delta['normal']['cells']>=limits['coverage_normal']
       and delta['trap']['cells']>=limits['coverage_trap']
       and delta['squeeze']['cells']>=limits['coverage_squeeze'])
 # Score used *only* among candidates passing all hard constraints.
 score=(.40*delta['normal']['cells']+.40*delta['trap']['cells']
       +.20*delta['squeeze']['cells']+.10*delta['normal']['branches']
       +.10*delta['trap']['branches'])
 return {'admissible':bool(good),'delta':delta,'score':round(float(score),6),'limits':limits}

def fit(student,data,steps,seed,branch_strength,anchor_strength=.03,lr=.0012):
 """Adam optimizer on legal teacher KL + positive/negative branch margin.

 Sources: held-out-from-evaluation D018 teacher-labelled synthetic states. For
 each action, the target is teacher D017's host-masked argmax, not a new d1
 prediction. Regularize both branch-positive and branch-negative cases: naive
 unconditional branch bonus is the explicit behavioural negative control.
 """
 if not (1<=steps<=600):raise ValueError('optimizer budget exceeds bounded run')
 assert set(data)=={'x','t','m','y'}
 xall,tall,mall,yall=(data[k] for k in ('x','t','m','y'))
 if len(xall)!=len(yall) or xall.shape[1]!=100:raise ValueError('corpus shape')
 if not np.all(mall[np.arange(len(yall)),yall]):raise AssertionError('illegal teacher target')
 branchidx=np.flatnonzero(yall==BRANCH)
 nonbranchidx=np.flatnonzero((yall!=BRANCH)&(mall[:,BRANCH]>0))
 if len(branchidx)<100 or len(nonbranchidx)<100:raise ValueError('branch support insufficient')
 pars=[student.w1,student.b1,student.w2,student.b2]
 original=[p.copy() for p in pars]
 m=[np.zeros_like(p) for p in pars];v=[np.zeros_like(p) for p in pars]
 rng=np.random.default_rng(seed)
 losses=[]
 for it in range(1,steps+1):
  # Independent 1/3 teacher-branch and 1/3 non-branch-with-branch-legal.
  # Remaining observations retain full policy competence and STOP/retrace.
  ids=np.concatenate((rng.choice(branchidx,96),rng.choice(nonbranchidx,96),
                      rng.integers(len(yall),size=96)))
  x=xall[ids];t=tall[ids];legal=mall[ids];yb=yall[ids];n=len(ids)
  h=np.tanh(x@pars[0]+pars[1]);z=h@pars[2]+pars[3]
  # Host mask KL, no probability mass assigned to illegal actions.
  a=np.where(legal>0,t/.8,-1e5);a-=a.max(axis=1,keepdims=True)
  q=np.exp(a);q/=q.sum(axis=1,keepdims=True)
  b=np.where(legal>0,z/.8,-1e5);b-=b.max(axis=1,keepdims=True)
  p=np.exp(b);p/=p.sum(axis=1,keepdims=True)
  dz=(.65/(n*.8))*(p-q)
  # Contrastive hinge on legal action ranking, preserving conditionality.
  ids2=np.arange(n)
  other=np.where(legal>0,z,-1e5).copy();other[:,BRANCH]=-1e5
  max_other=other.argmax(axis=1)
  wants_branch=(yb==BRANCH)&(legal[:,BRANCH]>0)
  legal_nonbranch=(yb!=BRANCH)&(legal[:,BRANCH]>0)
  margin=.16
  pos=np.flatnonzero(wants_branch)
  neg=np.flatnonzero(legal_nonbranch)
  hp=np.maximum(0,z[pos,max_other[pos]]-z[pos,BRANCH]+margin)
  hn=np.maximum(0,z[neg,BRANCH]-z[neg,yb[neg]]+margin)
  strength=branch_strength/n
  np.add.at(dz,(pos,max_other[pos]),strength*(hp>0))
  np.add.at(dz,(pos,np.full(len(pos),BRANCH)), -strength*(hp>0))
  np.add.at(dz,(neg,np.full(len(neg),BRANCH)),strength*(hn>0))
  np.add.at(dz,(neg,yb[neg]),-strength*(hn>0))
  # Gentle checkpoint trust; avoids runaway interleaved corrective updates.
  ref=np.tanh(x@original[0]+original[1])@original[2]+original[3]
  dz+=(anchor_strength/n)*(z-ref)
  gh=(dz@pars[2].T)*(1-h*h)
  grads=[x.T@gh,gh.sum(0),h.T@dz,dz.sum(0)]
  for j,(par,grad) in enumerate(zip(pars,grads)):
   grad=np.clip(grad,-5,5);m[j]=.9*m[j]+.1*grad;v[j]=.999*v[j]+.001*grad*grad
   par-=lr*(m[j]/(1-.9**it))/(np.sqrt(v[j]/(1-.999**it))+1e-8)
  if it==1 or it==steps:losses.append({'step':it,'teacher_kl':round(float(np.mean(np.sum(q*np.log(np.maximum(q,1e-8)/np.maximum(p,1e-8)),axis=1))),6),'branch_hinge_pos':round(float(hp.mean()),6),'branch_hinge_neg':round(float(hn.mean()),6)})
 return {'updates':steps,'branch_strength':branch_strength,'anchor':anchor_strength,
         'batch':288,'training_branch_observations':int(len(branchidx)),
         'training_nonbranch_legal':int(len(nonbranchidx)), 'losses':losses}

def main(argv=None):
 ap=argparse.ArgumentParser()
 for key in ('d007','d017','q8','d018','d019b','corpus'):ap.add_argument('--'+key,required=True,type=Path)
 ap.add_argument('--out',required=True,type=Path)
 ap.add_argument('--steps',type=int,default=180)
 args=ap.parse_args(argv)
 if not 1<=args.steps<=600:raise ValueError('bounded updates per candidate')
 sources={key:sha(getattr(args,key)) for key in ('d007','d017','q8','d018','d019b','corpus')}
 if [sources['d007'],sources['d017'],sources['q8']]!=[D007_SHA,D017_SHA,Q8_SHA]:raise ValueError('source model hash mismatch')
 if len(set(VALID)|set(TEST)|set(CONFIRM))!=len(VALID)+len(TEST)+len(CONFIRM):raise AssertionError('seed collision')
 parent=load(args.d007);style=parents(args.q8)
 teacher=Controller(parent,style,Gate.read(args.d017),.5,.58)
 base=Controller(parent,style)
 with np.load(args.corpus,allow_pickle=False) as f:
  data={k:f[k] for k in ('x','t','m','y')}
 out=args.out;out.mkdir(parents=True,exist_ok=True)
 specs=[('d020_anchor',args.d018,.12),('d020_low',args.d019b,.16),('d020_high',args.d019b,.5)]
 validation={};models={};fit_info={};updates=0
 for i,(name,initial,strength) in enumerate(specs):
  student=load(initial)
  if student.w1.shape!=(100,44):raise ValueError('expected 4,714-param standalone compact student')
  tr=fit(student,data,args.steps,202000+i,strength);updates+=tr['updates']
  ck=out/f'{name}.npz';student.save(ck);models[name]=student
  rows,_,_=rollout(VALID,{'d017':teacher,name:student})
  val=admissibility(summary(rows),name)
  validation[name]={'guard':val,'fit':tr,'checkpoint_sha256':sha(ck),'summary':summary(rows)}
  print(js({'candidate':name,'validation':val}),flush=True)
 eligible=[name for name in models if validation[name]['guard']['admissible']]
 # If none qualifies, select a DIAGNOSTIC comparator only, not a promoted model.
 selected=max(eligible or models,key=lambda k:validation[k]['guard']['score'])
 allmod={'d007':base,'d017':teacher,'d018':load(args.d018),'d019b':load(args.d019b),selected:models[selected]}
 selected_keys=list(allmod)
 chosen=tuple(TEST.start+i for i in (0,4,8,12))
 test,actions,_=rollout(TEST,allmod,draw=out/'pngs',receipt_seeds=chosen)
 conf,_,_=rollout(CONFIRM,allmod)
 gallery(out/'pngs',out,selected_keys,chosen)
 for case in CASES:(out/f'd018_{case}_gallery.png').rename(out/f'd020_{case}_gallery.png')
 dumpjsonl(out/'d020_test_rollouts.jsonl',test)
 dumpjsonl(out/'d020_confirmation_rollouts.jsonl',conf)
 dumpjsonl(out/'d020_actions.jsonl',actions)
 metrics={'schema':SCHEMA,'environment':'external_synthetic_host','native_game_executed':False,
  'new_parent_model_forwards':0,'real_q8_source_seed':570,
  'frankenbrain_is_style_prior_only':True,
  'sources_sha256':sources,'train_corpus_generation':'D018 source-only original frozen-D017 synthetic teacher',
  'train_seed_range':[35001,35032],
  'validation_seeds':[VALID.start,VALID.stop-1], 'test_seeds':[TEST.start,TEST.stop-1],
  'confirmation_seeds':[CONFIRM.start,CONFIRM.stop-1],
  'candidates':validation,'candidate_count':len(specs),'optimizer_updates':updates,
  'selected_for_diagnostic':selected,'selected_admissible':bool(eligible),
  'test_guard':admissibility(summary(test),selected),
  'confirmation_guard':admissibility(summary(conf),selected),
  'test':summary(test),'confirmation':summary(conf),
  'test_ci_vs_d017':bootstrap(test,selected,TEST),'confirmation_ci_vs_d017':bootstrap(conf,selected,CONFIRM)}
 metrics['promotion']='REJECTED_out_of_sample_regression' if (not eligible or not metrics['test_guard']['admissible'] or not metrics['confirmation_guard']['admissible']) else 'RESEARCH_ELIGIBLE_not_native_validated'
 (out/'d020_metrics.json').write_text(js(metrics)+'\n')
 (out/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(out)}\n' for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.txt'))
 print(js({'selected':selected,'admissible':bool(eligible),'test':metrics['test'],'confirmation':metrics['confirmation']}),flush=True)
 return metrics

if __name__=='__main__':main()
