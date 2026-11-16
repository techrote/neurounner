"""D017: source-Q8-anchored counterfactual branch-gate student.

Research-only synthetic host. Historical Q8 source seed 570 establishes style
preference only. Counterfactual labels come from replay of the external Game,
not from parent inferences on unseen worlds. Do not modify the native game.
"""
from __future__ import annotations
import argparse, collections, copy, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .lab import ACTIONS, STYLES, Game, feats
from .repair import adverse, load

ORIGINAL_SHA = '140342c16a0b8c6c8d2c3bc6071767885b15096d5a53bbf304979f3015c86522'
D007_SHA = '1a4a35f8f15f979b8460121d31021d562a457de78d86de9b9d2fedbcec05e38f'
CASES=('normal','trap','squeeze')
TRAIN=range(31001,31033); VALID=range(32001,32009)
TEST=range(33001,33017); CONFIRM=range(34001,34017)
HORIZON=8; MAX_STEPS=80
SCHEMA='cybersand.driv3r.distill.source-q8-counterfactual-branch-gate/1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)

def parents(path):
 if sha(path)!=ORIGINAL_SHA:raise ValueError('original Q8 source SHA mismatch')
 rows=[json.loads(s) for s in Path(path).read_text().splitlines()]
 if len(rows)!=144:raise ValueError('original source count')
 by=collections.defaultdict(list)
 for r in rows:
  if r['schema']!='cybersand.driv3r.distill.original-teacher/1' or r['source_seed']!=570 or r['source_kind']!='original_model_inference':raise ValueError('source provenance failure')
  if r['action'] not in r['observation']['legal']:raise ValueError('illegal original teacher receipt')
  if r['model_sha256']!='073313d92b53b0ac8193db6e906cb3c78a98183d244b15b7b8f5426115c6738c':raise ValueError('wrong d1 parent')
  if set(r['probabilities'])!=set(ACTIONS) or abs(sum(r['probabilities'].values())-1)>1e-5:raise ValueError('bad parent probabilities')
  by[r['teacher']].append(r)
 if {k:len(v) for k,v in by.items()}!={'stock':48,'greyhound':48,'branchwitch':48}:raise ValueError('unexpected parent groups')
 return {k:{'n':len(v),'branch_top1':sum(q['action']=='BRANCH' for q in v)/len(v),
  'branch_prob_mean':float(np.mean([q['probabilities']['BRANCH'] for q in v])),
  'action_counts':dict(collections.Counter(q['action'] for q in v))} for k,v in sorted(by.items())}

def forward(policy,g,obs,bonus=0.):
 sc=policy.scores(feats(obs)).copy()
 if 'BRANCH' in obs['legal']:sc[3]+=bonus
 sc[~np.asarray([a in obs['legal'] for a in ACTIONS])]=-1e5
 return ACTIONS[int(np.argmax(sc))]

def step_horizon(policy,game,case,first,steps=HORIZON):
 g=copy.deepcopy(game)
 for h in range(steps):
  if h>0:adverse(g,case)
  o=g.obs();a=first if h==0 else policy.predict(o)
  if a not in o['legal']:raise AssertionError('counterfactual used illegal action')
  g.step(a)
  if g.stopped:break
 return g

def advantage(policy,g,case,base_action):
 if base_action=='BRANCH' or 'BRANCH' not in g.legal():raise ValueError('not a branch alternative')
 b=step_horizon(policy,g,case,base_action)
 a=step_horizon(policy,g,case,'BRANCH')
 delta=(len(a.painted)-len(b.painted)) + .20*(a.branch_count-b.branch_count) - .18*(a.retraces-b.retraces) - .25*(a.revisits-b.revisits)
 if a.stopped and not b.stopped:delta-=1.5
 return float(delta)

def corpus(policy,source,output):
 X=[];Y=[];W=[];receipts=[];counts=collections.Counter()
 for seed in TRAIN:
  for style in STYLES:
   for case in CASES:
    g=Game(seed,style)
    for tick in range(MAX_STEPS):
     adverse(g,case);o=g.obs();a=policy.predict(o)
     qualifies=style in ('branchwitch','greyhound') and 'BRANCH' in o['legal'] and a!='BRANCH'
     sample=qualifies and ((tick+seed)%3==0)
     if sample:
      d=advantage(policy,g,case,a);x=feats(o);good=int(d>0.35)
      X.append(x);Y.append(good);W.append(1.0);counts[(style,case,'candidate')]+=1
      counts[(style,case,'positive' if good else 'negative')]+=1
      receipts.append({'seed':seed,'style':style,'case':case,'tick':tick,'base_action':a,
        'label':good,'counterfactual_advantage':d,'state_feature_sha256':hashlib.sha256(x.tobytes()).hexdigest()})
     g.step(a)
     if g.stopped:break
 if len(Y)<100 or min(Y)==max(Y):raise ValueError('counterfactual label collapse')
 data={'x':np.asarray(X,np.float32),'y':np.asarray(Y,np.float32),'weight':np.asarray(W,np.float32)}
 np.savez_compressed(output,**data)
 return data,receipts,{'n':len(Y),'positive':int(sum(Y)),'negative':int(len(Y)-sum(Y)),
   'by_style_case':{f'{s}/{c}/{t}':v for (s,c,t),v in sorted(counts.items())},'sha256':sha(output)}

class Gate:
 def __init__(self,hidden=8,seed=17017):
  rng=np.random.default_rng(seed)
  self.w=(rng.standard_normal((100,hidden))*.06).astype('float32')
  self.b=np.zeros(hidden,dtype='float32')
  self.v=(rng.standard_normal((hidden,))*.06).astype('float32')
  self.c=np.float32(0.)
 def predict(self,x):
  h=np.tanh(x@self.w+self.b);z=h@self.v+self.c
  return 1/(1+np.exp(-np.clip(z,-30,30)))
 def save(self,path):np.savez_compressed(path,w=self.w,b=self.b,v=self.v,c=np.asarray(self.c))
 @classmethod
 def read(cls,path):
  with np.load(path,allow_pickle=False) as z:
   obj=cls(hidden=len(z['b']))
   for k in ('w','b','v','c'):setattr(obj,k,np.asarray(z[k]).copy())
  return obj

def train(data,hidden=8,steps=200,seed=17017):
 if steps>320 or steps<1:raise ValueError('optimizer budget')
 gate=Gate(hidden,seed)
 pars=[gate.w,gate.b,gate.v,np.asarray(gate.c).reshape(())]
 m=[np.zeros_like(p) for p in pars];v=[np.zeros_like(p) for p in pars]
 X,Y=data['x'],data['y'];rng=np.random.default_rng(seed)
 pos=np.flatnonzero(Y>0);neg=np.flatnonzero(Y==0);losses=[]
 for t in range(1,steps+1):
  ix=np.r_[rng.choice(pos,96,replace=True),rng.choice(neg,96,replace=True)]
  rng.shuffle(ix);x=X[ix];y=Y[ix]
  h=np.tanh(x@pars[0]+pars[1]);z=h@pars[2]+pars[3]
  p=1/(1+np.exp(-np.clip(z,-30,30)))
  dz=(p-y)/len(ix)
  grads=[x.T@((dz[:,None]*pars[2][None,:])*(1-h*h)),
    ((dz[:,None]*pars[2][None,:])*(1-h*h)).sum(0),h.T@dz,dz.sum()]
  for k,(par,grad) in enumerate(zip(pars,grads)):
   m[k]=.9*m[k]+.1*grad;v[k]=.999*v[k]+.001*(grad*grad)
   par-=.008*(m[k]/(1-.9**t))/(np.sqrt(v[k]/(1-.999**t))+1e-8)
  losses.append(float(np.mean(-(y*np.log(p+1e-8)+(1-y)*np.log(1-p+1e-8)))))
 gate.c=pars[3].astype('float32')
 return gate,{'updates':steps,'parameters':sum(np.size(q) for q in pars),
  'loss_first20':float(np.mean(losses[:20])),'loss_last20':float(np.mean(losses[-20:]))}

class Controller:
 def __init__(self,base,source,gate=None,bonus=0.,threshold=.5,kind='gate'):
  self.base=base;self.source=source;self.gate=gate;self.bonus=bonus;self.threshold=threshold;self.kind=kind
 def bonus_for(self,o):
  style=o['style']
  if style not in ('branchwitch','greyhound') or 'BRANCH' not in o['legal']:return 0.
  strength=max(0.,self.source[style]['branch_top1']-self.source['stock']['branch_top1'])
  if self.kind=='constant':return strength*self.bonus
  if self.gate is None:return 0.
  p=float(self.gate.predict(feats(o)))
  return strength*self.bonus if p>=self.threshold else 0.
 def predict(self,o):return forward(self.base,None,o,self.bonus_for(o))

def evaluate(controllers,seeds,draw=None,log_actions=False):
 rolls=[];receipts=[]
 if draw:draw.mkdir(parents=True,exist_ok=True)
 selected={seeds.start,seeds.start+4,seeds.start+8,seeds.start+12}
 for seed in seeds:
  for style in STYLES:
   for case in CASES:
    for name,model in controllers.items():
     g=Game(seed,style)
     for t in range(MAX_STEPS):
      adverse(g,case);o=g.obs();action=model.predict(o)
      if action not in o['legal']:raise AssertionError('host illegal action')
      if log_actions and seed in selected and style=='branchwitch':receipts.append({'seed':seed,'style':style,'case':case,'model':name,'tick':t,
        'legal':o['legal'],'action':action,'pos':[o['x'],o['y']], 'bonus':round(model.bonus_for(o),6)})
      g.step(action)
      if g.stopped:break
     rolls.append({'seed':seed,'style':style,'case':case,'model':name,'cells':len(g.painted),
       'branches':g.branch_count,'retraces':g.retraces,'revisits':g.revisits,'decisions':g.tick,'illegal':g.invalid,'stopped':int(g.stopped)})
     if draw and seed in selected and style=='branchwitch':g.png().save(draw/f'{case}_{seed}_{style}_{name}.png')
 return rolls,receipts

def summary(rolls):
 result={}
 for case in CASES:
  for model in sorted(set(r['model'] for r in rolls)):
   x=[r for r in rolls if r['case']==case and r['model']==model]
   result[f'{case}/{model}']={k:round(float(np.mean([r[k] for r in x])),4) for k in ('cells','branches','retraces','revisits','decisions','stopped','illegal')}
 return result

def rank(s,name):
 d={k:s[f'{k}/{name}'] for k in CASES};b={k:s[f'{k}/d007'] for k in CASES}
 cov={k:d[k]['cells']-b[k]['cells'] for k in CASES}
 bonus=sum((.22 if k!='squeeze' else .10)*(d[k]['branches']-b[k]['branches']) for k in CASES)
 score=sum(w*cov[k] for k,w in [('normal',.35),('trap',.4),('squeeze',.25)])+bonus
 keep=all(cov[k]>=-2 for k in ('normal','trap')) and cov['squeeze']>=-1.0
 return {'objective':round(float(score),5),'coverage_delta':cov,'nonregression_guard':keep}

def gallery(draw,out,controllers,seeds):
 chosen=(seeds.start,seeds.start+4,seeds.start+8,seeds.start+12)
 for case in CASES:
  im=Image.new('RGB',(192*len(controllers),216*4),'#e7e8eb');d=ImageDraw.Draw(im)
  for j,seed in enumerate(chosen):
   for i,name in enumerate(controllers):
    im.paste(Image.open(draw/f'{case}_{seed}_branchwitch_{name}.png'),(i*192,j*216+24))
    d.text((i*192+4,j*216+4),f'{seed} {name}',fill='#161a22')
  im.save(out/f'd017_{case}_gallery.png')

def paired_ci(rolls,model,seed_start,seed_stop):
 rng=np.random.default_rng(17171);result={}
 for case in CASES:
  result[case]={}
  for key in ('cells','branches','revisits'):
   vals=[]
   for seed in range(seed_start,seed_stop):
    group=[]
    for st in STYLES:
     ref=next(r[key] for r in rolls if r['seed']==seed and r['style']==st and r['case']==case and r['model']=='d007')
     val=next(r[key] for r in rolls if r['seed']==seed and r['style']==st and r['case']==case and r['model']==model)
     group.append(val-ref)
    vals.append(float(np.mean(group)))
   a=np.asarray(vals);boots=a[rng.integers(len(a),size=(2000,len(a)))].mean(1)
   result[case][key]={'delta':round(float(a.mean()),4),'seed_block_95ci':[round(float(q),4) for q in np.quantile(boots,[.025,.975])]}
 return result

def main(argv=None):
 ap=argparse.ArgumentParser();ap.add_argument('--corpus',required=True,type=Path)
 ap.add_argument('--d007',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
 ap.add_argument('--steps',type=int,default=200);a=ap.parse_args(argv)
 if any(set(x)&set(y) for i,x in enumerate((TRAIN,VALID,TEST,CONFIRM)) for y in (TRAIN,VALID,TEST,CONFIRM)[i+1:]):raise AssertionError('split')
 source=parents(a.corpus)
 if sha(a.d007)!=D007_SHA:raise ValueError('D007 must be immutable hosted checkpoint')
 policy=load(a.d007);out=a.out;out.mkdir(parents=True,exist_ok=True)
 data,receipts,corpusinfo=corpus(policy,source,out/'d017_counterfactual_corpus.npz')
 (out/'d017_counterfactual_receipts.jsonl').write_text(''.join(js(r)+'\n' for r in receipts))
 variants={};candidates=[]
 choices=(('gate8_lo',8,.5,.58),('gate8_hi',8,1.3,.58),('gate16',16,1.1,.67))
 for i,(name,width,amplitude,threshold) in enumerate(choices):
  gate,fit=train(data,width,a.steps,17017+i)
  path=out/f'{name}.npz';gate.save(path)
  variants[name]=Controller(policy,source,gate,amplitude,threshold)
  val,_=evaluate({'d007':Controller(policy,source),name:variants[name]},VALID)
  stat=summary(val);r=rank(stat,name)
  candidates.append({'name':name,'width':width,'max_branch_bonus':amplitude,'threshold':threshold,
    'training':fit,'sha256':sha(path),'validation':r,'validation_summary':stat})
  print(js({'candidate':name,'fit':fit,'validation':r}),flush=True)
 eligible=[x for x in candidates if x['validation']['nonregression_guard']]
 selected=max(eligible or candidates,key=lambda q:q['validation']['objective'])['name']
 model=variants[selected]
 controls={'d007':Controller(policy,source),'fixed_0p5':Controller(policy,source,bonus=.5,kind='constant'),selected:model}
 rolls,actions=evaluate(controls,TEST,out/'pngs',True)
 confirm,_=evaluate(controls,CONFIRM)
 gallery(out/'pngs',out,controls,TEST)
 (out/'d017_test_rollouts.jsonl').write_text(''.join(js(r)+'\n' for r in rolls))
 (out/'d017_confirm_rollouts.jsonl').write_text(''.join(js(r)+'\n' for r in confirm))
 (out/'d017_action_receipts.jsonl').write_text(''.join(js(r)+'\n' for r in actions))
 su=summary(rolls);conf=summary(confirm)
 metrics={'schema':SCHEMA,'genuine_parent_forward_count_in_this_run':0,'historical_original_q8_receipts':144,
  'original_parent_seed':570,'true_frankenbrain_on_synthetic_game':False,'synthetic_host_only':True,
  'original_corpus_sha256':sha(a.corpus),'source_checkpoint_sha256':sha(a.d007),'original_teacher_action_distribution':source,
  'train_seeds':[TRAIN.start,TRAIN.stop-1],'validation_seeds':[VALID.start,VALID.stop-1],
  'test_seeds':[TEST.start,TEST.stop-1],'confirm_seeds':[CONFIRM.start,CONFIRM.stop-1],
  'counterfactual_horizon':HORIZON,'advantage_threshold':.35,'teacher_only_privileged_labels':True,
  'counterfactual_corpus':corpusinfo,'optimizer_updates':sum(q['training']['updates'] for q in candidates),
  'candidates':candidates,'selected_on_validation':selected,'validation_promotable':bool(eligible),
  'test_summary':su,'confirm_summary':conf,'test_paired_bootstrap':paired_ci(rolls,selected,TEST.start,TEST.stop),
  'confirm_paired_bootstrap':paired_ci(confirm,selected,CONFIRM.start,CONFIRM.stop),
  'selected_checkpoint_sha256':sha(out/f'{selected}.npz'),
  'qualified_for_native_use':False,'merge_authorized':False}
 (out/'d017_metrics.json').write_text(js(metrics)+'\n')
 (out/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(out)}\n' for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.txt'))
 print(js({'selected':selected,'valid_nonregression':bool(eligible),'test':su,'confirm':conf,'source_anchors':source}),flush=True)
 return metrics
if __name__=='__main__':main()
