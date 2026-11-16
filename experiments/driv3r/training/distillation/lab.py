"""CPU-only DRIV3R game-policy distillation lab (synthetic W0-like host, no CyberSand writes).

Genuine archived d1 parent probability anchors plus explicitly synthesized
obstacle and adverse-state corrections. This never calls a parent on a synthetic
state, so results are NOT faithful closed-loop parent distillation.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,random
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ACTIONS=('STRAIGHT','LEFT','RIGHT','BRANCH','RETRACE','STOP')
STYLES=('stock','greyhound','branchwitch','aerohound_proxy')
DIRS=((1,0),(0,1),(-1,0),(0,-1))
DATASET_SCHEMA='cybersand.driv3r.distill.student-curriculum/1'
ROOT=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
def js(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def load_original(path=None):
 path=Path(path or os.getenv('DRIV3R_TEACHER_CORPUS',str(ROOT/'original_teachers.jsonl')))
 rows=[json.loads(s) for s in path.read_text().splitlines()]
 if len(rows)!=144 or len({(r['teacher'],r['family'],r['tick']) for r in rows})!=144:raise ValueError('Original teacher manifest invalid')
 out={}
 for r in rows:
  if r['schema']!='cybersand.driv3r.distill.original-teacher/1' or r['source_kind']!='original_model_inference':raise ValueError('Impersonation')
  if r['action'] not in r['observation']['legal'] or abs(sum(r['probabilities'].values())-1)>1e-4:raise ValueError('Invalid original teacher distribution')
  out[(r['teacher'],r['family'],r['tick'])]=r
 return out

class Game:
 def __init__(self,seed,style='branchwitch',size=24):
  self.seed=seed;self.style=style;self.size=size;rng=random.Random(seed)
  self.blocks=set();self.protected=set()
  for y in range(size):
   for x in range(size):
    if x in (0,size-1) or y in (0,size-1) or rng.random()<.075:self.blocks.add((x,y))
    elif x>size//3 and y in (size//3,2*size//3) and rng.random()<.52:self.blocks.add((x,y))
    elif x in (size//2,size//2+1) and 3<y<size-4 and rng.random()<.16:self.protected.add((x,y))
  self.pos=(2,size//2);self.blocks.discard(self.pos);self.protected.discard(self.pos)
  self.history=[self.pos];self.painted={self.pos};self.heading=0
  self.tick=0;self.stopped=False;self.branch_count=0;self.retraces=0;self.revisits=0;self.invalid=0;self.actions=[]
 def open(self,p):return 0<p[0]<self.size-1 and 0<p[1]<self.size-1 and p not in self.blocks and p not in self.protected
 def target(self,hd):return (self.pos[0]+DIRS[hd][0],self.pos[1]+DIRS[hd][1])
 def branch_target(self):
  dirs=[(self.heading+2)%4,(self.heading+1)%4,(self.heading+3)%4,self.heading]
  opts=[(self.target(d),d) for d in dirs if self.open(self.target(d)) and self.target(d) not in self.painted]
  return opts[0] if opts else None
 def legal(self):
  if self.stopped:return ['STOP']
  a=[]
  for i,dir in enumerate((self.heading,(self.heading-1)%4,(self.heading+1)%4)):
   if self.open(self.target(dir)):a.append(ACTIONS[i])
  if self.branch_target() is not None:a.append('BRANCH')
  if len(self.history)>1:a.append('RETRACE')
  a.append('STOP');return a
 def obs(self):
  x,y=self.pos;rows=[]
  for dy in range(-2,3):
   row=''
   for dx in range(-2,3):
    p=(x+dx,y+dy)
    row+='@' if p==self.pos else ('P' if p in self.protected else '#' if p in self.blocks or not (0<=p[0]<self.size and 0<=p[1]<self.size) else '*' if p in self.painted else '.')
   rows.append(row)
  return {'seed':self.seed,'style':self.style,'tick':self.tick,'x':x,'y':y,'heading':self.heading,
   'local_5x5':rows,'legal':self.legal(),'last_actions':self.actions[-6:],
   'visited_fraction':len(self.painted)/float(self.size*self.size),'last_revisit':self.revisits>0}
 def step(self,action):
  legal=self.legal();self.tick+=1;self.actions.append(action)
  if action not in legal:self.invalid+=1;return False
  if action=='STOP':self.stopped=True;return False
  if action=='RETRACE':
   self.history.pop();self.pos=self.history[-1];self.retraces+=1;return True
  if action=='BRANCH':
   t=self.branch_target();assert t is not None
   p,newheading=t;self.branch_count+=1
  else:
   d={'STRAIGHT':0,'LEFT':-1,'RIGHT':1}[action];newheading=(self.heading+d)%4;p=self.target(newheading)
  if p in self.painted:self.revisits+=1
  self.heading=newheading;self.pos=p;self.history.append(p);self.painted.add(p);return True
 def metrics(self):
  return {'cells':len(self.painted),'coverage':len(self.painted)/(self.size*self.size),'branches':self.branch_count,
  'revisits':self.revisits,'retraces':self.retraces,'raw_invalid':self.invalid,'decisions':self.tick,
  'stop':self.stopped,'repetition':1.-len(self.painted)/max(1,len(self.actions))}
 def png(self,scale=8):
  im=Image.new('RGB',(self.size,self.size),(20,23,29));p=im.load()
  for q in self.blocks:p[q]=(72,76,84)
  for q in self.protected:p[q]=(170,64,78)
  for q in self.painted:p[q]=(225,196,94)
  p[self.pos]=(93,213,234)
  return im.resize((self.size*scale,self.size*scale),Image.Resampling.NEAREST)

def feats(o):
 f=[]
 for row in o['local_5x5']:
  for char in row:f.extend([int(char=='#'),int(char=='P'),int(char=='*')])
 f.extend(float(a in o['legal']) for a in ACTIONS)
 f.extend(float(o['last_actions'].count(a))/6 for a in ACTIONS)
 f.extend([min(1,o['tick']/40),o['x']/24,o['y']/24,o['visited_fraction'],float(o['last_revisit'])])
 f.extend(float(o['heading']==h) for h in range(4))
 f.extend(float(o['style']==s) for s in STYLES)
 return np.asarray(f,dtype=np.float32)
def teacher(o,original):
 """Source parent probabilities are ORIGINAL only on archived source observations."""
 legal=o['legal'];style=o['style'];tick=o['tick']
 scores={'STRAIGHT':.7,'LEFT':.4,'RIGHT':.4,'BRANCH':.4,'RETRACE':-.5,'STOP':-5.}
 if style=='stock':scores['LEFT']+=1.5;scores['RIGHT']+=.6
 elif style=='greyhound':scores['BRANCH']+=2.4
 elif style=='branchwitch':scores['BRANCH']+=1.9;scores['LEFT']+=.8
 else:
  scores['LEFT']+=(2.2 if tick%4<2 else -.2)
  scores['RIGHT']+=(2.2 if tick%4>=2 else -.2)
  scores['STRAIGHT']+=.35
 for a in ('STRAIGHT','LEFT','RIGHT'):
  if a in legal:
   idx={'STRAIGHT':0,'LEFT':-1,'RIGHT':1}[a]
   direction=(o['heading']+idx)%4;dx,dy=DIRS[direction]
   if o['local_5x5'][2+dy][2+dx]=='*':scores[a]-=.85
 if set(legal)<=set(('RETRACE','STOP')):scores['RETRACE']=4.;scores['STOP']=-4
 if style!='aerohound_proxy':
  row=original[(style,('greebles','pipework','architecture')[o['seed']%3],tick%16)]
  for a in ACTIONS:scores[a]+=.50*math.log(max(.0001,row['probabilities'][a]))
 vals=np.asarray([scores[a] if a in legal else -1e5 for a in ACTIONS],dtype=np.float32)
 vals-=np.max(vals);p=np.exp(vals);return (p/sum(p)).astype('float32')
def curriculum(original,seed_begin,seed_end,turns=26):
 examples=[]
 for seed in range(seed_begin,seed_end):
  for style in STYLES:
   g=Game(seed,style);rng=np.random.default_rng(seed*31+STYLES.index(style))
   for _ in range(turns):
    o=g.obs();target=teacher(o,original);examples.append((feats(o),target,o))
    action=ACTIONS[int(rng.choice(len(ACTIONS),p=target))]
    if rng.random()<.08:action=ACTIONS[int(rng.choice([i for i,a in enumerate(ACTIONS) if a in o['legal']]))]
    g.step(action)
    if g.stopped:break
 return examples
class Student:
 def __init__(self,nf=100,hidden=48,seed=19):
  g=np.random.default_rng(seed);self.w1=(g.standard_normal((nf,hidden))*.08).astype('float32')
  self.b1=np.zeros(hidden,dtype='float32');self.w2=(g.standard_normal((hidden,6))*.08).astype('float32')
  self.b2=np.zeros(6,dtype='float32')
 def scores(self,x):return np.tanh(x@self.w1+self.b1)@self.w2+self.b2
 def predict(self,o,mask=True):
  s=self.scores(feats(o))
  if mask:s=np.asarray([v if a in o['legal'] else -1e5 for a,v in zip(ACTIONS,s)])
  return ACTIONS[int(s.argmax())]
 def train(self,x,y,iterations=450,batch=128,lr=.018,seed=71):
  rng=np.random.default_rng(seed)
  m=[np.zeros_like(p) for p in (self.w1,self.b1,self.w2,self.b2)];v=[p.copy() for p in m]
  losses=[];N=len(x)
  for step in range(1,iterations+1):
   idx=rng.integers(N,size=batch);xx=x[idx];yy=y[idx]
   h=np.tanh(xx@self.w1+self.b1);z=h@self.w2+self.b2;z-=z.max(axis=1,keepdims=True)
   pred=np.exp(z);pred/=pred.sum(axis=1,keepdims=True)
   losses.append(float(-(yy*np.log(np.maximum(pred,1e-7))).sum(axis=1).mean()))
   d=(pred-yy)/batch;gw2=h.T@d;gb2=d.sum(0);dh=(d@self.w2.T)*(1-h*h)
   gw1=xx.T@dh;gb1=dh.sum(0)
   for k,(p,g) in enumerate(zip((self.w1,self.b1,self.w2,self.b2),(gw1,gb1,gw2,gb2))):
    m[k]=.9*m[k]+.1*g;v[k]=.999*v[k]+.001*(g*g)
    p-=lr*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
  return {'steps':iterations,'initial_loss':float(np.mean(losses[:10])),'final_loss':float(np.mean(losses[-20:])),
   'parameters':sum(p.size for p in (self.w1,self.b1,self.w2,self.b2))}
 def save(self,path):np.savez_compressed(path,w1=self.w1,b1=self.b1,w2=self.w2,b2=self.b2)
def rollout(seed,style,which,student=None,steps=40):
 g=Game(seed,style);rng=np.random.default_rng(seed*1009+STYLES.index(style))
 for i in range(steps):
  o=g.obs();legal=o['legal']
  if which=='student':act=student.predict(o)
  elif which=='flat':act=legal[int(rng.integers(len(legal)))]
  elif which=='flat_continue':
   offered=[a for a in legal if a!='STOP'] or ['STOP'];act=offered[int(rng.integers(len(offered)))]
  elif which=='wall':act=next((a for a in ('LEFT','STRAIGHT','RIGHT','BRANCH','RETRACE','STOP') if a in legal),'STOP')
  else:
   preference=['LEFT','RIGHT','STRAIGHT','RIGHT','LEFT','BRANCH'][i%6]
   act=preference if preference in legal else next((a for a in ('RETRACE','STRAIGHT','RIGHT','LEFT','BRANCH','STOP') if a in legal),'STOP')
  g.step(act)
  if g.stopped:break
 return g
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=450);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=True)
 original=load_original();training=curriculum(original,1,65);val=curriculum(original,1001,1017)
 x=np.stack([z[0] for z in training]);y=np.stack([z[1] for z in training])
 xv=np.stack([z[0] for z in val]);yv=np.stack([z[1] for z in val])
 model=Student(x.shape[1]);fresh=Student(x.shape[1]);metrics=model.train(x,y,iterations=a.steps)
 metrics.update({'train_samples':len(training),'validation_samples':len(val),'train_seed_range':[1,64],
 'validation_seed_range':[1001,1016],'heldout_test_seeds':[2001,2016],
 'original_parent_calls':len(original),'parent_seed':570,'synthetic_teacher_calls':len(training)+len(val),
 'original_parent_eval_heldout':False,'student_kind':'numpy-f32-mlp','feature_count':x.shape[1],
 'teacher_corpus_sha256':sha(Path(os.getenv('DRIV3R_TEACHER_CORPUS',str(ROOT/'original_teachers.jsonl'))).read_bytes()),
 'dataset_schema':DATASET_SCHEMA,'synthetic_states_are_real_parent_forwards':False,'native_cybersand_writes':0})
 z=model.scores(xv);z-=z.max(axis=1,keepdims=True);pr=np.exp(z);pr/=pr.sum(axis=1,keepdims=True)
 z0=fresh.scores(xv);z0-=z0.max(axis=1,keepdims=True);pr0=np.exp(z0);pr0/=pr0.sum(axis=1,keepdims=True)
 metrics['untrained_val_soft_ce']=float(-(yv*np.log(np.maximum(pr0,1e-7))).sum(1).mean())
 metrics['untrained_val_teacher_top1']=float(np.mean(pr0.argmax(1)==yv.argmax(1)))
 metrics['val_soft_ce']=float(-(yv*np.log(np.maximum(pr,1e-7))).sum(1).mean())
 metrics['val_teacher_top1']=float(np.mean(pr.argmax(1)==yv.argmax(1)))
 metrics['val_raw_illegal_rate']=float(np.mean([ACTIONS[int(i)] not in o['legal'] for i,(_,_,o) in zip(pr.argmax(1),val)]))
 allrows=[]
 for seed in range(2001,2017):
  for style in STYLES:
   for policy in ('student','untrained','flat','flat_continue','wall','fidget'):
    g=rollout(seed,style,'student' if policy=='untrained' else policy,fresh if policy=='untrained' else model)
    allrows.append({'seed':seed,'style':style,'controller':policy,**g.metrics()})
 grouped={}
 for pcy in ('student','untrained','flat','flat_continue','wall','fidget'):
  rows=[r for r in allrows if r['controller']==pcy]
  grouped[pcy]={k:float(np.mean([r[k] for r in rows])) for k in ('cells','branches','revisits','retraces','raw_invalid','decisions','repetition')}
 metrics['heldout_rollouts']=len(allrows);metrics['rollout_summary']=grouped
 model.save(a.out/'student.npz');metrics['checkpoint_sha256']=sha((a.out/'student.npz').read_bytes())
 (a.out/'metrics.json').write_text(js(metrics)+'\n')
 (a.out/'rollouts.jsonl').write_text(''.join(js(r)+'\n' for r in allrows))
 seeds=[2001,2005,2009,2013];canvas=Image.new('RGB',(4*192,4*220),(240,241,243));d=ImageDraw.Draw(canvas)
 for row,(seed,style) in enumerate(zip(seeds,STYLES)):
  for col,policy in enumerate(('student','flat','wall','fidget')):
   im=rollout(seed,style,policy,model).png();canvas.paste(im,(col*192,row*220+20))
   d.text((col*192+4,row*220+3),f'{style} / {policy}',fill=(10,10,10))
 canvas.save(a.out/'synthetic_student_controls.png')
 (a.out/'evidence.json').write_text(js({'source_corpus_sha256':metrics['teacher_corpus_sha256'],
  'checkpoint_sha256':metrics['checkpoint_sha256'],
  'gallery_sha256':sha((a.out/'synthetic_student_controls.png').read_bytes()),
  'metrics_sha256':sha((a.out/'metrics.json').read_bytes()),
  'source_code_sha256':sha(Path(__file__).read_bytes()),
  'policy':'research_only_synthetic_external_host',
  'real_original_teacher_decisions':144,'new_original_parent_inferences':0,'optimizer_updates':a.steps})+'\n')
 print(js({'train':metrics['train_samples'],'steps':metrics['steps'],'val_top1':metrics['val_teacher_top1'],
  'val_ce':metrics['val_soft_ce'],'rollouts':metrics['rollout_summary'],'checkpoint_sha':metrics['checkpoint_sha256']}))
if __name__=='__main__':main()
