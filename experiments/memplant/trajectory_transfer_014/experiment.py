"""DG014: trained neural donor fingerprints for trajectory-level game planning.
Research-only synthetic grid; never imports or modifies CyberSand.
"""
import argparse, json, hashlib, time, random, heapq, itertools
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SIDE=9; A=6; FEATURES=11; ACTIONS=('LEFT','RIGHT','UP','DOWN','PAINT','RETRACE')
DIRS=((-1,0),(1,0),(0,-1),(0,1)); HELD=(1,1,1)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def neighbors(world,p):
  x,y=p
  for a,(dx,dy) in enumerate(DIRS):
    nx,ny=x+dx,y+dy
    if 0<=nx<SIDE and 0<=ny<SIDE and not world['wall'][ny,nx]: yield a,(nx,ny)

def distances(world,goal,wet_cost=0.,visits=None):
  # Backward Dijkstra; stepping into a cell pays that cell's cost.
  dist={goal:0.}; queue=[(0.,goal)]
  while queue:
    val,p=heapq.heappop(queue)
    if val>dist[p]+1e-8:continue
    for _,q in neighbors(world,p):
      cost=1+wet_cost*float(world['wet'][p[1],p[0]])
      if visits:cost+=visits.get(p,0)*.55
      nv=val+cost
      if nv+1e-8<dist.get(q,1e8):dist[q]=nv;heapq.heappush(queue,(nv,q))
  return dist

def make_world(seed):
  r=np.random.default_rng(seed)
  for _ in range(30):
    w={'wall':r.random((SIDE,SIDE))<.17,'wet':r.random((SIDE,SIDE))<.43,'protected':r.random((SIDE,SIDE))<.2}
    start=(SIDE//2,SIDE//2);goal=(int(r.integers(SIDE)),int(r.integers(SIDE)))
    alt=(int(r.integers(SIDE)),int(r.integers(SIDE)))
    for q in (start,goal,alt):
      w['wall'][q[1],q[0]]=False;w['protected'][q[1],q[0]]=False
    w['start']=start;w['goal']=goal;w['alternate']=alt
    if goal!=start and alt!=goal and alt!=start and start in distances(w,goal) and start in distances(w,alt):return w
  raise RuntimeError('unable to generate reachable world')

class Game:
  def __init__(self,world,laws):
    self.world=world;self.laws=laws;self.pos=world['start'];self.history=[self.pos]
    self.visits={self.pos:1};self.painted=set();self.done=False;self.steps=0;self.illegal=0;self.wet_steps=0;self.travel_cost=0.
  @property
  def goal(self):return self.world['alternate'] if self.laws[1]>0 else self.world['goal']
  def legal(self):
    mask=np.zeros(A,dtype=bool)
    for a,_ in neighbors(self.world,self.pos):mask[a]=True
    mask[4]=not self.world['protected'][self.pos[1],self.pos[0]] and self.pos not in self.painted
    mask[5]=len(self.history)>1
    return mask
  def obs(self):
    w=self.world;g=w['goal'];h=w['alternate'];d1=distances(w,g);d2=distances(w,h)
    x=np.zeros((A,FEATURES),np.float32)
    for a,q in neighbors(w,self.pos):
      x[a,0]=min(1.,d1.get(q,99)/16)
      x[a,1]=float(w['wet'][q[1],q[0]])
      x[a,2]=float(w['protected'][q[1],q[0]])
      x[a,3]=min(1.,self.visits.get(q,0)/3)
      x[a,4]=float(q==g)
      x[a,5]=float(q==h)
      x[a,6]=float(self.pos==w['start'])
      x[a,7]=max(-1.,min(1.,(abs(self.pos[0]-g[0])+abs(self.pos[1]-g[1])-abs(q[0]-g[0])-abs(q[1]-g[1]))/2))
      x[a,8]=min(1.,d2.get(q,99)/16)
      x[a,9]=float(w['start'] in self.painted)
      x[a,10]=1.
    x[4,4]=float(self.pos==g);x[4,5]=float(self.pos==h)
    x[4,6]=float(self.pos==w['start']);x[4,9]=float(w['start'] in self.painted);x[4,10]=1.
    x[5,4]=float(self.pos==g);x[5,5]=float(self.pos==h);x[5,6]=float(self.pos==w['start']);x[5,9]=float(w['start'] in self.painted);x[5,10]=1.
    return x,self.legal()
  def oracle(self):
    if self.done:return None
    w=self.world;wet,branch,paint=self.laws
    if paint<0 and self.pos==w['start'] and w['start'] not in self.painted:return 4
    if self.pos==self.goal and paint>0:return 4
    if self.pos==self.goal and paint<0:return None
    wet_cost=6. if wet>0 else .05
    cost=distances(w,self.goal,wet_cost)
    options=[]
    for a,q in neighbors(w,self.pos):
      qcost=1+wet_cost*float(w['wet'][q[1],q[0]])
      options.append((qcost+cost.get(q,1e6),a))
    if not options:return 5 if self.legal()[5] else None
    return min(options)[1]
  def step(self,action):
    if action is None:return
    m=self.legal()
    if not m[action]:self.illegal+=1;raise ValueError('illegal game action')
    if action<4:
      dx,dy=DIRS[action];self.pos=(self.pos[0]+dx,self.pos[1]+dy)
      self.history.append(self.pos);self.visits[self.pos]=self.visits.get(self.pos,0)+1
      self.wet_steps+=int(self.world['wet'][self.pos[1],self.pos[0]])
      self.travel_cost+=1+(6. if self.laws[0]>0 else .05)*int(self.world['wet'][self.pos[1],self.pos[0]])
    elif action==4:self.painted.add(self.pos)
    elif len(self.history)>1:self.history.pop();self.pos=self.history[-1]
    self.steps+=1
    self.done=self.pos==self.goal and (self.goal in self.painted if self.laws[2]>0 else self.world['start'] in self.painted)

class Donor(nn.Module):
  def __init__(self,gated=False):
    super().__init__();self.f=nn.Sequential(nn.Linear(4,24 if gated else 12),nn.Tanh() if gated else nn.ReLU(),nn.Linear(24 if gated else 12,16),nn.Tanh(),nn.Linear(16,1))
  def forward(self,x):return self.f(x).squeeze(-1)

def train_donor(seed,family,sign,gated,steps=140):
  rng=np.random.default_rng(seed);torch.manual_seed(seed)
  m=Donor(gated);perm=rng.permutation(4);flip=rng.choice([-1.,1.],4).astype('float32')
  opt=torch.optim.Adam(m.parameters(),lr=.009)
  for _ in range(steps):
    x=rng.uniform(-1,1,(96,4)).astype('float32')
    target=sign*(1.7*x[:,family] + .2*x[:,(family+1)%4]**2)
    z=torch.from_numpy(x[:,perm]*flip)
    loss=F.mse_loss(m(z),torch.tensor(target))
    opt.zero_grad();loss.backward();opt.step()
  with torch.no_grad():
    x=rng.uniform(-1,1,(256,4)).astype('float32')
    error=float(F.mse_loss(m(torch.from_numpy(x[:,perm]*flip)),torch.tensor(sign*(1.7*x[:,family]+.2*x[:,(family+1)%4]**2))))
  m.eval()
  return dict(model=m,perm=perm,flip=flip,error=error)

def fingerprint(donor):
  # Canonical calibration inputs known to receiver, private maps never exposed.
  r=np.random.default_rng(883721);x=r.uniform(-1,1,(8,4)).astype('float32')
  with torch.no_grad():y=donor['model'](torch.from_numpy(x[:,donor['perm']]*donor['flip'])).numpy()
  return y.astype('float32')

def signatures(bank,kind,laws):
  return np.concatenate([fingerprint(bank[(kind,j,laws[j])]) for j in range(3)]).astype('float32')

class Policy(nn.Module):
  def __init__(self,with_donor):
    super().__init__();self.with_donor=with_donor
    self.net=nn.Sequential(nn.Linear(A*FEATURES+(24 if with_donor else 0),128),nn.ReLU(),nn.Linear(128,96),nn.ReLU(),nn.Linear(96,A))
  def forward(self,x,legal,sig=None):
    flat=x.flatten(1)
    if self.with_donor:flat=torch.cat((flat,sig),dim=1)
    return self.net(flat).masked_fill(~legal,-1e5)

def sample_data(seed,bank,train_worlds=600):
  # Replan as the oracle acts, with matched law interventions in each source world.
  xs=[];ls=[];ss=[];ys=[];keylist=[]
  laws=[a for a in itertools.product((-1,1),repeat=3) if a!=HELD]
  for i in range(train_worlds):
    w=make_world(seed+37*i)
    for law in laws:
      game=Game(w,law);sig=signatures(bank,'narrow',law)
      for t in range(36):
        obs,legal=game.obs();a=game.oracle()
        if a is None or game.done:break
        xs.append(obs);ls.append(legal);ss.append(sig);ys.append(a);keylist.append(i)
        game.step(a)
  return torch.tensor(np.stack(xs)),torch.tensor(np.stack(ls)),torch.tensor(np.stack(ss)),torch.tensor(ys,dtype=torch.long),len(set(keylist))

def fit_policy(seed,data,with_donor,steps=420):
  torch.manual_seed(seed);model=Policy(with_donor);opt=torch.optim.AdamW(model.parameters(),lr=.002)
  x,l,s,y,_=data;rng=np.random.default_rng(seed+100)
  for i in range(steps):
    ix=torch.tensor(rng.integers(len(y),size=128))
    logits=model(x[ix],l[ix],s[ix] if with_donor else None)
    loss=F.cross_entropy(logits,y[ix]);opt.zero_grad();loss.backward();opt.step()
  model.eval();return model

def rollouts(seed,bank,models,worlds=80,horizon=64):
  modes=['inherited','blind','wrong_donor','wrong_wet','oracle']
  results={k:[] for k in modes};traces=[]
  for i in range(worlds):
    w=make_world(seed+717*i)
    law=HELD if i%2==0 else tuple((-1,1)[int(q)] for q in np.random.default_rng(seed+314+i).integers(0,2,3))
    cf=(law[0],-law[1],law[2])
    for mode in modes:
      game=Game(w,law);history=[];visited=[]
      sig=signatures(bank,'gated',cf if mode=='wrong_donor' else (-law[0],law[1],law[2]) if mode=='wrong_wet' else law)
      for t in range(horizon):
        x,legal=game.obs()
        if mode=='oracle':a=game.oracle()
        else:
          with torch.no_grad():
            logits=models['inherited' if mode!='blind' else 'blind'](torch.tensor(x[None]),torch.tensor(legal[None]),torch.tensor(sig[None]) if mode!='blind' else None)
            a=int(logits.argmax(-1)[0])
        if a is None:break
        game.step(a)
        visited.append(game.pos)
        if i<3:history.append([t,ACTIONS[a],*game.pos])
        if game.done:break
      result=dict(success=int(game.done),actions=game.steps,visited_unique=len(set(visited)),repeated=max(0,len(visited)-len(set(visited))),painted=len(game.painted),wet_steps=game.wet_steps,travel_cost=round(game.travel_cost,4),illegal=game.illegal)
      results[mode].append(result)
      if i<3:traces.append(dict(world_seed=seed+717*i,laws=law,mode=mode,path=history,metrics=result))
  summary={m:{k:round(float(np.mean([v[k] for v in vals])),4) for k in vals[0]} for m,vals in results.items()}
  for mode,vals in results.items():
    summary[mode]['heldout_success']=round(float(np.mean([v['success'] for j,v in enumerate(vals) if j%2==0])),4)
    summary[mode]['mixed_success']=round(float(np.mean([v['success'] for j,v in enumerate(vals) if j%2==1])),4)
  summary={m:dict(**v,worlds=worlds) for m,v in summary.items()}
  return summary,traces

def gallery_svg(traces,path):
  # Fully original vector evidence of the actual trained-policy trajectories.
  colors={'inherited':'#2866c6','blind':'#cf7b23','wrong_donor':'#be3d70','wrong_wet':'#47a99d','oracle':'#252529'}
  svg=['<svg xmlns="http://www.w3.org/2000/svg" width="970" height="660" viewBox="0 0 970 660">',
       '<rect width="970" height="660" fill="#f7f8fa"/>',
       '<text x="22" y="26" font-size="18" font-family="sans-serif">DG014 · held-out policy trajectories / neural donor grafts</text>']
  for i,t in enumerate(traces):
    row=i//5;col=i%5;ox=20+col*190;oy=42+row*204;w=make_world(t['world_seed'])
    svg.append(f'<text x="{ox}" y="{oy+13}" font-size="11" font-family="sans-serif">{t["mode"]} · {t["laws"]}</text>')
    for y in range(SIDE):
      for x in range(SIDE):
        fill='#303849' if w['wall'][y,x] else '#b1dfe4' if w['wet'][y,x] else '#e6e9ec' if w['protected'][y,x] else '#ffffff'
        svg.append(f'<rect x="{ox+17*x}" y="{oy+20+17*y}" width="16" height="16" fill="{fill}" stroke="#d9dce0" stroke-width=".6"/>')
    for q,c in [(w['goal'],'#53af38'),(w['alternate'],'#a96cc9')]:
      svg.append(f'<circle cx="{ox+17*q[0]+8}" cy="{oy+20+17*q[1]+8}" r="5" fill="{c}"/>')
    pts=[w['start']]+[(a[2],a[3]) for a in t['path']]
    poly=' '.join(f'{ox+17*x+8},{oy+20+17*y+8}' for x,y in pts)
    svg.append(f'<polyline points="{poly}" fill="none" stroke="{colors[t["mode"]]}" stroke-width="2" stroke-opacity=".85"/>')
    svg.append(f'<text x="{ox}" y="{oy+185}" font-size="10" font-family="sans-serif">goal {t["metrics"]["success"]} · actions {t["metrics"]["actions"]} · wet {t["metrics"]["wet_steps"]}</text>')
  svg.append('</svg>')
  path.write_text('\n'.join(svg)+'\n')

def selftest():
  w=make_world(10);g=Game(w,(1,1,1));
  for _ in range(40):
    x,l=g.obs();assert x.shape==(A,FEATURES) and l.shape==(A,)
    a=g.oracle()
    if a is None:break
    assert l[a];g.step(a)
    if g.done:break
  assert g.done
  print('DG014 SELFTEST: host legal movement, goal paint and oracle PASS')

def run(seed,out,train_worlds=600,steps=420,test_worlds=80,horizon=64,donor_steps=140):
  t0=time.monotonic();torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
  out=Path(out);out.mkdir(parents=True,exist_ok=True)
  bank={};errors={}
  for kind in ('narrow','gated'):
    for f in range(3):
      for s in (-1,1):
        d=train_donor(seed+1000+100*f+10*(s+1)+(0 if kind=='narrow' else 500),f,s,kind=='gated',donor_steps)
        bank[kind,f,s]=d; errors[f'{kind}/{f}/{s}']=round(d['error'],6)
        torch.save(dict(weights=d['model'].state_dict(),private_permutation=d['perm'].tolist(),private_reflections=d['flip'].tolist(),validation_mse=d['error']),out/f'donor-{kind}-{f}-{s}.pt')
  data=sample_data(seed+2000,bank,train_worlds)
  models={'inherited':fit_policy(seed+3000,data,True,steps),'blind':fit_policy(seed+4000,data,False,steps)}
  for k,v in models.items():torch.save(v.state_dict(),out/f'policy-{k}.pt')
  summary,traces=rollouts(seed+100000,bank,models,test_worlds,horizon)
  train_seeds={seed+2000+37*i for i in range(train_worlds)}
  sealed_seeds={seed+100000+717*i for i in range(test_worlds)}
  assert not train_seeds & sealed_seeds, 'train/sealed world overlap'
  (out/'trajectories.json').write_text(json.dumps(traces,indent=2)+'\n')
  gallery_svg(traces,out/'gallery.svg')
  report={'experiment':'DG014','seed':seed,'source_sha256':sha(__file__), 'pytorch':torch.__version__,
          'donor_training_steps':donor_steps,'policy_training_steps':steps,'train_worlds':train_worlds,
          'training_states':len(data[3]),'training_unique_worlds':data[4],
          'sealed_worlds':test_worlds,'horizon':horizon,'donor_mse':errors,'trajectories':summary,
          'train_test_world_seed_overlap':0,'holdout':'+++ entirely withheld from trajectory-policy training; gated donor architecture never used in receiver fitting',
          'limitations':'External synthetic grid; supervised expert Dijkstra; shortest-path potential given as feature; raw donor output calibration, not KV transfer; oracle is scripted and donors learn authored laws; wrong-donor swaps branch goal, wrong-wet swaps material law.',
          'duration_seconds':round(time.monotonic()-t0,2)}
  (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
  paths={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='sha256.json'}
  (out/'sha256.json').write_text(json.dumps(paths,indent=2)+'\n')
  print(json.dumps(report,indent=2))

if __name__=='__main__':
  a=argparse.ArgumentParser();a.add_argument('--selftest',action='store_true');a.add_argument('--seed',type=int,default=26011624);a.add_argument('--out',default='/tmp/dg014');a.add_argument('--train-worlds',type=int,default=600);a.add_argument('--steps',type=int,default=420);a.add_argument('--test-worlds',type=int,default=80);a.add_argument('--horizon',type=int,default=64);a.add_argument('--donor-steps',type=int,default=140)
  x=a.parse_args()
  if x.selftest:selftest()
  else:run(x.seed,x.out,x.train_worlds,x.steps,x.test_worlds,x.horizon,x.donor_steps)
