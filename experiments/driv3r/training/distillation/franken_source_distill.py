"""D-STUDENT-014: graph-conditioned 10-real-Frankenbrain source-parent distillation.

RESEARCH ONLY. Parses pinned immutable 2026-11-15 #630 Q8 System One
original inference artifacts. Bounded genuine Adam CPU students and honest
unseen-teacher graph test; NO new parent forwards or native CyberSand execution.
Model/checkpoints/evidence live in Actions artifacts, not repository history.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image,ImageDraw
from .lab import ACTIONS,STYLES,Game,feats,Student
from .repair import CASES,adverse,load

# GitHub Actions run 43769010943 on exact source head 8759900d20805ce87aec5a3a023780e808f5ed4d.
# Original artifact digests, not synthetic teacher checkpoints.
PINS={
 'branchwitch_drop21':(14183378085,'3e82f4f0234a144672a70e832d44fa5653dc7b771f093c29b79e3fe91353829e'),
 'greyhound_restore7':(14183572553,'2ee7cf79939bd554b77335d32e10bed23188d1ac7179dbe522fcbdd3cb49d63b'),
 'branchwitch_no_repeat10':(14183612319,'5d8762a08a6a29b5d8d5f5edb9987df06bf63d8623b4afd260b9db409dfb3dba'),
 'branchwitch_restore12':(14183671417,'58dddde6572bcccf9c6260617a05ee46ca9518de9c4219c78d336857835603fb'),
 'branchwitch':(14183687090,'2ca50e798d2008b8574564d952d18cc73340a12205d902b1f00b9fd4efb983d9'),
 'greyhound_restore16':(14183852344,'4d46531124e55d96ecfcfc75360e2fc4569fc72bf6d5b6d8985e417b91ee0918'),
 'greyhound':(14184017182,'9476bd7e6c3f7c5652949fd79fec5a4aa554a7347b31b8d2a1748678cf70b99b'),
 'greyhound_restore12':(14184211222,'01da1ea69612dd303e05a3b4e65ea6894060f925d19d36e9648835d615f6d24c'),
 'branchwitch_restore8':(14184295982,'76b9c14fa724f504bf829e1038d1c05dccdc66dc7333df68243872186314b7bf'),
 'greyhound_restore5':(14184395599,'24487e879e91acc07b1682ae38aa0b48958223f7bf33d9800672a5cb836cf23b'),
}
FAMILIES=('greebles','pipework','architecture')
# Entire unseen graph IDs: explicit behavioural inversion challenges.
TRAIN=('stock','greyhound','greyhound_restore5','greyhound_restore16','branchwitch','branchwitch_no_repeat10','branchwitch_restore12')
VALID=('greyhound_restore12','branchwitch_drop21')
TEST=('greyhound_restore7','branchwitch_restore8')
ALL=('stock',)+tuple(PINS)
Q8='073313d92b53b0ac8193db6e906cb3c78a98183d244b15b7b8f5426115c6738c'
LLAMA='8642541b255bfb5dc39cd351a178be09013a55d6'
SCHEMA='cybersand.driv3r.distill.franken-source-graph/1'

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha(p):return sha_bytes(Path(p).read_bytes())
def jd(o):return json.dumps(o,sort_keys=True,separators=(',',':'),allow_nan=False)
def softmax(s):
 s=np.asarray(s,dtype=np.float64);s=s-s.max(axis=-1,keepdims=True)
 z=np.exp(np.clip(s,-80,30));return z/z.sum(axis=-1,keepdims=True)

def inspect_archive(z,name):
 if z.testzip() is not None:raise ValueError('bad source zip integrity')
 checksum=z.read('SHA256SUMS.txt').decode()
 seen=set()
 for line in checksum.splitlines():
  expected,filename=line.split(maxsplit=1);filename=filename.removeprefix('./')
  if filename in seen or filename=='SHA256SUMS.txt' or filename.startswith('/') or '..' in Path(filename).parts:raise ValueError('unsafe manifest')
  seen.add(filename)
  if sha_bytes(z.read(filename))!=expected:raise ValueError(f'bad internal hash {name}:{filename}')
 if set(z.namelist())!=seen|{'SHA256SUMS.txt'}:raise ValueError('manifest incompleteness')
 req=json.loads(z.read('request.json'))
 if req.get('q8_sha256')!=Q8 or req.get('llama_commit')!=LLAMA:raise ValueError('source model mismatch')
 if req.get('max_parallel')!=10 or req.get('paid_api')!=False:raise ValueError('source campaign mismatch')
 flows=req['flows']
 if name not in flows:raise ValueError('unknown graph')
 comparison=json.loads(z.read('inference/comparison.json'))
 if comparison.get('mode')!=name or not comparison.get('actual_systemone_inference') or comparison.get('llama_source_commit')!=LLAMA:raise ValueError('not genuine source inference')
 return flows

def graph_features(flow):
 # A 24-slot graph fingerprint + repeat count + source graph length
 bits=np.zeros(26,dtype=np.float32)
 for node in flow:
  if type(node)!=int or node not in range(24):raise ValueError('invalid graph layer')
  bits[node]=1
 bits[24]=(len(flow)-len(set(flow)))/4.
 bits[25]=len(flow)/24.
 return bits

def convert_choice(entry,variant,artifact_id,zipsha,graph):
 if entry['status']!='ok' or entry['http_status']!=200:raise ValueError('failed original source forward')
 o=entry['observation'];family=o['family'];act=entry['choice']
 if family not in FAMILIES or act not in o['legal']:raise ValueError('illegal original source action')
 if entry['state'].get('seed')!=570:raise ValueError('unexpected original seed')
 if entry['state'].get('tick')!=o['tick']:raise ValueError('source tick mismatch')
 ps=entry['parsed']['probabilities'];values=np.array([ps.get(a,0) for a in ACTIONS],dtype=float)
 mask=np.array([a in o['legal'] for a in ACTIONS])
 if any(values[~mask]>1e-8) or abs(values.sum()-1)>1e-6 or ACTIONS[int(values.argmax())]!=act:raise ValueError('malformed original probability receipt')
 return {'variant':variant,'family':family,'tick':o['tick'],'action':act,'observation':o,
  'probabilities':{a:float(values[i]) for i,a in enumerate(ACTIONS)},'graph':graph,
  'artifact_id':artifact_id,'artifact_zip_sha256':zipsha,'source_seed':570,'source_kind':'original_franken_inference',
  'producer_model_sha256':Q8,'producer_llama_commit':LLAMA,'schema':SCHEMA}

def extract(sourcedir):
 records=[];manifest=[];stock_saved=False
 for name,(artifact_id,pinned_sha) in PINS.items():
  path=sourcedir/f'source-parent-{artifact_id}.zip'
  if not path.is_file():raise FileNotFoundError(path)
  if sha(path)!=pinned_sha:raise ValueError(f'changed immutable source ZIP for {name}')
  with zipfile.ZipFile(path) as z:
   graphs=inspect_archive(z,name)
   graph=graphs[name]
   files=sorted(n for n in z.namelist() if n.startswith(f'inference/{name}/') and n.endswith('-choices.json'))
   if len(files)!=3:raise ValueError('missing source trajectories')
   ct=Counter()
   for fname in files:
    entries=json.loads(z.read(fname))
    for a in entries:
     row=convert_choice(a,name,artifact_id,pinned_sha,graph)
     records.append(row);ct[row['family']]+=1
   # Keep single explicitly pinned stock baseline; other archives' stock is
   # independently re-run same seed, not independent game-state coverage.
   if not stock_saved:
    for fam in FAMILIES:
     filename=f'inference/stock/stock-{fam}-choices.json'
     for a in json.loads(z.read(filename)):
      records.append(convert_choice(a,'stock',artifact_id,pinned_sha,list(range(24))))
    stock_saved=True
   n=sum(ct.values())
   expected=3 if name=='branchwitch_restore12' else 48
   if n!=expected:raise ValueError(f'{name}: expected {expected} real receipts, received {n}')
   manifest.append({'variant':name,'artifact_id':artifact_id,'zip_sha256':pinned_sha,'source_calls':n,
     'family_calls':dict(ct),'graph':graph})
 if len(records)!=483 or any(v not in ALL for v in (r['variant'] for r in records)):raise ValueError('source expected 483 records')
 return records,manifest

def encode(r):
 o=r['observation'];v=r['variant'];sty='stock' if v=='stock' else ('greyhound' if v.startswith('greyhound') else 'branchwitch')
 rows=o['local_5x5']
 oo={**o,'style':sty,'local_5x5':[row.replace('+','*') for row in rows],
  'visited_fraction':sum(row.count('+') for row in rows)/25.,'last_revisit':o['last_actions'][-1:]==['RETRACE']}
 # 100 local features + 3 original requested-family labels + 26 *public layer graph* fingerprints.
 # This family/graph information is NOT hidden model internal state or a new parent inference.
 xx=np.r_[feats(oo),np.eye(3,dtype=np.float32)[FAMILIES.index(r['family'])],graph_features(r['graph'])]
 if xx.shape!=(129,) or not np.isfinite(xx).all():raise ValueError('incorrect 129-feature source adapter')
 return xx.astype(np.float32)

def arrays(records):
 x=np.stack([encode(r) for r in records]);y=np.array([[r['probabilities'][a] for a in ACTIONS] for r in records],dtype=np.float32)
 legal=np.array([[a in r['observation']['legal'] for a in ACTIONS] for r in records],bool)
 return x,y,legal

def train(x,y,legal,width,steps,seed,hard_weight):
 if width not in (16,32,64) or not (1<=steps<=900) or hard_weight not in (0.,.18,.35):raise ValueError('bounded source trainer only')
 model=Student(nf=x.shape[1],hidden=width,seed=seed)
 params=[model.w1,model.b1,model.w2,model.b2]
 m=[np.zeros_like(p) for p in params];v=[np.zeros_like(p) for p in params]
 hard=np.eye(6,dtype=np.float32)[np.argmax(y,axis=1)]
 target=(1-hard_weight)*y+hard_weight*hard
 rng=np.random.default_rng(seed);losses=[]
 for tick in range(1,steps+1):
  idx=rng.integers(len(x),size=96);xx=x[idx];yy=target[idx];mask=legal[idx]
  h=np.tanh(xx@model.w1+model.b1);z=h@model.w2+model.b2
  p=softmax(np.where(mask,z,-1e5));dz=np.asarray((p-yy)/len(idx),dtype=np.float32)
  best=np.argmax(np.where(mask,z,-1e5),axis=1)
  margin=np.clip(z-z[np.arange(len(idx)),best,None]+.5,-30,30)
  sig=1/(1+np.exp(-margin));sig*=~mask
  dz+=.08*sig/len(idx);dz[np.arange(len(idx)),best]-=.08*sig.sum(axis=1)/len(idx)
  dh=(dz@model.w2.T)*(1-h*h)
  grads=[xx.T@dh,dh.sum(0),h.T@dz,dz.sum(0)]
  for j,(w,g) in enumerate(zip(params,grads)):
   m[j]=.9*m[j]+.1*g;v[j]=.999*v[j]+.001*(g*g)
   w-=.004*(m[j]/(1-.9**tick))/(np.sqrt(v[j]/(1-.999**tick))+1e-8)
  losses.append(float(-np.mean(np.sum(yy*np.log(p+1e-12),axis=1))))
 return model,{'updates':steps,'parameters':sum(p.size for p in params),
    'ce_first20':float(np.mean(losses[:20])),'ce_last20':float(np.mean(losses[-20:])),
    'hard_weight':hard_weight}

def family_prior(records,y,inds,groupby):
 # No graph-aware neural modeling; honest controlled source-conditioned priors.
 out={}
 for i in inds:
  key=tuple(records[i][k] for k in groupby)
  out.setdefault(key,[]).append(y[i])
 return {k:np.mean(arr,axis=0) for k,arr in out.items()}

def evaluate(model,records,x,y,legal,ids,prior=None,nearest_graph=None):
 out=[]
 for i in ids:
  if prior is not None:
   v=records[i]['variant']
   if nearest_graph is not None:v=nearest_graph.get(v,v)
   logits=np.log(np.clip(prior.get(v,np.ones(6)/6),1e-8,1))
  else:logits=model.scores(x[i])
  p=softmax(np.where(legal[i],logits,-1e5))
  out.append({'variant':records[i]['variant'],'family':records[i]['family'],'tick':int(records[i]['tick']),
   'action':records[i]['action'],'predicted':ACTIONS[int(p.argmax())],
   'ce':float(-np.sum(y[i]*np.log(p+1e-12))),
   'kl':float(np.sum(y[i]*(np.log(y[i]+1e-12)-np.log(p+1e-12)))),
   'action_agreement':int(p.argmax()==np.argmax(y[i])),
   'raw_illegal':int(np.argmax(logits) not in np.flatnonzero(legal[i]))})
 return out

def smry(rows):
 return {k:float(np.mean([r[k] for r in rows])) for k in ('ce','kl','action_agreement','raw_illegal')}

def nearest_graph_dict(records):
 variants={v:r['graph'] for r in records if (v:=r['variant'])}
 out={}
 for v in VALID+TEST:
  flow=variants[v]
  dst={vv:(sum(a!=b for a,b in zip(graph_features(flow),graph_features(variants[vv]))),vv)
       for vv in TRAIN}
  out[v]=min(dst,key=dst.get)
 return out

def plot_variants(out,rows,models):
 # Use real source-parent PNGs separately in the original immutable source artifacts.
 # These are external synthetic host rollouts (not native/GPU paints).
 gallery_seeds=(27001,27005,27009,27013)
 for case in CASES:
  im=Image.new('RGB',(192*len(models),216*len(gallery_seeds)),(240,241,243));d=ImageDraw.Draw(im)
  for j,seed in enumerate(gallery_seeds):
   for k,model in enumerate(models):
    pp=out/'pngs'/f'{case}_{seed}_{model}.png'
    im.paste(Image.open(pp),(192*k,216*j+24));d.text((192*k+2,216*j+4),f'{seed}/{model}',fill=(15,17,23))
  im.save(out/f'd014_{case}_gallery.png')

def synthetic_play(out,model):
 out.mkdir(parents=True,exist_ok=True);(out/'pngs').mkdir(exist_ok=True)
 # Demonstrate two graph-pinned personalities, including TEST-only source variants
 # (the final model has seen them AFTER the heldout source generalization gate).
 variants=('greyhound_restore7','branchwitch_restore8','branchwitch_no_repeat10')
 families=('greebles','architecture','pipework')
 rows=[];receipts=[]
 for seed in range(27001,27017):
  for vv,fa in zip(variants,families):
   flow=next(r['graph'] for r in model['records'] if r['variant']==vv)
   for case in CASES:
    sty='greyhound' if vv.startswith('greyhound') else 'branchwitch'
    g=Game(seed,sty)
    for step in range(80):
     adverse(g,case);obs=g.obs()
     dummy={'variant':vv,'family':fa,'graph':flow,'observation':obs}
     xx=encode(dummy)
     zz=model['student'].scores(xx)
     top=ACTIONS[int(np.argmax(zz))]
     act=ACTIONS[int(np.argmax(np.where([a in obs['legal'] for a in ACTIONS],zz,-1e5)))]
     if act not in obs['legal']:raise AssertionError('illegal host action')
     if seed in (27001,27005,27009,27013):
      receipts.append({'seed':seed,'variant':vv,'case':case,'tick':step,'raw':top,'action':act,'legal':obs['legal'],'x':obs['x'],'y':obs['y']})
     g.step(act)
     if g.stopped:break
    rows.append({'seed':seed,'variant':vv,'case':case,'cells':len(g.painted),'branches':g.branch_count,
      'revisits':g.revisits,'retraces':g.retraces,'decisions':g.tick,'stopped':int(g.stopped),'illegal':g.invalid})
    if seed in (27001,27005,27009,27013):
     g.png().save(out/'pngs'/f'{case}_{seed}_{vv}.png')
 plot_variants(out,rows,variants)
 (out/'synthetic_rollouts.jsonl').write_text(''.join(jd(r)+'\n' for r in rows))
 (out/'synthetic_action_receipts.jsonl').write_text(''.join(jd(r)+'\n' for r in receipts))
 return {f'{v}/{case}':{k:float(np.mean([r[k] for r in rows if r['variant']==v and r['case']==case])) for k in ('cells','branches','revisits','decisions','stopped','illegal')}
   for v in variants for case in CASES}

def main(argv=None):
 p=argparse.ArgumentParser()
 p.add_argument('--sources',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--steps',type=int,default=360)
 args=p.parse_args(argv)
 out=args.out;out.mkdir(parents=True,exist_ok=True)
 rec,artifacts=extract(args.sources);x,y,legal=arrays(rec)
 (out/'source_parent_corpus.jsonl').write_text(''.join(jd(r)+'\n' for r in rec))
 np.savez_compressed(out/'parent_features.npz',x=x,y=y,legal=legal.astype(np.uint8),
   variant=np.array([ALL.index(r['variant']) for r in rec],np.uint8),
   family=np.array([FAMILIES.index(r['family']) for r in rec],np.uint8),
   tick=np.array([r['tick'] for r in rec],np.uint8))
 idx={k:np.array([i for i,r in enumerate(rec) if r['variant'] in seq],int) for k,seq in [('train',TRAIN),('valid',VALID),('test',TEST)]}
 if any(set(idx[a])&set(idx[b]) for a in idx for b in idx if a!=b):raise AssertionError('variant split collision')
 nearest=nearest_graph_dict(rec)
 prior={k:v for (k,),v in family_prior(rec,y,idx['train'],('variant',)).items()}
 validprior=evaluate(None,rec,x,y,legal,idx['valid'],prior=prior,nearest_graph=nearest)
 testprior=evaluate(None,rec,x,y,legal,idx['test'],prior=prior,nearest_graph=nearest)
 candidates=[]
 for width,hard_weight in [(16,0.),(32,0.),(32,.18),(64,.18)]:
  model,fit=train(x[idx['train']],y[idx['train']],legal[idx['train']],width,args.steps,
                  seed=14014+width+round(hard_weight*100),hard_weight=hard_weight)
  path=out/f'candidate_w{width}_h{int(hard_weight*100)}.npz';model.save(path)
  ev=evaluate(model,rec,x,y,legal,idx['valid'])
  candidates.append({'width':width,'hard_weight':hard_weight,'fit':fit,'valid':smry(ev),
       'checkpoint_sha256':sha(path),'filename':path.name})
 # Unseen-variant validation controls candidate selection, TEST identities untouched.
 winner=max(candidates,key=lambda o:(o['valid']['action_agreement']-.3*o['valid']['kl'], -o['fit']['parameters']))
 selected=load(out/winner['filename'])
 test_rows=evaluate(selected,rec,x,y,legal,idx['test'])
 (out/'heldout_teacher_test.jsonl').write_text(''.join(jd(r)+'\n' for r in test_rows))
 # Final all-source model is explicitly AFTER and distinct from unseen-graph test.
 final,finalfit=train(x,y,legal,winner['width'],args.steps,seed=14414,hard_weight=winner['hard_weight'])
 finalpath=out/'final_all_graph_student.npz';final.save(finalpath)
 # Honest in-source final agreement includes training examples; not generalization.
 final_fit=smry(evaluate(final,rec,x,y,legal,np.arange(len(rec))))
 synthetic=synthetic_play(out,{'student':final,'records':rec})
 stats={v:{'source_calls':sum(r['variant']==v for r in rec),
            'action_counts':dict(Counter(r['action'] for r in rec if r['variant']==v))} for v in ALL}
 report={'schema':SCHEMA,'source_head_sha':'8759900d20805ce87aec5a3a023780e808f5ed4d',
   'source_workflow_run':43769010943,'source_original_teacher_calls':len(rec),
   'source_seed':570,'source_artifact_pins':artifacts,'source_variant_stats':stats,
   'original_parent_inference':'YES, archived #630 source evidence (not new forwards)',
   'new_parent_forwards':0,'native_cybersand_runs':0,'paid_gpu_spend':0,
   'source_corpus_sha256':sha(out/'source_parent_corpus.jsonl'),
   'feature_corpus_sha256':sha(out/'parent_features.npz'),
   'student_encoder':'129 features: 100 lossy local host fields + 3 family one-hot + 26 layer-graph indicators; NOT native contract',
   'train_variants':TRAIN,'validation_variants':VALID,'test_variants':TEST,
   'graph_nearest_source_baseline':nearest,
   'nearest_graph_prior_valid':smry(validprior),'nearest_graph_prior_test':smry(testprior),
   'candidate_models':candidates,'validation_selected':winner['filename'],
   'heldout_graph_test':smry(test_rows),
   'heldout_test_per_variant':{v:smry([r for r in test_rows if r['variant']==v]) for v in TEST},
   'optimizer_updates':(len(candidates)+1)*args.steps,
   'final_all_source_student_sha256':sha(finalpath),'final_fit':finalfit,
   'synthetic_seed_range':[27001,27016],'synthetic_summary':synthetic,
   'synthetic_rollout_sha256':sha(out/'synthetic_rollouts.jsonl'),
   'honest_limitations':['all original parent inference shares seed 570','ten unique executed layer flows are NOT ten independently trained weights',
      'one STOP-only variant has exactly 3 original decisions','heldout variants do not imply seed-independent trajectories',
      'external simulator is NOT real CyberSand native W0','AEROHOUND/LFM/Sprocket were not newly inferred']}
 (out/'metrics.json').write_text(jd(report)+'\n')
 (out/'SHA256SUMS.txt').write_text(''.join(f'{sha(pp)}  {pp.relative_to(out).as_posix()}\n' for pp in sorted(out.rglob('*')) if pp.is_file() and pp.name!='SHA256SUMS.txt'))
 print(jd({'source_count':len(rec),'selected':winner['filename'],'validation':winner['valid'],
   'unseen_teacher_graph_test':report['heldout_graph_test'],'prior_test':report['nearest_graph_prior_test'],
   'final_checkpoint':report['final_all_source_student_sha256'],'updates':report['optimizer_updates']}),flush=True)
 return report

if __name__=='__main__':main()
