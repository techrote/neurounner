"""D-STUDENT-002: CPU on-policy repair. All new states and targets synthetic.
Archived stock/GREYHOUND/BRANCHWITCH distributions are priors, NOT new inference.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .lab import ACTIONS,STYLES,DIRS,Game,Student,feats,teacher,curriculum,load_original,sha,js

SCHEMA="cybersand.driv3r.distill.onpolicy-repair/1"
TRAIN=range(1,65);VALID=range(1001,1017);TEST=range(2001,2017)
CASES=("normal","squeeze","trap")

def load(path):
 with np.load(path,allow_pickle=False) as z:
  if set(z.files)!={"w1","b1","w2","b2"}:raise ValueError("Unexpected checkpoint members")
  m=Student(nf=z["w1"].shape[0],hidden=z["w1"].shape[1])
  for k in ("w1","b1","w2","b2"):
   a=np.asarray(z[k],dtype=np.float32)
   if a.shape!=getattr(m,k).shape or not np.isfinite(a).all():raise ValueError("Bad tensor "+k)
   setattr(m,k,a.copy())
  return m

def adverse(g,case):
 if case=="normal":return
 if case=="squeeze":
  x,y=g.pos;near=((x+1,y-1),(x+1,y+1),(x+2,y))
  for p in near:
   if p!=g.pos and p not in g.painted and g.open(p):g.protected.add(p)
 elif case=="trap":
  if g.tick%7 in (0,4):
   for d in ((g.heading+1)%4,(g.heading+3)%4):
    p=g.target(d)
    if p!=g.pos and p not in g.painted and g.open(p):g.blocks.add(p)
 else:raise ValueError(case)

def corrected(g,original,mix=.25):
 o=g.obs();legal=o["legal"];prob=teacher(o,original)
 s={a:-1000. for a in ACTIONS};fresh=0
 for a,d in (("STRAIGHT",0),("LEFT",-1),("RIGHT",1)):
  if a not in legal:continue
  p=g.target((g.heading+d)%4);novel=p not in g.painted
  fresh+=int(novel)
  exits=sum(g.open((p[0]+dx,p[1]+dy)) and (p[0]+dx,p[1]+dy) not in g.painted for dx,dy in DIRS)
  s[a]=(.9 if novel else -2.1)+.3*min(3,exits)-.7*(o["last_actions"].count(a)>=4)
 if "BRANCH" in legal:
  fresh+=1
  s["BRANCH"]=1.05+(.4 if g.style=="branchwitch" else 0)-.45*o["last_actions"].count("BRANCH")
 if "RETRACE" in legal:s["RETRACE"]=2.7 if fresh==0 else -1.7
 if "STOP" in legal:s["STOP"]=4. if len(legal)==1 else -6.
 if g.style=="aerohound_proxy":
  a="LEFT" if g.tick%4<2 else "RIGHT"
  if a in legal:s[a]+=.3
 z=np.array([s[a] for a in ACTIONS],dtype=np.float32);z-=z.max()
 q=np.exp(z);q/=q.sum()
 p=mix*prob+(1-mix)*q
 for i,a in enumerate(ACTIONS):
  if a not in legal:p[i]=0.
 p/=p.sum()
 return p.astype(np.float32)

def corpus(base,original):
 x=[];y=[];kinds={}
 for seed in TRAIN:
  for style in STYLES:
   for case in CASES:
    for driver in ("base","corrective"):
     g=Game(seed,style)
     rng=np.random.default_rng(seed*7919+STYLES.index(style)*31+CASES.index(case)*7+(driver=="base"))
     for _ in range(22):
      adverse(g,case);o=g.obs();target=corrected(g,original,.37 if case=="normal" else .17)
      x.append(feats(o));y.append(target)
      key=case+"/"+driver;kinds[key]=kinds.get(key,0)+1
      a=base.predict(o) if driver=="base" else ACTIONS[int(rng.choice(len(ACTIONS),p=target))]
      if a not in o["legal"]:raise ValueError("Host-legal action projection failed")
      g.step(a)
      if g.stopped:break
 return np.stack(x),np.stack(y),kinds

def episode(seed,style,case,policy,base,student,budget):
 g=Game(seed,style);rng=np.random.default_rng(seed*1009+STYLES.index(style))
 raw_invalid=deadends=recovery=0;counts={a:0 for a in ACTIONS};trace=[]
 for _ in range(budget):
  adverse(g,case);o=g.obs();legal=o["legal"]
  raw=None
  if policy in ("base","repaired"):
   m=base if policy=="base" else student
   raw=m.predict(o,mask=False)
   raw_invalid+=int(raw not in legal)
   act=m.predict(o,mask=True)
  elif policy=="flat_continue":
   offered=[a for a in legal if a!="STOP"] or ["STOP"]
   act=offered[int(rng.integers(len(offered)))]
  else:act=next(a for a in ("LEFT","STRAIGHT","RIGHT","BRANCH","RETRACE","STOP") if a in legal)
  if act not in legal:raise ValueError("Host selected illegal action")
  dead="RETRACE" in legal and not any(a in legal for a in ("STRAIGHT","LEFT","RIGHT","BRANCH"))
  deadends+=int(dead);recovery+=int(dead and act=="RETRACE")
  trace.append({"tick":g.tick,"pos":list(g.pos),"legal":list(legal),"raw":raw,"action":act})
  counts[act]+=1;g.step(act)
  if g.stopped:break
 metrics=g.metrics()
 metrics.update(seed=seed,style=style,case=case,controller=policy,budget=budget,
       raw_preprojection_invalid=raw_invalid,deadend_exposures=deadends,
       deadend_recoveries=recovery,action_counts=counts,action_trace=trace)
 return g,metrics

def eval_student(base,student):
 records=[];grids={}
 for budget in (40,80):
  for case in CASES:
   for seed in TEST:
    for style in STYLES:
     for policy in ("base","repaired","flat_continue","wall"):
      g,r=episode(seed,style,case,policy,base,student,budget);records.append(r)
      if budget==80 and case=="squeeze" and seed in (2001,2005,2009,2013):
       grids[(seed,style,policy)]=g
 summary={}
 for budget in (40,80):
  for case in CASES:
   bypolicy={}
   for policy in ("base","repaired","flat_continue","wall"):
    a=[r for r in records if r["budget"]==budget and r["case"]==case and r["controller"]==policy]
    bypolicy[policy]={k:float(np.mean([r[k] for r in a])) for k in
          ("cells","branches","revisits","retraces","raw_preprojection_invalid",
           "deadend_exposures","deadend_recoveries","decisions","repetition")}
   diffs=[]
   for seed in TEST:
    for style in STYLES:
     ar={r["controller"]:r for r in records if r["budget"]==budget and r["case"]==case
         and r["seed"]==seed and r["style"]==style}
     diffs.append(ar["repaired"]["cells"]-ar["base"]["cells"])
   summary[str(budget)+"/"+case]={"policies":bypolicy,
      "paired_cells_delta_mean":float(np.mean(diffs)),"wins":sum(d>0 for d in diffs),
      "losses":sum(d<0 for d in diffs),"ties":sum(d==0 for d in diffs)}
 return records,summary,grids

def legacy_val(model,examples):
 x=np.stack([r[0] for r in examples]);y=np.stack([r[1] for r in examples])
 z=model.scores(x);z-=z.max(axis=1,keepdims=True)
 p=np.exp(z);p/=p.sum(axis=1,keepdims=True)
 return {"soft_ce":float(-(y*np.log(np.maximum(p,1e-7))).sum(1).mean()),
         "top1":float(np.mean(p.argmax(1)==y.argmax(1)))}

def gallery(grids,path):
 seeds=(2001,2005,2009,2013);policies=("base","repaired","flat_continue","wall")
 im=Image.new("RGB",(4*192,4*215),(235,238,241));draw=ImageDraw.Draw(im)
 for iy,(seed,style) in enumerate(zip(seeds,STYLES)):
  for ix,policy in enumerate(policies):
   im.paste(grids[(seed,style,policy)].png(),(ix*192,iy*215+23))
   draw.text((ix*192+4,iy*215+3),style+" / "+policy,fill=(5,6,7))
 im.save(path)

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--input",required=True,type=Path);ap.add_argument("--out",required=True,type=Path)
 ap.add_argument("--steps",type=int,default=300);args=ap.parse_args()
 args.out.mkdir(parents=True,exist_ok=True)
 original=load_original();base=load(args.input);student=load(args.input)
 x,y,kinds=corpus(base,original)
 np.savez_compressed(args.out/"repair_train_corpus.npz",features=x.astype('float32'),soft_targets=y.astype('float32'))
 training=student.train(x,y,iterations=args.steps,seed=109,lr=.007)
 validation=curriculum(original,VALID.start,VALID.stop)
 traces,summary,grids=eval_student(base,student)
 student.save(args.out/"repair_student.npz")
 (args.out/"repair_rollouts.jsonl").write_text("".join(js(r)+"\n" for r in traces))
 gallery(grids,args.out/"repair_squeeze_gallery.png")
 import os
 source=Path(os.environ["DRIV3R_TEACHER_CORPUS"])
 metrics={"schema":SCHEMA,"research_kind":"external_synthetic_repair","parent_inference_new_seeds":0,
   "original_parent_receipts":len(original),"parent_seed":570,
   "original_corpus_sha256":sha(source.read_bytes()),"base_sha256":sha(args.input.read_bytes()),
   "repair_sha256":sha((args.out/"repair_student.npz").read_bytes()),
   "gallery_sha256":sha((args.out/"repair_squeeze_gallery.png").read_bytes()),
   "rollouts_sha256":sha((args.out/"repair_rollouts.jsonl").read_bytes()),
   "train_corpus_sha256":sha((args.out/"repair_train_corpus.npz").read_bytes()),
   "train_seed_range":[TRAIN.start,TRAIN.stop-1],"validation_seed_range":[VALID.start,VALID.stop-1],
   "test_seed_range":[TEST.start,TEST.stop-1],"onpolicy_synthetic_examples":len(x),
   "adversity_examples":kinds,"optimizer":training,"legacy_val_base":legacy_val(base,validation),
   "legacy_val_repair":legacy_val(student,validation),"matched_rollouts":len(traces),"evaluation":summary}
 (args.out/"repair_metrics.json").write_text(js(metrics)+"\n")
 print(js({"optimizer":training,"matched":summary,"model":metrics["repair_sha256"]}))
if __name__=="__main__":main()
