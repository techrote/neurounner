"""D-STUDENT-007: synthetic hard-negative STOP repair of D006 (research Draft/HOLD).

The full-map BFS is a teacher/evaluation diagnostic, never an MLP input.
Source parent d1 receipts remain the historical 144 seed-570 decisions.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .lab import ACTIONS,STYLES,Game,feats
from .repair import adverse,load
from .trap_rebalance import CASES,masked_softmax,adam_update
from .exhaustion_recovery import reachable_frontier,evaluate_model,summarize
SCHEMA='cybersand.driv3r.distill.stop-repair/1'
TRAIN=range(225,273);VALID=range(1057,1065);TEST=range(6001,6017)
BUDGET=80;EXHAUST_STOP=(.65,.82,.95)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(o):return json.dumps(o,sort_keys=True,separators=(',',':'),allow_nan=False)

def collect(old,base,seeds=TRAIN,budget=BUDGET):
 d={k:[] for k in ('features','legal','frontier_live','hard_live','false_stop','case_id')}
 for seed in seeds:
  for style in STYLES:
   for case in CASES:
    for model in (old,base):
     g=Game(seed,style)
     for _ in range(budget):
      adverse(g,case);o=g.obs();menu=o['legal']
      if 'STOP' not in menu:raise AssertionError('invalid host menu')
      live=bool(reachable_frontier(g) and len(menu)>1)
      d['features'].append(feats(o));d['legal'].append([a in menu for a in ACTIONS])
      d['frontier_live'].append(live)
      d['hard_live'].append(live and sum(row.count('.') for row in o['local_5x5'])<=3)
      d['false_stop'].append(live and base.predict(o)=='STOP')
      d['case_id'].append(CASES.index(case))
      action=model.predict(o)
      if action not in menu:raise AssertionError('illegal action')
      g.step(action)
      if g.stopped:break
 return {k:np.asarray(v,dtype=(np.float32 if k=='features' else np.uint8 if k=='case_id' else bool)) for k,v in d.items()}

def resample_index(d,seed=7007):
 live=d['frontier_live'];hard=d['hard_live'];bad=d['false_stop']
 pools=(np.where(~live)[0],np.where(bad)[0],np.where(hard&~bad)[0],np.where(live&~hard&~bad)[0])
 if any(len(p)<10 for p in pools):raise ValueError('insufficient positive/negative strata')
 n=len(live);counts=(int(n*.27),int(n*.20),int(n*.30));rng=np.random.default_rng(seed)
 ix=np.concatenate([rng.choice(p,size=c,replace=True) for p,c in zip(pools,(*counts,n-sum(counts)))])
 return ix[rng.permutation(n)]

def target_probs(p005,p006,menu,live,exhaust_stop):
 if not 0<exhaust_stop<=1:raise ValueError('invalid STOP weight')
 if p005.shape!=p006.shape or p005.shape!=menu.shape or p005.shape[1]!=6:raise ValueError('bad shape')
 if not np.all(menu[:,5]) or not np.all(menu.any(1)) or not np.all(np.isfinite(p005+p006)):
  raise ValueError('invalid host menus/prior')
 soft=(.58*p005+.42*p006)*menu
 soft[live,5]=0
 norms=soft.sum(1,keepdims=True)
 if np.any(live&(norms[:,0]<=0)):raise AssertionError('live STOP-only menu')
 soft=np.divide(soft,norms,out=np.zeros_like(soft),where=norms>0)
 stopped=~live;soft[stopped]*=(1-exhaust_stop);soft[stopped,5]=exhaust_stop
 only=~menu[:,:5].any(1);soft[only]=0;soft[only,5]=1
 soft*=menu;soft/=soft.sum(1,keepdims=True)
 if np.max(abs(soft.sum(1)-1))>2e-5 or np.any(soft[~menu]!=0):raise AssertionError('unsafe target')
 return soft.astype(np.float32)

def selection_score(v,name='repair_007'):
 score=0.
 for case,w in (('normal',.25),('trap',.30),('squeeze',.45)):
  a=v[case+'/'+name];b=v[case+'/recovery_006']
  score+=w*((a['cells']-b['cells'])*.22+(a['branches']-b['branches'])*.05
            -a['premature_stop']*6-a['wasted_after_exhaustion']*.12)
 return float(score)

def paired(rows):
 stats={};rng=np.random.default_rng(7007)
 for case in CASES:
  blocks=[]
  for seed in TEST:
   diffs=[]
   for style in STYLES:
    a=next(r for r in rows if (r['seed'],r['case'],r['style'],r['policy'])==(seed,case,style,'recovery_006'))
    b=next(r for r in rows if (r['seed'],r['case'],r['style'],r['policy'])==(seed,case,style,'repair_007'))
    diffs.append({k:b[k]-a[k] for k in ('cells','branches','premature_stop','wasted_after_exhaustion')})
   blocks.append({k:float(np.mean([v[k] for v in diffs])) for k in diffs[0]})
  stats[case]={}
  for k in blocks[0]:
   ds=np.asarray([v[k] for v in blocks]);boot=ds[rng.integers(len(ds),size=(2000,len(ds)))].mean(1)
   stats[case][k]={'mean_delta':float(ds.mean()),'seed_block_bootstrap_95pct':
                   [float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]}
 return stats

def pictures(models,out):
 pics={};names=tuple(models)
 for case in CASES:
  canvas=Image.new('RGB',(len(names)*192,4*218),(237,239,242));draw=ImageDraw.Draw(canvas)
  for i,(seed,style) in enumerate(zip((6001,6005,6009,6013),STYLES)):
   for j,(name,model) in enumerate(models.items()):
    g=Game(seed,style)
    for _ in range(BUDGET):
     adverse(g,case);o=g.obs()
     live=bool(reachable_frontier(g) and len(o['legal'])>1)
     action='STOP' if name=='oracle_stop' and not live else model.predict(o)
     if action not in o['legal']:raise AssertionError('illegal gallery action')
     g.step(action)
     if g.stopped:break
    fn=f'original_{case}_{seed}_{style}_{name}.png';pics[fn]=g.png()
    canvas.paste(pics[fn],(j*192,i*218+24))
    draw.text((j*192+2,i*218+4),style+'/'+name,fill=(12,12,12))
  canvas.save(out/f'stop_repair_{case}_gallery.png')
 return pics

def main(argv=None):
 p=argparse.ArgumentParser()
 for n in ('frontier','exhaustion','exhaustion-metrics','out'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--steps',type=int,default=120);a=p.parse_args(argv)
 if not 1<=a.steps<=300:raise ValueError('unbounded optimizer')
 old=json.loads(a.exhaustion_metrics.read_text())
 if digest(a.frontier)!=old['source_d005_sha256'] or digest(a.exhaustion)!=old['checkpoint_sha256']:
  raise ValueError('source pin mismatch')
 if old.get('new_parent_model_forwards') or old.get('new_native_executions'):
  raise ValueError('unsafe provenance')
 if set(TRAIN)&set(VALID) or set(TRAIN)&set(TEST) or set(VALID)&set(TEST):raise AssertionError('split collision')
 a.out.mkdir(parents=True,exist_ok=True);p005=load(a.frontier);p006=load(a.exhaustion)
 data=collect(p005,p006);ix=resample_index(data);x=data['features'];legal=data['legal']
 pa=masked_softmax(p005.scores(x),legal);pb=masked_softmax(p006.scores(x),legal)
 np.savez_compressed(a.out/'stop_repair_corpus.npz',**data,prior_005=pa,prior_006=pb,train_indices=ix)
 candidates=[]
 for fraction in EXHAUST_STOP:
  y=target_probs(pa,pb,legal,data['frontier_live'],fraction)
  model=load(a.exhaustion);opt=adam_update(model,x[ix],y[ix],steps=a.steps,lr=.001,seed=7007,batch=192)
  name=f'stop_candidate_{fraction:.2f}.npz';model.save(a.out/name)
  validation,_,_=evaluate_model({'frontier_005':p005,'recovery_006':p006,'repair_007':model},VALID)
  s=summarize(validation)
  candidates.append({'fraction':fraction,'checkpoint':name,'sha256':digest(a.out/name),
                     'optimizer':opt,'validation_score':selection_score(s),'validation_summary':s})
 chosen=max(candidates,key=lambda v:(v['validation_score'],-v['fraction']))
 winner=load(a.out/chosen['checkpoint']);winner.save(a.out/'stop_repaired_student.npz')
 models={'frontier_005':p005,'recovery_006':p006,'repair_007':winner,'oracle_stop':p005}
 rows,actions,_=evaluate_model(models,TEST,record=True)
 (a.out/'stop_repair_rollouts.jsonl').write_text(''.join(js(r)+'\n' for r in rows))
 (a.out/'stop_repair_actions.jsonl').write_text(''.join(js(r)+'\n' for r in actions))
 imgdir=a.out/'original_pngs';imgdir.mkdir(exist_ok=True)
 pics=pictures(models,a.out)
 for fn,im in pics.items():im.save(imgdir/fn)
 counts={k:int(v.sum()) for k,v in data.items() if k in ('frontier_live','hard_live','false_stop')}
 report={'schema':SCHEMA,'source_d005_sha256':digest(a.frontier),'source_d006_sha256':digest(a.exhaustion),
 'source_d006_metrics_sha256':digest(a.exhaustion_metrics),
 'train_seeds':[TRAIN.start,TRAIN.stop-1],'validation_seeds':[VALID.start,VALID.stop-1],
 'test_seeds':[TEST.start,TEST.stop-1],'training_examples':len(x),'training_strata':counts,
 'balanced_examples':len(ix),'dataset_sha256':digest(a.out/'stop_repair_corpus.npz'),
 'new_parent_forwards':0,'new_native_game_executions':0,'actual_optimizer_updates':a.steps*len(EXHAUST_STOP),
 'candidates':candidates,'selected_fraction':chosen['fraction'],
 'student_sha256':digest(a.out/'stop_repaired_student.npz'),
 'test_episodes':len(rows),'test_summary':summarize(rows),'paired_seed_blocks':paired(rows),
 'original_png_sha256':{k:digest(imgdir/k) for k in sorted(pics)},
 'action_receipts':len(actions),'action_sha256':digest(a.out/'stop_repair_actions.jsonl'),
 'research_limit':'Synthetic 24x24 host only; full-map oracle never fed to 100-feature student'}
 (a.out/'stop_repair_metrics.json').write_text(js(report)+'\n')
 print(js({'selected':chosen['fraction'],'candidate_validation_scores':[(c['fraction'],c['validation_score']) for c in candidates],
 'training_examples':len(x),'training_strata':counts,'checkpoint_sha256':report['student_sha256'],
 'squeeze':{name:report['test_summary']['squeeze/'+name] for name in models}}))
 return report
if __name__=='__main__':main()
