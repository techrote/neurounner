"""Independent read-only D007 action, original PNG, metric and checkpoint audit."""
from __future__ import annotations
import argparse,collections,hashlib,io,json
from pathlib import Path
from .lab import Game
from .repair import adverse,load
from .exhaustion_recovery import reachable_frontier
from .stop_repair import digest,js

def audit(out,frontier,exhaustion):
 m=json.loads((out/'stop_repair_metrics.json').read_text())
 if digest(frontier)!=m['source_d005_sha256'] or digest(exhaustion)!=m['source_d006_sha256']:
  raise ValueError('upstream checkpoint changed')
 if digest(out/'stop_repaired_student.npz')!=m['student_sha256'] or digest(out/'stop_repair_actions.jsonl')!=m['action_sha256']:
  raise ValueError('original student or action receipt changed')
 models={'frontier_005':load(frontier),'recovery_006':load(exhaustion),
         'repair_007':load(out/'stop_repaired_student.npz'),'oracle_stop':load(frontier)}
 actions=[json.loads(s) for s in (out/'stop_repair_actions.jsonl').read_text().splitlines()]
 grouped=collections.defaultdict(list)
 for row in actions:grouped[(row['seed'],row['style'],row['case'],row['policy'])].append(row)
 rows=[json.loads(s) for s in (out/'stop_repair_rollouts.jsonl').read_text().splitlines()]
 indexed={(r['seed'],r['style'],r['case'],r['policy']):r for r in rows}
 if len(indexed)!=m['test_episodes'] or len(actions)!=m['action_receipts']:
  raise AssertionError('missing action or episode evidence')
 action_count=png_count=0
 for key,trace in grouped.items():
  seed,style,case,name=key;g=Game(seed,style);first_exhaust=None;wrong=correct=invalid=0
  for tick,r in enumerate(trace):
   adverse(g,case);o=g.obs();front=bool(reachable_frontier(g) and len(o['legal'])>1)
   if not front and first_exhaust is None:first_exhaust=tick
   raw='STOP' if name=='oracle_stop' and not front else models[name].predict(o,mask=False)
   chosen='STOP' if name=='oracle_stop' and not front else models[name].predict(o,mask=True)
   if not (r['tick']==tick and r['position']==list(g.pos) and r['heading']==g.heading
           and r['frontier_exists_oracle']==front and r['legal']==o['legal']
           and r['local_5x5']==o['local_5x5'] and r['raw']==raw and r['chosen']==chosen
           and chosen in o['legal']):
    raise AssertionError('independent action replay mismatch: '+str(key)+' '+str(tick))
   invalid+=int(raw not in o['legal']);wrong+=int(chosen=='STOP' and front)
   correct+=int(chosen=='STOP' and not front);g.step(chosen);action_count+=1
   if g.stopped and tick!=len(trace)-1:raise AssertionError('post-STOP action')
  original=indexed[key]
  metrics={'cells':len(g.painted),'branches':g.branch_count,'revisits':g.revisits,
           'retraces':g.retraces,'decisions':g.tick,'stop':int(g.stopped),
           'premature_stop':wrong,'exhausted_stop':correct,
           'wasted_after_exhaustion':max(0,g.tick-first_exhaust-1) if first_exhaust is not None else 0,
           'raw_invalid':invalid}
  for metric,value in metrics.items():
   if original[metric]!=value:raise AssertionError('metric mismatch '+metric)
  filename=f'original_{case}_{seed}_{style}_{name}.png'
  if filename in m['original_png_sha256']:
   buf=io.BytesIO();g.png().save(buf,format='PNG')
   if hashlib.sha256(buf.getvalue()).hexdigest()!=m['original_png_sha256'][filename]:
    raise AssertionError('original PNG mismatch '+filename)
   png_count+=1
 if len(grouped)!=len(indexed) or png_count!=len(m['original_png_sha256']):
  raise AssertionError('incomplete episode or PNG replay')
 return {'original_model_decisions_replayed':action_count,'original_episodes_replayed':len(grouped),
         'original_pngs_byte_exact':png_count,'selected_student_sha256':m['student_sha256'],
         'native_game_executions':0,'new_parent_forwards':0}

def main():
 p=argparse.ArgumentParser()
 for n in ('out','frontier','exhaustion'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();result=audit(a.out,a.frontier,a.exhaustion)
 (a.out/'stop_repair_audit.json').write_text(js(result)+'\n');print(js(result))
if __name__=='__main__':main()
