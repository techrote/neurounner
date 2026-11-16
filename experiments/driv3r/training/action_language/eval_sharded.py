"""GAL-014 immutable-checkpoint sharded evaluation with independent host receipt replay."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from start_semantics import (SCHEMA,EVAL_BASE,Adapted,episode,audit_row,summarise)

POLICIES=('raw','always','conditional','adapted','adapted_sample','teacher')

def evaluate_seed(task):
    seed,base,adapter=task
    frozen=Adapted(base);adapted=Adapted(base);adapted.adapter=list(adapter)
    kind=('tree','path')[(seed-EVAL_BASE)%2]
    rows=[];receipts=[];verified=0
    for required in (False,True):
        for policy in POLICIES:
            model=frozen if policy in ('raw','always','conditional') else adapted
            w,row=episode(seed,kind,required,model,policy)
            verified+=audit_row(row,w.trace)
            rows.append(row)
            receipts.append(dict(seed=seed,kind=kind,required=required,policy=policy,actions=w.trace))
    return rows,receipts,verified

def run(out,base_path,checkpoint_path,start=0,count=32,workers=4):
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    base=json.loads(Path(base_path).read_text())['weights']
    checkpoint=json.loads(Path(checkpoint_path).read_text());adapter=checkpoint['adapter']
    tasks=[(EVAL_BASE+j,base,adapter) for j in range(start,start+count)]
    rows=[];receipts=[];verified=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for rs,ts,n in pool.map(evaluate_seed,tasks):
            rows.extend(rs);receipts.extend(ts);verified+=n
    summary={p:{'start_not_goal':summarise([r for r in rows if r['policy']==p and not r['start_required']]),
                'start_is_goal':summarise([r for r in rows if r['policy']==p and r['start_required']])}
             for p in POLICIES}
    (root/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    (root/'receipts.json').write_text(json.dumps(receipts,separators=(',',':'))+'\n')
    meta=dict(schema=SCHEMA,checkpoint_sha256=hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      first_seed=EVAL_BASE+start,last_seed=EVAL_BASE+start+count-1,
      heldout_episodes=len(rows),audited_receipts=verified,summary=summary,
      native_receipts=0,game_code_edited=False)
    (root/'metrics.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2));return meta

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True);p.add_argument('--frozen',required=True)
    p.add_argument('--trained',required=True);p.add_argument('--start',type=int,default=0)
    p.add_argument('--count',type=int,default=32);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.out,a.frozen,a.trained,a.start,a.count,a.workers)
