"""P2: Train prefix-to-future W0 rankers from archived model/host trajectories.

Source archives were independently audited locally. Hosted CI only sees frozen
derived features, NOT original ZIPs. All ranking labels are synthetic future
events, never human preference, and no game/native execution is performed.
"""
from __future__ import annotations
import csv, hashlib, json, math
from itertools import combinations
from pathlib import Path

HERE=Path(__file__).resolve().parent
DATA_SHA='32b233be29a960e544ff978721b70c1c1625e35020bac961962a989310c157c6'
SRC={'lfm_w13':(43771476464,'245290cab2352381dbd1eaf16374c577138b4ade',
      '0e69b23c100a00b438d4e2d9d07fa4550e276368e0ad4635e38a6c858f79b95e',
      '36fa8876f249b0263d19dc40bea17731e8a4776da9cc82400189a876980380ff'),
     'sprocket_f3':(43769025429,'e56e0aac49b310cc1e00aafe3ffeddb90eca6e38',
      '5eb64e5f7d5335809f631aa11caf9edbfcf1a67f546601d5d14e795cc61f7af6')}
SEEDS={'lfm_w13':{6111,6127,6143,6167},'sprocket_f3':{8111,8123,8147,8161}}
ROLES={'learned','random','wall','fidget'}
PREFIX=('prefix_new','prefix_recent_new','prefix_turns','prefix_branch','prefix_retrace',
        'prefix_near_obstacle','prefix_constrained','prefix_stall_recover')
OUTCOMES={'future_novelty':('future_new',3),'future_stall_escape':('future_recover',1)}

def load(path=None):
    path=Path(path or HERE/'turn_features_p2.csv')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=DATA_SHA:
        raise ValueError('frozen receipt-feature SHA256 mismatch')
    rows=list(csv.DictReader(raw.decode().splitlines()))
    groups={}
    if len(rows)!=32:raise ValueError('expected 32 original episodes')
    for r in rows:
        family=r['cohort'];seed=int(r['seed']);role=r['role']
        if family not in SRC or seed not in SEEDS[family] or role not in ROLES:
            raise ValueError('unrecognised original source/seed/role')
        if len(r['trace_sha256'])!=64 or any(c not in '0123456789abcdef' for c in r['trace_sha256']):
            raise ValueError('missing trace byte identity')
        for k in (*PREFIX,'future_new','future_recover','future_max_stall','all_stop'):
            if not r[k].isdigit() or int(r[k])>128:raise ValueError('bad chronological source feature')
        if int(r['prefix_recent_new'])>int(r['prefix_new']) or int(r['future_new'])>64 or int(r['prefix_new'])>64:
            raise ValueError('invalid 64/128 prefix/future cut')
        groups.setdefault((family,seed),[]).append(r)
    if len(groups)!=8 or any({r['role'] for r in g}!=ROLES for g in groups.values()):
        raise ValueError('incomplete paired source worlds')
    return groups

def pairs(groups,axis,without_wall=False):
    field,margin=OUTCOMES[axis];pp=[];ties=0
    for world,rows in sorted(groups.items()):
        rows=sorted([r for r in rows if not without_wall or r['role']!='wall'],key=lambda r:r['role'])
        for a,b in combinations(rows,2):
            delta=int(a[field])-int(b[field])
            if abs(delta)<margin:ties+=1;continue
            pp.append((world,a,b,int(delta>0)))
    return pp,ties

def fit(pp,epochs=300,lr=.035,reg=.1):
    if not pp:raise ValueError('no train pairs')
    n=len(PREFIX)
    xs=[[int(row[k])/64 for k in PREFIX] for _,a,b,_ in pp for row in (a,b)]
    std=[max(.03,(sum((x[j]-sum(v[j] for v in xs)/len(xs))**2 for x in xs)/len(xs))**.5) for j in range(n)]
    data=[([(int(a[k])-int(b[k]))/64/std[j] for j,k in enumerate(PREFIX)],y) for _,a,b,y in pp]
    w=[0.]*n;m=[0.]*n;v=[0.]*n;loss=[]
    for ep in range(1,epochs+1):
        grad=[reg*t for t in w];total=0.
        for x,y in data:
            z=max(-30,min(30,sum(t*q for t,q in zip(x,w))))
            p=1/(1+math.exp(-z));total-=y*math.log(max(1e-12,p))+(1-y)*math.log(max(1e-12,1-p))
            for j in range(n):grad[j]+=(p-y)*x[j]/len(data)
        for j,g in enumerate(grad):
            m[j]=.9*m[j]+.1*g;v[j]=.999*v[j]+.001*g*g
            w[j]-=lr*(m[j]/(1-.9**ep))/(math.sqrt(v[j]/(1-.999**ep))+1e-8)
        if ep in (1,20,100,epochs):loss.append([ep,round(total/len(data),6)])
    return {'weights':w,'std':std,'train_loss':loss,'epochs':epochs,'l2':reg,'prefix_only_features':PREFIX}

def evaluate(model,pp):
    good=0;brier=0
    for _,a,b,y in pp:
        z=sum((int(a[k])-int(b[k]))/64/model['std'][j]*model['weights'][j] for j,k in enumerate(PREFIX))
        p=1/(1+math.exp(-max(-30,min(30,z))))
        good+=int((p>=.5)==bool(y));brier+=(p-y)**2
    return {'pairs':len(pp),'correct':good,'accuracy':round(good/len(pp),4) if pp else None,'brier':round(brier/len(pp),4) if pp else None}

def simple_baseline(pp,feature):
    usable=[(a,b,y) for _,a,b,y in pp if int(a[feature])!=int(b[feature])]
    return {'correct':sum((int(a[feature])>int(b[feature]))==bool(y) for a,b,y in usable),
            'evaluated':len(usable),'ties':len(pp)-len(usable)}

def run(output_dir,source_path=None):
    groups=load(source_path);dest=Path(output_dir);dest.mkdir(parents=True,exist_ok=True)
    report={'schema':'driv3r-gameplay-receipt-p2/v1','source_feature_sha256':DATA_SHA,
        'evidence':'frozen original 128-turn Python W0 model-scored and model-free host traces',
        'worlds':8,'episodes':32,'model_forward_episodes':8,'native_receipts':0,'owner_human_votes':0,
        'new_model_forward_executions':0,'game_policy_optimizer_steps':0,
        'current_run_original_producer_zip_bytes_downloaded':False,
        'labels':'weak synthetic future-event measures, NOT human preference','axes':{}}
    for axis in OUTCOMES:
        for no_wall in (False,True):
            pp,ties=pairs(groups,axis,no_wall)
            full=fit(pp);folds=[]
            for family in SRC:
                train=[p for p in pp if p[0][0]!=family];hold=[p for p in pp if p[0][0]==family]
                folds.append({'held_out_family':family,'train_pairs':len(train),**evaluate(fit(train),hold)})
            label=axis+('-no_wall' if no_wall else '-all')
            report['axes'][label]={'pairs':len(pp),'abstained_ties':ties,'full_fit':evaluate(full,pp),
                'leave_model_family_out':folds,'prefix_new_only':simple_baseline(pp,'prefix_new'),
                'prefix_recent_new_only':simple_baseline(pp,'prefix_recent_new'),
                'real_cpu_optimizer_epochs':300}
            (dest/(label+'-trained_head.json')).write_text(json.dumps(full,sort_keys=True,indent=2)+'\n')
    (dest/'P2_receipt_report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output-dir',default='/tmp/driv3r-pref-P2');p.add_argument('--source-path');args=p.parse_args()
    print(json.dumps(run(args.output_dir,args.source_path),indent=2))
