#!/usr/bin/env python3
"""Bounded CPU SGD diagnostic for two TRAIN-only worm corpora; no game access.

World-disjoint 80/20 *TRAIN-INTERNAL* fold, never official DEV/held-out.
Family-label ablation and legal-action masking; no checkpoint promotion.
Requires numpy and scikit-learn; writes JSON only, no model binaries.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import zipfile

ACTIONS = ('STRAIGHT', 'LEFT', 'RIGHT', 'BRANCH', 'RETRACE', 'STOP')
MEMBERS = {'v1': ('train.messages.jsonl','train.provenance.jsonl','world_group_id','synthetic-host-heuristic'),
           'probe': ('train.family_probes.messages.jsonl','train.family_probes.provenance.jsonl',
                     'source_world_group_id','train-synthetic-unexecuted-single-step-style-probe')}
OBS = {'x','y','heading','local_5x5','local_material_5x5','local_protected_5x5',
       'legal','last_actions','family','tick','budget_remaining','goal_delta','goal_reached'}


def digest(b):
    return hashlib.sha256(b).hexdigest()


def load(path, kind):
    m,p,g,expected = MEMBERS[kind]
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if z.testzip() or len(names)!=len(set(names)) or m not in names or p not in names:
            raise ValueError('bad source ZIP inventory/CRC')
        if sum(z.getinfo(n).file_size for n in names)>40_000_000:
            raise ValueError('oversized source ZIP')
        msgs=z.read(m).splitlines(); prov=z.read(p).splitlines()
    if not msgs or len(msgs)!=len(prov):
        raise ValueError('source/provenance count mismatch')
    result=[]
    for i,(line,receipt) in enumerate(zip(msgs,prov)):
        messages=json.loads(line)['messages']; source=json.loads(receipt)
        if len(messages)!=3 or [x['role'] for x in messages]!=['system','user','assistant']:
            raise ValueError('invalid chat roles')
        obs=json.loads(messages[1]['content']); label=messages[2]['content']
        if set(obs)!=OBS or label not in ACTIONS or label not in obs['legal']:
            raise ValueError('unsafe observation or illegal teacher action')
        if (source.get('index')!=i or source.get('evidence_class')!=expected or
                not source.get(g)):
            raise ValueError('invalid source lineage')
        if kind=='v1' and (source.get('label')!=label or not source.get('record_id')):
            raise ValueError('TRAIN teacher mismatch')
        if kind=='probe' and (source.get('counterfactual_label')!=label or
                source.get('counterfactual_family')!=obs['family'] or
                source.get('was_host_applied') is not False or
                source.get('has_native_receipt') is not False):
            raise ValueError('counterfactual label misrepresented')
        result.append((obs,label,source[g]))
    return result


def split(rows):
    flag=lambda g:int(digest(('diagnostic-worldfold-v1:'+g).encode())[:8],16)%5==0
    train=[r for r in rows if not flag(r[2])]
    test=[r for r in rows if flag(r[2])]
    if not train or not test or ({r[2] for r in train}&{r[2] for r in test}):
        raise ValueError('nonindependent world folds')
    signatures={digest(json.dumps(o,sort_keys=True).encode()) for o,_,_ in train}
    if any(digest(json.dumps(o,sort_keys=True).encode()) in signatures for o,_,_ in test):
        raise ValueError('exact prompt collision across world folds')
    return train,test


def features(o, family):
    f={'heading:'+str(o['heading']):1,'tick2:'+str(o['tick']%2):1,
       'tick4:'+str(o['tick']%4):1,'tick9:'+str(o['tick']%9):1,
       'tick18:'+str(o['tick']%18):1,'tick32:'+str(o['tick']//32):1,
       'budget32:'+str(o['budget_remaining']//32):1,
       'goal_reached:'+str(o['goal_reached']):1}
    if family:f['family:'+o['family']]=1
    for a in o['legal']:f['legal:'+a]=1
    for i,a in enumerate(o['last_actions']):f[f'last{i}:{a}']=1
    for field in ('local_5x5','local_protected_5x5'):
        for y,row in enumerate(o[field]):
            for x,val in enumerate(row):f[f'{field}:{x}:{y}:{val}']=1
    for y,row in enumerate(o['local_material_5x5']):
        for x,val in enumerate(row):f[f'material:{x}:{y}:{val}']=1
    for axis,value in zip('xy',o['goal_delta']):
        f[f'goal{axis}sign:{(value>0)-(value<0)}']=1
        f[f'goal{axis}bin:{min(8,abs(value)//12)}']=1
    return f


def evaluate(rows, family, epochs):
    import numpy as np
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import SGDClassifier
    train,test=split(rows)
    vectorizer=DictVectorizer(sparse=True,sort=True)
    x=vectorizer.fit_transform(features(o,family) for o,_,_ in train)
    xt=vectorizer.transform(features(o,family) for o,_,_ in test)
    y=np.array([label for _,label,_ in train]); truth=[label for _,label,_ in test]
    if not set(truth).issubset(set(y)) or len(set(y))<2:
        raise ValueError('unseen class in diagnostic evaluation')
    model=SGDClassifier(loss='log_loss',alpha=0.0001,max_iter=epochs,tol=None,
                        random_state=17,shuffle=True,learning_rate='optimal')
    model.fit(x,y)
    scores=model.decision_function(xt)
    pred=[max((a for a in model.classes_ if a in obs['legal']),
              key=lambda a:score[list(model.classes_).index(a)])
          for (obs,_,_),score in zip(test,scores)]
    prior=Counter(y)
    baseline=[max(obs['legal'],key=lambda a:(prior[a],-ACTIONS.index(a))) for obs,_,_ in test]
    count=Counter(truth);good=Counter(a for a,b in zip(truth,pred) if a==b)
    output={'train_rows':len(train),'eval_rows':len(test),
            'train_world_groups':len({r[2] for r in train}),
            'eval_world_groups':len({r[2] for r in test}),
            'features':x.shape[1],'optimizer_epochs':int(model.n_iter_),
            'samples_seen':int(model.t_),
            'accuracy':round(sum(a==b for a,b in zip(truth,pred))/len(test),5),
            'legal_majority_accuracy':round(sum(a==b for a,b in zip(truth,baseline))/len(test),5),
            'recall':{a:round(good[a]/count[a],4) if count[a] else None for a in ACTIONS}}
    groups=defaultdict(list)
    for (obs,label,world),picked in zip(test,pred):
        key=digest(json.dumps({k:v for k,v in obs.items() if k!='family'},sort_keys=True).encode())
        groups[(world,key)].append((obs['family'],label,picked))
    triples=[v for v in groups.values() if len(v)==3 and len({f for f,_,_ in v})==3]
    if triples:
        output['equal_state_triples']=len(triples)
        output['target_family_divergence']=round(sum(len({y for _,y,_ in t})>1 for t in triples)/len(triples),5)
        output['prediction_family_divergence']=round(sum(len({p for _,_,p in t})>1 for t in triples)/len(triples),5)
    return output


def main():
    import numpy,sklearn
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--v1',type=Path,required=True)
    ap.add_argument('--probe',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=24)
    a=ap.parse_args()
    if not 1<=a.epochs<=100 or a.output.exists():ap.error('bounded epochs and new output required')
    result={'schema':'cybersand/corpus-train-internal-sgd-probe-v1',
            'warning':'Internal world-group fold of TRAIN only; NOT official DEV, held-out or native evidence',
            'optimizer':'scikit-learn SGDClassifier/log_loss, CPU only, no checkpoint persisted',
            'runtime':{'numpy':numpy.__version__,'sklearn':sklearn.__version__,'epochs':a.epochs},
            'sources':{},'experiments':{}}
    for kind,path in (('v1',a.v1),('probe',a.probe)):
        rows=load(path,kind)
        result['sources'][kind]={'zip_sha256':digest(path.read_bytes()),'rows':len(rows),
                                 'world_groups':len({r[2] for r in rows}),
                                 'evidence_class':MEMBERS[kind][3]}
        result['experiments'][kind]={label:evaluate(rows,include,a.epochs)
                                    for label,include in (('without_family',False),('with_family',True))}
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result['experiments'],indent=2))


if __name__=='__main__':main()
