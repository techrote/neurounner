"""DG004: real-Qwen-state-conditioned, trained synthetic game-policy experiment.

A supervised two-scalar DG002 source readout is NOT direct cross-model native KV.
The four-way rule addressing is an explicit bias, NOT emergent LLM reasoning.
No CyberSand game implementation or content is modified.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from experiments.memplant.dreamgraft002.source import make_cases

SOURCE_SHA = '473cdb663922705cbddd99b57bfc5ea63048e113ec318e238574b2420511ae75'
SOURCE_RUN = 43736088404
TRAIN_OPS = (0,1,2,3,4,7,8,11,12,13,14,15)
VALID_OPS = (5,10)
TEST_OPS = (6,9)  # XOR, XNOR; unseen in training and checkpoint selection
SEED = 260116


def digest(b): return sha256(b).hexdigest()


def donor_scores(path):
    root = Path(path)
    manifest = json.loads((root/'manifest.json').read_text())
    cases = make_cases()
    if manifest['cases'] != cases: raise ValueError('DG002 source cases changed')
    archive = root/'features.npz'
    if digest(archive.read_bytes()) != manifest['feature_container_hash']:
        raise ValueError('DG002 NPZ archive digest mismatch')
    with np.load(archive, allow_pickle=False) as z:
        v = np.array(z['v'], dtype=np.float32, copy=True)
    if v.shape != (128,2048) or not np.isfinite(v).all():
        raise ValueError('DG002 Qwen V dimensions/finiteness failed')
    if digest(v.tobytes()) != SOURCE_SHA or digest(v.tobytes()) != manifest['channels']['v']['sha256']:
        raise ValueError('DG002 Qwen V REAL source provenance mismatch')
    v = torch.from_numpy(v)
    x = (v-v.mean(1,keepdim=True))/(v.std(1,keepdim=True,unbiased=False)+1e-6)
    train = [i for i,r in enumerate(cases) if r['split']=='train']
    y = torch.tensor([[1 if r['heat']=='hardens' else -1,
                       1 if r['vibration']=='floats' else -1] for r in cases],dtype=torch.float)
    alpha = torch.linalg.solve(x[train]@x[train].T/2048+torch.eye(len(train)),y[train])
    scores = x@x[train].T/2048@alpha
    return cases, scores.detach(), manifest


def laws(row): return int(row['heat']=='hardens'),int(row['vibration']=='floats')


def gate(op,h,v): return (op >> ((1-h)*2+(1-v))) & 1


def scenes(cases,split,ops,turns,seed):
    out=[]
    for i,row in enumerate(cases):
        if row['split']!=split: continue
        for op in ops:
            for turn in range(turns):
                # Same geometry across all four counterfactual law variants;
                # removes potential label leaks through pseudo-random terrain.
                key=f'{seed}|{row["family"]}|{op}|{turn}'
                r=random.Random(int(digest(key.encode())[:16],16))
                inv=r.randrange(2); actions=[]; legal=[]
                for j in range(4):
                    edge=j%2 if turn%3==0 else r.randrange(2)
                    actions.append([float(edge),r.random(),r.random(),r.random(),float(j==2)])
                    legal.append(not(r.random()<0.18 or r.random()<0.08))
                if not any(legal): legal[r.randrange(4)]=True
                out.append((i,op,turn,inv,actions,legal))
    return out


def target(s,row):
    _,op,_,inv,actions,legal=s
    desired=gate(op,*laws(row))^inv
    scores=[(3*(int(a[0])==desired)+.38*a[1]+.23*a[2]-.27*a[3]+.1*a[4])
            if legal[j] else -1e8 for j,a in enumerate(actions)]
    return max(range(4),key=lambda j:scores[j])


def batch(sc,cases,scores,mode='real'):
    idx={r['case_id']:i for i,r in enumerate(cases)}
    x=[]; tables=[];inv=[];act=[];mask=[];labels=[];ids=[]
    for s in sc:
        i,op,turn,flip,actions,legal=s
        row=cases[i]; injected=i
        if mode in ('heat_swap','vibration_swap','both_swap'):
            which={'heat_swap':'heat_twin','vibration_swap':'vibe_twin','both_swap':'both_twin'}[mode]
            injected=idx[row[which]]
        elif mode=='crossover':
            pool=[j for j,r in enumerate(cases) if r['split']==row['split'] and r['family']!=row['family']]
            injected=pool[(i+op+turn)%len(pool)]
        truth=cases[injected]
        state=scores[injected].clone()
        if mode=='crossover':
            h,_=laws(row);_,v=laws(truth)
            state[0]=scores[i,0]
            truth={'heat':'hardens' if h else 'softens','vibration':'floats' if v else 'sinks'}
        if mode=='oracle':
            state=torch.tensor([1 if k else -1 for k in laws(truth)],dtype=torch.float)/15
        if mode=='zero': state=torch.zeros(2)
        if mode=='shuffled':
            pool=[j for j,r in enumerate(cases) if r['split']==row['split']]
            j=pool[int(digest(f'shuffle|{i}|{op}|{turn}'.encode())[:16],16)%len(pool)]
            state=scores[j]
        x.append((state*15).clamp(-1,1).tolist())
        tables.append([float((op>>k)&1) for k in range(4)])
        inv.append([float(flip)]);act.append(actions);mask.append(legal)
        labels.append(target(s,truth));ids.append((row['case_id'],cases[injected]['case_id'],op,turn))
    return {'x':torch.tensor(x), 'table':torch.tensor(tables), 'inv':torch.tensor(inv),
            'action':torch.tensor(act), 'legal':torch.tensor(mask,dtype=torch.bool),
            'y':torch.tensor(labels,dtype=torch.long), 'identities':ids}


class Factored(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale=nn.Parameter(torch.ones(2)*2)
        self.offset=nn.Parameter(torch.zeros(2))
        self.head=nn.Sequential(nn.Linear(5,24),nn.Tanh(),nn.Linear(24,1))

    def forward(self,b):
        p=torch.sigmoid(b['x']*self.scale+self.offset)
        h,v=p[:,0],p[:,1]
        slot=torch.stack((h*v,h*(1-v),(1-h)*v,(1-h)*(1-v)),-1)
        g=(slot*b['table']).sum(-1)
        wanted=torch.where(b['inv'][:,0]>0.5,1-g,g)
        edge=b['action'][:,:,0]
        match=wanted[:,None]*edge+(1-wanted[:,None])*(1-edge)
        features=torch.cat((match[:,:,None],b['action'][:,:,1:]),-1)
        return self.head(features).squeeze(-1).masked_fill(~b['legal'],-1e4)


class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(12,32),nn.Tanh(),nn.Linear(32,1))

    def forward(self,b):
        common=torch.cat((b['x'],b['table'],b['inv']),-1)
        a=torch.cat((common[:,None,:].expand(-1,4,-1),b['action']),-1)
        return self.head(a).squeeze(-1).masked_fill(~b['legal'],-1e4)


def select(b,inds):
    return {k:v[inds] if isinstance(v,torch.Tensor) else v for k,v in b.items()}


def infer(model,b):
    model.eval()
    with torch.no_grad(): return model(b).argmax(-1).tolist()


def train(kind,b,valid,updates,seed):
    torch.manual_seed(seed)
    model=Factored() if kind=='factored' else Dense()
    optim=torch.optim.AdamW(model.parameters(),lr=.004,weight_decay=.0001)
    rng=torch.Generator().manual_seed(seed+1000)
    best=-1;history=[];weights=None;best_step=0
    for step in range(1,updates+1):
        model.train();ii=torch.randint(len(b['y']),(256,),generator=rng)
        item=select(b,ii);optim.zero_grad()
        loss=nn.functional.cross_entropy(model(item),item['y'])
        if not torch.isfinite(loss): raise RuntimeError('nonfinite optimizer loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);optim.step()
        if step%25==0 or step==updates:
            pred=infer(model,valid)
            a=sum(p==int(y) for p,y in zip(pred,valid['y']))/len(pred)
            history.append({'step':step,'loss':round(loss.detach().item(),5),'validation_accuracy':a})
            if a>best:
                best=a;best_step=step
                weights={k:v.detach().clone() for k,v in model.state_dict().items()}
    assert weights is not None
    model.load_state_dict(weights)
    return model, {'updates':updates,'best_step':best_step,'best_validation':best,'history':history}


def evaluate(model,b):
    predictions=infer(model,b);truth=b['y'].tolist()
    return {'correct':sum(a==y for a,y in zip(predictions,truth)),
            'n':len(truth), 'accuracy':sum(a==y for a,y in zip(predictions,truth))/len(truth),
            'predictions':predictions,'gold':truth}


def produce(source,out,updates=450,seed=SEED):
    torch.set_num_threads(min(4,torch.get_num_threads()))
    t=time.monotonic();cases,scores,source_manifest=donor_scores(source)
    tr=scenes(cases,'train',TRAIN_OPS,4,seed)
    va=scenes(cases,'validation',VALID_OPS,6,seed)
    te=scenes(cases,'sealed',TEST_OPS,8,seed)
    dest=Path(out);dest.mkdir(parents=True,exist_ok=True)
    report={'schema':'memplant-dg004-game-policy-v1','status':'STARTED',
            'source_run':SOURCE_RUN,'source_v_sha256':SOURCE_SHA,
            'donor_revision':source_manifest['donor']['revision'],
            'source_ridge':'DG002 lambda=1, 80 train cases ONLY',
            'train_operators':TRAIN_OPS,'valid_operators':VALID_OPS,'test_operators':TEST_OPS,
            'train_families':20,'valid_families':6,'test_families':6,
            'previously_inspected_test_source_families':True,
            'recipient_input_is_two_supervised_scalars':True,
            'test_task_labels_used_for_training_or_selection':False,
            'receiver_receives_donor_text_or_world_law_labels':False,
            'game': 'external deterministic synthetic CyberSand-like action choice, not native',
            'models':{}}
    models={}
    for name,kind,mode,salt in (('factored','factored','real',0),
                                ('zero_trained','factored','zero',51),
                                ('dense','dense','real',107)):
        model,hist=train(kind,batch(tr,cases,scores,mode),
                         batch(va,cases,scores,mode),updates,seed+salt)
        saved=dest/f'{name}.pt'
        torch.save({'state_dict':model.state_dict(),'kind':kind,'mode':mode,
                    'source_sha256':SOURCE_SHA},saved)
        hist['checkpoint_sha256']=digest(saved.read_bytes())
        hist['test']={k:v for k,v in evaluate(model,batch(te,cases,scores,mode)).items()
                      if k not in ('predictions','gold')}
        report['models'][name]=hist;models[name]=model
    factored=models['factored'];raw={};condition={}
    for mode in ('real','zero','shuffled','oracle','heat_swap','vibration_swap','both_swap','crossover'):
        b=batch(te,cases,scores,mode);ev=evaluate(factored,b)
        condition[mode]={k:v for k,v in ev.items() if k not in ('predictions','gold')}
        raw[mode]=[{'base':base,'injected':src,'operator':op,'turn':turn,
                    'predicted':p,'correct_action':g,'legal':s[-1]}
                   for (base,src,op,turn),p,g,s in zip(b['identities'],ev['predictions'],ev['gold'],te)]
    causal={}
    for mode in ('heat_swap','vibration_swap','both_swap','crossover'):
        pairs=[]
        for a,b in zip(raw['real'],raw[mode]):
            assert (a['base'],a['operator'],a['turn'])==(b['base'],b['operator'],b['turn'])
            pairs.append({'base':a['base'],'operator':a['operator'],'turn':a['turn'],
                          'expected_change':a['correct_action']!=b['correct_action'],
                          'observed_change':a['predicted']!=b['predicted'],
                          'both_correct':(a['correct_action']==a['predicted'] and
                                          b['correct_action']==b['predicted'])})
        causal[mode]={'n':len(pairs),'expected_changes':sum(x['expected_change'] for x in pairs),
                      'observed_changes':sum(x['observed_change'] for x in pairs),
                      'correct_change_flags':sum(x['expected_change']==x['observed_change'] for x in pairs),
                      'both_correct':sum(x['both_correct'] for x in pairs)}
    (dest/'raw_decisions.json').write_text(json.dumps(raw,indent=2)+'\n')
    report['conditions']=condition;report['causal']=causal
    report['raw_decisions_sha256']=digest((dest/'raw_decisions.json').read_bytes())
    report['elapsed_seconds']=round(time.monotonic()-t,2)
    report['status']='TRAINED_EXPLORATORY'
    report['limitations']={'native_KV_transfer':False,'two_scalar_supervised_channel':True,
                           'explicit_logic_inductive_bias':True,'genuine_game_receipts':False,
                           'two_parent_graft_is_scalar_only':True}
    (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'models':{k:v['test'] for k,v in report['models'].items()},
                      'conditions':condition,'causal':causal,'elapsed_seconds':report['elapsed_seconds']},indent=2))
    return report


def selftest():
    cases=make_cases()
    assert len(cases)==128 and len(TRAIN_OPS)==12
    assert not(set(TRAIN_OPS)&set(TEST_OPS) or set(VALID_OPS)&set(TEST_OPS))
    assert all(gate(6,h,v)==h^v and gate(9,h,v)==1-(h^v) for h in (0,1) for v in (0,1))
    src=torch.tensor([[1 if r['heat']=='hardens' else -1,
                       1 if r['vibration']=='floats' else -1] for r in cases],dtype=torch.float)/15
    test=scenes(cases,'sealed',TEST_OPS,3,SEED)
    a=batch(test,cases,src)
    assert all(any(s[-1]) for s in test)
    for kind in (Factored,Dense):
        choices=infer(kind(),a)
        assert all(s[-1][j] for s,j in zip(test,choices))
    for mode in ('heat_swap','vibration_swap','both_swap','crossover'):
        b=batch(test,cases,src,mode)
        assert torch.equal(a['action'],b['action']) and torch.equal(a['legal'],b['legal'])
        assert all(s[-1][int(y)] for s,y in zip(test,b['y']))
        if mode=='crossover':
            assert all(x[0].rsplit('-',1)[0]!=x[1].rsplit('-',1)[0] for x in b['identities'])
    tr=batch(scenes(cases,'train',TRAIN_OPS[:3],2,SEED),cases,src)
    val=batch(scenes(cases,'validation',VALID_OPS,2,SEED),cases,src)
    model,rec=train('factored',tr,val,25,SEED)
    assert rec['updates']==25 and 0<=rec['best_validation']<=1
    print('DG004 SOURCE/CAUSAL/LEGAL/AUTOGRAD SELFTEST PASSED')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--selftest',action='store_true')
    parser.add_argument('--source')
    parser.add_argument('--out')
    parser.add_argument('--updates',type=int,default=450)
    parser.add_argument('--seed',type=int,default=SEED)
    args=parser.parse_args()
    if args.selftest: selftest();return
    if not args.source or not args.out or not 25<=args.updates<=3000:
        parser.error('source, out and 25..3000 bounded optimizer steps required')
    produce(args.source,args.out,args.updates,args.seed)


if __name__=='__main__': main()
