"""DG006: three noncommuting game-law edits from real donor-state-derived scalars.

Extends DG005. Qwen2.5-0.5B attention-V -> frozen supervised DG005 decoder
(two bits); protected/material rule is an external synthetic observation.
Only atomic edits are trained; all composed sequences are held out.
Research-only: not native KV graft, real-game inference or physics discovery.
"""
import argparse
from hashlib import sha256
import itertools
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.memplant.game_state_lab.game_policy import donor_scores, SOURCE_SHA, laws

OPS = "HVP"

def digest(p):
    return sha256(Path(p).read_bytes()).hexdigest()

class SourceDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(2,24),nn.Tanh(),nn.Linear(24,2))
    def forward(self,x):
        return self.net(x)

class Editor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(6,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,3))
    def forward(self,state,op):
        return self.net(torch.cat((torch.sigmoid(state),F.one_hot(op,3).float()),-1))

class Additive(nn.Module):
    def __init__(self):
        super().__init__()
        self.delta=nn.Parameter(torch.zeros(3,3))
    def forward(self,state,op):
        return state + self.delta[op]

def atomic(s,op):
    h,v,p=s
    if op==0: return (1-h, v^h, p)
    if op==1: return (h^v, 1-v, p^(h&v))
    if op==2: return (h^p, v^p, 1-p)
    raise ValueError("Invalid edit")

def truth_after(s,seq):
    for op in seq:s=atomic(s,op)
    return s

def train(model,seed,updates):
    torch.manual_seed(seed)
    gen=torch.Generator().manual_seed(seed+41)
    optim=torch.optim.AdamW(model.parameters(),lr=.008,weight_decay=0)
    history=[]
    for step in range(1,updates+1):
        bits=torch.randint(2,(192,3),generator=gen)
        ops=torch.randint(3,(192,),generator=gen)
        desired=torch.tensor([atomic(tuple(b),int(o)) for b,o in zip(bits.tolist(),ops.tolist())],dtype=torch.float)
        confidence=1+4*torch.rand((192,3),generator=gen)
        logits=(2*bits.float()-1)*confidence
        optim.zero_grad()
        loss=F.binary_cross_entropy_with_logits(model(logits,ops),desired)
        loss.backward();optim.step()
        if step%100==0 or step==updates:
            history.append([step,round(float(loss.detach()),5)])
    return model,history

def advance(model,initial,seq):
    x=initial
    for op in seq:
        x=model(x,torch.full((len(x),),op,dtype=torch.long))
    return x

def sequence_set():
    rng=random.Random(606);seqs=[];seen=set()
    for n in (2,3,4,6,8):
        for _ in range(8):
            while True:
                seq=tuple(rng.randrange(3) for _ in range(n))
                if seq not in seen:break
            seqs.append(seq);seen.add(seq)
    return seqs

def game_scene(cases,i,seq,seed,turn):
    # Shared external geometry for every competing state hypothesis.
    key=f'dg006|{seed}|{cases[i]["family"]}|{turn}|{seq}'
    r=random.Random(int(sha256(key.encode()).hexdigest()[:16],16))
    actions=[[r.random() for _ in range(4)] for _ in range(5)]
    legal=[r.random()>0.2 for _ in range(5)]
    if not any(legal):legal[0]=True
    return (i,turn,actions,legal)

def choose_action(bits,scene):
    _,_,actions,legal=scene
    h,v,p=[int(x) for x in bits]
    wanted=(h^(v&p))^(int(actions[0][1]>0.5)&p)
    values=[]
    for i,a in enumerate(actions):
        valid=legal[i] and not(p and a[2]>0.7 and a[3]>0.5)
        values.append((2*(int(a[0]>0.5)==wanted)+0.22*a[1]+0.11*a[2]-0.15*a[3]) if valid else -1e6)
    if all(v==-1e6 for v in values):
        values=[-1e6 if not legal[i] else 0 for i in range(len(legal))]
    return max(range(len(values)),key=values.__getitem__)

def evaluate(model,decoder,cases,y,scores,seed):
    idx=[i for i,r in enumerate(cases) if r['split']=='sealed']
    world=torch.tensor([((j*7+seed)%2) for j in range(len(idx))],dtype=torch.float)[:,None]
    truth=torch.cat((y[idx],world),-1)
    torch.manual_seed(seed+77)
    with torch.no_grad():
        initial=torch.cat((decoder(scores[idx]),(2*world-1)*6),-1)
        zero=initial.clone();zero[:,:2]=0
        shuffled=initial.clone();shuffled[:,:2]=initial[torch.randperm(len(initial)),:2]
        oracle=(2*truth-1)*6
        results={}
        for mode,x in (('real',initial),('zero',zero),('shuffled',shuffled),('oracle',oracle)):
            lengths={}
            for n in (2,3,4,6,8):
                joint=0;ntotal=0;right=0;nactions=0
                for seq in (s for s in sequence_set() if len(s)==n):
                    gold=torch.tensor([truth_after(tuple(map(int,row)),seq) for row in truth.tolist()])
                    pred=(torch.sigmoid(advance(model,x,seq))>=.5).int()
                    joint+=int((gold==pred).all(-1).sum());ntotal+=len(idx)
                    for j,i in enumerate(idx):
                        scene=game_scene(cases,i,seq,seed,0)
                        g=choose_action(gold[j],scene);p=choose_action(pred[j],scene)
                        assert scene[-1][p]
                        right+=g==p;nactions+=1
                lengths[str(n)]={'joint_state_accuracy':round(joint/ntotal,6),
                    'state_n':ntotal,'game_action_accuracy':round(right/nactions,6),
                    'action_n':nactions}
            results[mode]=lengths
    return results

def produce(source,reference,out,seed,updates):
    if not 100<=updates<=2500:raise ValueError("updates must be 100..2500")
    torch.set_num_threads(2)
    start=time.monotonic()
    cases,scores,manifest=donor_scores(source)  # validates authentic source V bytes
    y=torch.tensor([laws(c) for c in cases],dtype=torch.float)
    scores=(scores*15).clamp(-1,1)
    checkpoint=Path(reference)/'source_decoder.pt'
    raw=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if raw['source_v_sha256']!=SOURCE_SHA:raise ValueError("decoder donor mismatch")
    decoder=SourceDecoder()
    decoder.load_state_dict(raw['state_dict']);decoder.eval()
    output=Path(out);output.mkdir(parents=True,exist_ok=True)
    models={};checksums={}
    for name,model in (('contextual',Editor()),('additive',Additive())):
        fitted,curve=train(model,seed+(name=='additive'),updates)
        fitted.eval()
        dst=output/(name+'.pt')
        torch.save({'state_dict':fitted.state_dict(),'seed':seed,
                    'experiment':'DG006','atomic_only':True,
                    'real_qwen_attention_v':SOURCE_SHA},dst)
        checksums[name]=digest(dst)
        models[name]={'atomic_loss':curve,
                      'sequences':evaluate(fitted,decoder,cases,y,scores,seed)}
    report={'schema':'DG006-three-law-counterfactual-v1','seed':seed,'updates':updates,
        'donor_run':43736088404,'qwen_V_sha256':SOURCE_SHA,
        'dg005_source_decoder_sha256':digest(checkpoint),
        'source_manifest_sha256':digest(Path(source)/'manifest.json'),
        'checkpoint_sha256':checksums,'trained':'only individually supervised H,V,P atomic edits',
        'sequences_held_out':[2,3,4,6,8],
        'world_third_law':'external observed synthetic protected/material bit',
        'sealed_source_families_previously_inspected':True,'results':models,
        'elapsed_seconds':round(time.monotonic()-start,2),
        'limits':['two supervised Qwen-derived bits','symbolic edit targets',
                  'synthetic action projection','no native model/KV transfer',
                  'not independent source-held-out validation','no CyberSand game edits']}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{m:{n:v['joint_state_accuracy'] for n,v in d.items()}
                       for m,d in x['sequences'].items()} for k,x in models.items()},indent=2))
    return report

def selftest():
    states=list(itertools.product((0,1),repeat=3))
    assert any(truth_after(s,(0,1))!=truth_after(s,(1,0)) for s in states)
    assert all(q in (0,1) for s in states for op in range(3) for q in atomic(s,op))
    m=Additive();x=torch.randn(4,3);a=torch.zeros(4,dtype=torch.long);b=torch.ones(4,dtype=torch.long)
    assert torch.allclose(m(m(x,a),b),m(m(x,b),a))
    for i in range(3):
        s=(0,0,[[.3,.5,.8,.7]]*5,[False,False,True,False,False])
        assert choose_action((i%2,0,1),s)==2
    assert len(sequence_set())==40
    print("DG006 ATOMIC / NONCOMMUTING / LEGAL / SEQUENCE SELFTEST PASS")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--selftest',action='store_true')
    parser.add_argument('--source');parser.add_argument('--reference');parser.add_argument('--out')
    parser.add_argument('--seed',type=int,default=26011607)
    parser.add_argument('--updates',type=int,default=300)
    args=parser.parse_args()
    if args.selftest:selftest()
    else:
        if not args.source or not args.reference or not args.out:parser.error('source/reference/out required')
        produce(args.source,args.reference,args.out,args.seed,args.updates)
