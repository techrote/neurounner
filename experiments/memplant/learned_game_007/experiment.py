"""DG007: real-donor-state-conditioned LEARNED game-policy head.

Not native CyberSand gameplay; Qwen V source + DG005 decoder + DG006 editor
are frozen. Only a separately trained synthetic game action policy is new here.
No engine/Workbench/game files are changed.
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
from torch.nn import functional as F

from experiments.memplant.three_law.experiment import (
    SourceDecoder, Editor, Additive, advance, atomic, digest,
    sequence_set, truth_after,
)
from experiments.memplant.game_state_lab.game_policy import donor_scores, SOURCE_SHA, laws

N_ACTIONS=6
N_FEATURES=15
DIRS=((-1,0),(1,0),(0,-1),(0,1))
ACTIONS=('LEFT','RIGHT','UP','DOWN','PAINT','RETRACE')


def world(seed):
    rng=random.Random(seed)
    tiles=np.array([rng.choices([0,1,2,3],[42,24,22,12])[0] for _ in range(81)],dtype=np.int8).reshape(9,9)
    protect=np.array([rng.random()<.1 for _ in range(81)],dtype=np.bool_).reshape(9,9)
    cells=[(x,y) for x in range(9) for y in range(9) if tiles[y,x]!=3 and not protect[y,x]]
    start=rng.choice(cells);goal=rng.choice([c for c in cells if c!=start])
    return {'tiles':tiles,'protect':protect,'start':start,'goal':goal,'seed':seed}


def offers(w,pos,history,painted,heading):
    tiles,protected=w['tiles'],w['protect'];gx,gy=w['goal'];x,y=pos
    olddist=abs(gx-x)+abs(gy-y)
    f=np.zeros((N_ACTIONS,N_FEATURES),dtype=np.float32)
    mask=np.zeros(N_ACTIONS,dtype=np.bool_)
    for j in range(N_ACTIONS):
        nx,ny=(x+DIRS[j][0],y+DIRS[j][1]) if j<4 else ((history[-2] if len(history)>1 else pos) if j==5 else pos)
        inside=0<=nx<9 and 0<=ny<9
        tile=int(tiles[ny,nx]) if inside else 3
        mask[j]=inside and not protected[ny,nx] and tile!=3 if j<4 else (not protected[y,x] if j==4 else len(history)>1)
        newdist=abs(gx-nx)+abs(gy-ny)
        f[j,0]=(olddist-newdist)/2
        f[j,1]=newdist/16
        f[j,2]=float((nx,ny) not in history)
        f[j,3]=float(j==heading)
        f[j,4]=float(tile==0)
        f[j,5]=float(tile==1)
        f[j,6]=float(tile==2)
        f[j,7]=float((nx,ny)==w['goal'])
        f[j,8]=float(j==4)
        f[j,9]=float(j==5)
        f[j,10]=float(j<4)
        f[j,11]=float(pos in painted)
        f[j,12]=float(old_dist_near:=olddist<=2)
        f[j,13]=float(len(history)>2 and history[-2]==(nx,ny))
        f[j,14]=1.0
    assert mask.any(), 'Every supplied world must offer a legal action'
    return f,mask


def expert(law,features,mask):
    h,v,p=map(int,law)
    scores=np.full(N_ACTIONS,-1e7,dtype=np.float32)
    for i in range(N_ACTIONS):
        if not mask[i]:continue
        a=features[i]
        if i<4:
            scores[i]=1.9*a[0]+.33*a[2]+.06*a[3]+.28*a[7]-.18*a[13]
            scores[i]+=a[5]*(1.12 if h else -1.12)+a[6]*(1.12 if v else -1.12)
        elif i==4:
            # The protection/material law decides whether a new mark is useful;
            # host authority still forbids protected-cell painting for every law.
            scores[i]=.7+1.1*a[12]+.8*a[7]+(1.65 if (a[5] if p else a[4]) else -1.1)-2.5*a[11]
        else:
            scores[i]=-.65+.5*a[13]
    return int(np.argmax(scores))


def frames(seed,count,training=True):
    """World-separated, reproducible independent observations."""
    rng=random.Random(seed); fs=[]; masks=[]; states=[]; labels=[]; identities=[]
    for k in range(count):
        wi=seed*100000+k
        w=world(wi)
        candidates=[(x,y) for y in range(9) for x in range(9) if w['tiles'][y,x]!=3 and not w['protect'][y,x]]
        pos=rng.choice(candidates)
        hist=[pos]
        for _ in range(rng.randrange(4)):
            step=rng.choice(DIRS);n=(hist[-1][0]+step[0],hist[-1][1]+step[1])
            if n in candidates:hist.append(n)
        pos=hist[-1]
        painted={rng.choice(candidates) for _ in range(rng.randrange(4))}
        law=tuple(rng.randrange(2) for _ in range(3))
        f,m=offers(w,pos,hist,painted,rng.randrange(4))
        fs.append(f);masks.append(m);states.append(law);labels.append(expert(law,f,m));identities.append((wi,pos,w['goal']))
    return {'features':torch.from_numpy(np.stack(fs)), 'legal':torch.from_numpy(np.stack(masks)),
            'laws':torch.tensor(states,dtype=torch.float32), 'target':torch.tensor(labels,dtype=torch.long),
            'identities':identities}


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(N_FEATURES+3,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
    def forward(self,laws,features,legal):
        full=torch.cat((features,laws[:,None,:].expand(-1,N_ACTIONS,-1)),-1)
        return self.head(full).squeeze(-1).masked_fill(~legal,-1e4)


def pred(model,data,state=None,blind=False):
    with torch.no_grad():
        law=data['laws'] if state is None else state
        if blind:law=torch.zeros_like(law)+.5
        return model(law,data['features'],data['legal']).argmax(-1)


def accuracy(model,data,state=None,blind=False):
    p=pred(model,data,state,blind)
    assert bool(data['legal'].gather(1,p[:,None]).all())
    return float((p==data['target']).float().mean())


def train(seed,steps=500,blind=False):
    torch.manual_seed(seed+int(blind)*711)
    tr=frames(seed+10,4800);val=frames(seed+20,1000)
    policy=Policy();opt=torch.optim.AdamW(policy.parameters(),lr=.003,weight_decay=.00005)
    gen=torch.Generator().manual_seed(seed+1599)
    best=-1;weights=None;curve=[]
    for step in range(1,steps+1):
        ix=torch.randint(0,len(tr['target']),(256,),generator=gen)
        law=tr['laws'][ix] if not blind else torch.zeros_like(tr['laws'][ix])+.5
        opt.zero_grad()
        logits=policy(law,tr['features'][ix],tr['legal'][ix])
        loss=F.cross_entropy(logits,tr['target'][ix]);loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(),2)
        opt.step()
        if step%50==0 or step==steps:
            a=accuracy(policy,val,blind=blind)
            curve.append({'step':step,'loss':round(float(loss.detach()),6),'valid_accuracy':round(a,6)})
            if a>best:
                best=a;weights={k:v.detach().clone() for k,v in policy.state_dict().items()}
    policy.load_state_dict(weights);policy.eval()
    return policy,{'curve':curve,'best_validation':best,'synthetic_train_worlds':4800,'synthetic_valid_worlds':1000}


def donor_states(source,ref005,ref006,seed):
    cases,scores,manifest=donor_scores(source)
    torch.manual_seed(seed)
    dec=SourceDecoder();d=torch.load(Path(ref005)/'source_decoder.pt',map_location='cpu',weights_only=True)
    if d['source_v_sha256']!=SOURCE_SHA:raise ValueError('DG005 Qwen donor provenance mismatch')
    dec.load_state_dict(d['state_dict']);dec.eval()
    models={}
    for kind,cls in [('contextual',Editor),('additive',Additive)]:
        m=cls();o=torch.load(Path(ref006)/(kind+'.pt'),map_location='cpu',weights_only=True)
        if o['real_qwen_attention_v']!=SOURCE_SHA or o['experiment']!='DG006':raise ValueError('DG006 frozen model mismatch')
        m.load_state_dict(o['state_dict']);m.eval();models[kind]=m
    idx=[i for i,c in enumerate(cases) if c['split']=='sealed']
    true2=torch.tensor([laws(cases[i]) for i in idx],dtype=torch.float32)
    third=torch.tensor([((j*7+seed)%2) for j in range(len(idx))],dtype=torch.float32)[:,None]
    truth=torch.cat((true2,third),-1)
    with torch.no_grad():
        initial=torch.cat((dec((scores[idx]*15).clamp(-1,1)),(third*2-1)*6),-1)
    return cases,idx,truth,initial,models,{'donor_v_sha256':SOURCE_SHA,'dg005_decoder_sha256':digest(Path(ref005)/'source_decoder.pt'),
        'dg006_contextual_sha256':digest(Path(ref006)/'contextual.pt'),'dg006_additive_sha256':digest(Path(ref006)/'additive.pt')}


def test_inheritance(policy,donors,seed):
    cases,idx,truth,initial,parents,provenance=donors
    sequences=[seq for seq in sequence_set() if len(seq) in (4,8)]
    rows=[];totals={}
    # Consistent worlds and legal masks across modes and sequence variants.
    for seq_id,seq in enumerate(sequences):
        gold=torch.tensor([truth_after(tuple(map(int,t)),seq) for t in truth.tolist()],dtype=torch.float32)
        with torch.no_grad():
            zero=initial.clone();zero[:,:2]=0
            perm=torch.randperm(len(idx),generator=torch.Generator().manual_seed(seed+seq_id+222))
            shuffle=initial.clone();shuffle[:,:2]=initial[perm,:2]
            sources={'real':initial,'zero':zero,'shuffled':shuffle,'oracle':(2*truth-1)*6}
            predicted={}
            for model_name,model in parents.items():
                for mode,input_state in sources.items():
                    predicted[model_name+'/'+mode]=(torch.sigmoid(advance(model,input_state,seq))>=.5).float()
        feat=[];masks=[];labels=[]
        for j,i in enumerate(idx):
            w=world(seed*1000000+seq_id*1000+j)
            f,m=offers(w,w['start'],[w['start']],set(),j%4)
            feat.append(f);masks.append(m);labels.append(expert(gold[j],f,m))
        f=torch.from_numpy(np.stack(feat));mask=torch.from_numpy(np.stack(masks));target=torch.tensor(labels)
        for name,state in predicted.items():
            with torch.no_grad():choice=policy(state,f,mask).argmax(-1)
            assert bool(mask.gather(1,choice[:,None]).all())
            key=name+'/'+str(len(seq))
            t=totals.setdefault(key,{'actions_right':0,'state_right':0,'n':0,'action_by_class':[0]*N_ACTIONS})
            t['actions_right']+=int((choice==target).sum());t['state_right']+=int((state==gold).all(-1).sum());t['n']+=len(idx)
            for c in choice.tolist():t['action_by_class'][c]+=1
        if seq_id<3:
            rows.append({'seq':list(seq),'world_seed':seed*1000000+seq_id*1000,
                'source_cases':[cases[i]['case_id'] for i in idx[:3]],
                'truth':gold[:3].int().tolist(),'real_contextual':predicted['contextual/real'][:3].int().tolist(),
                'oracle_actions':target[:3].tolist()})
    for t in totals.values():
        t['action_accuracy']=round(t['actions_right']/t['n'],6)
        t['state_accuracy']=round(t['state_right']/t['n'],6)
    return totals,rows,provenance


def paired_counterfactual(policy,donors,seed):
    """Paired counterfactual laws on IDENTICAL scenes, with raw change flags."""
    cases,idx,truth,initial,parents,_=donors
    sequences=(((0,1,2,0),(2,1,0,0)),
               ((0,1,2,0,1,2,0,1),(1,0,2,1,0,2,1,0)))
    result={}
    for name,model in parents.items():
        for mode in ('real','zero','oracle'):
            data=[]
            for pair_id,(left,right) in enumerate(sequences):
                with torch.no_grad():
                    injected=initial.clone()
                    if mode=='zero':injected[:,:2]=0
                    if mode=='oracle':injected=(2*truth-1)*6
                    a=(torch.sigmoid(advance(model,injected,left))>=.5).float()
                    b=(torch.sigmoid(advance(model,injected,right))>=.5).float()
                for j,i in enumerate(idx):
                    w=world(seed*300000+pair_id*1000+j)
                    f,m=offers(w,w['start'],[w['start']],set(),j%4)
                    f=torch.tensor(f[None]);m=torch.tensor(m[None]);
                    ga=expert(truth_after(tuple(map(int,truth[j])),left),f[0].numpy(),m[0].numpy())
                    gb=expert(truth_after(tuple(map(int,truth[j])),right),f[0].numpy(),m[0].numpy())
                    with torch.no_grad():
                        pa=int(policy(a[j:j+1],f,m).argmax(-1))
                        pb=int(policy(b[j:j+1],f,m).argmax(-1))
                    assert m[0,pa] and m[0,pb]
                    data.append((ga!=gb,pa!=pb,pa==ga and pb==gb))
            key=name+'/'+mode
            result[key]={'n':len(data),'expected_changes':sum(q[0] for q in data),
                'observed_changes':sum(q[1] for q in data),
                'change_flag_accuracy':round(sum(q[0]==q[1] for q in data)/len(data),6),
                'paired_both_actions_correct':sum(q[2] for q in data)}
    return result


def render_rollout(policy,law,seed,out):
    from PIL import Image,ImageDraw
    w=world(seed);pos=w['start'];hist=[pos];painted=set();heading=0;actions=[]
    for step in range(22):
        f,legal=offers(w,pos,hist,painted,heading)
        with torch.no_grad():
            ix=int(policy(torch.tensor([law],dtype=torch.float32),torch.tensor(f[None]),torch.tensor(legal[None])).argmax(-1))
        assert legal[ix]
        actions.append(ACTIONS[ix])
        if ix<4:
            pos=(pos[0]+DIRS[ix][0],pos[1]+DIRS[ix][1]);hist.append(pos);heading=ix
        elif ix==4:painted.add(pos)
        elif len(hist)>1:hist.pop();pos=hist[-1]
    im=Image.new('RGB',(9*24,9*24),(20,20,20));d=ImageDraw.Draw(im)
    colours=[(128,132,136),(187,156,95),(64,115,159),(25,29,32)]
    for y in range(9):
        for x in range(9):
            c=(x,y);v=colours[int(w['tiles'][y,x])]
            if w['protect'][y,x]:v=(90,41,53)
            if c in painted:v=(220,179,76)
            d.rectangle((x*24,y*24,x*24+23,y*24+23),fill=v,outline=(37,40,43))
    for k,(x,y) in enumerate(hist):d.ellipse((x*24+6,y*24+6,x*24+17,y*24+17),fill=(122,205,142) if k<len(hist)-1 else (255,255,255))
    gx,gy=w['goal'];d.rectangle((gx*24+8,gy*24+8,gx*24+15,gy*24+15),fill=(234,71,67))
    im.save(out)
    return {'seed':seed,'true_law':list(law),'moves':actions,'final_position':pos,'painted':len(painted),'image_sha256':digest(out)}


def produce(source,ref005,ref006,out,seed,updates):
    if not 100<=updates<=1500:raise ValueError('updates out of budget')
    torch.set_num_threads(2);t0=time.monotonic();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    policy,record=train(seed,steps=updates,blind=False)
    blind,blindrec=train(seed,steps=updates,blind=True)
    ck={}
    for name,model in [('game_policy',policy),('blind_policy',blind)]:
        path=out/(name+'.pt');torch.save({'model':model.state_dict(),'kind':name,'seed':seed,'schema':'DG007'},path);ck[name]=digest(path)
    sealed=frames(seed+30,1800)
    sealed_accuracy={'learned_oracle_laws':round(accuracy(policy,sealed),6),'blind_laws':round(accuracy(blind,sealed,blind=True),6)}
    donor=donor_states(source,ref005,ref006,seed)
    totals,receipts,provenance=test_inheritance(policy,donor,seed)
    counterfactual=paired_counterfactual(policy,donor,seed)
    imgs=[]
    for i,law in enumerate([(1,0,1),(0,1,0),(1,1,1)]):
        path=out/f'world-{i}.png';imgs.append(render_rollout(policy,law,seed+970+i,path))
    report={'schema':'DG007-trained-game-action-inheritance-v1','seed':seed,'updates_each_policy':updates,
        'model_training':{'learned':record,'blind':blindrec},'sealed_worlds':1800,
        'sealed_accuracy':sealed_accuracy,'donor_provenance':provenance,'held_out_dg006_sequences':[4,8],
        'source_modes':totals,'matched_world_counterfactuals':counterfactual,'sample_receipts':receipts,'sample_synthetic_worlds':imgs,'checkpoints_sha256':ck,
        'elapsed_seconds':round(time.monotonic()-t0,2),
        'limitations':['3-bit manually specified synthetic laws','true law labels train policy','no native CyberSand gameplay',
          'state parent supervised atomic truth table','historically inspected Qwen held-out source families',
          'test sequences withheld from DG006 editor training, but not independent law universe',
          'synthetic host-authority legality independent of donor state','two authentic donor-derived bits plus synthetic third bit',
          'no cross-model native KV transplant']}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'sealed':sealed_accuracy,'source_modes':totals,'counterfactuals':counterfactual,'secs':report['elapsed_seconds']},indent=2))
    return report


def selftest():
    a=frames(261,32);b=frames(261,32)
    assert torch.equal(a['features'],b['features']) and torch.equal(a['target'],b['target'])
    assert not torch.equal(a['target'],frames(262,32)['target'])
    assert all(bool(a['legal'][j,int(y)]) for j,y in enumerate(a['target']))
    p=Policy();choices=pred(p,a);assert bool(a['legal'].gather(1,choices[:,None]).all())
    for law in ((0,0,0),(1,1,1)):
        w=world(781);f,m=offers(w,w['start'],[w['start']],set(),0)
        assert m.any() and m[4];assert m[expert(law,f,m)]
    model,r=train(311,steps=5);assert len(r['curve'])==1 and 0<=accuracy(model,a)<=1
    print('DG007 DETERMINISTIC WORLDS / HOST LEGALITY / OPTIMIZER SELFTEST PASS')


if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--selftest',action='store_true')
    pa.add_argument('--source');pa.add_argument('--reference-dg005');pa.add_argument('--reference-dg006');pa.add_argument('--out')
    pa.add_argument('--seed',type=int,default=26011609);pa.add_argument('--updates',type=int,default=500)
    a=pa.parse_args()
    if a.selftest:selftest()
    else:
        if not (a.source and a.reference_dg005 and a.reference_dg006 and a.out):pa.error('source + both frozen parent checkpoints + out required')
        produce(a.source,a.reference_dg005,a.reference_dg006,a.out,a.seed,a.updates)
