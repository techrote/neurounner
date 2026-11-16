"""P15: seed-robust goal-switch memory; scratch-trained synthetic CyberSand policy study.

Strictly model-side research using the P0 World host menu. No native game
integration. Preserves failed training seeds and complete fixed evaluation runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from population_lab import FEATURES, NICHES, stable_seed
from population_p10 import ACTIONS, require_torch
from population_p12 import split_seeds, tensorize
from population_p14 import (dataset, build_model as pairwise_model, fit,
                            evaluate, rollout)

MODES = ('pairwise_goal', 'factorized_order_goal', 'learned_attention',
         'shuffled_order', 'state_only')
GOAL = FEATURES.index('goal_progress')
EAST, WEST = ACTIONS.index('E'), ACTIONS.index('W')


def build_model(mode, width=32):
    torch, nn = require_torch()
    if mode not in MODES: raise ValueError(mode)
    if width % 4: raise ValueError('width must be multiple of four')
    if mode == 'pairwise_goal': return pairwise_model(mode, width)
    if mode == 'factorized_order_goal':
        class FactorizedOrderGoal(nn.Module):
            """Explicit branch-order and goal conjunction, learned niche sign.

            This grants engineered E/W action order and goal direction; it is
            an upper-bias comparator, NOT emergent temporal reasoning.
            """
            def __init__(self):
                super().__init__()
                self.local = nn.Sequential(nn.Linear(len(FEATURES),width), nn.Tanh(),
                                           nn.Linear(width,1))
                self.niche = nn.Embedding(len(NICHES),1)
                nn.init.normal_(self.niche.weight, 0, .2)
                self.scale = nn.Parameter(torch.tensor(1.))
            def forward(self,xs,mask,hist,niche):
                # For each pair of cue tokens, E before W contributes +1,
                # W before E contributes -1. Ignore padding (ID zero).
                e=(hist==EAST+1).float(); w=(hist==WEST+1).float()
                order=(e.cumsum(1)*w - w.cumsum(1)*e).sum(1)
                order=order / (e.sum(1)*w.sum(1)).clamp(min=1.)
                toward_e=(xs[:,EAST,GOAL]-xs[:,WEST,GOAL])*.5
                signed=order*toward_e*self.niche(niche).squeeze(1)*self.scale
                logits=self.local(xs).squeeze(-1)
                bias=torch.zeros_like(logits)
                bias[:,EAST]=signed
                bias[:,WEST]=-signed
                return (logits+bias).masked_fill(~mask,-1e9)
        return FactorizedOrderGoal()
    class TemporalAttention(nn.Module):
        """Trainable token/position cross-attention; no pairwise-order statistic."""
        def __init__(self):
            super().__init__()
            self.feature=nn.Linear(len(FEATURES),width)
            self.niche=nn.Embedding(len(NICHES),width)
            self.tokens=nn.Embedding(len(ACTIONS)+1,width,padding_idx=0)
            self.positions=nn.Embedding(32,width)
            self.attention=nn.MultiheadAttention(width,4,batch_first=True)
            self.fusion=nn.Sequential(nn.Linear(2*width,width),nn.Tanh())
            self.base=nn.Linear(width,1)
        def forward(self,xs,mask,hist,niche):
            local=torch.tanh(self.feature(xs))
            # Query depends on present legal candidate state and current niche.
            denom=mask.sum(1,keepdim=True).clamp(min=1)
            query=(local*mask.unsqueeze(-1)).sum(1)/denom + self.niche(niche)
            seq=self.tokens(hist)+self.positions(torch.arange(hist.shape[1],device=hist.device))
            pad=(hist==0)
            # A blank input is used for the state-only ablation. Leave the
            # final token as a dummy to avoid all-masked softmax NaNs.
            pad=pad.clone();pad[:,-1]=False
            memory,_=self.attention(query.unsqueeze(1),seq,seq,
                                    key_padding_mask=pad,need_weights=False)
            fused=self.fusion(torch.cat((query,memory.squeeze(1)),dim=-1))
            logits=(local*fused.unsqueeze(1)).sum(-1)/math.sqrt(local.shape[-1])
            return (logits+self.base(local).squeeze(-1)).masked_fill(~mask,-1e9)
    return TemporalAttention()


def counterfactual_interventions(model, rows):
    """First-step causal swaps: history, goal features, and both together.

    Rows must be P14 dataset() blocks: E/W goal sides, two counterfactual
    survey orders, and four teacher-forced actions per episode (16/block).
    Every intervention retains the original legal menu and target, and never
    reads metadata as a model input. The simultaneous swap is an XOR control.
    """
    torch,_=require_torch()
    if len(rows)%16: raise ValueError('not complete matched goal/order blocks')
    X,M,H,N,Y=tensorize(rows)
    first=torch.arange(0,len(rows),4)
    order=H.clone(); goal=X.clone()
    for begin in range(0,len(rows),8):
        order[[begin,begin+4]]=order[[begin+4,begin]]
    cols=[FEATURES.index('goal_progress'),FEATURES.index('guidance')]
    for begin in range(0,len(rows),16):
        for off in (0,4):
            a,b=begin+off,begin+8+off
            goal[a,:,cols]=X[b,:,cols]
            goal[b,:,cols]=X[a,:,cols]
    variants={
       'original':(X,H),
       'swap_history':(X,order),
       'swap_goal':(goal,H),
       'swap_both':(goal,order),
    }
    result={}
    model.eval()
    with torch.no_grad():
        for key,(features,history) in variants.items():
            pred=model(features,M,history,N).argmax(-1)
            if not M[torch.arange(len(rows)),pred].all():
                raise AssertionError('illegal causal probe')
            result[key]=round(float((pred[first]==Y[first]).float().mean()),6)
    return result


def run(output, seeds=tuple(range(658,664)), train_worlds=4, validation_worlds=2,
        heldout_worlds=2, epochs=12, width=32, modes=MODES):
    if (not 1 <= len(seeds) <= 16 or len(set(seeds)) != len(seeds) or
        not 3 <= epochs <= 25 or not 16 <= width <= 96 or width % 4 or
        not 2 <= train_worlds <= 24 or not 2 <= validation_worlds <= 16 or
        not 2 <= heldout_worlds <= 16 or not modes or any(m not in MODES for m in modes)):
        raise ValueError('outside bounded CPU research envelope')
    torch,_=require_torch();torch.set_num_threads(1)
    folder=Path(output);folder.mkdir(parents=True,exist_ok=False)
    records=[]
    all_start=time.monotonic()
    for seed in seeds:
        cohort=folder/f'cohort_{seed}';cohort.mkdir()
        (cohort/'models').mkdir();(cohort/'gallery').mkdir()
        splits=split_seeds(seed,train_worlds,validation_worlds,heldout_worlds)
        train=dataset(splits['train'],('train_a','train_b'),(5,7),('train_low','train_high'))
        validation=dataset(splits['validation'],('train_a','train_b'),(5,),('train_low','train_high'))
        cases={
            'seen': dataset(splits['holdout'],('train_a','train_b'),(7,),('train_low',)),
            'unseen_grammar': dataset(splits['holdout'],('heldout_c',),(7,),('train_high',)),
            'unseen_goal_gap': dataset(splits['holdout'],('heldout_c',),(9,),('heldout_far',))
        }
        tr,val=tensorize(train),tensorize(validation)
        outputs={}
        for mode in modes:
            torch.manual_seed(stable_seed('p15',seed,'initial',mode))
            model=build_model(mode,width)
            before={k:v.clone() for k,v in model.state_dict().items()}
            started=time.monotonic()
            info=fit(model,mode,tr,val,stable_seed('p15',seed,'optimizer'),epochs)
            if not any(not torch.equal(before[k],v) for k,v in model.state_dict().items()):
                raise AssertionError(f'{seed} {mode} model unchanged after training')
            ck=cohort/'models'/f'{mode}.pt'
            torch.save(dict(format='population-p15',seed=seed,mode=mode,
                            width=width,features=FEATURES,actions=ACTIONS,
                            state_dict=model.state_dict(),native=False),ck)
            info['sha256']=hashlib.sha256(ck.read_bytes()).hexdigest()
            info['elapsed_s']=round(time.monotonic()-started,3)
            tests={k:evaluate(model,mode,rows) for k,rows in cases.items()}
            interventions=(counterfactual_interventions(model,cases['unseen_goal_gap'])
                           if mode in ('pairwise_goal','factorized_order_goal','learned_attention')
                           else None)
            mission=rollout(model,mode,splits['holdout'],('heldout_c',),(9,),
                            ('heldout_far',),cohort/'gallery'/mode,gallery_limit=6)
            outputs[mode]=dict(training=info,tests=tests,interventions=interventions,closed_loop=mission)
            records.append(dict(seed=seed,mode=mode,
                                first=tests['unseen_goal_gap']['first'],
                                paired_both=tests['unseen_goal_gap']['paired_both'],
                                mission=mission['mission_success'],
                                legal=mission['legal_rate'],
                                updates=info['updates'],elapsed_s=info['elapsed_s'],
                                by_niche=mission['by_niche']))
        data=dict(seed=seed,splits=splits,train_rows=len(train),val_rows=len(validation),
                  cases={k:len(v) for k,v in cases.items()},models=outputs)
        (cohort/'results.json').write_text(json.dumps(data,indent=2)+'\n')
        write_manifest(cohort)
    aggregates={}
    for mode in modes:
        r=[x for x in records if x['mode']==mode]
        aggregates[mode]=dict(seeds=len(r),first_mean=round(float(np.mean([x['first'] for x in r])),6),
                              first_range=[min(x['first'] for x in r),max(x['first'] for x in r)],
                              paired_mean=round(float(np.mean([x['paired_both'] for x in r])),6),
                              mission_mean=round(float(np.mean([x['mission'] for x in r])),6),
                              mission_perfect=sum(x['mission']==1 for x in r),
                              legal_min=min(x['legal'] for x in r),
                              updates=sum(x['updates'] for x in r),
                              niche_success={n:round(float(np.mean([x['by_niche'][n] for x in r])),6)
                                             for n in NICHES})
    report=dict(format='population-p15-seed-replication',source='synthetic-P0',
                native=False,pretrained_slm=False,trained=True,
                matched_conditions=True,seeds=list(seeds),modes=list(modes),
                dataset={'train_worlds':train_worlds,'validation_worlds':validation_worlds,
                         'heldout_worlds':heldout_worlds,'epochs':epochs,'width':width},
                torch=torch.__version__,numpy=np.__version__,
                aggregate=aggregates,cohorts=records,
                total_updates=sum(x['updates'] for x in records),
                elapsed_s=round(time.monotonic()-all_start,3),
                limitations=['P0 synthetic host; not independently native-source-verified',
                             'Four-step scripted teacher, not self-play',
                             'Goal/order factorization is explicitly engineered',
                             'No SLM pretraining and no real artistic niche validation',
                             'Small correlated world families, results are not confidence bounds'])
    (folder/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    with (folder/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['seed','mode','first','paired_both','mission','legal','updates','elapsed_s'])
        writer.writeheader()
        for row in records:writer.writerow({k:row[k] for k in writer.fieldnames})
    write_manifest(folder)
    return report


def write_manifest(folder):
    (folder/'manifest.sha256').write_text(''.join(
        hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(folder).as_posix()+'\n'
        for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='manifest.sha256'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(658,664)))
    p.add_argument('--train-worlds',type=int,default=4)
    p.add_argument('--validation-worlds',type=int,default=2)
    p.add_argument('--heldout-worlds',type=int,default=2)
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--width',type=int,default=32)
    r=run(**vars(p.parse_args()))
    print(json.dumps({'seeds':r['seeds'],'updates':r['total_updates'],
                      'mission':{k:v['mission_mean'] for k,v in r['aggregate'].items()}},sort_keys=True))

if __name__=='__main__':main()
