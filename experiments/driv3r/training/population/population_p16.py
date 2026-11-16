"""P16: controlled grammar extrapolation, explicit-vs-soft order memory.

CPU-only synthetic P0 host; never imports or writes CyberSand runtime. The
minimal monotone-cue model is an engineered inductive bias and is NOT evidence
of emergent relational logic. Every checkpoint and failed cohort is preserved.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, time
from pathlib import Path
import numpy as np
from population_lab import FEATURES,NICHES,stable_seed
from population_p10 import ACTIONS,require_torch
from population_p12 import split_seeds,tensorize
from population_p14 import dataset,fit,evaluate,rollout
from population_p15 import build_model as p15_model,counterfactual_interventions,write_manifest

EAST,WEST=ACTIONS.index('E'),ACTIONS.index('W')
GOAL=FEATURES.index('goal_progress')
MODES=('factorized_order_goal','soft_first_cue','learned_attention','attention_grammar_complete')


def build_model(mode,width=32):
    torch,nn=require_torch()
    if mode not in MODES or width%4 or not 16<=width<=96: raise ValueError(mode)
    if mode == 'factorized_order_goal':return p15_model(mode,width)
    if mode in ('learned_attention','attention_grammar_complete'):
        return p15_model('learned_attention',width)
    class SoftFirstCue(nn.Module):
        """Learn E/W token polarity with a monotone earliest-cue prior.

        E/W cue *selection* and goal-direction multiplication are engineered;
        cue sign, niche modulation, and positional sharpness are trained.
        The cue comparator is not a general SLM attention mechanism.
        """
        def __init__(self):
            super().__init__()
            self.local=nn.Sequential(nn.Linear(len(FEATURES),width),nn.Tanh(),nn.Linear(width,1))
            self.action_embedding=nn.Embedding(len(ACTIONS)+1,1,padding_idx=0)
            nn.init.normal_(self.action_embedding.weight,0,.2)
            self.niche=nn.Embedding(len(NICHES),1)
            nn.init.normal_(self.niche.weight,0,.2)
            self.temperature=nn.Parameter(torch.tensor(1.0))
            self.scale=nn.Parameter(torch.tensor(1.0))
        def forward(self,xs,mask,hist,niche):
            # Time order is ordinal left-to-right in right-aligned, padded history.
            valid=(hist==EAST+1)|(hist==WEST+1)
            pos=torch.arange(hist.shape[1],device=hist.device,dtype=xs.dtype)
            # Exponential preference for earlier valid direction; fully
            # differentiable temperature, with no 'E before W' comparison.
            scores=-torch.nn.functional.softplus(self.temperature)*pos
            scores=scores.unsqueeze(0).expand(hist.shape[0],-1)
            scores=scores.masked_fill(~valid,-1e4)
            cue=(torch.softmax(scores,dim=1)*self.action_embedding(hist).squeeze(-1)).sum(1)
            toward_e=(xs[:,EAST,GOAL]-xs[:,WEST,GOAL])*.5
            signed=cue*toward_e*self.niche(niche).squeeze(1)*self.scale
            out=self.local(xs).squeeze(-1)
            bias=torch.zeros_like(out)
            bias[:,EAST]=signed;bias[:,WEST]=-signed
            return (out+bias).masked_fill(~mask,-1e9)
    return SoftFirstCue()


def run(output,seeds=tuple(range(664,670)),train_worlds=4,validation_worlds=2,
        heldout_worlds=2,epochs=12,width=32,modes=MODES,gallery_limit=4):
    if (not 1<=len(seeds)<=12 or len(set(seeds))!=len(seeds) or
        not 2<=train_worlds<=20 or not 2<=validation_worlds<=12 or
        not 2<=heldout_worlds<=12 or not 3<=epochs<=25 or
        width%4 or not 16<=width<=96 or
        not modes or any(m not in MODES for m in modes) or not 0<=gallery_limit<=12):
        raise ValueError('outside CPU-bound P16 experimental envelope')
    torch,_=require_torch();torch.set_num_threads(1)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    records=[];started=time.monotonic()
    for seed in seeds:
        folder=output/f'cohort_{seed}';folder.mkdir()
        (folder/'models').mkdir();(folder/'gallery').mkdir()
        splits=split_seeds(seed,train_worlds,validation_worlds,heldout_worlds)
        train_base=dataset(splits['train'],('train_a','train_b'),(5,7),('train_low','train_high'))
        # Explicit extra-grammar curriculum is *not* eligible for an
        # unseen-grammar claim. It is a positive exposure comparator.
        train_plus=train_base+dataset(splits['train'],('heldout_c',),(5,7),('train_low','train_high'))
        val=dataset(splits['validation'],('train_a','train_b'),(5,),('train_low','train_high'))
        holdout=dataset(splits['holdout'],('heldout_c',),(9,),('heldout_far',))
        seen=dataset(splits['holdout'],('train_a',),(7,),('train_low',))
        V=tensorize(val);heldout_results={}
        split_sha256=hashlib.sha256(json.dumps(splits,sort_keys=True).encode()).hexdigest()
        for mode in modes:
            initial_seed=stable_seed('p16',seed,'initial',mode)
            optimizer_seed=stable_seed('p16',seed,'optimizer')
            torch.manual_seed(initial_seed)
            model=build_model(mode,width)
            initial={k:v.clone() for k,v in model.state_dict().items()}
            base='learned_attention' if mode=='attention_grammar_complete' else mode
            train=tensorize(train_plus if mode=='attention_grammar_complete' else train_base)
            t0=time.monotonic()
            fitinfo=fit(model,base,train,V,optimizer_seed,epochs)
            if not any(not torch.equal(initial[k],v) for k,v in model.state_dict().items()):
                raise AssertionError('no optimizer parameter change')
            ck=folder/'models'/f'{mode}.pt'
            torch.save(dict(format='population-p16',seed=seed,mode=mode,width=width,
                            actions=ACTIONS,features=FEATURES,state_dict=model.state_dict(),
                            source='synthetic-P0',native=False,grammar_exposed=(mode=='attention_grammar_complete')),ck)
            fitinfo['checkpoint_sha256']=hashlib.sha256(ck.read_bytes()).hexdigest()
            fitinfo['elapsed_s']=round(time.monotonic()-t0,3)
            # No held-out rows used for training or checkpoint selection.
            test=evaluate(model,base,holdout)
            control=evaluate(model,base,seen)
            counterfactual=counterfactual_interventions(model,holdout)
            gallery=folder/'gallery'/mode
            mission=rollout(model,base,splits['holdout'],('heldout_c',),(9,),('heldout_far',),gallery,gallery_limit=gallery_limit)
            heldout_results[mode]=dict(
                lineage=dict(parent='scratch',initial_seed=initial_seed,
                             optimizer_seed=optimizer_seed,split_sha256=split_sha256,
                             weight_mutation=False,adapter_mixing=False),
                training=fitinfo,heldout=test,seen=control,
                                      intervention=counterfactual,mission=mission,
                                      training_grammar=['train_a','train_b','heldout_c'] if mode=='attention_grammar_complete'
                                                      else ['train_a','train_b'])
            records.append(dict(seed=seed,mode=mode,updates=fitinfo['updates'],
                                first=test['first'],paired_both=test['paired_both'],
                                mission=mission['mission_success'],legal=mission['legal_rate'],
                                parameters=fitinfo['parameters'],elapsed_s=fitinfo['elapsed_s']))
        (folder/'results.json').write_text(json.dumps(dict(seed=seed,splits=splits,
            train_rows=len(train_base),extra_grammar_rows=len(train_plus)-len(train_base),
            split_sha256=split_sha256,
            validation_rows=len(val),heldout_rows=len(holdout),models=heldout_results),indent=2)+'\n')
        write_manifest(folder)
    aggregate={}
    for mode in modes:
        rs=[r for r in records if r['mode']==mode]
        aggregate[mode]=dict(first_mean=round(float(np.mean([r['first'] for r in rs])),6),
            paired_mean=round(float(np.mean([r['paired_both'] for r in rs])),6),
            mission_mean=round(float(np.mean([r['mission'] for r in rs])),6),
            perfect=sum(r['mission']==1 for r in rs),
            legal_min=min(r['legal'] for r in rs),updates=sum(r['updates'] for r in rs),
            parameters=rs[0]['parameters'],first_by_seed={str(r['seed']):r['first'] for r in rs})
    result=dict(format='population-p16-soft-ordinal-memory',native=False,
        pretrained_slm=False,trained=True,seeds=list(seeds),modes=list(modes),
        dataset=dict(train_worlds=train_worlds,validation_worlds=validation_worlds,
            heldout_worlds=heldout_worlds,epochs=epochs,width=width),
        torch=torch.__version__,numpy=np.__version__,aggregate=aggregate,
        cohorts=records,total_updates=sum(r['updates'] for r in records),
        elapsed_s=round(time.monotonic()-started,3),
        design='Matched seeds and data except explicitly labelled grammar-complete attention comparator',
        limitations=['P0 synthetic host, native game correspondence unverified',
            'Only four-step scripted goal-reversal missions, not autonomous self-play',
            'Soft cue explicitly gates E/W and multiplies engineered goal-direction feature',
            'Factorized baseline explicitly encodes E-before-W ordering',
            'Grammar-complete treatment trains on heldout_c and cannot claim unseen grammar',
            'Six niche labels encode XOR parity, not independent artistic personalities'])
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (output/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write_manifest(output)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    ap.add_argument('--seeds',nargs='+',type=int,default=list(range(664,670)))
    ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--train-worlds',type=int,default=4)
    ap.add_argument('--validation-worlds',type=int,default=2)
    ap.add_argument('--heldout-worlds',type=int,default=2)
    args=ap.parse_args()
    result=run(**vars(args))
    print(json.dumps(dict(seeds=result['seeds'],updates=result['total_updates'],
                          mission={m:r['mission_mean'] for m,r in result['aggregate'].items()}),sort_keys=True))
if __name__=='__main__':main()
