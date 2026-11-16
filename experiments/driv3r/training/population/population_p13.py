"""P13: held-out relational-order memory and closed-loop four-decision missions.

Model-only CyberSand-like synthetic research. Never modifies the game. Uses P12's
frozen source-faithful World and host-legal action catalog. Historical cues
encode a hidden mission instruction; a trained policy must issue the full
E/W -> RETRACE -> remaining E/W -> STOP mission, not just classify one probe.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from population_lab import FEATURES, NICHES, draw_svg, stable_seed
from population_p10 import ACTIONS, require_torch, shuffled
from population_p12 import (HISTORY, GRAMMARS, split_seeds, pair, encode_history,
                            tensorize, world)
from population_p11 import build_model as build_gru

MODES = ('pairwise_order', 'recurrent_gru', 'shuffled_order', 'state_only')
PHASES = ('first_branch','retrace','second_branch','stop')


def snap(w, row, phase, target):
    if target not in w.legal():
        raise AssertionError(f'host-illegal mission teacher: {phase} {target}')
    x=np.zeros((len(ACTIONS),len(FEATURES)),np.float32)
    mask=np.zeros(len(ACTIONS),np.bool_)
    for a in w.legal():
        i=ACTIONS.index(a); x[i]=w.feature(a); mask[i]=True
    history=[v['action'] for v in w.history]
    return dict(xs=x,mask=mask,history=encode_history(history),
                niche=row['niche'],target=ACTIONS.index(target),
                phase=phase,seed=row['seed'],depth=row['depth'],
                grammar=row['grammar'],order_id=row['order_id'])


def episode_rows(seed,card,depth,grammar,niche):
    paired=pair(seed,card,depth,grammar,niche)
    out=[]
    for row in paired:
        w=deepcopy(row['world']); first=ACTIONS[row['target']]
        second='W' if first=='E' else 'E'
        sequence=(first,'RETRACE',second,'STOP')
        rows=[]
        for phase,action in zip(PHASES,sequence):
            rows.append(snap(w,row,phase,action))
            w.step(action)
        if not w.stopped or (w.painted & (w.obstacles|w.protected)):
            raise AssertionError('invalid synthetic mission result')
        out.append(rows)
    assert np.array_equal(out[0][0]['xs'],out[1][0]['xs'])
    assert np.array_equal(out[0][0]['history'][-8:],out[1][0]['history'][-8:])
    assert out[0][0]['target']!=out[1][0]['target']
    return out


def make_data(seeds,grammars,depths,cards=(0,1),niches=NICHES):
    # Flat 8-row blocks (two counterfactual histories x four mission steps).
    return [sample for seed in seeds for card in cards for depth in depths
            for grammar in grammars for niche in niches
            for trajectory in episode_rows(seed,card,depth,grammar,niche)
            for sample in trajectory]


def build_model(mode,width=40):
    torch,nn=require_torch()
    if mode not in MODES: raise ValueError(mode)
    if mode!='pairwise_order':
        return build_gru('state_only' if mode=='state_only' else 'token_gru',width)
    class PairwiseMemory(nn.Module):
        """Learnable classifier on direction-agnostic pairwise action ordering.

        Position/order information is supplied as an antisymmetric action-pair
        co-occurrence matrix; *which* relation matters is learned end-to-end
        from next-action loss. This inductive bias is not an emergent learned
        temporal representation and must be reported as such.
        """
        def __init__(self):
            super().__init__()
            self.width=width
            self.features=nn.Linear(len(FEATURES),width)
            self.niche=nn.Embedding(len(NICHES),width)
            self.rel=nn.Sequential(nn.Linear(len(ACTIONS)**2,width),nn.Tanh(),
                                   nn.Linear(width,width),nn.Tanh())
            self.fusion=nn.Sequential(nn.Linear(width*2,width),nn.Tanh(),
                                      nn.Linear(width,width),nn.Tanh())
            self.base=nn.Linear(width,1)
        def forward(self,xs,mask,hist,niche):
            # Each valid earlier token votes for an ordered pair. Token ID zero
            # is padding. Normalize each pair by the maximum available count.
            oh=torch.nn.functional.one_hot(hist,num_classes=len(ACTIONS)+1).float()[...,1:]
            prefix=oh.cumsum(dim=1)-oh
            pairs=torch.einsum('bti,btj->bij',prefix,oh)
            counts=oh.sum(dim=1)
            denom=counts.unsqueeze(2)*counts.unsqueeze(1)
            antisym=(pairs-pairs.transpose(1,2))/denom.clamp(min=1)
            ctx=self.rel(antisym.reshape(len(hist),-1))
            state=torch.tanh(self.features(xs))
            fused=self.fusion(torch.cat((self.niche(niche),ctx),dim=-1))
            logits=(state*fused.unsqueeze(1)).sum(-1)/(self.width**.5)+self.base(state).squeeze(-1)
            return logits.masked_fill(~mask,-1e9)
    return PairwiseMemory()


def transform(hist,mode,sham=None):
    if mode=='state_only': return hist*0
    if mode=='shuffled_order':
        if sham is None: raise ValueError('sham generator required')
        return shuffled(hist,sham)
    return hist


def fit(model,mode,train,val,seed,epochs=14,batch_size=128,lr=.004):
    torch,_=require_torch()
    g=torch.Generator().manual_seed(seed)
    sham=torch.Generator().manual_seed(stable_seed('p13',seed,mode,'shuffle'))
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
    curves=[]; updates=0
    for epoch in range(epochs):
        model.train();losses=[]
        for ids in torch.randperm(len(train[0]),generator=g).split(batch_size):
            X,M,H,N,Y=(q[ids] for q in train)
            logits=model(X,M,transform(H,mode,sham),N)
            loss=torch.nn.functional.cross_entropy(logits,Y)
            if not bool(torch.isfinite(loss)):
                raise AssertionError('nonfinite loss')
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            opt.step();updates+=1;losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            X,M,H,N,Y=val
            CE=float(torch.nn.functional.cross_entropy(model(X,M,transform(H,mode,sham),N),Y))
        curves.append(dict(epoch=epoch,train_ce=round(float(np.mean(losses)),6),validation_ce=round(CE,6)))
    return dict(epochs=epochs,updates=updates,curve=curves,
                parameters=sum(p.numel() for p in model.parameters()))


def evaluate_static(model,mode,rows):
    torch,_=require_torch()
    model.eval(); X,M,H,N,Y=tensorize(rows)
    sham=torch.Generator().manual_seed(stable_seed('p13',mode,'evaluate-sham'))
    with torch.no_grad():
        P=model(X,M,transform(H,mode,sham),N).argmax(-1).numpy()
    if not bool(M.numpy()[np.arange(len(rows)),P].all()):
        raise AssertionError('host-illegal model action')
    correct=P==Y.numpy()
    by_phase={phase:round(float(correct[np.array([r['phase']==phase for r in rows])].mean()),6)
              for phase in PHASES}
    # Each 8-row block is two identical initial states with opposite targets.
    first=np.array([i for i,r in enumerate(rows) if r['phase']=='first_branch'])
    pairs=correct[first].reshape(-1,2)
    return dict(accuracy=round(float(correct.mean()),6),
                first_branch_accuracy=by_phase['first_branch'],
                first_branch_both_correct=round(float(pairs.all(axis=1).mean()),6),
                by_phase=by_phase,legal_action_rate=1.0,
                decisions=len(rows),paired_worlds=len(first)//2)


def choose(model,mode,w,niche,history,sham):
    torch,_=require_torch()
    if not w.legal(): raise AssertionError('empty host menu')
    X=np.zeros((len(ACTIONS),len(FEATURES)),np.float32)
    M=np.zeros(len(ACTIONS),np.bool_)
    for a in w.legal():
        ix=ACTIONS.index(a);X[ix]=w.feature(a);M[ix]=True
    x=torch.tensor(X).unsqueeze(0)
    m=torch.tensor(M).unsqueeze(0)
    h=torch.tensor(encode_history(history)).unsqueeze(0)
    n=torch.tensor([niche])
    model.eval()
    with torch.no_grad():
        pred=model(x,m,transform(h,mode,sham),n).argmax(-1).item()
    action=ACTIONS[pred]
    if action not in w.legal():raise AssertionError('host-illegal generated action')
    return action


def rollout(model,mode,seeds,grammars,depths,out=None, gallery_pairs=6):
    """Run four autonomous model choices, without teacher forcing/recovery.

    Failure on a wrong first decision counts as a failed mission even if later
    actions look useful. Wrong early actions may terminate; trace is retained.
    """
    sham=require_torch()[0].Generator().manual_seed(stable_seed('p13',mode,'closed-loop'))
    episodes=[]
    for seed in seeds:
        for card in (0,1):
            for grammar in grammars:
                for depth in depths:
                    for niche in NICHES:
                        for ref in pair(seed,card,depth,grammar,niche):
                            w=deepcopy(ref['world'])
                            history=list(ref['precursor'])
                            target=ACTIONS[ref['target']]
                            desired=[target,'RETRACE','W' if target=='E' else 'E','STOP']
                            actual=[]
                            for tick in range(4):
                                if w.stopped: break
                                action=choose(model,mode,w,ref['niche'],history,sham)
                                actual.append(action)
                                w.step(action)
                                history.append(action)
                            success=actual==desired
                            record=dict(seed=seed,card=card,grammar=grammar,depth=depth,
                                        niche=NICHES[ref['niche']],order_id=ref['order_id'],
                                        target=desired,actions=actual,mission_success=success,
                                        steps=len(actual),painted=len(w.painted),
                                        stopped=w.stopped,
                                        legal_actions=True,trajectory=w.history)
                            if out and len(episodes)<gallery_pairs*2:
                                folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
                                stem=f'episode{len(episodes):03d}'
                                (folder/(stem+'.json')).write_text(json.dumps(record,indent=2)+'\n')
                                (folder/(stem+'.svg')).write_text(draw_svg(w))
                            episodes.append(record)
    rates={n:round(float(np.mean([e['mission_success'] for e in episodes if e['niche']==n])),6)
           for n in NICHES}
    summary=dict(episodes=len(episodes),mission_success=round(float(np.mean([e['mission_success'] for e in episodes])),6),
                 legal_action_rate=1.,by_niche=rates,
                 premature_stop=round(float(np.mean([len(e['actions'])<4 for e in episodes])),6),
                 mean_painted=round(float(np.mean([e['painted'] for e in episodes])),4))
    if out:
        (Path(out)/'mission_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def run(output,seed=651,train_worlds=32,validation_worlds=12,heldout_worlds=16,
        epochs=14,width=40):
    if not (8<=train_worlds<=80 and 4<=validation_worlds<=40 and
            4<=heldout_worlds<=60 and 3<=epochs<=30 and 16<=width<=96):
        raise ValueError('campaign outside bounded CPU research envelope')
    torch,_=require_torch();torch.set_num_threads(1)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'models').mkdir();(output/'gallery').mkdir()
    splits=split_seeds(seed,train_worlds,validation_worlds,heldout_worlds)
    train_rows=make_data(splits['train'],('train_a','train_b'),(5,7))
    val_rows=make_data(splits['validation'],('train_a','train_b'),(5,7))
    tests={
        'seen_gap7':make_data(splits['holdout'],('train_a','train_b'),(7,)),
        'unseen_grammar_gap7':make_data(splits['holdout'],('heldout_c',),(7,)),
        'unseen_grammar_gap9':make_data(splits['holdout'],('heldout_c',),(9,)),
    }
    T=tensorize(train_rows); V=tensorize(val_rows)
    results={};total=0
    for mode in MODES:
        torch.manual_seed(stable_seed('p13',seed,'matched-init'))
        model=build_model(mode,width)
        initial={k:v.detach().clone() for k,v in model.state_dict().items()}
        learning=fit(model,mode,T,V,stable_seed('p13',seed,'matched-order'),epochs)
        if not any(not torch.equal(initial[k],v) for k,v in model.state_dict().items()):
            raise AssertionError('optimizer failed to train model')
        checkpoint=output/'models'/f'{mode}.pt'
        torch.save(dict(format='population-p13',seed=seed,mode=mode,width=width,
                        actions=ACTIONS,features=FEATURES,niches=NICHES,
                        history=HISTORY,weights=model.state_dict()),checkpoint)
        learning['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        total+=learning['updates']
        cases={}
        for case,rows in tests.items():
            case_result=evaluate_static(model,mode,rows)
            mission=rollout(model,mode,splits['holdout'],('heldout_c',),(9,),
                            out=output/'gallery'/mode if case=='unseen_grammar_gap9' else None) if case=='unseen_grammar_gap9' else None
            cases[case]=dict(probe=case_result,mission=mission)
        results[mode]=dict(training=learning,heldout=cases)
    report=dict(format='population-p13-relational-mission',native=False,
                actual_game_agent=False,pretrained_slm=False,real_optimizer=True,
                seed=seed,training='CPU PyTorch AdamW',torch=torch.__version__,numpy=np.__version__,
                split_seeds=splits,train_rows=len(train_rows),validation_rows=len(val_rows),
                optimizer_updates=total,models=results,protocol={
                'train_grammars':['train_a','train_b'],'unseen_grammar':'heldout_c',
                'train_depths':[5,7],'unseen_depth':9,'history_tokens':HISTORY,
                'mission_phases':PHASES,'world':'P0 synthetic host-legal World'},
                limitations=['Not real CyberSand native episodes','Mission scripted by hidden cue teacher for training',
                  'Pairwise comparator has explicit human-designed second-order inductive bias',
                  'Four-step local closed-loop mission, not open-ended autonomous gameplay',
                  'No genuine self-play, material semantics, or pretrained SLM adapters'])
    (output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    with (output/'metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(('mode','case','static_accuracy','first_probe','paired_correct','mission_success','legal'))
        for mode,d in results.items():
            for case,v in d['heldout'].items():
                writer.writerow((mode,case,v['probe']['accuracy'],v['probe']['first_branch_accuracy'],
                                 v['probe']['first_branch_both_correct'],
                                 v['mission']['mission_success'] if v['mission'] else '',1))
    (output/'manifest.sha256').write_text(''.join(
        hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(output).as_posix()+'\n'
        for p in sorted(output.rglob('*')) if p.is_file() and p.name!='manifest.sha256'))
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--seed',type=int,default=651)
    parser.add_argument('--train-worlds',type=int,default=32)
    parser.add_argument('--validation-worlds',type=int,default=12)
    parser.add_argument('--heldout-worlds',type=int,default=16)
    parser.add_argument('--epochs',type=int,default=14)
    parser.add_argument('--width',type=int,default=40)
    x=run(**vars(parser.parse_args()))
    print(json.dumps(dict(seed=x['seed'],updates=x['optimizer_updates'],
        mission={k:v['heldout']['unseen_grammar_gap9']['mission']['mission_success']
                 for k,v in x['models'].items()},
        first={k:v['heldout']['unseen_grammar_gap9']['probe']['first_branch_accuracy']
               for k,v in x['models'].items()}),sort_keys=True))

if __name__=='__main__':main()