"""P11: matched *real* W16 auxiliary-head inference with STOP-only clock surgery.

Keep all five movement logits from the original clock-intact forward. Replace
ONLY the STOP logit with a second inference under a counterfactual clock basis.
The decoder still uses the source-pinned W0 host legal menu and independent
STOP vs conditional-movement CRNs. This is NOT root Sprocket inference,
not native CyberSand, not reward learning, and NOT a human preference judgment.

The 15 clock inputs occupy W9 model-feature indices [107:122]. The action
input/architecture/checkpoint are immutable and independently replay checked.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import torch
import p10_live_head as src
from p8_stop_retention import ACTIONS, FAMILIES, choice, stop_probability, digest

MODES=('intact','stop_zero_clock','stop_frozen_clock','stop_no_periodics',
       'stop_no_linear','full_zero_clock')
ORIGINAL_SEEDS=src.SEEDS
HELDOUT_SEEDS=(2711,2729,2741,2753,2767,2789,2801,2819,2833)
TEMP=0.7


def get_logits(head, obs, mode):
    if mode not in MODES: raise ValueError('unsupported clock surgery mode')
    orig=src.model_features(obs,'intact')
    if mode=='intact':
        with torch.no_grad(): raw=head(torch.tensor([orig],dtype=torch.float32))[0].tolist()
        return raw,raw,1
    other=list(orig)
    if mode=='stop_zero_clock' or mode=='full_zero_clock':
        other[107:122]=src.model_features(obs,'zeroed')[107:122]
    elif mode=='stop_frozen_clock':
        other[107:122]=src.model_features(obs,'frozen_t0')[107:122]
    elif mode=='stop_no_periodics':
        other[107:121]=[0.]*14
    elif mode=='stop_no_linear':
        other[121]=0.
    with torch.no_grad():
        logits=head(torch.tensor([orig,other],dtype=torch.float32)).tolist()
    full,ablated=logits
    effective=ablated if mode=='full_zero_clock' else [*full[:5],ablated[5]]
    if mode!='full_zero_clock' and effective[:5]!=full[:5]:raise AssertionError('STOP-only changed movement logits')
    return full,effective,2


def rollout(w0,head,seed,family,gate,mode):
    if gate not in ('late','open'):raise ValueError('invalid STOP-gate condition')
    decisions=[]
    def selector(obs):
        full,used,n=get_logits(head,obs,mode)
        move_u=src.deterministic_uniform(seed,family,obs['tick'],'w11-CRN')
        stop_u=src.deterministic_uniform(seed,family,obs['tick'],'p10-independent-STOP-CRN-v1')
        p=stop_probability(used,obs['legal'],TEMP)
        if (gate=='open' or obs['tick']>=124) and stop_u<p:
            action='STOP'
        else:
            nonstop=[a for a in obs['legal'] if a!='STOP']
            action=choice(used,nonstop,TEMP,move_u) if nonstop else 'STOP'
        decisions.append({'tick':obs['tick'],'host_legal':obs['legal'],
            'full_model_logits':full,'effective_stop_logit':used[5],
            'original_stop_logit':full[5],
            'stop_p':p,'stop_draw':stop_u,'move_draw':move_u,
            'action':action,'forwards':n,'aux_head':True,'root_slm':False})
        return action
    rec=w0.run(seed,'callback',family,128,selector=selector)
    if len(rec['events'])!=len(decisions) or any(
        not ev['raw_legal'] or ev['applied']!=dec['action'] or
        ev['observation']['legal']!=dec['host_legal'] or
        ev['observation']['tick']!=dec['tick']
        for ev,dec in zip(rec['events'],decisions)):
        raise ValueError('host did not apply exact legal model decisions')
    rec['mode']=f'P11-{mode}-{gate}';rec['engine']='frozen-PyTorch-W16-aux-head-in-exact-pinned-synthetic-W0'
    rec['head_decisions']=decisions
    return rec


def paired(late,open_,mode):
    a=[e['applied'] for e in late['events']]
    b=[e['applied'] for e in open_['events']]
    if b==a:
        first=None
    elif len(b)<=len(a) and b and b[-1]=='STOP' and b[:-1]==a[:len(b)-1] and (
          open_['trajectory'][:-1]==late['trajectory'][:len(b)]):
        first=len(b)-1
    else:
        raise ValueError('STOP-only gate caused pre-termination movement divergence')
    if open_['metrics']['new_steps']>late['metrics']['new_steps'] or open_['metrics']['branches']>late['metrics']['branches']:
        raise ValueError('STOP-open increased construction against matched prefix')
    return {'mode':mode,'first_difference_tick':first,
        'pre_gated_stop':first is not None and first<124,
        'lost_novel':late['metrics']['new_steps']-open_['metrics']['new_steps'],
        'lost_branches':late['metrics']['branches']-open_['metrics']['branches']}


def run(source, out, seeds=ORIGINAL_SEEDS+HELDOUT_SEEDS):
    torch.set_num_threads(1)
    z,manifest,w0,head,ck=src.open_original(source)
    controls=src.replay_original(z,manifest,w0,head)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[];pairs=[]
    for family in FAMILIES:
        for seed in seeds:
            groups={}
            for mode in MODES:
                for gate in ('late','open'):
                    rec=rollout(w0,head,seed,family,gate,mode)
                    stem=f'{family}-{seed}-{mode}-{gate}'
                    trace=json.dumps(rec,sort_keys=True,separators=(',',':')).encode()+b'\n'
                    png=w0.png_bytes(rec)
                    (out/(stem+'.json')).write_bytes(trace)
                    (out/(stem+'.png')).write_bytes(png)
                    groups[(mode,gate)]=rec
                    rows.append({'family':family,'seed':seed,'cohort':'discovery' if seed in ORIGINAL_SEEDS else 'heldout',
                        'mode':mode,'gate':gate,'steps':rec['steps_executed'],
                        'stopped':rec['metrics']['terminated_by_stop'],
                        'early_stop':rec['metrics']['terminated_by_stop'] and rec['steps_executed']<=124,
                        'stop_tick':rec['steps_executed']-1 if rec['metrics']['terminated_by_stop'] else None,
                        'new':rec['metrics']['new_steps'],'branches':rec['metrics']['branches'],
                        'retraces':rec['metrics']['retraces'],
                        'invalid':rec['metrics']['invalid_raw_choices'],
                        'head_forwards':sum(x['forwards'] for x in rec['head_decisions']),
                        'trace_sha256':digest(trace),'png_sha256':digest(png)})
                pairs.append({'family':family,'seed':seed,
                    'cohort':'discovery' if seed in ORIGINAL_SEEDS else 'heldout',
                    **paired(groups[(mode,'late')],groups[(mode,'open')],mode)})
            # STOP-only surgery is not allowed to change the original *nonSTOP*
            # action choices, even on divergent STOP paths.
            base=groups[('intact','late')]
            for mode in MODES[1:-1]:
                comparator=groups[(mode,'late')]
                aa=[d['action'] for d in base['head_decisions']]
                bb=[d['action'] for d in comparator['head_decisions']]
                common=min(len(aa),len(bb))
                for t in range(common):
                    if aa[t]=='STOP' or bb[t]=='STOP':break
                    if aa[t]!=bb[t]:raise ValueError(f'nonSTOP divergence {mode} {family}/{seed}/{t}')
    summary=[]
    for cohort in ('discovery','heldout'):
        for mode in MODES:
            for gate in ('late','open'):
                sub=[r for r in rows if r['cohort']==cohort and r['mode']==mode and r['gate']==gate]
                sp=[p for p in pairs if p['cohort']==cohort and p['mode']==mode]
                if not sub:continue
                summary.append({'cohort':cohort,'mode':mode,'gate':gate,'n':len(sub),
                    'early_stops':sum(r['early_stop'] for r in sub),
                    'mean_new':sum(r['new'] for r in sub)/len(sub),
                    'mean_branch':sum(r['branches'] for r in sub)/len(sub),
                    'mean_retrace':sum(r['retraces'] for r in sub)/len(sub),
                    'mean_steps':sum(r['steps'] for r in sub)/len(sub),
                    'paired_mean_novel_lost':sum(p['lost_novel'] for p in sp)/len(sp)})
    report={'schema':'DRIV3R-P11-real-W16-STOP-only-clock-interventions/v1',
        'source_original_sha256':digest(source),'checkpoint_sha256':ck,
        'source_original_positive_replays':controls,
        'W0_sha256':src.W0_SHA,'source_class':'real trained auxiliary W16 PyTorch decision MLP / synthetic W0 host / no root SLM or native',
        'experiment':'STOP-only second frozen-head forward using modified clock basis vs full clock/zeroed inputs; conditional movement CRN unchanged',
        'human_ratings':0,'optimiser_updates':0,'cohorts':{'discovery':list(ORIGINAL_SEEDS),'heldout':list(HELDOUT_SEEDS)},
        'model_head_forwards':sum(r['head_forwards'] for r in rows),'runs':rows,'paired':pairs,'summary':summary}
    (out/'P11-results.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--original',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--discovery-only',action='store_true')
    args=ap.parse_args()
    r=run(args.original.read_bytes(),args.out,ORIGINAL_SEEDS if args.discovery_only else ORIGINAL_SEEDS+HELDOUT_SEEDS)
    print(json.dumps({'forwards':r['model_head_forwards'],'summary':r['summary']},indent=2))
