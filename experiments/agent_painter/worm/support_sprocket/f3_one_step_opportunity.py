#!/usr/bin/env python3
"""F3 original-score, legal-state, ONE-ACTION intervention test (no model forwards).

Pinned, archived SPROCKET F3 states only. A one-action legal counterfactual is
applied by the *actual F3 W0 code* after exact original prefix replay; never
pretend any counterfactual future was produced by the learned model.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, io, json, math, zipfile
from collections import Counter
from pathlib import Path

ARCHIVE_SHA='5eb64e5f7d5335809f631aa11caf9edbfcf1a67f546601d5d14e795cc61f7af6'
W0_SHA='81c3923cb38d7f196b99986806216388c34ebc27fd4976e168b3b370c25a323e'
PRODUCER='e56e0aac49b310cc1e00aafe3ffeddb90eca6e38'
ACTIONS=('STRAIGHT','LEFT','RIGHT','BRANCH','RETRACE','STOP')
WORLD={(0,8111,'greebles'),(0,8123,'pipework'),(1,8147,'architecture'),(1,8161,'greebles')}

def require(p,reason):
    if not p:raise ValueError(reason)
def sha(b):return hashlib.sha256(b).hexdigest()
def load_w0(path:Path):
    require(path.is_file() and sha(path.read_bytes())==W0_SHA,'pinned W0 digest mismatch')
    spec=importlib.util.spec_from_file_location('pinned_f3_w0',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    require(tuple(mod.ACTIONS)==ACTIONS,'pinned legal-action contract drift')
    return mod

def sampled(logits,legal,u):
    require(set(logits)==set(ACTIONS) and all(math.isfinite(x) for x in logits.values()),'invalid original logits')
    require(0<=u<1 and legal and len(legal)==len(set(legal)) and not set(legal)-set(ACTIONS),'bad host mask/draw')
    choices=[a for a in ACTIONS if a in legal]
    vals=[logits[a]/2.25 for a in choices]
    expo=[math.exp(x-max(vals)) for x in vals]; total=sum(expo)
    for a,prob in zip(choices,expo):
        u-=prob/total
        if u<0:return a
    return choices[-1]

def original_worlds(raw):
    require(sha(raw)==ARCHIVE_SHA,'original F3 ZIP SHA mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names=z.namelist()
        require(len(names)==len(set(names))==65,'original ZIP inventory mismatch')
        worlds=[]
        for p in names:
            if p.endswith('.json') and '/original-png-and-json/' in p and 'sprocket-f1-' in p:
                doc=json.loads(z.read(p)); world=(int(p.split('/')[0].split('-')[1]),doc['seed'],doc['family'])
                require(world in WORLD,'unexpected original world')
                require(doc['size']==96 and doc['steps_executed']==doc['step_budget']==128,'unbounded original walk')
                require(len(doc['score_receipts'])==len(doc['events'])==128,'missing actual score forward receipts')
                worlds.append((world,doc,p,sha(z.read(p[:-5]+'.png'))))
        require({w[0] for w in worlds}==WORLD and len(worlds)==4,'missing four original worm worlds')
    return sorted(worlds)

def effect(w0,seed,family,prefix,candidate):
    # Full host apply, including RETRACE's history pop and BRANCH's priority.
    rec=w0.run(seed,'replay',family,steps=len(prefix)+1,raw_actions=prefix+[candidate])
    event=rec['events'][-1]
    before=set(map(tuple,rec['trajectory'][:-1]))
    after=tuple(event['position_after'])
    return {'new_cell':int(event['applied']!='STOP' and after not in before),
            'revisit':int(event['applied']!='STOP' and after in before),
            'action':event['applied'],'position':list(after)}

def analyze(archive:bytes,w0,output:Path|None=None):
    rows=[]; galleries=[]
    for (slot,seed,family),original,path,png_sha in original_worlds(archive):
        # Unmodified original actions and PNG must replay exactly with pinned W0.
        applied=[e['applied'] for e in original['events']]
        replay=w0.run(seed,'replay',family,steps=128,raw_actions=applied)
        for key in ('trajectory','events','metrics','obstacles','branch_points'):
            require(replay[key]==original[key],'original pinned W0 replay differs: '+key)
        require(sha(w0.png_bytes(original))==png_sha,'original F3 PNG drift')
        for tick,(e,r) in enumerate(zip(original['events'],original['score_receipts'])):
            obs=e['observation']; require(r['tick']==obs['tick']==tick,'original tick mismatch')
            require(sha(json.dumps(obs,sort_keys=True,separators=(',',':')).encode())==r['observation_sha256'],
                    'original observation hash mismatch')
            legal=[a for a in obs['legal'] if a!='STOP'] if tick<126 and len(obs['legal'])>1 else obs['legal']
            require(r['host_policy_legal']==legal and r['original_W0_legal']==obs['legal'], 'legal mask mismatch')
            draw=int.from_bytes(hashlib.sha256(f'612:sprocket:score-crn-s2:{seed}:{tick}'.encode()).digest()[:8],'big')/2**64
            require(r['host_uniform_draw']==draw,'original host PRNG mismatch')
            chosen=sampled(r['full_six_token_logits'],legal,draw)
            flat=sampled(dict.fromkeys(ACTIONS,0.0),legal,draw)
            require(chosen==e['applied']==r['host_selected'] and flat==r['same_observation_flat'],
                    'saved learned/flat choice mismatch')
            prefix=applied[:tick]
            # One isolated source-authoritative counterfactual, not a policy rollout.
            m=effect(w0,seed,family,prefix,chosen)
            f=m if flat==chosen else effect(w0,seed,family,prefix,flat)
            choices={a: (m if a==chosen else f if a==flat else effect(w0,seed,family,prefix,a))
                     for a in legal}
            best=max(v['new_cell'] for v in choices.values())
            stats={'slot':slot,'seed':seed,'family':family,'tick':tick,'phase':min(tick//32,3),
                   'chosen':chosen,'flat':flat,'differs':chosen!=flat,
                   'learned_new':m['new_cell'],'flat_new':f['new_cell'],
                   'novel_possible':best,'missed_novel':int(best>m['new_cell']),
                   'flat_missed_novel':int(best>f['new_cell']),
                   'chosen_revisit':m['revisit'],'flat_revisit':f['revisit'],
                   'legal':legal,'wall_pixels':sum(s.count('#') for s in obs['local_5x5']),
                   'visited_pixels':sum(s.count('+') for s in obs['local_5x5']),
                   'start':list(original['trajectory'][tick]),
                   'learned_end':m['position'],'flat_end':f['position']}
            rows.append(stats)
            if chosen!=flat and m['new_cell']!=f['new_cell']:
                galleries.append((seed,family,tick,'LEARNED_GAIN' if m['new_cell'] else 'LEARNED_LOSS',
                                  prefix,chosen,flat))
    require(len(rows)==512 and sum(x['differs'] for x in rows)==149,'F3 actual score population mismatch')
    def tally(xs):
        return {'n':len(xs),'different':sum(x['differs'] for x in xs),
                'learned_new':sum(x['learned_new'] for x in xs),'flat_new':sum(x['flat_new'] for x in xs),
                'novel_possible':sum(x['novel_possible'] for x in xs),
                'learned_missed_novel':sum(x['missed_novel'] for x in xs),
                'flat_missed_novel':sum(x['flat_missed_novel'] for x in xs),
                'learned_revisit':sum(x['chosen_revisit'] for x in xs),
                'flat_revisit':sum(x['flat_revisit'] for x in xs)}
    divergence=[x for x in rows if x['differs']]
    gains=[x for x in divergence if x['learned_new']>x['flat_new']]
    losses=[x for x in divergence if x['learned_new']<x['flat_new']]
    ties=[x for x in divergence if x['learned_new']==x['flat_new']]
    report={'schema':'cybersand.worm.sprocket-support-f3-one-step-opportunity/1',
            'original_producer':PRODUCER,'original_archive_sha256':ARCHIVE_SHA,
            'pinned_W0_sha256':W0_SHA,'world_group_seeds':sorted(seed for _,seed,_ in WORLD),
            'model_forwards_in_original_archive':512,'new_model_forwards':0,'new_optimizers':0,
            'not_closed_loop_counterfactual':True,'not_native':True,'STOP_censoring_disclosed':True,
            'population':tally(rows),'differing_same_observation':tally(divergence),
            'learned_novel_gains_vs_flat':len(gains),'learned_novel_losses_vs_flat':len(losses),
            'different_but_same_novel_outcome':len(ties),
            'by_seed':{str(seed):tally([x for x in rows if x['seed']==seed]) for seed in sorted({x['seed'] for x in rows})},
            'by_32tick_quartile':{str(t):tally([x for x in rows if x['phase']==t]) for t in range(4)},
            'by_local_wall_presence':{str(b):tally([x for x in rows if bool(x['wall_pixels'])==b]) for b in (False,True)},
            'by_local_trail_density':{name:tally([x for x in rows if low<=x['visited_pixels']<=high])
                      for name,low,high in [('0-4',0,4),('5-9',5,9),('10-24',10,24)]},
            'action_counts':dict(Counter(x['chosen'] for x in rows)),
            'exact_same_observation_paired_only':True,
            'causal_claim':'One isolated legal action applied at the exact archived original W0 prefix; no divergent model re-forward or future rollout.'}
    if output:
        output.mkdir(parents=True,exist_ok=False)
        (output/'per_decision.jsonl').write_text(''.join(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n' for row in rows))
        selected=[]
        for seed in sorted({x['seed'] for x in rows}):
            for outcome in ('LEARNED_GAIN','LEARNED_LOSS'):
                choices=[g for g in galleries if g[0]==seed and g[3]==outcome]
                if choices: selected.append(choices[0])
        manifest=[]
        for seed,family,tick,outcome,prefix,chosen,flat in selected:
            for name,action in [('learned',chosen),('flat',flat)]:
                trial=w0.run(seed,'replay',family,steps=tick+1,raw_actions=prefix+[action])
                data=w0.png_bytes(trial)
                fn=f'seed{seed}-tick{tick:03}-{outcome.lower()}-{name}-{action}.png'
                (output/fn).write_bytes(data)
                manifest.append({'seed':seed,'family':family,'tick':tick,'outcome':outcome,
                                 'arm':name,'action':action,'sha256':sha(data),'filename':fn,
                                 'one_action_only':True})
        (output/'gallery-manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
        report['gallery_pngs']=len(manifest)
        (output/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('original_zip',type=Path);a.add_argument('--w0',required=True,type=Path)
    a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    result=analyze(args.original_zip.read_bytes(),load_w0(args.w0),args.output)
    print(json.dumps(result,sort_keys=True))
