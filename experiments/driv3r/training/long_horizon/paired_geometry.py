"""Post-run original-trajectory geometry comparison; DEVELOPMENT only.

No student fine-tune, native execution, held-out read, or image synthesis.
The gallery crops are existing material PNGs generated from independently
replayed source-fenced synthetic worlds.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from .verify_intervention import verify_run
from .world import digest


def ink_set(obj):
    return {tuple(p) for e in obj['events'] for p in e['painted']}


def pair_metric(intact,forced):
    if (intact['world_id'],intact['arm'],intact['horizon'],intact['policy_seed']) != (
        forced['world_id'],forced['arm'],forced['horizon'],forced['policy_seed']):
        raise ValueError('Unmatched original world/seed/arm/horizon')
    if intact['perturbed'] or not forced['perturbed']:
        raise ValueError('Wrong intervention identity')
    a,b=ink_set(intact),ink_set(forced)
    first=next((i for i in range(max(len(intact['events']),len(forced['events'])))
        if i>=len(intact['events']) or i>=len(forced['events']) or
        intact['events'][i]['post_sha256'] != forced['events'][i]['post_sha256']),None)
    return {'world_id':intact['world_id'],'family':intact['family'],
            'mode':intact['mode'],'challenge':intact['challenge'],
            'arm':intact['arm'],'horizon':intact['horizon'],
            'first_consequential_divergence':first,
            'intersection_ink':len(a&b),'union_ink':len(a|b),
            'jaccard_ink':round(len(a&b)/len(a|b),6) if a|b else 1,
            'symmetric_difference_cells':len(a^b),
            'forced_actions':len(forced['actual_interventions']),
            'intact_ink':len(a),'forced_ink':len(b),
            'intact_turns':intact['metrics']['turns'],
            'forced_turns':forced['metrics']['turns'],
            'intact_recoveries':intact['metrics']['recoveries'],
            'forced_recoveries':forced['metrics']['recoveries'],
            'intact_early_stop':intact['metrics']['stopped_early'],
            'forced_early_stop':forced['metrics']['stopped_early'],
            'intact_trace_sha256':intact['trajectory_sha256'],
            'forced_trace_sha256':forced['trajectory_sha256']}


def analyze(root:Path):
    validated=verify_run(root)
    by=defaultdict(dict)
    for path in sorted((root/'trajectories').rglob('*.json')):
        obj=json.loads(path.read_text())
        k=(obj['horizon'],obj['arm'],obj['world_id'],obj['policy_seed'])
        if obj['perturbed'] in by[k]:raise ValueError('Duplicate policy/synthetic identities')
        by[k][obj['perturbed']]=obj
    results=[]
    for pair in by.values():
        if set(pair)!={False,True}:raise ValueError('Missing paired complete original trajectory')
        results.append(pair_metric(pair[False],pair[True]))
    cohorts=defaultdict(list)
    for pair in results:cohorts[(pair['horizon'],pair['arm'])].append(pair)
    group={}
    for (horizon,arm),rows in sorted(cohorts.items()):
        group[f'{horizon}/{arm}']={
            'n':len(rows),
            'mean_ink_jaccard':round(statistics.mean(r['jaccard_ink'] for r in rows),6),
            'mean_symmetric_difference_cells':round(statistics.mean(r['symmetric_difference_cells'] for r in rows),4),
            'first_divergence_histogram':{str(x):sum(r['first_consequential_divergence']==x for r in rows)
                                          for x in sorted(set(r['first_consequential_divergence'] for r in rows),key=lambda x:-1 if x is None else x)},
            'mean_turn_delta':round(statistics.mean(r['forced_turns']-r['intact_turns'] for r in rows),4),
            'mean_ink_delta':round(statistics.mean(r['forced_ink']-r['intact_ink'] for r in rows),4),
            'early_stop_delta':sum(r['forced_early_stop'] for r in rows)-sum(r['intact_early_stop'] for r in rows),
        }
    return {'version':'driv3r-forced-geometry-pairs/1',
            'input_run_sha256':digest([r['world_id'] for r in results]),
            'verified':validated,'pair_count':len(results),
            'cohorts':group,'pairs':results,'heldout_read':False,'native_execution':False}


def gallery(root:Path,report:dict,dest:Path):
    from PIL import Image, ImageDraw, ImageFont
    arms=('ordinary_offpolicy_013','stress_offpolicy_040','teacher_mixture','random_legal_persistent')
    cases=(('islands','motif','normal'),('pickets','wall','trap_corridor'))
    base=Path(root)/'trajectories'/'128'
    canvas=Image.new('RGB',(1440,980),(23,27,36))
    draw=ImageDraw.Draw(canvas)
    draw.text((20,8),'128-decision DEVELOPMENT | original replayed material states | intact vs forced',(248,244,233))
    for col, (fam,mode,challenge) in enumerate(cases):
        for n,perturb in enumerate((False,True)):
            x=col*720+n*355+12
            draw.text((x,36),f'{fam}/{mode}/{challenge} | {"forced" if perturb else "intact"}',(194,216,231))
    for row,arm in enumerate(arms):
        y=70+row*223
        draw.text((12,y),arm,(233,218,164))
        for col, (fam,mode,challenge) in enumerate(cases):
            records=[r for r in report['pairs'] if r['horizon']==128 and r['arm']==arm
                     and (r['family'],r['mode'],r['challenge'])==(fam,mode,challenge)]
            if len(records)!=1:raise ValueError('Gallery exact scenario not unique')
            pair=records[0]
            for n,perturb in enumerate((False,True)):
                files=list((base/arm/('forced' if perturb else 'intact')).glob(f'{pair["world_id"]}-128-{arm}-{int(perturb)}.png'))
                if len(files)!=1:raise ValueError('Original PNG not found')
                with Image.open(files[0]) as img:
                    frame=img.convert('RGB').resize((186,186),resample=Image.Resampling.NEAREST)
                x=col*720+n*355+12
                canvas.paste(frame,(x,y+23))
                draw.text((x+190,y+24),f'ink {pair["forced_ink" if perturb else "intact_ink"]}',(216,224,230))
                draw.text((x+190,y+45),f'turn {pair["forced_turns" if perturb else "intact_turns"]}',(216,224,230))
                if n==1:draw.text((x+190,y+67),f'J {pair["jaccard_ink"]:.3f}',(216,224,230))
    dest.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(dest)


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('root',type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--gallery',required=True,type=Path)
    a=ap.parse_args(argv)
    if a.output.exists() or a.gallery.exists():raise FileExistsError('Never overwrite analysis or original gallery')
    report=analyze(a.root)
    gallery(a.root,report,a.gallery)
    a.output.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'pairs':report['pair_count'],'cohorts':report['cohorts']},sort_keys=True))

if __name__=='__main__':main()
