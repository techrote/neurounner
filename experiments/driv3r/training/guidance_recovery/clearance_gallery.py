"""Source-reconstructable gallery for real synthetic policy decisions; no native art."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path

from experiment import World, sha


def svg_world(record, size=315):
    """Reconstruct source geometry from seed+family, not an invented screenshot."""
    w=World(record['seed'],record['family'],record['personality'])
    assert sha(sorted(w.protected))==record['material_sha256']
    step=size/15
    canvas=[f'<svg viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg" role="img">',
      f'<rect width="{size}" height="{size}" fill="#161d27"/>']
    for x,y in sorted(w.protected):
        material=w.material_at((x,y))
        color='#545c67' if material=='SOLID_ROCK' else '#777c87'
        canvas.append(f'<rect x="{x*step}" y="{y*step}" width="{step}" height="{step}" fill="{color}" stroke="#28313b" stroke-width=".4"/>')
    moves=record['events']
    pts=[(1,1)]+[tuple(e[3]) for e in moves]
    for tick,kind,old,pos,target,cue,goal in moves:
        if kind=='INK':
            x,y=pos
            canvas.append(f'<rect x="{x*step+2}" y="{y*step+2}" width="{step-4}" height="{step-4}" fill="#69b2b3" opacity=".9"/>')
    poly=' '.join(f'{(x+.5)*step},{(y+.5)*step}' for x,y in pts)
    canvas.append(f'<polyline points="{poly}" fill="none" stroke="#f6ce69" stroke-width="1.8" opacity=".88" stroke-linejoin="round"/>')
    if moves:
        x,y=pts[-1]
        canvas.append(f'<circle cx="{(x+.5)*step}" cy="{(y+.5)*step}" r="5.5" fill="#ff8069"/>')
    for x,y in [(13,13),(13,1)]:
        canvas.append(f'<circle cx="{(x+.5)*step}" cy="{(y+.5)*step}" r="3" fill="#9fe2b4"/>')
    canvas.append('</svg>')
    return ''.join(canvas)


def build(before,after,out,per_family=2):
    original=json.loads(Path(before).read_text());adapted=json.loads(Path(after).read_text())
    assert len(original)==len(adapted)
    left={(r['seed'],r['family'],r['personality'],r['arm'],r['cue']):r for r in original}
    right={(r['seed'],r['family'],r['personality'],r['arm'],r['cue']):r for r in adapted}
    assert left.keys()==right.keys()
    chosen=[]
    for fam in ('forks','switchback','clumps'):
        seed_order=sorted({r['seed'] for r in original if r['family']==fam})[:per_family]
        for seed in seed_order:
            for persona in (-1,1):
                key=(seed,fam,persona,1.0,True)
                for label,index in [('Frozen prior',left),('Evidence adapter',right)]:
                    chosen.append((label,index[key]))
    head='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DRIV3R clearance | source-reconstructed trajectories</title><style>body{background:#10151d;color:#ecf0f5;font:14px system-ui,sans-serif;margin:24px;max-width:1400px}p{line-height:1.5;color:#b4c0cd}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(245px,1fr));gap:18px}.card{padding:12px;background:#202a35;border:1px solid #425467;border-radius:8px}.card svg{width:100%;height:auto}h3{margin:6px 0 2px}small{display:block;color:#b2bdc9;line-height:1.5}code{font-size:11px;overflow-wrap:anywhere}h1{font-size:24px}</style></head><body><h1>DRIV3R — bounded-clearance worm traces</h1><p>Original synthetic, seed-matched CPU policy decisions. Grey cells are host-protected material; teal cells were actually selected for INK; yellow is the cursor path including retracing; red is final cursor. Green dots indicate initial/final spatial goals. Identical starting map/seed and RNG across paired prior/adapted panels; trajectories cease being same-state after their choices diverge. These are not native CyberSand screenshots or actual d1 visual inference.</p><div class="grid">'''
    cards=[]
    for label,r in chosen:
        cards.append('<article class="card"><h3>'+html.escape(label)+'</h3><small>'+html.escape(f"{r['family']} | seed {r['seed']} | persona {r['personality']:+d} | strong hint + recovery cue")+'</small>'+svg_world(r)+'<small>'+html.escape(f"{r['steps']} turns; {r['unique_ink']} painted cells; final L1 goal distance {r['goal_distance']}; STOP {r['stopped']}")+'</small><code>source '+html.escape(r['source_sha256'])+'</code></article>')
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(head+''.join(cards)+'</div></body></html>')
    return len(cards)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(build(a.before,a.after,a.out),'panels')

if __name__=='__main__':main()
