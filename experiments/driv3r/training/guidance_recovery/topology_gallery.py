"""Source-verified static gallery of C8/C7 synthetic cursor trajectories."""
from __future__ import annotations
import argparse,json
from html import escape
from pathlib import Path
from experiment import N,sha
from adversarial_cues import source_hash
from topology_transfer import ALL_FAMILIES,build

def render(receipts,output):
    panels=[]
    for fam in ALL_FAMILIES:
        for outcome in ('reverse','ahead'):
            hits=[r for r in receipts if r['family']==fam and r['outcome']==outcome
                    and r['arm']==.35 and r['cue'] and r['personality']==1]
            if not hits:continue
            seed=min(r['seed'] for r in hits)
            for model in ('C7','C8'):
                r=next(z for z in hits if z['seed']==seed and z['model']==model)
                # Rotation/mirror are deterministically assigned by seed index.
                # This gallery requires stored orientation in the receipt.
                w=build(r['seed'],r['personality'],fam,r['rotation'],r['mirror'],outcome)
                assert source_hash(w)==r['source_sha256']
                assert sha(sorted(w.protected))==r['material_sha256']
                step=12;blocks=[]
                for x,y in sorted(w.protected):
                    blocks.append(f'<rect x="{x*step}" y="{y*step}" width="{step}" height="{step}" fill="#5b6570"/>')
                def coords(points):return ' '.join(f'{x*step+6},{y*step+6}' for x,y in points)
                cursor=[w.pos]+[tuple(t['cursor']) for t in r['trace']]
                svg=f'<svg viewBox="0 0 {N*step} {N*step}"><rect width="100%" height="100%" fill="#edf1f4"/>'
                svg+=''.join(blocks)
                svg+=f'<polyline points="{coords(w.history+[w.pos])}" fill="none" stroke="#9aa8b6" stroke-width="3"/>'
                svg+=f'<polyline points="{coords(cursor)}" fill="none" stroke="#216fb1" stroke-width="3"/>'
                gx,gy=w.goal
                svg+=f'<circle cx="{gx*step+6}" cy="{gy*step+6}" r="5" fill="#128654"/></svg>'
                title=escape(f'{fam} / {outcome} / {model}')
                status='goal reached' if r['reached'] else f'goal distance {r["end_distance"]}'
                panels.append(f'<article><h2>{title}</h2>{svg}<p>{status}; {r["steps"]} turns; {r["unwinds"]} backtracks</p></article>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>C8 synthetic worm trajectory gallery</title>
<style>body{font:14px system-ui,sans-serif;background:#f3f4f6;color:#24292f;margin:2em}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:1em}article{border:1px solid #c6cdd4;border-radius:6px;padding:1em;background:white}svg{width:100%;image-rendering:pixelated}h2{font-size:1em}</style>
<h1>C8 vs C7 — synthetic source-reconstructed cursor traces</h1>
<p>Gray cells are protected; light cells legal. Pale line: initial painted history. Blue: cursor trajectory. Green: goal. BACKTRACK does not erase ink. Matched initial source/seed; paths may diverge. No native game execution.</p><main>'''+''.join(panels)+'</main></html>'
    Path(output).write_text(page)
    return len(panels)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('receipts');p.add_argument('output');a=p.parse_args()
    print(render(json.loads(Path(a.receipts).read_text()),a.output))
