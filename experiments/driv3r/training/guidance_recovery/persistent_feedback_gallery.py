"""Render source-reconstructed C8/C9 synthetic cursor traces from raw receipts."""
from __future__ import annotations
import argparse
import json
from html import escape
from pathlib import Path
from experiment import N, sha
from adversarial_cues import source_hash
from topology_transfer import ALL_FAMILIES, build


def render(records, destination):
    panels=[];step=12
    for family in ALL_FAMILIES:
        for outcome in ('reverse','ahead'):
            group=[r for r in records if r['family']==family and r['outcome']==outcome and
                   r['arm']==.35 and r['cue'] and r['personality']==1]
            seed=min(r['seed'] for r in group)
            i=(seed-(1000000 if outcome=='ahead' else 0)-1010001-
               ALL_FAMILIES.index(family)*300000)//1607
            assert i>=0
            rot=i%4;mirror=bool((i//4)%2)
            for policy in ('C8','C9'):
                row=next(r for r in group if r['seed']==seed and r['model']==policy)
                world=build(row['seed'],row['personality'],family,rot,mirror,outcome)
                assert source_hash(world)==row['source_sha256']
                assert sha(sorted(world.protected))==row['material_sha256']
                coord=lambda ps:' '.join(f'{x*step+step/2},{y*step+step/2}' for x,y in ps)
                occupied=''.join(f'<rect x="{x*step}" y="{y*step}" width="{step}" height="{step}" fill="#5b6570"/>'
                    for x,y in sorted(world.protected))
                walked=[world.pos]+[tuple(t['cursor']) for t in row['trace']]
                svg=(f'<svg viewBox="0 0 {N*step} {N*step}" aria-label="synthetic worm trajectory">'
                     '<rect width="100%" height="100%" fill="#edf1f4"/>'+occupied+
                     f'<polyline points="{coord(world.history+[world.pos])}" fill="none" stroke="#b4aab8" stroke-width="3"/>'+
                     f'<polyline points="{coord(walked)}" fill="none" stroke="#216fb1" stroke-width="2.8"/>'+
                     f'<circle cx="{world.goal[0]*step+6}" cy="{world.goal[1]*step+6}" r="5" fill="#128654"/></svg>')
                status='reached' if row['reached'] else 'not reached'
                panels.append(f'<article><h2>{escape(family)} / {outcome} / {policy}</h2>{svg}'
                   f'<p>{status}, {row["steps"]} steps, max consecutive BACKTRACK {row["longest_backtrack_run"]}</p></article>')
    page=('<!doctype html><html lang="en"><meta charset="utf-8"><title>C9 guidance recovery trajectories</title>'
          '<style>body{font:14px system-ui,sans-serif;background:#f3f4f6;color:#24292f;margin:2em}'
          'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:1em}'
          'article{border:1px solid #c6cdd4;border-radius:6px;padding:1em;background:#fff}'
          'svg{width:100%;image-rendering:pixelated}h2{font-size:1em}</style>'
          '<h1>C8 vs C9 — matched synthetic worm recovery</h1>'
          '<p>Protected material: gray. Prepainted history: pale purple. Cursor path: blue. Goal: green. '
          'BACKTRACK moves the cursor but does not undo painted materials. Initial seeds and host permissions '
          'match, but worlds may diverge after policy decisions. This is not native CyberSand gameplay.</p><main>'+
          ''.join(panels)+'</main></html>')
    Path(destination).write_text(page)
    return len(panels)


def main():
    a=argparse.ArgumentParser();a.add_argument('receipts');a.add_argument('out');args=a.parse_args()
    print(render(json.loads(Path(args.receipts).read_text()),args.out))

if __name__=='__main__':main()
