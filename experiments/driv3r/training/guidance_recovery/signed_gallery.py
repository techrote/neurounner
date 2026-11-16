"""Deterministic external synthetic trail gallery; reads raw C11 receipts."""
from __future__ import annotations
import html, json
from pathlib import Path
from topology_transfer import build, ALL_FAMILIES

CELL=13

def svg_panel(row):
    fam=row['family'];idx=ALL_FAMILIES.index(fam)
    seed=row['seed']-1000000 if row['outcome']=='ahead' else row['seed']
    i=(seed-2560001-idx*300000)//1901
    # Raw source SHA verification guards against picturing a different world.
    rot=i%4;mirror=bool((i//4)%2)
    w=build(row['seed'],row['personality'],fam,rot,mirror,row['outcome'])
    assert row['source_sha256']==__import__('adversarial_cues').source_hash(w)
    svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 195 195" width="195" height="195" role="img">',
         '<rect width="195" height="195" fill="#f9fafb"/>']
    for x,y in sorted(w.protected):
        svg.append(f'<rect x="{x*CELL}" y="{y*CELL}" width="13" height="13" fill="#262d39"/>')
    for x,y in sorted(w.inked):
        svg.append(f'<rect x="{x*CELL+3}" y="{y*CELL+3}" width="7" height="7" fill="#b2bac5"/>')
    pts=[w.pos]+[tuple(t['cursor']) for t in row['trace']]
    poly=' '.join(f'{x*CELL+6.5},{y*CELL+6.5}' for x,y in pts)
    svg.append(f'<polyline points="{poly}" fill="none" stroke="#ef6f37" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"/>')
    for t in row['trace']:
        if t['active_directive']<0:
            x,y=t['cursor'];svg.append(f'<circle cx="{x*CELL+6.5}" cy="{y*CELL+6.5}" r="2.5" fill="#317fb7"/>')
    x,y=w.goal;svg.append(f'<circle cx="{x*CELL+6.5}" cy="{y*CELL+6.5}" r="4.7" fill="#3a9a69"/>')
    svg.append('</svg>')
    return ''.join(svg)

def write_gallery(input_path,output_path):
    data=json.loads(Path(input_path).read_text())
    # Fixed distinct family/personality/arm cases; never cherry pick by success.
    selected=[]
    for family in ALL_FAMILIES:
      for mode in ('release_0','release_2','release_4'):
        rows=[r for r in data if r['family']==family and r['mode']==mode and r['outcome']=='ahead'
             and r['personality']==1 and r['arm']==.35]
        seed=min(r['seed'] for r in rows)
        selected.extend([r for r in rows if r['seed']==seed])
    assert len(selected)==24
    parts=['<!DOCTYPE html><html><head><meta charset="utf-8"/><title>C11 matched synthetic trails</title>',
      '<style>body{font-family:system-ui,sans-serif;background:#e7edf3;color:#18222e;margin:28px}',
      '.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:13px}',
      'figure{margin:0;background:white;border:1px solid #c7d1de;border-radius:6px;padding:12px}',
      'figcaption{font-size:12px;margin-top:8px;line-height:1.5}svg{width:100%;max-width:195px;height:auto}</style></head><body>',
      '<h1>C11 / C9: release-time paired synthetic trajectories</h1>',
      '<p>24 panels; fixed seeds per topology and timing. Black=protected, grey=initial ink, orange=chosen cursor path, blue=host CONTINUE advice active, green=goal. No native simulation or rollback.</p><div class="gallery">']
    for r in selected:
      cap=f"{r['family']} / {r['mode']} / {r['model']} / goal {r['goal_reached']} / steps {r['turns']} / post-release backs {r['post_release_backtracks']}"
      parts.append('<figure>'+svg_panel(r)+'<figcaption>'+html.escape(cap)+'</figcaption></figure>')
    parts.append('</div></body></html>')
    Path(output_path).write_text(''.join(parts))

if __name__=='__main__':
    import sys
    write_gallery(*sys.argv[1:])
