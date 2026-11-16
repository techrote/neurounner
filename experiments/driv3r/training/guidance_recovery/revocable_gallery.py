"""Deterministic SVG gallery of synthetic C10 receipts, never native screenshots."""
from __future__ import annotations
import argparse, html, json, hashlib
from pathlib import Path
from topology_transfer import ALL_FAMILIES, build
from adversarial_cues import source_hash

def reconstruct(r, seed_base=1893301):
    ix=ALL_FAMILIES.index(r['family'])
    offset=1000000 if r['outcome']=='ahead' else 0
    j=(r['seed']-seed_base-ix*250000-offset)//1709
    assert 0 <= j <= 64 and seed_base+ix*250000+offset+j*1709==r['seed']
    w=build(r['seed'],r['personality'],r['family'],j%4,bool((j//4)%2),r['outcome'])
    assert source_hash(w)==r['source_sha256']
    return w

def render(r):
    w=reconstruct(r)
    f=['<svg viewBox="0 0 360 360" role="img" xmlns="http://www.w3.org/2000/svg">',
       '<rect width="360" height="360" fill="#141e2d"/>']
    step=24
    for x,y in sorted(w.protected):
        f.append(f'<rect x="{x*step}" y="{y*step}" width="24" height="24" fill="#64748b" stroke="#1c2737" stroke-width="1"/>')
    for x,y in sorted(w.inked):
        f.append(f'<rect x="{x*step+7}" y="{y*step+7}" width="10" height="10" rx="2" fill="#47bfbb" opacity=".8"/>')
    start=w.pos
    pts=[start]+[tuple(t['cursor']) for t in r['trace']]
    for i,t in enumerate(r['trace']):
        x1,y1=pts[i];x2,y2=pts[i+1]
        co='#f5b449' if t['action'][0]=='BACKTRACK' else ('#e67683' if t['revoked'] else '#8fe5cb')
        f.append(f'<line x1="{(x1+.5)*step}" y1="{(y1+.5)*step}" x2="{(x2+.5)*step}" y2="{(y2+.5)*step}" stroke="{co}" stroke-width="3" stroke-linecap="round"/>')
    gx,gy=w.goal
    f.append(f'<circle cx="{(gx+.5)*step}" cy="{(gy+.5)*step}" r="8" fill="none" stroke="#f6e68d" stroke-width="3"/>')
    x,y=start;f.append(f'<circle cx="{(x+.5)*step}" cy="{(y+.5)*step}" r="5" fill="#fff"/>')
    if pts:
       x,y=pts[-1];f.append(f'<circle cx="{(x+.5)*step}" cy="{(y+.5)*step}" r="6" fill="#e88c9c"/>')
    return ''.join(f)+'</svg>'

def build_gallery(trace_file,out,selected=16):
    arr=json.loads(Path(trace_file).read_text())
    # Deterministic stratification, with explicit held-out/topology and cue mode.
    groups={}
    for r in arr:
        key=(r['family'],r['seed'],r['personality'],r['arm'],r['outcome'],r['cue_mode'])
        groups.setdefault(key,{})[r['model']]=r
    order=[]
    for fam in ALL_FAMILIES:
      for outc in ('reverse','ahead'):
       for mode in ('live','countermand_2'):
        keys=sorted(k for k in groups if k[0]==fam and k[4]==outc and k[5]==mode and k[3]==.35)
        if keys:order.append(keys[0])
    order=order[:selected]
    css='body{font:14px system-ui;background:#0e1725;color:#e8eff7;margin:24px}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}article{background:#1b2a3d;border-radius:12px;padding:12px}svg{width:100%;max-width:360px}h2{font-size:16px;margin:8px 0}p{line-height:1.45}small{color:#b1c3d2}'
    parts=[f'<html><head><meta charset="utf-8"><title>C10 synthetic countermand trajectories</title><style>{css}</style></head><body>',
     '<h1>C10 synthetic worm traces — C9 full vs C10 weak-feedback</h1><p>Original source-reconstructed external 15×15 worlds; no native CyberSand screenshots. Gold movement = BACKTRACK; red = host countermanded; teal = movement; yellow ring = goal. Matched initial source seeds and action RNG, not same-state trajectories after divergence.</p><section>']
    for key in order:
       family,seed,personality,arm,outcome,mode=key
       for label in ('C9_full','C10_weak'):
        r=groups[key][label]
        parts.append(f'<article><h2>{html.escape(label)} · {family} · {outcome}</h2><p>seed={seed}, personality={personality}, scalar={arm}, mode={mode}</p>{render(r)}<p>Reached: <b>{r["reached"]}</b> · decisions: {r["steps"]} · backtracks: {r["backtracks"]} · end distance: {r["end_distance"]}</p><small>Source SHA-256 {r["source_sha256"][:24]}…</small></article>')
    parts.append('</section></body></html>')
    result=''.join(parts)+'\n'; Path(out).write_text(result)
    return {'panels':len(order)*2,'sha256':hashlib.sha256(result.encode()).hexdigest()}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('receipts');p.add_argument('output');a=p.parse_args()
    print(build_gallery(a.receipts,a.output))
