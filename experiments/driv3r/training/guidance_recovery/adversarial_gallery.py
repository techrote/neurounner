"""Render source-traceable *synthetic* guidance-recovery first-choice gallery.

No model execution, network, native materials or external rendering dependencies.
Each SVG is reconstructed from the exact source seed and held-out topology.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from adversarial_cues import near_wall_false_world, decoy_true_world, source_hash
from positive_correction import warning_world
from hard_recovery import false_cue_world


def world_for(row):
    name=row['family'];seed=row['seed'];persona=row['personality']
    rot=row['rotation'];mirror=row['mirror']
    if name=='true':return warning_world(seed,persona,4,rot,mirror)[0]
    if name=='straight_false':return near_wall_false_world(seed+100000,persona,4,rot,mirror,'straight')[0]
    if name=='corner_false':return near_wall_false_world(seed+200000,persona,4,rot,mirror,'corner')[0]
    if name=='decoy_straight_true':return decoy_true_world(seed+100000,persona,4,rot,mirror,'straight')[0]
    if name=='decoy_corner_true':return decoy_true_world(seed+200000,persona,4,rot,mirror,'corner')[0]
    if name=='open_false':return false_cue_world(seed+300000,persona)
    raise ValueError('family')


def svg(w,choice,target):
    size=16;elements=["<svg viewBox='0 0 240 240' xmlns='http://www.w3.org/2000/svg' role='img' aria-label='Synthetic host 15 by 15 material grid'>"]
    for y in range(15):
        for x in range(15):
            p=(x,y);color='#565d70' if p in w.protected else '#ebf2f0'
            if p in w.inked:color='#b77b43'
            elements.append(f"<rect x='{x*size}' y='{y*size}' width='{size}' height='{size}' fill='{color}' stroke='#bdc9cc' stroke-width='.5'/>")
    if w.history:
        positions=list(w.history)+[w.pos]
        points=' '.join(f'{x*size+8},{y*size+8}' for x,y in positions)
        elements.append(f"<polyline points='{points}' stroke='#24a5bd' stroke-width='2.5' stroke-dasharray='4 2' fill='none'/>")
    gx,gy=w.goal;elements.append(f"<circle cx='{gx*size+8}' cy='{gy*size+8}' r='5' fill='#00b56a' stroke='#00593a' stroke-width='2'/>")
    px,py=w.pos;elements.append(f"<circle cx='{px*size+8}' cy='{py*size+8}' r='5' fill='#287dff' stroke='#083367' stroke-width='2'/>")
    tx,ty=target;elements.append(f"<line x1='{px*size+8}' y1='{py*size+8}' x2='{tx*size+8}' y2='{ty*size+8}' stroke='#f0415a' stroke-width='3'/>")
    elements.append(f"<circle cx='{tx*size+8}' cy='{ty*size+8}' r='4' fill='#f0415a'/>")
    elements.append('</svg>');return ''.join(elements)


def render(path,output,seeds=(800001,800001+977),arm=0.35):
    d=json.loads(Path(path).read_text());cards=[]
    for group in ('before','after'):
        for row in d[group]:
            if row['seed'] not in seeds or row['arm']!=arm or not row['cue']:continue
            w=world_for(row)
            assert source_hash(w)==row['world_sha256']
            title=f"{row['family'].replace('_',' ')} · {group} · seed {row['seed']}"
            choice=row['action'];target=row['target'];status='Expected' if row['correct_first_action'] else 'Missed'
            cards.append(f"<article><h2>{html.escape(title)}</h2>{svg(w,choice,target)}"
               f"<p><strong>{html.escape(status)}</strong> — {html.escape(str(choice[0]))}, target {html.escape(str(target))}</p>"
               f"<small>World SHA256 {row['world_sha256']}<br>Legal menu SHA256 {row['legal_menu_sha256']}</small></article>")
    result='''<!doctype html><html lang="en"><meta charset="utf-8"><title>DRIV3R synthetic adversarial recovery gallery</title>
<style>body{background:#101820;color:#ecf1f4;font:14px system-ui;margin:24px}h1{font-size:22px}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px}article{border:1px solid #526570;padding:12px;border-radius:8px;background:#1b2b36}article svg{width:100%;max-width:270px}h2{font-size:14px}small{word-break:break-all;color:#b3c9d1;font:11px monospace}p{line-height:1.4}</style>
<h1>DRIV3R worm — synthetic near-wall warning choices</h1><p>Slate = protected material, pale = traversable, blue = cursor, green = goal, cyan dashes = cursor history, red = selected action. These are reconstructed synthetic host states, not actual CyberSand screenshots, native Undo or resident model trajectories. Genuine warnings and decoy false alternatives can have identical 25-feature observations despite differing deeper material topology.</p>
<section>'''+''.join(cards)+'</section></html>\n'
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(result)
    return len(cards)


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipts',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();print(f'Rendered {render(a.receipts,a.out)} source-verified panels')

if __name__=='__main__':main()
