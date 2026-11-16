"""Read-only SVG gallery of saved synthetic-host goal-switch receipts.

Every panel reconstructs source material from its stored seed/rotation and
asserts the original world/material SHA before drawing any action trace.
"""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
from experiment import sha
from adversarial_cues import source_hash
from goal_switch_recovery import world


def panel(rec, label):
    shape=rec['branch_shape'];w=world(rec['seed'],rec['personality'],rec['rotation'],
        rec['mirror'],'reverse',branch=shape!='corridor',
        branch_shape=shape if shape!='corridor' else 'upper')
    assert source_hash(w)==rec['source_sha256']
    assert sha(sorted(w.protected))==rec['material_sha256']
    scale=18;rects=[]
    for y in range(15):
        for x in range(15):
            p=(x,y)
            c='#293746' if p in w.protected else ('#c69b65' if p in w.inked else '#d9e3e7')
            rects.append(f'<rect x="{x*scale}" y="{y*scale}" width="{scale}" height="{scale}" fill="{c}"/>')
    points=' '.join(f'{int(z["pos"][0])*scale+scale//2},{int(z["pos"][1])*scale+scale//2}' for z in rec['trace'])
    rects.append(f'<polyline points="{points}" fill="none" stroke="#e04939" stroke-width="3.4" stroke-linejoin="round"/>')
    sx,sy=rec['trace'][0]['pos'];gx,gy=rec['goal']
    rects.append(f'<circle cx="{sx*scale+9}" cy="{sy*scale+9}" r="4" fill="#146f96"/>')
    rects.append(f'<circle cx="{gx*scale+9}" cy="{gy*scale+9}" r="6" fill="none" stroke="#148642" stroke-width="3"/>')
    first=rec['trace'][1]['action'];title=f'{label} | {shape} | guidance {rec["arm"]} | {"cue" if rec["cue"] else "uncued"}'
    footer=f'first {first} · goal {rec["goal_reached"]} · unwind {rec["history_unwound"]} · illegal {rec["host_rejected"]}'
    return f'<figure><figcaption>{html.escape(title)}</figcaption><svg viewBox="0 0 270 270" role="img" aria-label="{html.escape(title)}">{"".join(rects)}</svg><small>{html.escape(footer)}</small></figure>'


def build(before,after,out):
    a=json.loads(Path(before).read_text());b=json.loads(Path(after).read_text())
    assert len(a)==len(b)
    aa={(r['branch_shape'],r['seed'],r['arm'],r['cue']):r for r in a}
    bb={(r['branch_shape'],r['seed'],r['arm'],r['cue']):r for r in b}
    assert aa.keys()==bb.keys()
    selected=[]
    for shape in ('corridor','upper','lower'):
        for arm in (0.0,.35,1.0):
            key=next(k for k in aa if k[0]==shape and k[2]==arm and k[3])
            selected.extend((panel(aa[key],'Frozen C4'),panel(bb[key],'C6 goal-switch')))
    style='''body{font:15px system-ui;background:#f4f7f9;color:#172432;max-width:1400px;margin:auto;padding:25px}
    main{display:grid;grid-template-columns:repeat(auto-fit,minmax(265px,1fr));gap:18px}
    figure{background:white;border:1px solid #ced7de;border-radius:10px;margin:0;padding:12px}
    figcaption{font-weight:650;margin-bottom:8px}svg{display:block;width:100%;height:auto}
    small{display:block;margin-top:8px;color:#4c5d68}p{max-width:900px;line-height:1.5}'''
    contents='<!doctype html><html lang="en"><meta charset="utf-8"><title>Goal-switch synthetic trajectories</title>'
    contents+=f'<style>{style}</style><h1>DRIV3R C6: goal-switch recovery</h1>'
    contents+='<p>Source-reconstructed offline synthetic worlds. Dark cells are protected. Tan cells are pre-existing ink. Red is the host-accepted cursor path; blue is start and green is goal. BACKTRACK does <b>not</b> erase painted material. C6 is trained from frozen C4, not a live worm or native game execution.</p>'
    contents+='<main>'+''.join(selected)+'</main></html>'
    Path(out).write_text(contents)
    return len(selected)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--before',required=True);p.add_argument('--after',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(build(a.before,a.after,a.out))
