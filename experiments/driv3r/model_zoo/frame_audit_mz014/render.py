"""Render original SVG trajectories from MZ-014 evidence with no external deps."""
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

def render(metrics):
    names=("diagonal","gru","reactive")
    modes=("original","horizontal_mirror","vertical_mirror","rotation_180")
    width,height=1250,640
    elements=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="1250" height="640" fill="#101928"/>',
              '<text x="18" y="29" font-size="20" fill="#f9fafb">MZ-014: frozen game-policy orientation transfer</text>',
              '<text x="18" y="51" font-size="12" fill="#a7bfd2">Same 700-update checkpoint / same legal host / coordinate-frame only changed</text>']
    colors={"diagonal":"#38bdf8","gru":"#a78bfa","reactive":"#fbbf24"}
    for c,mode in enumerate(modes):
        elements.append(f'<text x="{158+c*270}" y="84" font-size="14" fill="#fff">{escape(mode)}</text>')
    for r,name in enumerate(names):
        y=108+r*170
        elements.append(f'<text x="15" y="{y+22}" font-size="15" fill="#fff">{name}</text>')
        for c,mode in enumerate(modes):
            x=150+c*270
            record=metrics["scores"][name][mode]
            trace=record["sample"][0]
            points=trace["path"]
            elements.append(f'<rect x="{x}" y="{y}" width="254" height="148" rx="5" fill="#213047"/>')
            # Normalize physical coordinates independently for every unique map.
            xmax=max([px for px,py in points]+[2])
            scale=min(228/max(1,xmax), 12)
            xy=" ".join(f"{x+10+px*scale:.1f},{y+18+py*10:.1f}" for px,py in points)
            elements.append(f'<polyline points="{xy}" stroke="{colors[name]}" stroke-width="2" fill="none"/>')
            for sx in (trace.get("stations") or []):
                elements.append(f'<circle cx="{x+10+sx*scale:.1f}" cy="{y+48}" r="2" fill="#e5e7eb"/>')
            elements.append(f'<text x="{x+8}" y="{y+115}" font-size="12" fill="#e2e8f0">clean {record["clean"]}/{record["episodes"]}  paints {record["paints"]}</text>')
            elements.append(f'<text x="{x+8}" y="{y+133}" font-size="11" fill="#bbcfe4">collision {record["protected_collisions"]}  raw-illegal {record["raw_illegal"]}</text>')
    elements.append('<text x="16" y="630" font-size="11" fill="#94a3b8">Original synthetic evidence, not native CyberSand; no new training. Research Draft/HOLD.</text>')
    elements.append('</svg>')
    return "\n".join(elements)+"\n"

def main():
    path=Path(sys.argv[1])
    data=json.loads((path/"metrics.json").read_text())
    (path/"trajectories.svg").write_text(render(data))
    print(path/"trajectories.svg")

if __name__=="__main__":
    main()
