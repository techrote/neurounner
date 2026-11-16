"""Recompute image geometry from unchanged original W0 PNG bytes; never use crops.

Requires all 32 original RGB 384x384 PNGs named '<family>-<seed>-<controller>.png'.
Missing or altered evidence fails closed. Pillow is a decoder; no native replay.
"""
import argparse, hashlib
from pathlib import Path
from PIL import Image
import rank

BG=(14,19,29)
OBSTACLE=(78,81,96)

def image_features(path):
    im=Image.open(path)
    if im.format!='PNG' or im.size!=(384,384) or im.mode!='RGB':
        raise ValueError(f'original full W0 PNG RGB 384x384 required: {path}')
    px=im.load(); cells=set();rocks=set()
    for y in range(96):
        for x in range(96):
            c=px[4*x+2,4*y+2]
            for dy in range(4):
                for dx in range(4):
                    if px[4*x+dx,4*y+dy]!=c:
                        raise ValueError(f'nonuniform source pixel block ({x},{y}): {path}')
            if c==BG:continue
            if c==OBSTACLE:rocks.add((x,y))
            else:cells.add((x,y))
    if not cells:raise ValueError('empty original material drawing')
    def neighbors(x,y):return [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
    degree={p:sum(q in cells for q in neighbors(*p)) for p in cells}
    found=set();components=0
    for p in sorted(cells):
        if p in found:continue
        components+=1;todo=[p]
        while todo:
            z=todo.pop()
            if z in found:continue
            found.add(z);todo.extend(q for q in neighbors(*z) if q in cells)
    edge=sum(degree.values())//2
    return {'cells':len(cells),'xspan':max(x for x,y in cells)-min(x for x,y in cells)+1,
            'yspan':max(y for x,y in cells)-min(y for x,y in cells)+1,
            'edges':edge,'junctions':sum(d>=3 for d in degree.values()),
            'cycles':max(0,edge-len(cells)+components),'endpoints':sum(d==1 for d in degree.values()),
            'components':components,
            'obstacle_contacts':sum(any(q in rocks for q in neighbors(*p)) for p in cells),
            'diagonal_contacts':sum(sum(q in cells for q in [(x-1,y+1),(x+1,y+1)]) for x,y in cells),
            'western_cells':sum(x<16 for x,y in cells)}

def audit(directory,data):
    worlds=rank.validate(data)
    source={x['id']:x for rows in worlds.values() for x in rows}
    originals=list(Path(directory).glob('*.png'))
    required={k+'.png' for k in source}
    if not required.issubset({p.name for p in originals}):
        raise ValueError('missing original PNGs: '+str(sorted(required-{p.name for p in originals})))
    checked=0
    for key,entry in sorted(source.items()):
        p=Path(directory)/(key+'.png')
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        if actual!=entry['original_png_sha256']:
            raise ValueError(f'original SHA mismatch {key}: {actual}')
        features=image_features(p)
        if features!=entry['features']:
            raise ValueError(f'original geometry mismatch {key}: {features} != {entry["features"]}')
        checked+=1
    return checked

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--image-dir',required=True)
    ap.add_argument('--data',default=str(Path(__file__).with_name('source_features.csv')))
    a=ap.parse_args();n=audit(a.image_dir,a.data)
    print(f'PASS: {n}/32 original PNG SHA and full reconstructed 96x96 geometry matches')
