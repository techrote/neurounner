"""GAL-003 external synthetic worm morphology descriptors.

No aesthetic score, neural inference or native CyberSand claims.
Metrics count the geometric topology of actual host-painted cells and
physical changes of the painted cursor path, not inferred intention.
"""
from __future__ import annotations
from lab import DIRS, W, H

def direction(a,b):
    dx=b[0]-a[0];dy=b[1]-a[1]
    for name,delta in DIRS.items():
        if delta==(dx,dy):return name
    raise ValueError('non-adjacent painted path')


def morphology(ink, trace):
    """Observable geometry, not a single aesthetic score or A* proxy."""
    ink=set(map(tuple,ink))
    for x,y in ink:
        if not (0 <= x < W and 0 <= y < H):raise ValueError('bad ink coordinate')
    adjacency={p:{(p[0]+dx,p[1]+dy) for dx,dy in DIRS.values()} & ink for p in ink}
    remaining=set(ink);components=0
    while remaining:
        components+=1; stack=[remaining.pop()]
        while stack:
            for q in adjacency[stack.pop()] & remaining:
                remaining.remove(q);stack.append(q)
    edges=sum(map(len,adjacency.values()))//2
    cycles=edges-len(ink)+components  # undirected 4-neighbour graph cycle rank
    xs=[p[0] for p in ink];ys=[p[1] for p in ink]
    bbox=[min(xs),min(ys),max(xs),max(ys)] if ink else None
    span_x=max(xs)-min(xs)+1 if ink else 0
    span_y=max(ys)-min(ys)+1 if ink else 0
    longest_x=longest_y=0
    for y in range(H):
        cur=0
        for x in range(W):
            cur=cur+1 if (x,y) in ink else 0
            longest_x=max(longest_x,cur)
    for x in range(W):
        cur=0
        for y in range(H):
            cur=cur+1 if (x,y) in ink else 0
            longest_y=max(longest_y,cur)
    straight_lengths=[];bends=0;prior_end=None;prior_heading=None;length=0
    for rec in trace:
        name=rec['action_id']
        if name.startswith(('INK_','JOG_')):
            positions=[tuple(p) for p in rec['path']]
            src=tuple(rec['cursor_before'])
            if src!=prior_end:prior_heading=None;length=0
            for dst in positions:
                h=direction(src,dst)
                if prior_heading and h!=prior_heading:
                    bends+=1;straight_lengths.append(length);length=0
                length+=1;prior_heading=h;src=dst
            prior_end=src
        elif name.startswith(('LIFT_','RETRACE_')):
            if length:straight_lengths.append(length)
            prior_heading=None;prior_end=None;length=0
    if length:straight_lengths.append(length)
    return {
        'new_ink':len(ink),'components':components,'graph_cycle_rank':cycles,
        'endpoints':sum(len(n)==1 for n in adjacency.values()),
        'junctions':sum(len(n)>=3 for n in adjacency.values()),
        'isolated_pixels':sum(not n for n in adjacency.values()),
        'edges':edges,'bbox':bbox,'span_x':span_x,'span_y':span_y,
        'bbox_area':span_x*span_y,'bbox_fill_fraction':round(len(ink)/max(1,span_x*span_y),5),
        'longest_horizontal_run':longest_x,'longest_vertical_run':longest_y,
        'paint_path_bends':bends,'longest_paint_run':max(straight_lengths,default=0),
    }

