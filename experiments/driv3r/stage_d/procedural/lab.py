"""Standalone D-2 synthetic raster laboratory. No native/model IO; Python stdlib only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import zlib

VERSION = "d2.offline/1"
DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def packed(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha(x):
    return hashlib.sha256(x).hexdigest()


def points_hash(points):
    return sha(packed(sorted(map(list, points))))


class Random32:
    def __init__(self, seed):
        self.state = (int(seed) & 0xffffffff) or 0x9e3779b9

    def next(self):
        x = self.state
        x ^= (x << 13) & 0xffffffff
        x ^= x >> 17
        x ^= (x << 5) & 0xffffffff
        self.state = x & 0xffffffff
        return self.state

    def choose(self, items):
        return items[self.next() % len(items)]


def raster(a, b):
    x, y = a
    X, Y = b
    dx, dy = abs(X - x), -abs(Y - y)
    sx, sy = (1 if x < X else -1), (1 if y < Y else -1)
    err = dx + dy
    while True:
        yield x, y
        if (x, y) == (X, Y):
            break
        e = 2 * err
        if e >= dy:
            err += dy
            x += sx
        if e <= dx:
            err += dx
            y += sy


def minkowski(cx, cy, size, depth):
    h = size / 2
    p = [(cx-h, cy-h), (cx+h, cy-h), (cx+h, cy+h), (cx-h, cy+h), (cx-h, cy-h)]
    for _ in range(depth):
        q = [p[0]]
        for (ax, ay), (bx, by) in zip(p, p[1:]):
            tx, ty = (bx-ax)/4, (by-ay)/4
            nx, ny = -ty, tx
            for t, n in ((1,0),(1,1),(2,1),(2,0),(2,-1),(3,-1),(3,0),(4,0)):
                q.append((ax+t*tx+n*nx, ay+t*ty+n*ny))
        p = q
    return [(round(x), round(y)) for x, y in p]


def within(x, y, polygon):
    result = False
    for (ax, ay), (bx, by) in zip(polygon, polygon[1:]):
        if (ay > y) != (by > y) and x < (bx-ax)*(y-ay)/(by-ay)+ax:
            result = not result
    return result


def hilbert(order):
    n = 1 << order
    points = []
    for index in range(n*n):
        x = y = 0
        s, t = 1, index
        while s < n:
            rx, ry = (t // 2) & 1, (t ^ (t // 2)) & 1
            if ry == 0:
                if rx:
                    x, y = s-1-x, s-1-y
                x, y = y, x
            x += s*rx
            y += s*ry
            t //= 4
            s *= 2
        points.append((x, y))
    return points


def moore(order):
    """Four Hilbert tiles with cardinal-joined ends and closed final seam."""
    base = hilbert(order)
    n = 1 << order
    def transform(point, mode):
        x, y = point
        if mode & 4:
            x, y = y, x
        if mode & 1:
            x = n-1-x
        if mode & 2:
            y = n-1-y
        return x, y
    tiles = ((0,0), (n,0), (n,n), (0,n))
    variants = [[[(x+ox, y+oy) for x,y in (transform(p,m) for p in base)]
                 for m in range(8)] for ox,oy in tiles]
    def adjacent(a,b):
        return abs(a[0]-b[0])+abs(a[1]-b[1]) == 1
    for a in variants[0]:
        for b in variants[1]:
            if not adjacent(a[-1],b[0]):
                continue
            for c in variants[2]:
                if not adjacent(b[-1],c[0]):
                    continue
                for d in variants[3]:
                    if adjacent(c[-1],d[0]) and adjacent(d[-1],a[0]):
                        return a+b+c+d
    raise ValueError("Moore tiling is not contiguous")


def htree(x, y, size, depth):
    result = []
    def recurse(cx,cy,span,n):
        if n == 0 or span < 2:
            return
        a,b = max(1,span//2), max(1,span//3)
        result.extend([[(cx-a,cy),(cx+a,cy)],
                       [(cx-a,cy-b),(cx-a,cy+b)],
                       [(cx+a,cy-b),(cx+a,cy+b)]])
        for dx in (-a,a):
            for dy in (-b,b):
                recurse(cx+dx,cy+dy,span//2,n-1)
    recurse(x,y,size,depth)
    return result


def guidance(w, h, seed):
    out = {}
    for y in range(h):
        for x in range(w):
            r = math.hypot(x-w//2, y-h//2)
            wave = 0.32*(1+math.sin((x*3+y*2+seed%17)*0.21))
            out[x,y] = round(max(0,min(1,(1-r/(w*.72))*.8+wave))*65535)
    return out


def footprint(point, radius, w, h):
    x,y=point
    return {(a,b) for a in range(x-radius,x+radius+1)
            for b in range(y-radius,y+radius+1)
            if 0<=a<w and 0<=b<h and (a-x)**2+(b-y)**2<=radius**2}


def image(w,h,cells):
    scan = bytearray()
    for y in range(h):
        scan.append(0)
        for x in range(w):
            scan.extend((232,222,199,255) if (x,y) in cells else (24,33,43,255))
    def chunk(tag,data):
        return struct.pack(">I",len(data))+tag+data+struct.pack(">I",zlib.crc32(tag+data)&0xffffffff)
    return (b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",w,h,8,6,0,0,0))
            +chunk(b"IDAT",zlib.compress(bytes(scan),9))+chunk(b"IEND",b""))


def simulate(recipe):
    assert recipe["version"]==VERSION
    family=recipe["family"]
    w,h=recipe["canvas"]
    assert 24<=w<=256 and 24<=h<=256
    seed=recipe["seed"]
    budget=recipe["steps"]
    radius=recipe.get("brush_radius",0)
    assert 1<=budget<=2048 and 0<=radius<=4
    policy=recipe.get("recovery","backtrack")
    assert policy in ("backtrack","relocate")
    rng=Random32(seed)
    scalar=guidance(w,h,seed)
    material=set()
    original=set()
    visited=set()
    track=[]
    contiguous=[]
    records=[]
    midpoint=None
    geometry={}
    cursor=w//2,h//2

    def log(kind, **kwargs):
        records.append(dict(sequence=len(records),kind=kind,**kwargs))

    def stroke(point, effect, authorized=None):
        nonlocal cursor,midpoint
        if not (0<=point[0]<w and 0<=point[1]<h):
            log("REJECT_OUT_OF_BOUNDS",point=point)
            return
        old=set(material)
        sweep=footprint(point,radius,w,h)
        admitted=sweep if authorized is None else sweep & authorized
        if effect=="add":
            material.update(admitted)
        else:
            material.difference_update(admitted)
        cursor=point
        visited.add(point)
        track.append(point)
        contiguous.append(point)
        if len(track)==max(1,budget//2):
            midpoint=set(material)
        log("STEP",point=point,effect=effect,brush_radius=radius,
            swept=sorted(map(list,sweep)),admitted=sorted(map(list,admitted)),
            actual_changed=len(old^material),guidance_u16=scalar[point],
            material_before_sha256=points_hash(old),material_after_sha256=points_hash(material),
            native_receipt=None)

    def reverse(point,kind):
        nonlocal cursor
        old=points_hash(material)
        if kind=="BACKTRACK_CURSOR":
            assert point in contiguous
            index=len(contiguous)-1-contiguous[::-1].index(point)
            trace=list(reversed(contiguous[index:]))
            assert all(abs(a[0]-b[0])+abs(a[1]-b[1])<=1 for a,b in zip(trace,trace[1:]))
            del contiguous[index+1:]
        else:
            assert kind=="RELOCATE_TO_BRANCH" and point in visited
            trace=[]
            contiguous.clear()
            contiguous.append(point)
        cursor=point
        log("REVERSAL",semantics=kind,brush="up",to=point,
            reverse_trace=trace,material_before_sha256=old,
            material_after_sha256=points_hash(material),native_receipt=None)

    def next_to(point,cells):
        return [(point[0]+dx,point[1]+dy) for dx,dy in DIRS
                if (point[0]+dx,point[1]+dy) in cells and
                (point[0]+dx,point[1]+dy) not in visited]

    if family=="weathering_worm":
        polygon=minkowski(w//2,h//2,min(w,h)//2,recipe.get("depth",1))
        original={(x,y) for y in range(h) for x in range(w)
                  if within(x+.5,y+.5,polygon)}
        material=set(original)
        edge={p for a,b in zip(polygon,polygon[1:]) for p in raster(a,b)}
        geometry=dict(generator="minkowski_square",original_perimeter=polygon,
                      original_volume_sha256=points_hash(original))
        cursor=min(original,key=lambda p:(min(abs(p[0]-q[0])+abs(p[1]-q[1]) for q in edge),p))
        visited.add(cursor)
        walker_policy=recipe.get("walker_policy","boundary_biased")
        assert walker_policy in ("boundary_biased","random_legal")
        for _ in range(budget):
            options=next_to(cursor,original)
            if not options:
                candidates=([p for p in contiguous[:-1] if next_to(p,original)]
                            if policy=="backtrack"
                            else sorted(p for p in visited if next_to(p,original)))
                if not candidates:
                    log("TRAPPED",reason="no eligible original-rock frontier")
                    break
                reverse(candidates[-1] if policy=="backtrack" else rng.choose(candidates),
                        "BACKTRACK_CURSOR" if policy=="backtrack" else "RELOCATE_TO_BRANCH")
                options=next_to(cursor,original)
            def score(p):
                distance=min(abs(p[0]-q[0])+abs(p[1]-q[1]) for q in edge)
                return abs(distance-recipe.get("boundary_offset",2)),rng.next()%7,p
            selected=(min(options,key=score) if walker_policy=="boundary_biased"
                      else rng.choose(sorted(options)))
            log("SYNTHETIC_CHOICE",generator="weathering_worm",
                legal_points=sorted(map(list,options)),chosen=selected,
                policy=walker_policy,legal_catalog_sha256=points_hash(options),
                native_legal_catalog=False,model_decision=False)
            stroke(selected,"erase")
    elif family=="branching_growth":
        lines=htree(w//2,h//2,min(w,h)//2,recipe.get("depth",3))
        skeleton={p for a,b in lines for p in raster(a,b)}
        authorized={p for q in skeleton for p in footprint(q,recipe.get("volume_radius",1),w,h)}
        geometry=dict(generator="h_tree",segments=lines,
                      authorized_volume=sorted(map(list,authorized)))
        stroke(cursor,"add",authorized if recipe.get("containment","contained")=="contained" else None)
        while len(track)<budget:
            options=next_to(cursor,skeleton)
            if not options:
                candidates=([p for p in contiguous[:-1] if next_to(p,skeleton)]
                            if policy=="backtrack"
                            else sorted(p for p in visited if next_to(p,skeleton)))
                if not candidates:
                    log("TRAPPED",reason="no remaining reachable branch frontier")
                    break
                reverse(candidates[-1] if policy=="backtrack" else rng.choose(candidates),
                        "BACKTRACK_CURSOR" if policy=="backtrack" else "RELOCATE_TO_BRANCH")
                continue
            stroke(rng.choose(sorted(options)),"add",
                   authorized if recipe.get("containment","contained")=="contained" else None)
        log("GROWTH_END",unvisited_skeleton=len(skeleton-visited),
            deposited_outside_authorized_volume=len(material-authorized))
    elif family=="guided_coverage":
        order=recipe.get("order",3)
        assert 2<=order<=6
        curve=recipe.get("curve","hilbert")
        if curve=="hilbert":
            coordinates=hilbert(order)
        elif curve=="moore":
            coordinates=moore(order)
        elif curve=="serpentine":
            n=1<<order
            coordinates=[(x if y%2==0 else n-1-x,y) for y in range(n) for x in range(n)]
        else:
            raise ValueError("unsupported coverage curve")
        size=(1<<order)*(2 if curve=="moore" else 1)
        assert size<=min(w,h)
        ox,oy=(w-size)//2,(h-size)//2
        coordinates=[(x+ox,y+oy) for x,y in coordinates]
        geometry=dict(generator=curve,order=order,ordered_curve=coordinates,
                      closed=curve=="moore")
        low=recipe.get("exit_below",48000)
        high=recipe.get("reenter_at",52000)
        assert 0<=low<high<=65535
        i=0
        while i<len(coordinates) and len(track)<budget:
            point=coordinates[i]
            if scalar[point]<low:
                legal=["STOP"]+(["BACKTRACK_CURSOR"] if contiguous else [])
                log("GUIDANCE_EXIT",at_index=i,scalar_u16=scalar[point],
                    legal_candidates=legal+["SEEK_FUTURE"],policy="synthetic_deterministic",
                    legal_catalog_sha256=sha(packed(sorted(legal+["SEEK_FUTURE"]))))
                # Require a future four-point interval that is both new and tolerable.
                candidates=[j for j in range(i+1,min(i+33,len(coordinates)-3))
                            if scalar[coordinates[j]]>=high and
                            all(q not in visited and scalar[q]>=low for q in coordinates[j:j+4])]
                if not candidates:
                    log("TRAPPED",reason="no viable future four-point reentry",at_index=i)
                    break
                j=candidates[0]
                # Backtracking a single cursor step is different from a
                # future-point reposition; neither paints or undoes material.
                if policy=="backtrack" and len(contiguous)>1:
                    reverse(contiguous[-2],"BACKTRACK_CURSOR")
                old_hash=points_hash(material)
                log("BRUSH_UP_SEEK_FUTURE",from_cursor=cursor,to=coordinates[j],
                    policy=policy,virtual_offline_transfer=True,
                    continuous_physical_path_verified=False,material_preserved=True,
                    material_before_sha256=old_hash,material_after_sha256=old_hash,
                    native_receipt=None)
                cursor=coordinates[j]
                contiguous.clear()  # a new logical segment, never a hidden brush stroke
                assert points_hash(material)==old_hash
                log("GUIDANCE_REENTRY",from_index=i,to_index=j,skipped=j-i,
                    lookahead_unvisited=4,reentry_guidance_u16=scalar[coordinates[j]])
                i=j
                continue
            if point in visited:
                log("REJECT_REVISIT",at_index=i)
                break
            stroke(point,"add")
            i+=1
        log("COVERAGE_END",index=i,total=len(coordinates))
    else:
        raise ValueError("unsupported family")

    # This simulator has no real native edit history, and must never fake Undo.
    log("UNDO_MATERIAL_EDIT_UNAVAILABLE",executed=False,native_receipt=None,
        reason="real host-owned native history and readback required")
    summary=dict(family=family,seed=seed,steps=len(track),initial_cells=len(original),
                 final_cells=len(material),net_delta=len(material)-len(original),
                 visited=len(visited),reversals=sum(e["kind"]=="REVERSAL" for e in records),
                 guidance_exits=sum(e["kind"]=="GUIDANCE_EXIT" for e in records),
                 trapped=sum(e["kind"]=="TRAPPED" for e in records),
                 material_sha256=points_hash(material),trajectory_sha256=sha(packed(track)),
                 event_sha256=sha(packed(records)))
    return dict(recipe=recipe,summary=summary,records=records,track=track,
                geometry=geometry,visited=sorted(map(list,visited)),guidance=scalar,
                before=original,midpoint=midpoint if midpoint is not None else material.copy(),
                after=material,midpoint_captured=midpoint is not None)


def artifact_bytes(recipe, result):
    """All synthetic artifacts are rederived on replay, not merely hash checked."""
    w,h=recipe["canvas"]
    field=b"".join(struct.pack("<H",result["guidance"][x,y])
                   for y in range(h) for x in range(w))
    files={
        "recipe.json":packed(recipe),"events.json":packed(result["records"]),
        "trajectory.json":packed(result["track"]),"geometry.json":packed(result["geometry"]),
        "visited.json":packed(result["visited"]),"guidance.u16le":field,
        "guidance.json":packed(dict(encoding="u16le",canvas=[w,h],zero="neutral",
                                    producer="envelope_bands/v1",seed=recipe["seed"])),
        "guidance-preview.png":image(w,h,{p for p,v in result["guidance"].items() if v>=38000}),
        "before.png":image(w,h,result["before"]),
        "midpoint.png":image(w,h,result["midpoint"]),
        "after.png":image(w,h,result["after"])}
    return files


def write(recipe, folder):
    result=simulate(recipe)
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    files=artifact_bytes(recipe,result)
    manifest=dict(version=VERSION,summary=result["summary"],
        qualification="OFFLINE_SYNTHETIC_ONLY",native_replay_verified=False,
        model_decisions=0,provenance=dict(native=None,model=None,native_receipt=None,
            source_python_sha256=sha(Path(__file__).read_bytes()),
            runtime=f"Python {sys.version_info.major}.{sys.version_info.minor}",
            images="synthetic material raster, NOT framebuffer"),
        midpoint_captured=result["midpoint_captured"],
        failures=[e for e in result["records"] if e["kind"] in
                  ("TRAPPED","REJECT_REVISIT","REJECT_OUT_OF_BOUNDS")],
        artifacts={name:dict(bytes=len(data),sha256=sha(data)) for name,data in files.items()})
    for name,data in files.items():
        (folder/name).write_bytes(data)
    (folder/"manifest.json").write_bytes(packed(manifest))
    return manifest


def replay(folder):
    folder=Path(folder)
    manifest=json.loads((folder/"manifest.json").read_text())
    assert manifest["version"]==VERSION
    assert manifest["qualification"]=="OFFLINE_SYNTHETIC_ONLY"
    assert manifest["native_replay_verified"] is False
    assert manifest["model_decisions"]==0
    provenance=manifest["provenance"]
    assert provenance["source_python_sha256"]==sha(Path(__file__).read_bytes())
    assert provenance["native"] is None and provenance["model"] is None
    assert provenance["native_receipt"] is None
    recipe=json.loads((folder/"recipe.json").read_text())
    fresh=simulate(recipe)
    assert fresh["summary"]==manifest["summary"]
    assert fresh["midpoint_captured"]==manifest["midpoint_captured"]
    assert manifest["failures"]==[e for e in fresh["records"] if e["kind"] in
                                  ("TRAPPED","REJECT_REVISIT","REJECT_OUT_OF_BOUNDS")]
    expected=artifact_bytes(recipe,fresh)
    assert set(expected)==set(manifest["artifacts"])
    # An attacker can recalculate a manifest hash; each item must also match
    # the independently reconstructed raster/field/trajectory byte-for-byte.
    for name,data in expected.items():
        meta=manifest["artifacts"][name]
        assert meta==dict(bytes=len(data),sha256=sha(data)),name
        assert (folder/name).read_bytes()==data,name
    return True


def default_recipes():
    common=dict(version=VERSION,canvas=[64,64],steps=96,seed=29,brush_radius=0,
                recovery="backtrack")
    return [
        dict(common,family="weathering_worm",depth=2,boundary_offset=2),
        dict(common,family="branching_growth",depth=3,volume_radius=1,
             containment="contained"),
        dict(common,family="guided_coverage",order=3,curve="moore",
             exit_below=48000,reenter_at=52000)]


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="/tmp/driv3r-d2")
    args=parser.parse_args()
    output=Path(args.out)
    for n,recipe in enumerate(default_recipes()):
        folder=output/f"{n:02d}-{recipe['family']}"
        summary=write(recipe,folder)["summary"]
        assert replay(folder)
        print(json.dumps(dict(folder=str(folder),**summary),sort_keys=True))
