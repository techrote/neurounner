#!/usr/bin/env python3
"""SPROCKET WORM: deterministic, decorative replay of ORIGINAL W0 action paths.

This tool NEVER infers actions or writes to native CyberSand. It reads original
stored W0 trajectories/observations, validates their event lineage, and renders
an explicitly *derived* stylized texture gallery. Original PNGs are never
changed/reclassified. This is a local-art experiment, not a trained-model test.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import struct
import zipfile
import zlib

SCALE = 8
BACK=(14,17,27)
ROCK=(49,54,70)
INK={
    "greebles": ((73,228,237),(255,117,173),(251,205,96)),
    "pipework": ((105,218,176),(197,248,235),(249,195,89)),
    "architecture": ((185,137,246),(226,205,248),(255,204,132)),
}

def sha(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()

class Raster:
    def __init__(self,side:int):
        self.w=side*SCALE
        self.h=side*SCALE
        self.rgb=bytearray(bytes(BACK)*(self.w*self.h))
    def pixel(self,x:int,y:int,c):
        if 0<=x<self.w and 0<=y<self.h:
            i=(y*self.w+x)*3
            self.rgb[i:i+3]=bytes(c)
    def disk(self,x,y,r,c):
        r=max(0,int(r))
        for dy in range(-r,r+1):
            for dx in range(-r,r+1):
                if dx*dx+dy*dy<=r*r:
                    self.pixel(x+dx,y+dy,c)
    def line(self,x0,y0,x1,y1,r,c):
        dx=abs(x1-x0);dy=abs(y1-y0)
        sx=1 if x0<x1 else -1
        sy=1 if y0<y1 else -1
        err=dx-dy
        while True:
            self.disk(x0,y0,r,c)
            if x0==x1 and y0==y1: break
            e2=err*2
            if e2>-dy:err-=dy;x0+=sx
            if e2< dx:err+=dx;y0+=sy
    def rect(self,x0,y0,x1,y1,c):
        for y in range(max(0,y0),min(self.h,y1)):
            for x in range(max(0,x0),min(self.w,x1)):
                self.pixel(x,y,c)
    def center(self,p): return int(p[0])*SCALE+SCALE//2,int(p[1])*SCALE+SCALE//2
    def png(self)->bytes:
        def chunk(tag,body):
            return struct.pack('!I',len(body))+tag+body+struct.pack('!I',zlib.crc32(tag+body)&0xffffffff)
        rows=[]
        width=self.w*3
        for y in range(self.h):rows.append(b'\0'+self.rgb[y*width:(y+1)*width])
        return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',self.w,self.h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(b''.join(rows),8))+chunk(b'IEND',b'')

def validate_record(record):
    assert record.get('contract')=='cybersand.worm.observation-action/0-experimental'
    side=record.get('size');assert side==96
    path=record['trajectory'];events=record['events']
    assert len(events)==record['steps_executed'] and len(path)==len(events)+1
    assert all(isinstance(p,list) and len(p)==2 and all(isinstance(v,int) and 0<=v<side for v in p) for p in path)
    rocks={tuple(p) for p in record['obstacles']}
    assert all(tuple(p) not in rocks for p in path)
    for i,e in enumerate(events):
        obs=e['observation']
        assert (obs['x'],obs['y'])==tuple(path[i])
        assert tuple(e['position_after'])==tuple(path[i+1])
        assert e['applied'] in obs['legal']
        assert e['applied'] in ('STRAIGHT','LEFT','RIGHT','BRANCH','RETRACE','STOP')
        assert e['raw_legal'] is True
        assert abs(path[i+1][0]-path[i][0])+abs(path[i+1][1]-path[i][1])<=1
    return rocks

def draw(record)->tuple[bytes,dict]:
    rocks=validate_record(record)
    family=record['family']
    main,light,accent=INK[family]
    raster=Raster(record['size'])
    for rx,ry in rocks:
        x=rx*SCALE;y=ry*SCALE
        raster.rect(x+1,y+1,x+SCALE-1,y+SCALE-1,ROCK)
    path=record['trajectory']; events=record['events']
    rng=random.Random(f"sprocket-etch/1:{record['seed']}:{record['family']}:{record['mode']}")
    touches=set()
    for i,event in enumerate(events):
        a=raster.center(path[i]);b=raster.center(path[i+1]);action=event['applied']
        touches.add(tuple(path[i]));touches.add(tuple(path[i+1]))
        if family=='pipework':
            raster.line(*a,*b,4,(20,64,59))
            raster.line(*a,*b,3,main)
            raster.line(*a,*b,1,light)
        elif family=='architecture':
            raster.line(*a,*b,5,(67,50,92))
            raster.line(*a,*b,3,main)
            raster.line(*a,*b,1,light)
        else:
            raster.line(*a,*b,3,(38,78,101))
            raster.line(*a,*b,2,main)
            raster.line(*a,*b,0,light)
        if action=='BRANCH':
            # Distinct stud, pipe-valve or corridor buttress at a REAL branch position.
            raster.disk(*b,4,accent)
            raster.disk(*b,2,main)
            raster.disk(*b,1,light)
        elif action=='RETRACE':
            raster.disk(*b,2,accent)
        elif action in ('LEFT','RIGHT') and i%3==0:
            raster.disk(*b,2,light)
        if family=='greebles' and action=='BRANCH':
            # Decorative topologically non-authoritative stud pixels, kept local.
            px,py=b
            for dx,dy in ((-6,0),(6,0),(0,-6),(0,6)):
                if rng.random()<0.7:
                    gx=(px+dx)//SCALE;gy=(py+dy)//SCALE
                    if (gx,gy) not in rocks:
                        raster.disk(px+dx,py+dy,1,accent)
        if family=='architecture' and i%6==0:
            x,y=b
            raster.rect(x-2,y-2,x+3,y+3,light)
            raster.disk(x,y,1,main)
        if family=='pipework' and i%7==0:
            raster.disk(*b,5,accent)
            raster.disk(*b,3,main)
            raster.disk(*b,1,light)
    if path:
        raster.disk(*raster.center(path[0]),3,light)
        raster.disk(*raster.center(path[-1]),3,accent)
    png=raster.png()
    return png,{"schema":"cybersand.sprocket.derived-texture-replay/1",
        "original_trajectory_steps":len(events),"original_unique_cells":len(set(tuple(p) for p in path)),
        "original_invalid_actions":record['metrics']['invalid_raw_choices'],
        "texture_family":family,"derived_not_original_W0":True,"not_native":True,
        "no_model_inference_in_replay":True,"no_optimizers_in_replay":True,
        "derived_footprint_pixels":sum(1 for i in range(0,len(raster.rgb),3) if tuple(raster.rgb[i:i+3]) not in (BACK,ROCK))}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--artifact',required=True,type=Path,help='Original W0 complete F2/F3 audit ZIP')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--strict-source-sha',help='Expected source artifact SHA-256')
    args=parser.parse_args()
    data=args.artifact.read_bytes(); archive_sha=sha(data)
    if args.strict_source_sha and archive_sha!=args.strict_source_sha:
        raise ValueError('original artifact SHA mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    outputs=[]
    with zipfile.ZipFile(io.BytesIO(data)) as zipf:
        if zipf.testzip() is not None:raise ValueError('original ZIP CRC failed')
        names=zipf.namelist()
        for member in sorted(n for n in names if n.endswith('.json') and '/original-png-and-json/' in n):
            record=json.loads(zipf.read(member))
            source_png=member[:-5]+'.png'
            if source_png not in names:raise ValueError('missing original source W0 PNG '+source_png)
            png,summary=draw(record)
            pos=member.split('/original-png-and-json/')[-1][:-5]
            path=args.output/(pos+'.derived.png')
            if path.exists():raise ValueError('nonunique source episode basename')
            path.write_bytes(png)
            outputs.append({"source_json":member,"source_json_sha256":sha(zipf.read(member)),
                            "source_original_png":source_png,"source_original_png_sha256":sha(zipf.read(source_png)),
                            "derived_png":path.name,"derived_png_sha256":sha(png),**summary})
    if not outputs:raise ValueError('no original images were found')
    inventory={"schema":"cybersand.sprocket.derived-texture-replay-inventory/1",
      "original_source_artifact_sha256":archive_sha,"original_source_artifact":args.artifact.name,
      "original_pngs_preserved_untouched":True,"not_native":True,"no_new_model_steps":True,
      "outputs":outputs}
    p=args.output/'derived-inventory.json'
    p.write_text(json.dumps(inventory,sort_keys=True,indent=2)+'\n')
    print(json.dumps({"derived_pngs":len(outputs),"original_artifact_sha256":archive_sha,
                      "invalid":sum(x['original_invalid_actions'] for x in outputs)},sort_keys=True))

if __name__=='__main__':main()
