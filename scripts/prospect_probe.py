#!/usr/bin/env python3
from __future__ import annotations
import base64, json, os, platform, struct, subprocess, sys, time, urllib.request, zlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CACHE=Path(os.environ.get("NEURO_CACHE",ROOT/".cache"/"neurounner"))
RESULTS=ROOT/"results"; LOGS=ROOT/"logs"

def load_env(p):
    d={}
    for raw in p.read_text().splitlines():
        if raw.strip() and not raw.lstrip().startswith("#"):
            k,v=raw.split("=",1); d[k]=v
    return d

def cpu_model():
    for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
        if line.lower().startswith("model name"): return line.split(":",1)[1].strip()
    return "unknown"

def rss_kib(pid):
    p=Path(f"/proc/{pid}/status")
    if not p.exists(): return None
    for line in p.read_text(errors="replace").splitlines():
        if line.startswith("VmRSS:"): return int(line.split()[1])
    return None

def make_png(path:Path):
    w=h=256
    pixels=[]
    for y in range(h):
        row=bytearray([0])
        for x in range(w):
            # light gray background/floor
            r=g=b=220
            # black obstacle: vertical rectangle at center
            if 112 <= x < 144 and 72 <= y < 220: r=g=b=20
            # red actor on left
            if 38 <= x < 82 and 154 <= y < 198: r,g,b=220,30,30
            # blue target on right
            if 180 <= x < 228 and 146 <= y < 194: r,g,b=30,80,230
            row += bytes((r,g,b))
        pixels.append(bytes(row))
    raw=b"".join(pixels)
    def chunk(t,data):
        return struct.pack(">I",len(data))+t+data+struct.pack(">I",zlib.crc32(t+data)&0xffffffff)
    png=b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",w,h,8,2,0,0,0))+chunk(b"IDAT",zlib.compress(raw,9))+chunk(b"IEND",b"")
    path.write_bytes(png)

def do_request(port:int, multimodal:bool, image_path:Path|None, prompt:str, max_tokens:int=220):
    if multimodal:
        b64=base64.b64encode(image_path.read_bytes()).decode()
        content=[
          {"type":"text","text":prompt},
          {"type":"image_url","image_url":{"url":"data:image/png;base64,"+b64}}
        ]
    else:
        content=prompt
    body={"model":"local","messages":[{"role":"user","content":content}],"max_tokens":max_tokens,"temperature":0.25,"top_p":0.9,"seed":9917,"stream":False}
    req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"},method="POST")
    t=time.monotonic()
    with urllib.request.urlopen(req,timeout=300) as r: payload=json.loads(r.read())
    elapsed=time.monotonic()-t
    msg=payload["choices"][0]["message"]; usage=payload.get("usage") or {}
    n=int(usage.get("completion_tokens") or 0)
    return {"elapsed_seconds":elapsed,"completion_tokens":n,"tokens_per_second":n/elapsed if n else None,
            "content":(msg.get("content") or "").strip(),"reasoning_content":(msg.get("reasoning_content") or "").strip(),
            "finish_reason":payload["choices"][0].get("finish_reason"),"usage":usage}

def main():
    if len(sys.argv) not in (3,4): raise SystemExit("usage: prospect_probe.py ID MODEL [MMPROJ]")
    ident=sys.argv[1]; model=Path(sys.argv[2]).resolve(); mmproj=Path(sys.argv[3]).resolve() if len(sys.argv)==4 else None
    paths=load_env(CACHE/"paths.env"); server=Path(paths["LLAMA_SERVER"]); bench=Path(paths["LLAMA_BENCH"])
    RESULTS.mkdir(exist_ok=True); LOGS.mkdir(exist_ok=True)
    bench_p=subprocess.run([str(bench),"-m",str(model),"-t","4","-p","64","-n","64","-r","2"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=360)
    (LOGS/f"prospect-{ident}-bench.log").write_text(bench_p.stdout)
    if bench_p.returncode: raise RuntimeError(bench_p.stdout[-5000:])
    args=[str(server),"-m",str(model),"--host","127.0.0.1","--port","18600","-t","4","-c","4096"]
    if mmproj: args += ["--mmproj",str(mmproj)]
    log_path=LOGS/f"prospect-{ident}-server.log"; log=log_path.open("w")
    t0=time.monotonic(); proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,text=True)
    try:
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            if proc.poll() is not None:
                log.close(); raise RuntimeError(log_path.read_text(errors="replace")[-8000:])
            try:
                with urllib.request.urlopen("http://127.0.0.1:18600/health",timeout=1) as r:
                    if r.status==200: break
            except Exception: pass
            time.sleep(.25)
        else: raise RuntimeError("startup timeout")
        startup=time.monotonic()-t0; resident=rss_kib(proc.pid)
        image=None
        if mmproj:
            image=Path(os.environ.get("RUNNER_TEMP","/tmp"))/"neurounner-scene.png"; make_png(image)
            prompt=("The image is a synthetic physics-sandbox scene: a RED square actor on the left, a BLUE target on the right, "
                    "and a BLACK vertical obstacle between them. Decide one immediate local action for the red actor from "
                    "MOVE_LEFT, MOVE_RIGHT, WAIT, or TURN. State the action first, then explain in at most 3 sentences what "
                    "you observed and why. Do not invent objects not visible in the image.")
        else:
            prompt=("We are selecting behaviour kernels for a physics sandbox. A ground actor stores one bounded vector when shoved. "
                    "Propose one deterministic local rule that makes this memory create an interesting visible habit. Give trigger, state update, action, "
                    "and one emergent failure mode in under 140 words.")
        result=do_request(18600,mmproj is not None,image,prompt)
        record={"experiment":"prospect-001","id":ident,"runner":{"platform":platform.platform(),"cpu_count":os.cpu_count(),"cpu_model":cpu_model()},
                "model_file":model.name,"model_bytes":model.stat().st_size,"model_sha256":subprocess.check_output(["sha256sum",str(model)],text=True).split()[0],
                "mmproj_file":mmproj.name if mmproj else None,"mmproj_bytes":mmproj.stat().st_size if mmproj else None,
                "mmproj_sha256":subprocess.check_output(["sha256sum",str(mmproj)],text=True).split()[0] if mmproj else None,
                "startup_seconds":startup,"rss_kib":resident,"bench_output":bench_p.stdout,"probe":result}
        (RESULTS/f"prospect-{ident}.json").write_text(json.dumps(record,indent=2,ensure_ascii=False)+"\n")
        print(json.dumps(record,indent=2,ensure_ascii=False))
    finally:
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(10)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait(5)
        if not log.closed: log.close()

if __name__=="__main__": main()
