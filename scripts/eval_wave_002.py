#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
import platform
import struct
import subprocess
import sys
import time
import urllib.request
import zlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("NEURO_CACHE", ROOT / ".cache" / "neurounner"))
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

def load_env(path: Path) -> dict[str, str]:
    out = {}
    for raw in path.read_text().splitlines():
        if raw.strip() and not raw.lstrip().startswith("#"):
            k, v = raw.split("=", 1)
            out[k] = v
    return out

def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"

def rss_kib(pid: int) -> int | None:
    p = Path(f"/proc/{pid}/status")
    if not p.exists():
        return None
    for line in p.read_text(errors="replace").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return None

def sha256(path: Path) -> str:
    return subprocess.check_output(["sha256sum", str(path)], text=True).split()[0]

def png(path: Path, actor_x: int, target_x: int, obstacle_x: tuple[int, int] | None) -> None:
    w = h = 256
    rows = []
    for y in range(h):
        row = bytearray([0])
        for x in range(w):
            r = g = b = 225
            if 150 <= y < 198 and actor_x <= x < actor_x + 36:
                r, g, b = 225, 25, 25
            if 150 <= y < 198 and target_x <= x < target_x + 36:
                r, g, b = 25, 80, 235
            if obstacle_x and obstacle_x[0] <= x < obstacle_x[1] and 125 <= y < 220:
                r = g = b = 20
            row += bytes((r, g, b))
        rows.append(bytes(row))
    raw = b"".join(rows)
    def chunk(t: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data) & 0xffffffff)
    blob = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(blob)

class Server:
    def __init__(self, name: str, model: Path, mmproj: Path | None, threads: int = 4, ctx: int = 8192):
        paths = load_env(CACHE / "paths.env")
        self.exe = Path(paths["LLAMA_SERVER"])
        self.name = name
        self.model = model
        self.mmproj = mmproj
        self.threads = threads
        self.ctx = ctx
        self.port = 18700
        self.proc: subprocess.Popen[str] | None = None
        self.log_path = LOGS / f"wave002-{name}.log"
        self.log_handle = None
        self.startup_seconds = None

    def start(self):
        LOGS.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("w")
        args = [
            str(self.exe), "-m", str(self.model),
            "--host", "127.0.0.1", "--port", str(self.port),
            "-t", str(self.threads), "-c", str(self.ctx),
        ]
        if self.mmproj:
            args += ["--mmproj", str(self.mmproj)]
        t0 = time.monotonic()
        self.proc = subprocess.Popen(args, stdout=self.log_handle, stderr=subprocess.STDOUT, text=True)
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self.log_handle.close()
                raise RuntimeError(self.log_path.read_text(errors="replace")[-8000:])
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1) as r:
                    if r.status == 200:
                        self.startup_seconds = time.monotonic() - t0
                        return
            except Exception:
                pass
            time.sleep(0.25)
        raise RuntimeError(f"{self.name}: startup timeout")

    @property
    def pid(self) -> int:
        assert self.proc is not None
        return self.proc.pid

    def request(self, body: dict[str, Any], timeout: int = 600) -> dict[str, Any]:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
        elapsed = time.monotonic() - t0
        msg = payload["choices"][0]["message"]
        usage = payload.get("usage") or {}
        n = int(usage.get("completion_tokens") or 0)
        return {
            "elapsed_seconds": elapsed,
            "completion_tokens": n,
            "tokens_per_second": n / elapsed if n else None,
            "content": (msg.get("content") or "").strip(),
            "reasoning_content": (msg.get("reasoning_content") or "").strip(),
            "finish_reason": payload["choices"][0].get("finish_reason"),
            "usage": usage,
        }

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(5)
        if self.log_handle and not self.log_handle.closed:
            self.log_handle.close()

def image_part(path: Path) -> dict[str, Any]:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}

def audio_part(path: Path) -> dict[str, Any]:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}

def parse_json(text: str):
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        return json.loads(text[a:b+1])
    except json.JSONDecodeError:
        return None

def bench(model: Path) -> str:
    exe = Path(load_env(CACHE / "paths.env")["LLAMA_BENCH"])
    p = subprocess.run([str(exe), "-m", str(model), "-t", "4", "-p", "64", "-n", "64", "-r", "2"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=360)
    if p.returncode:
        raise RuntimeError(p.stdout[-6000:])
    return p.stdout

VISUAL_SCENES = [
    {"id":"right_open", "actor_x":35, "target_x":185, "obstacle":None, "target":"RIGHT", "blocked":False, "action":"MOVE_RIGHT"},
    {"id":"left_open", "actor_x":185, "target_x":35, "obstacle":None, "target":"LEFT", "blocked":False, "action":"MOVE_LEFT"},
    {"id":"right_blocked", "actor_x":35, "target_x":185, "obstacle":(110,145), "target":"RIGHT", "blocked":True, "action":"WAIT"},
    {"id":"left_blocked", "actor_x":185, "target_x":35, "obstacle":(110,145), "target":"LEFT", "blocked":True, "action":"WAIT"},
    {"id":"right_clear_obstacle_behind", "actor_x":45, "target_x":185, "obstacle":(5,25), "target":"RIGHT", "blocked":False, "action":"MOVE_RIGHT"},
    {"id":"left_clear_obstacle_behind", "actor_x":175, "target_x":35, "obstacle":(225,248), "target":"LEFT", "blocked":False, "action":"MOVE_LEFT"},
]

VISUAL_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {"type":"string", "enum":["LEFT","RIGHT"]},
        "blocked": {"type":"boolean"},
        "action": {"type":"string", "enum":["MOVE_LEFT","MOVE_RIGHT","WAIT"]},
    },
    "required":["target","blocked","action"],
    "additionalProperties":False,
}

def visual_suite(model_id: str, model: Path, mmproj: Path) -> dict[str, Any]:
    temp = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / f"wave002-{model_id}"
    temp.mkdir(parents=True, exist_ok=True)
    s = Server(model_id, model, mmproj)
    b = bench(model)
    try:
        s.start()
        meta = {"startup_seconds":s.startup_seconds,"rss_kib":rss_kib(s.pid)}
        results = []
        for scene in VISUAL_SCENES:
            path = temp / f"{scene['id']}.png"
            png(path, scene["actor_x"], scene["target_x"], scene["obstacle"])
            prompt = (
                "This synthetic 2D physics scene contains a RED actor square and a BLUE target square. "
                "A BLACK rectangle, if present between them horizontally, blocks movement. "
                "Determine whether the target is LEFT or RIGHT of the red actor; whether a black obstacle blocks "
                "the horizontal path between actor and target; and choose the immediate action. "
                "If blocked, WAIT. If clear, move toward the target."
            )
            body = {
                "model":"local",
                "messages":[{"role":"user","content":[{"type":"text","text":prompt},image_part(path)]}],
                "max_tokens":80,"temperature":0.0,"seed":100,
                "response_format":{"type":"json_schema","schema":VISUAL_SCHEMA},
                "stream":False,
            }
            out = s.request(body, timeout=300)
            parsed = parse_json(out["content"])
            score = {
                "target": bool(parsed and parsed.get("target")==scene["target"]),
                "blocked": bool(parsed and parsed.get("blocked")==scene["blocked"]),
                "action": bool(parsed and parsed.get("action")==scene["action"]),
            }
            results.append({"scene":scene,"response":out,"parsed":parsed,"score":score})
        total_fields = len(results)*3
        correct = sum(sum(int(v) for v in x["score"].values()) for x in results)
        action_correct = sum(int(x["score"]["action"]) for x in results)
        return {"bench_output":b,"server":meta,"scenes":results,
                "field_accuracy":correct/total_fields,"action_accuracy":action_correct/len(results)}
    finally:
        s.stop()

def omni_suite(model_id: str, model: Path, mmproj: Path, audio: Path) -> dict[str, Any]:
    temp = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "wave002-omni"
    temp.mkdir(parents=True, exist_ok=True)
    image = temp / "right_open.png"
    png(image, 35, 185, None)
    s = Server(model_id, model, mmproj)
    b = bench(model)
    try:
        s.start()
        meta={"startup_seconds":s.startup_seconds,"rss_kib":rss_kib(s.pid)}
        vision = s.request({
            "model":"local","messages":[{"role":"user","content":[
                {"type":"text","text":"The RED square is the actor and BLUE square is the target. Is the target LEFT or RIGHT of the actor? Answer one word."},
                image_part(image)
            ]}],"max_tokens":32,"temperature":0.0,"stream":False
        }, timeout=300)
        audio_out = s.request({
            "model":"local","messages":[{"role":"user","content":[
                {"type":"text","text":"Transcribe the spoken instruction in this audio. Return only the words you hear."},
                audio_part(audio)
            ]}],"max_tokens":64,"temperature":0.0,"stream":False
        }, timeout=300)
        mixed = s.request({
            "model":"local","messages":[{"role":"user","content":[
                {"type":"text","text":"The image shows a red actor and blue target. The audio gives a movement instruction. Does the spoken instruction agree with the direction from actor to target? Answer AGREE or DISAGREE, then one sentence."},
                image_part(image), audio_part(audio)
            ]}],"max_tokens":96,"temperature":0.0,"stream":False
        }, timeout=300)
        return {
            "bench_output":b,"server":meta,"vision":vision,"audio":audio_out,"mixed":mixed,
            "vision_expected":"RIGHT","audio_expected_phrase":"the robot should move right","mixed_expected":"AGREE"
        }
    finally:
        s.stop()

BOLT_PROMPT = (
    "Design one deterministic local behaviour kernel for a ground robot. It stores a single bounded vector when "
    "shoved. Make that memory create an interesting visible habit. Return a compact JSON object with keys "
    "trigger, state_update, action, failure_mode. No global knowledge and no randomness."
)

def bolt_suite(model_id: str, model: Path) -> dict[str, Any]:
    s=Server(model_id, model, None)
    b=bench(model)
    try:
        s.start()
        meta={"startup_seconds":s.startup_seconds,"rss_kib":rss_kib(s.pid)}
        off=s.request({
            "model":"local","messages":[{"role":"user","content":BOLT_PROMPT}],
            "max_tokens":320,"temperature":0.2,"seed":12,"stream":False,
            "chat_template_kwargs":{"enable_thinking":False}
        },timeout=600)
        on=s.request({
            "model":"local","messages":[{"role":"user","content":BOLT_PROMPT}],
            "max_tokens":1600,"temperature":0.2,"seed":12,"stream":False,
            "chat_template_kwargs":{"enable_thinking":True}
        },timeout=900)
        return {"bench_output":b,"server":meta,"thinking_off":off,"thinking_on":on}
    finally:
        s.stop()

CREATIVE_PROMPTS = [
    "Invent an embodied NPC personality shown only through movement, hesitation, obsession and one local memory. No dialogue or backstory. Keep it under 140 words and make the behaviour mechanically implementable.",
    "A sandbox has scarce heat and shelter. Invent a survival instinct that creates a recognizable personality using only local sensing, bounded internal state and physical actions. No dialogue. Under 140 words.",
    "Start from the bland rule 'move toward a nearby target'. Mutate it into a distinctive physical character without adding global knowledge, narration, or dialogue. Explain the local rule and its most interesting failure mode."
]

def run_text_model(name: str, model: Path, threads: int = 4) -> dict[str, Any]:
    s=Server(name,model,None,threads=threads,ctx=4096)
    b=bench(model)
    try:
        s.start()
        meta={"startup_seconds":s.startup_seconds,"rss_kib":rss_kib(s.pid)}
        outs=[]
        for i,prompt in enumerate(CREATIVE_PROMPTS):
            outs.append(s.request({
                "model":"local","messages":[{"role":"user","content":prompt}],
                "max_tokens":180,"temperature":0.9,"top_p":0.95,"seed":700+i,"stream":False
            },timeout=600))
        return {"bench_output":b,"server":meta,"outputs":outs}
    finally:
        s.stop()

def creative_suite(model_id: str, storyteller: Path) -> dict[str, Any]:
    paths=load_env(CACHE/"paths.env")
    sprocket=Path(paths["SPROCKET_MODEL"])
    return {
        "storyteller":run_text_model("qwen-storyteller",storyteller),
        "sprocket":run_text_model("sprocket-control",sprocket),
        "prompts":CREATIVE_PROMPTS,
    }

def main():
    if len(sys.argv) < 4:
        raise SystemExit("usage: eval_wave_002.py MODE ID MODEL [MMPROJ] [AUDIO]")
    mode, ident, model = sys.argv[1], sys.argv[2], Path(sys.argv[3]).resolve()
    mmproj = Path(sys.argv[4]).resolve() if len(sys.argv) >= 5 and sys.argv[4] != "-" else None
    audio = Path(sys.argv[5]).resolve() if len(sys.argv) >= 6 and sys.argv[5] != "-" else None
    RESULTS.mkdir(parents=True,exist_ok=True)
    LOGS.mkdir(parents=True,exist_ok=True)
    common={
        "experiment":"eval-wave-002","mode":mode,"id":ident,
        "runner":{"platform":platform.platform(),"cpu_count":os.cpu_count(),"cpu_model":cpu_model()},
        "model_file":model.name,"model_bytes":model.stat().st_size,"model_sha256":sha256(model),
        "mmproj_file":mmproj.name if mmproj else None,
        "mmproj_bytes":mmproj.stat().st_size if mmproj else None,
        "mmproj_sha256":sha256(mmproj) if mmproj else None,
    }
    if mode=="visual":
        result=visual_suite(ident,model,mmproj)
    elif mode=="omni":
        if audio is None: raise SystemExit("omni mode requires audio path")
        result=omni_suite(ident,model,mmproj,audio)
    elif mode=="bolt":
        result=bolt_suite(ident,model)
    elif mode=="creative":
        result=creative_suite(ident,model)
    else:
        raise SystemExit(f"unknown mode {mode}")
    common["result"]=result
    out=RESULTS/f"wave002-{ident}.json"
    out.write_text(json.dumps(common,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(common,indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
