#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("NEURO_CACHE", ROOT / ".cache" / "neurounner"))
PATHS_FILE = CACHE / "paths.env"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"


def load_kv(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        data[key] = value
    return data


def runner_info() -> dict[str, Any]:
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    mem_kib = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_kib = int(line.split()[1])
                break
    return {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpu_model": cpu_model,
        "mem_total_kib": mem_kib,
    }


def rss_kib(pid: int) -> int | None:
    status = Path(f"/proc/{pid}/status")
    if not status.exists():
        return None
    for line in status.read_text(errors="replace").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return None


@dataclass
class Server:
    name: str
    executable: Path
    model: Path
    port: int
    threads: int
    context: int = 2048

    def __post_init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.log_handle = None
        self.log_path = LOGS / self.name / "server.log"
        self.startup_seconds: float | None = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("w", encoding="utf-8")
        cmd = [
            str(self.executable),
            "-m", str(self.model),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-t", str(self.threads),
            "-c", str(self.context),
        ]
        start = time.monotonic()
        self.process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + 90
        last_error = ""
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.log_handle.flush()
                tail = self.log_path.read_text(errors="replace")[-6000:]
                raise RuntimeError(f"{self.name} server exited during startup:\n{tail}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1) as response:
                    if response.status == 200:
                        self.startup_seconds = time.monotonic() - start
                        return
            except Exception as exc:
                last_error = repr(exc)
            time.sleep(0.25)
        raise RuntimeError(f"{self.name} server did not become ready: {last_error}")

    @property
    def pid(self) -> int:
        assert self.process is not None
        return self.process.pid

    def chat(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float = 0.95,
        seed: int = 1234,
    ) -> dict[str, Any]:
        body = {
            "model": "local",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "stream": False,
        }
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{self.name} chat HTTP {exc.code}: {detail}") from exc
        elapsed = time.monotonic() - start
        content = payload["choices"][0]["message"].get("content") or ""
        usage = payload.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        return {
            "speaker": self.name,
            "content": content.strip(),
            "elapsed_seconds": elapsed,
            "completion_tokens": completion_tokens,
            "tokens_per_second": (completion_tokens / elapsed) if completion_tokens else None,
            "finish_reason": payload["choices"][0].get("finish_reason"),
        }

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log_handle is not None:
            self.log_handle.close()


def print_turn(turn: dict[str, Any]) -> None:
    rate = turn.get("tokens_per_second")
    rate_text = f"{rate:.1f} tok/s" if rate is not None else "rate unavailable"
    print(f"\n--- {turn['speaker']} | {turn['elapsed_seconds']:.3f}s | {rate_text} ---")
    print(turn["content"], flush=True)


def save(mode: str, payload: dict[str, Any], markdown: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{mode}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (RESULTS / f"{mode}.md").write_text(markdown.rstrip() + "\n", encoding="utf-8")


def server_meta(servers: list[Server]) -> list[dict[str, Any]]:
    return [
        {
            "name": s.name,
            "port": s.port,
            "threads": s.threads,
            "pid": s.pid,
            "rss_kib_after_load": rss_kib(s.pid),
            "startup_seconds": s.startup_seconds,
            "model": s.model.name,
        }
        for s in servers
    ]


def run_sprocket3(server_bin: Path, sprocket: Path) -> None:
    servers = [
        Server(f"sprocket-{label}", server_bin, sprocket, 18101 + i, 1)
        for i, label in enumerate(("a", "b", "c"))
    ]
    prompts = [
        "Invent one strange, mechanically implementable NPC reflex for a physics sandbox. Use local sensing, at most two persistent state values, and movement/pushing/waiting actions. Give only the behaviour idea in a compact paragraph.",
        "Invent one tiny NPC behaviour based on a physical mistake: slipping, bumping, falling, getting stuck, or losing support. Turn the mistake into a persistent quirk using minimal state. Compact paragraph only.",
        "Invent one odd local interaction rule that could make a simple sandbox creature look surprisingly intentional. Keep it mechanically explicit and small enough to implement directly. Compact paragraph only.",
    ]
    try:
        for s in servers:
            s.start()
        meta = server_meta(servers)
        wall_start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(s.chat, p, max_tokens=160, temperature=0.95, seed=101 + i * 101)
                for i, (s, p) in enumerate(zip(servers, prompts))
            ]
            turns = [f.result() for f in futures]
        wall = time.monotonic() - wall_start
        for t in turns:
            print_turn(t)
        total_tokens = sum(t["completion_tokens"] for t in turns)
        payload = {
            "experiment": "sprocket3",
            "runner": runner_info(),
            "servers": meta,
            "wall_seconds": wall,
            "completion_tokens": total_tokens,
            "aggregate_completion_tokens_per_second": (total_tokens / wall) if total_tokens else None,
            "turns": turns,
        }
        md = ["# sprocket3", "", f"Concurrent wall time: **{wall:.3f}s**", ""]
        for t in turns:
            md += [f"## {t['speaker']}", "", t["content"], "", f"{t['completion_tokens']} tokens in {t['elapsed_seconds']:.3f}s.", ""]
        save("sprocket3", payload, "\n".join(md))
    finally:
        for s in servers:
            s.stop()


def run_lfm1(server_bin: Path, lfm: Path) -> None:
    server = Server("lfm-1.2b", server_bin, lfm, 18111, 4)
    prompts = [
        "Invent one compact behaviour kernel for a physics-sandbox NPC. It must use only local observations and at most three small persistent state variables. Describe trigger, state update, and visible action in one paragraph.",
        "Invent a second behaviour kernel that could create an emergent-looking habit without machine learning. Prefer hysteresis, delayed response, refractory periods, or competing drives. One concise paragraph.",
        "Invent a third tiny behaviour mechanism that would interact interestingly with obstacles or other actors. Keep it simple enough to implement as a deterministic state machine. One concise paragraph.",
    ]
    try:
        server.start()
        meta = server_meta([server])
        turns = []
        wall_start = time.monotonic()
        for i, prompt in enumerate(prompts):
            turn = server.chat(prompt, max_tokens=160, temperature=0.35, seed=410 + i)
            turns.append(turn)
            print_turn(turn)
        wall = time.monotonic() - wall_start
        total_tokens = sum(t["completion_tokens"] for t in turns)
        payload = {
            "experiment": "lfm1",
            "runner": runner_info(),
            "servers": meta,
            "wall_seconds": wall,
            "completion_tokens": total_tokens,
            "aggregate_completion_tokens_per_second": (total_tokens / wall) if total_tokens else None,
            "turns": turns,
        }
        md = ["# lfm1", "", f"Sequential inference wall time: **{wall:.3f}s**", ""]
        for i, t in enumerate(turns, 1):
            md += [f"## Generation {i}", "", t["content"], "", f"{t['completion_tokens']} tokens in {t['elapsed_seconds']:.3f}s.", ""]
        save("lfm1", payload, "\n".join(md))
    finally:
        server.stop()


def run_pair(server_bin: Path, sprocket: Path, lfm: Path) -> None:
    sprocket_server = Server("sprocket", server_bin, sprocket, 18121, 2)
    lfm_server = Server("lfm-1.2b", server_bin, lfm, 18122, 2)
    servers = [sprocket_server, lfm_server]
    prompts = [
        "Invent a weird tiny physical NPC instinct. Local sensing only, minimal state, concrete actions. Do not explain your process; just give the idea.",
        "Invent a compact deterministic behaviour kernel for a physics NPC that could look learned or habitual. State the trigger, memory/state, and action in one paragraph.",
    ]
    try:
        for s in servers:
            s.start()
        meta = server_meta(servers)
        wall_start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(sprocket_server.chat, prompts[0], max_tokens=180, temperature=0.95, seed=515)
            f2 = pool.submit(lfm_server.chat, prompts[1], max_tokens=180, temperature=0.35, seed=616)
            turns = [f1.result(), f2.result()]
        wall = time.monotonic() - wall_start
        for t in turns:
            print_turn(t)
        total_tokens = sum(t["completion_tokens"] for t in turns)
        payload = {
            "experiment": "pair",
            "runner": runner_info(),
            "servers": meta,
            "wall_seconds": wall,
            "completion_tokens": total_tokens,
            "aggregate_completion_tokens_per_second": (total_tokens / wall) if total_tokens else None,
            "turns": turns,
        }
        md = ["# pair", "", f"Concurrent wall time: **{wall:.3f}s**", ""]
        for t in turns:
            md += [f"## {t['speaker']}", "", t["content"], "", f"{t['completion_tokens']} tokens in {t['elapsed_seconds']:.3f}s.", ""]
        save("pair", payload, "\n".join(md))
    finally:
        for s in servers:
            s.stop()


def run_ring(server_bin: Path, sprocket: Path, lfm: Path) -> None:
    participants = [
        (
            Server("sprocket-a-mutator", server_bin, sprocket, 18131, 1),
            "You are the mutator. Distort the previous idea into a stranger but still mechanically implementable local NPC behaviour.",
            0.95,
        ),
        (
            Server("sprocket-b-instinct", server_bin, sprocket, 18132, 1),
            "You are the instinct specialist. Add one crude memory, reflex, or drive that changes the visible physical behaviour.",
            0.95,
        ),
        (
            Server("sprocket-c-physics", server_bin, sprocket, 18133, 1),
            "You are the physics goblin. Alter the idea using contact, support, velocity, obstacles, pushing, jumping, waiting, or turning.",
            0.95,
        ),
        (
            Server("lfm-1.2b-synthesizer", server_bin, lfm, 18134, 1),
            "You are the synthesizer. Preserve the most interesting mechanism, remove incoherent parts, and return one concrete implementable behaviour kernel.",
            0.35,
        ),
    ]
    servers = [p[0] for p in participants]
    message = (
        "A small ground robot becomes briefly obsessed with the last direction from which it was unexpectedly pushed. "
        "Develop this into an interesting tiny behaviour kernel."
    )
    turns: list[dict[str, Any]] = []
    try:
        for s in servers:
            s.start()
        meta = server_meta(servers)
        wall_start = time.monotonic()
        turn_no = 0
        for lap in range(2):
            for server, role, temperature in participants:
                turn_no += 1
                prompt = (
                    "We are passing one behaviour concept around a circular design conversation. "
                    "Reply in one to three short sentences. Advance the mechanism itself; no headings, bullets, greetings, or meta commentary.\n\n"
                    f"Your role: {role}\n\n"
                    f"Previous contribution:\n{message}"
                )
                turn = server.chat(
                    prompt,
                    max_tokens=96,
                    temperature=temperature,
                    seed=700 + turn_no,
                )
                turn["lap"] = lap + 1
                turn["turn"] = turn_no
                turns.append(turn)
                print_turn(turn)
                if turn["content"]:
                    message = turn["content"]
        wall = time.monotonic() - wall_start
        total_tokens = sum(t["completion_tokens"] for t in turns)
        payload = {
            "experiment": "ring",
            "runner": runner_info(),
            "servers": meta,
            "laps": 2,
            "wall_seconds": wall,
            "completion_tokens": total_tokens,
            "aggregate_completion_tokens_per_second": (total_tokens / wall) if total_tokens else None,
            "turns": turns,
            "final_contribution": message,
        }
        md = ["# ring", "", f"Two laps / eight turns in **{wall:.3f}s**.", ""]
        for t in turns:
            md += [f"## Turn {t['turn']} — {t['speaker']}", "", t["content"], ""]
        md += ["## Final contribution", "", message, ""]
        save("ring", payload, "\n".join(md))
    finally:
        for s in servers:
            s.stop()


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"sprocket3", "lfm1", "pair", "ring"}:
        print("usage: experiment.py sprocket3|lfm1|pair|ring", file=sys.stderr)
        return 2
    if not PATHS_FILE.exists():
        raise SystemExit("cache paths.env missing; run scripts/bootstrap.sh first")
    paths = load_kv(PATHS_FILE)
    server_bin = Path(paths["LLAMA_SERVER"])
    sprocket = Path(paths["SPROCKET_MODEL"])
    lfm = Path(paths["LFM_MODEL"])
    mode = sys.argv[1]

    print(json.dumps({"experiment": mode, "runner": runner_info()}, indent=2), flush=True)
    if mode == "sprocket3":
        run_sprocket3(server_bin, sprocket)
    elif mode == "lfm1":
        run_lfm1(server_bin, lfm)
    elif mode == "pair":
        run_pair(server_bin, sprocket, lfm)
    else:
        run_ring(server_bin, sprocket, lfm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
