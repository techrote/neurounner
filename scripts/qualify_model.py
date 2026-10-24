#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("NEURO_CACHE", ROOT / ".cache" / "neurounner"))
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"


def load_kv(path: Path) -> dict[str, str]:
    out = {}
    for raw in path.read_text().splitlines():
        if raw.strip() and not raw.lstrip().startswith("#"):
            key, value = raw.split("=", 1)
            out[key] = value
    return out


def rss_kib(pid: int) -> int | None:
    path = Path(f"/proc/{pid}/status")
    if not path.exists():
        return None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return None


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def chat(port: int, prompt: str, *, max_tokens: int, disable_thinking: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "top_p": 0.9,
        "stream": False,
        "seed": 314159,
    }
    if disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=240) as response:
        payload = json.loads(response.read())
    elapsed = time.monotonic() - start
    message = payload["choices"][0]["message"]
    usage = payload.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "elapsed_seconds": elapsed,
        "completion_tokens": completion_tokens,
        "tokens_per_second": completion_tokens / elapsed if completion_tokens else None,
        "content": (message.get("content") or "").strip(),
        "reasoning_content": (message.get("reasoning_content") or "").strip(),
        "finish_reason": payload["choices"][0].get("finish_reason"),
        "usage": usage,
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: qualify_model.py CANDIDATE_ID MODEL_PATH", file=sys.stderr)
        return 2
    candidate_id = sys.argv[1]
    model_path = Path(sys.argv[2]).resolve()
    candidates = json.loads((ROOT / "config/candidates.json").read_text())
    spec = candidates[candidate_id]
    paths = load_kv(CACHE / "paths.env")
    bench_bin = Path(paths["LLAMA_BENCH"])
    server_bin = Path(paths["LLAMA_SERVER"])

    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    bench_cmd = [
        str(bench_bin), "-m", str(model_path), "-t", "4",
        "-p", "64", "-n", "64", "-r", "2",
    ]
    bench_start = time.monotonic()
    bench = subprocess.run(bench_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    bench_elapsed = time.monotonic() - bench_start
    (LOGS / f"{candidate_id}-bench.log").write_text(bench.stdout)
    if bench.returncode != 0:
        raise RuntimeError(f"llama-bench failed for {candidate_id}:\n{bench.stdout[-5000:]}")

    port = 18300
    server_log = (LOGS / f"{candidate_id}-server.log").open("w")
    process = subprocess.Popen(
        [
            str(server_bin), "-m", str(model_path),
            "--host", "127.0.0.1", "--port", str(port),
            "-t", "4", "-c", "4096",
        ],
        stdout=server_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    startup_start = time.monotonic()
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if process.poll() is not None:
                server_log.close()
                raise RuntimeError(f"server failed for {candidate_id}:\n{Path(server_log.name).read_text(errors='replace')[-6000:]}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except Exception:
                pass
            time.sleep(0.25)
        else:
            raise RuntimeError(f"server startup timeout for {candidate_id}")

        startup_seconds = time.monotonic() - startup_start
        resident_kib = rss_kib(process.pid)
        prompt = (
            "Invent one deterministic behaviour kernel for a small physics-sandbox NPC. "
            "Use only local observations, at most three bounded state values, and actions such as "
            "moving, turning, jumping, pushing or waiting. Give a concrete trigger, state update, "
            "visible action and one interesting failure mode. Keep the final answer under 120 words."
        )
        concise = chat(port, prompt, max_tokens=192, disable_thinking=True)
        reasoning_probe = None
        if spec.get("reasoning"):
            reasoning_probe = chat(
                port,
                "Reason briefly, then improve this mechanism: after being pushed, a robot remembers the push direction and changes its next movement. Make the result deterministic and physically local.",
                max_tokens=256,
                disable_thinking=False,
            )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(5)
        if not server_log.closed:
            server_log.close()

    data = {
        "candidate_id": candidate_id,
        "spec": spec,
        "runner": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "cpu_model": cpu_model(),
        },
        "model_bytes": model_path.stat().st_size,
        "bench_elapsed_seconds": bench_elapsed,
        "bench_output": bench.stdout,
        "server_startup_seconds": startup_seconds,
        "server_rss_kib": resident_kib,
        "concise_probe": concise,
        "reasoning_probe": reasoning_probe,
    }
    (RESULTS / f"qualify-{candidate_id}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    md = [
        f"# Qualification: {spec['label']}",
        "",
        f"- File: \`{spec['file']}\`",
        f"- Size: {model_path.stat().st_size / (1024**3):.3f} GiB",
        f"- Server RSS after load: {(resident_kib or 0) / (1024**2):.3f} GiB",
        f"- Startup: {startup_seconds:.3f}s",
        "",
        "## llama-bench",
        "",
        "~~~text",
        bench.stdout.strip(),
        "~~~",
        "",
        "## Concise behaviour probe",
        "",
        concise["content"] or "(no final content)",
        "",
        f"Completion: {concise['completion_tokens']} tokens in {concise['elapsed_seconds']:.3f}s.",
    ]
    if concise["reasoning_content"]:
        md += ["", "Reasoning content:", "", concise["reasoning_content"]]
    if reasoning_probe is not None:
        md += [
            "",
            "## Reasoning probe",
            "",
            reasoning_probe["content"] or "(no final content)",
            "",
            f"Completion: {reasoning_probe['completion_tokens']} tokens in {reasoning_probe['elapsed_seconds']:.3f}s.",
        ]
        if reasoning_probe["reasoning_content"]:
            md += ["", "Reasoning content:", "", reasoning_probe["reasoning_content"]]
    (RESULTS / f"qualify-{candidate_id}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
