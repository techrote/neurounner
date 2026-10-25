#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RETRYABLE = {408, 429, 500, 502, 503, 504}


def sanitize(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return value.replace(secret, "<REDACTED>") if secret else value
    if isinstance(value, list):
        return [sanitize(x, secret) for x in value]
    if isinstance(value, dict):
        return {str(k): sanitize(v, secret) for k, v in value.items()}
    return value


def write_json(path: Path, value: Any, secret: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = sanitize(value, secret)
    text = json.dumps(clean, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if secret and secret in text:
        raise RuntimeError(f"secret scrub failed for {path}")
    path.write_text(text)


def build_payload(role: str, model: dict, max_tokens: int) -> dict:
    payload = {
        "model": model["id"],
        "messages": [
            {
                "role": "system",
                "content": "This is a transport/protocol preflight. Respond briefly.",
            },
            {
                "role": "user",
                "content": (
                    "Return a tiny response proving the endpoint accepts text chat. "
                    "Do not include secrets or external data."
                ),
            },
        ],
        "max_tokens": max_tokens,
        "temperature": float(model.get("temperature", 0.2)),
        "top_p": float(model.get("top_p", 0.95)),
        "stream": False,
    }
    if "reasoning_effort" in model:
        payload["reasoning_effort"] = model["reasoning_effort"]
    return payload


def call_one(
    *,
    endpoint: str,
    role: str,
    model: dict,
    key: str,
    outdir: Path,
    max_tokens: int,
    attempts_limit: int,
) -> dict:
    payload = build_payload(role, model, max_tokens)
    request_record = {
        "schema_version": 1,
        "provider": "nvidia",
        "role": role,
        "endpoint": endpoint,
        "remote_unpinned_backend": True,
        "payload": payload,
        "authorization_archived": False,
    }
    write_json(outdir / f"{role}-request.json", request_record, key)

    headers = {
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "neurounner-village001a-preflight",
    }
    body = json.dumps(payload).encode("utf-8")
    attempts = []

    for attempt in range(1, attempts_limit + 1):
        started = time.monotonic()
        req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                raw = response.read().decode("utf-8")
                elapsed = time.monotonic() - started
                obj = json.loads(raw)
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": response.status,
                        "elapsed_seconds": elapsed,
                    }
                )
                write_json(outdir / f"{role}-provider.json", obj, key)

                choices = obj.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise RuntimeError("provider JSON has no choices")
                message = choices[0].get("message")
                if not isinstance(message, dict):
                    raise RuntimeError("provider choice has no message object")

                content = message.get("content")
                reasoning = message.get("reasoning_content")
                if not isinstance(content, str) and not isinstance(reasoning, str):
                    raise RuntimeError("message has neither content nor reasoning_content")

                return {
                    "role": role,
                    "model": model["id"],
                    "ok": True,
                    "http_status": response.status,
                    "elapsed_seconds": elapsed,
                    "finish_reason": choices[0].get("finish_reason"),
                    "content_chars": len(content) if isinstance(content, str) else 0,
                    "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0,
                    "usage": obj.get("usage"),
                    "attempts": attempts,
                }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            elapsed = time.monotonic() - started
            attempts.append(
                {
                    "attempt": attempt,
                    "status": exc.code,
                    "elapsed_seconds": elapsed,
                    "detail": sanitize(detail[:8000], key),
                }
            )
            if exc.code not in RETRYABLE or attempt == attempts_limit:
                raise RuntimeError(
                    f"{role}/{model['id']} HTTP {exc.code}: {sanitize(detail[:1000], key)}"
                ) from exc
            retry_after = exc.headers.get("Retry-After")
            if retry_after and retry_after.replace(".", "", 1).isdigit():
                wait = float(retry_after)
            else:
                wait = min(30.0, 2.0 ** attempt)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as exc:
            elapsed = time.monotonic() - started
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "transport_error",
                    "elapsed_seconds": elapsed,
                    "detail": sanitize(repr(exc), key),
                }
            )
            if attempt == attempts_limit:
                raise RuntimeError(f"{role}/{model['id']} transport failure: {exc}") from exc
            time.sleep(min(30.0, 2.0 ** attempt))

    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/village-001/config.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--attempts", type=int, default=4)
    args = parser.parse_args()

    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        raise SystemExit("NVIDIA_API_KEY is required; provider gate fails closed")

    cfg = json.loads(args.config.read_text())
    nv = cfg["nvidia"]
    endpoint = nv["endpoint"]
    models = nv["models"]

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "experiment": "VILLAGE-001A",
        "provider": "nvidia",
        "endpoint": endpoint,
        "remote_unpinned_backend": True,
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "results": [],
    }

    try:
        for role in ("compiler", "critic", "adjudicator"):
            result = call_one(
                endpoint=endpoint,
                role=role,
                model=models[role],
                key=key,
                outdir=args.out,
                max_tokens=args.max_tokens,
                attempts_limit=args.attempts,
            )
            summary["results"].append(result)
            print(
                f"{role}: PASS model={result['model']} "
                f"status={result['http_status']} elapsed={result['elapsed_seconds']:.2f}s"
            )
    except Exception as exc:
        summary["ok"] = False
        summary["failure"] = sanitize(repr(exc), key)
        write_json(args.out / "summary.json", summary, key)
        raise

    summary["ok"] = True
    write_json(args.out / "summary.json", summary, key)
    print("VILLAGE-001A NVIDIA provider preflight: PASS")


if __name__ == "__main__":
    main()
