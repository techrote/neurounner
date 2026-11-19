#!/usr/bin/env python3
"""Immutable, source-bound Stage E provider checkpoints and missing-only repairs."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_value(value) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def source_identity(driver: Path) -> dict[str, str]:
    paths = [driver, Path(__file__), Path(__file__).with_name("village_001b_generate_member.py")]
    return {path.name: sha256_file(path) for path in paths}


def write_once(path: Path, value) -> None:
    data = json_bytes(value)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"immutable checkpoint mismatch: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def prepare_output(out: Path, resume: Path | None, identity: dict) -> None:
    if resume is not None:
        source, target = resume.resolve(), out.resolve()
        if source == target or source in target.parents or target in source.parents:
            raise RuntimeError("resume and output must be disjoint directories")
        if not source.is_dir():
            raise RuntimeError("resume directory missing")
        for path in [resume, *resume.rglob("*")]:
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise RuntimeError(f"unsafe resume entry: {path}")
        receipt = resume / "provider-identity.json"
        if not receipt.is_file() or json.loads(receipt.read_text()) != identity:
            raise RuntimeError("provider resume identity mismatch")
    if out.is_symlink() or (out.exists() and any(out.iterdir())):
        raise RuntimeError("output must be empty; use --resume with a separate output")
    if resume is not None:
        shutil.copytree(resume, out, dirs_exist_ok=True)
    else:
        out.mkdir(parents=True, exist_ok=True)
    write_once(out / "provider-identity.json", identity)


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


DECODER = json.JSONDecoder(object_pairs_hook=_object_pairs)


def partial_object(raw: str) -> dict:
    """Only complete fields/array-prefix elements in the first outer object.

    Never search for JSON inside malformed strings, nested objects, prose, or a
    suffix after a broken array element. A complete syntactic delimiter is needed
    before admitting a field or element. Duplicate keys stop admission.
    """
    header = re.match(r'\s*(?:```(?:json)?\s*)?\{', raw)
    if not header:
        return {}
    pos, result = header.end(), {}
    while pos < len(raw):
        while pos < len(raw) and raw[pos].isspace():
            pos += 1
        if pos == len(raw) or raw[pos] == "}":
            break
        try:
            key, pos = DECODER.raw_decode(raw, pos)
        except (ValueError, TypeError):
            break
        if not isinstance(key, str) or key in result:
            break
        while pos < len(raw) and raw[pos].isspace():
            pos += 1
        if pos == len(raw) or raw[pos] != ":":
            break
        pos += 1
        while pos < len(raw) and raw[pos].isspace():
            pos += 1
        start = pos
        try:
            value, end = DECODER.raw_decode(raw, start)
        except (ValueError, TypeError):
            if start < len(raw) and raw[start] == "[":
                values, pos = [], start + 1
                while pos < len(raw):
                    while pos < len(raw) and raw[pos].isspace():
                        pos += 1
                    if pos == len(raw) or raw[pos] == "]":
                        break
                    try:
                        value, end = DECODER.raw_decode(raw, pos)
                    except (ValueError, TypeError):
                        break
                    while end < len(raw) and raw[end].isspace():
                        end += 1
                    if end == len(raw) or raw[end] not in {",", "]"}:
                        break
                    values.append(value)
                    if raw[end] == "]":
                        break
                    pos = end + 1
                result[key] = values
            break
        while end < len(raw) and raw[end].isspace():
            end += 1
        if end == len(raw) or raw[end] not in {",", "}"}:
            break
        result[key] = value
        if raw[end] == "}":
            break
        pos = end + 1
    return result


def parse_rows(raw: str, expected: list[str], *, array_key: str, validator) -> dict:
    rows = partial_object(raw).get(array_key)
    if not isinstance(rows, list):
        return {}
    accepted = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        ident = row.get("id")
        if ident not in expected or ident in accepted:
            continue
        try:
            parsed = validator(json.dumps({array_key: [row]}), [ident])
        except (ValueError, TypeError, KeyError):
            continue
        accepted[ident] = parsed[ident]
    return accepted


def payload_for(model: dict, prompt: str, system: str, max_tokens: int, *, repair=False) -> dict:
    result = {
        "model": model["id"],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1 if repair else 0.2,
        "top_p": 0.9,
        "stream": False,
    }
    if "reasoning_effort" in model:
        result["reasoning_effort"] = "low" if repair else model["reasoning_effort"]
    return result


class Checkpoint:
    def __init__(self, out: Path, *, campaign: dict, key: str, offline: bool, api, logical_request_limit: int = 42):
        self.out = out
        self.rawdir = out / "provider"
        self.rawdir.mkdir(exist_ok=True)
        self.campaign = campaign
        self.key = key
        self.offline = offline
        self.api = api
        self.visited: set[str] = set()
        self.new_calls = 0
        self.replayed_calls = 0
        if not isinstance(logical_request_limit, int) or isinstance(logical_request_limit, bool) or logical_request_limit < 1:
            raise ValueError("logical request limit must be a positive integer")
        self.logical_request_limit = logical_request_limit
        self.logical_labels = {
            re.sub(r"-transport-resume-[0-9]{2}$", "", path.name.removesuffix("-request.json"))
            for path in self.rawdir.glob("*-request.json")
        }
        if len(self.logical_labels) > self.logical_request_limit:
            raise RuntimeError("archived logical requests exceed whole-lane provider budget")

    def request(self, label: str, payload: dict, role: str) -> tuple[str, str]:
        """Replay exact archived request; use a new name after failed transport.

        Three transport launches per logical request are permitted across all
        resumes. Each launch retains the campaign's fixed HTTP attempt budget.
        """
        if re.search(r"-transport-resume-[0-9]{2}$", label):
            raise RuntimeError("logical request label uses reserved transport suffix")
        if label not in self.logical_labels and len(self.logical_labels) >= self.logical_request_limit:
            raise RuntimeError(f"whole-lane provider budget exhausted ({self.logical_request_limit} logical requests): {label}")
        for generation in range(3):
            actual = label if generation == 0 else f"{label}-transport-resume-{generation:02d}"
            request_path = self.rawdir / f"{actual}-request.json"
            response_path = self.rawdir / f"{actual}-provider.json"
            content_path = self.rawdir / f"{actual}-content.txt"
            attempts_path = self.rawdir / f"{actual}-attempts.json"
            if request_path.exists():
                self.visited.add(actual)
                request = json.loads(request_path.read_text())
                if request.get("payload") != payload or request.get("role") != role or request.get("provider") != "nvidia":
                    raise RuntimeError(f"archived request/source mismatch: {actual}")
                if response_path.exists() or content_path.exists():
                    if not response_path.is_file() or not content_path.is_file() or not attempts_path.is_file():
                        raise RuntimeError(f"incomplete archived response custody: {actual}")
                    response = json.loads(response_path.read_text())
                    try:
                        content = (response["choices"][0]["message"].get("content") or "").strip()
                    except (KeyError, IndexError, TypeError, AttributeError) as exc:
                        raise RuntimeError(f"invalid archived provider response: {actual}") from exc
                    if content_path.read_text() != content + "\n":
                        raise RuntimeError(f"archived content/provider mismatch: {actual}")
                    self.replayed_calls += 1
                    return content, actual
                # Only transport failure or interrupted request exists. Retain it.
                continue
            if any(self.rawdir.glob(f"{actual}-*")):
                raise RuntimeError(f"provider history without request: {actual}")
            if self.offline or not self.key:
                raise RuntimeError(f"fresh provider work required for {label}; offline/key unavailable")
            self.visited.add(actual)
            self.logical_labels.add(label)
            self.new_calls += 1
            print(f"provider request {actual}", file=sys.stderr, flush=True)
            raw, _, _ = self.api(
                key=self.key, endpoint=self.campaign["nvidia"]["endpoint"],
                payload=payload, outdir=self.rawdir, label=actual,
                attempts_limit=int(self.campaign["nvidia"]["generation"]["attempts"]), role=role,
            )
            print(f"provider response {actual}", file=sys.stderr, flush=True)
            return raw, actual
        raise RuntimeError(f"transport launch budget exhausted for {label}")

    def collect(self, *, label: str, expected: list[str], prompt_factory, system: str,
                model: dict, max_tokens: int, role: str, parser) -> tuple[dict, list[str]]:
        if not expected or len(set(expected)) != len(expected):
            raise RuntimeError("provider expected IDs must be nonempty and unique")
        accepted, refs = {}, []

        def request(selected, call_label, repair):
            prompt = prompt_factory(selected, repair)
            raw, actual = self.request(call_label, payload_for(model, prompt, system, max_tokens, repair=repair), role + ("-repair" if repair else ""))
            refs.append(actual)
            parsed = parser(raw, selected)
            for ident in selected:
                if ident in parsed:
                    accepted.setdefault(ident, parsed[ident])

        request(expected, label, False)
        for round_index, size in enumerate((3, 2, 1), 1):
            missing = [ident for ident in expected if ident not in accepted]
            if not missing:
                break
            for offset in range(0, len(missing), size):
                selected = missing[offset:offset + size]
                request(selected, f"{label}-missing-r{round_index}-b{offset // size + 1:02d}", True)
        missing = [ident for ident in expected if ident not in accepted]
        if missing:
            raise ValueError(f"provider judgments incomplete after bounded missing-only repair: {label}: {missing}")
        return accepted, refs

    def finish(self, filename: str, result: dict) -> None:
        requests = {p.name.removesuffix("-request.json") for p in self.rawdir.glob("*-request.json")}
        if requests != self.visited:
            raise RuntimeError(f"unconsumed provider history: {sorted(requests - self.visited)}")
        # A copied terminal artifact is evidence to verify, never an authority to
        # bypass replay. Byte equality guards the terminal reconstruction.
        write_once(self.out / filename, result)
        files = {}
        for path in sorted([self.out / "provider-identity.json", *self.rawdir.rglob("*")]):
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise RuntimeError(f"unsafe provider evidence path: {path}")
            if path.is_file():
                files[path.relative_to(self.out).as_posix()] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        write_once(self.out / "provider-evidence-hashes.json", {"schema_version": 1, "files": files})
        # A per-invocation receipt is deliberately separate from immutable results.
        receipt = {"schema_version": 1, "offline": self.offline, "new_provider_requests": self.new_calls, "replayed_provider_responses": self.replayed_calls, "logical_request_limit": self.logical_request_limit, "logical_requests_archived": len(self.logical_labels)}
        suffix = 1
        while (self.out / f"execution-{suffix:03d}.json").exists():
            suffix += 1
        write_once(self.out / f"execution-{suffix:03d}.json", receipt)
