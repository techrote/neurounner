"""Append-only local-generation custody for the fixed VILLAGE-001E campaign.

Each row is bound to exact inputs and to its preceding checkpoint. Resume copies
the archived bytes into an empty, disjoint output directory; it never rewrites a
prior model response. This module deliberately has no ML dependencies.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Callable


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_immutable(path: Path, data: bytes) -> None:
    """Atomically publish new bytes, or verify an already-published exact copy."""
    if path.is_symlink():
        raise RuntimeError(f"symlink is not an evidence file: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise RuntimeError(f"refusing to replace existing evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as fh:
        temporary = Path(fh.name)
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        # Linking publishes atomically and refuses an intervening existing path.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_token_custody(row: dict, *, maximum: int) -> None:
    for field in ("input_token_ids", "generated_token_ids"):
        ids = row.get(field)
        if not isinstance(ids, list) or any(type(x) is not int or x < 0 for x in ids):
            raise RuntimeError(f"invalid token custody: {field}")
    if not row["input_token_ids"]:
        raise RuntimeError("empty native model prompt")
    if type(row.get("generated_tokens")) is not int:
        raise RuntimeError("invalid generated token count")
    if row["generated_tokens"] != len(row["generated_token_ids"]):
        raise RuntimeError("generated token count/custody mismatch")
    if not 0 <= row["generated_tokens"] <= maximum:
        raise RuntimeError("generated token budget exceeded")
    if not isinstance(row.get("rendered_prompt"), str) or not row["rendered_prompt"]:
        raise RuntimeError("missing rendered native prompt")


class GenerationCheckpoint:
    def __init__(
        self, *, out: Path, resume: Path | None, identity: dict,
        keys: list[str], terminal_name: str,
        validate_row: Callable[[str, dict, list[dict]], None],
    ) -> None:
        self.out = out
        self.keys = keys
        self.terminal_name = terminal_name
        self.validate_row = validate_row
        if not keys or len(keys) != len(set(keys)):
            raise RuntimeError("generation keys must be nonempty and unique")
        if any(not key or Path(key).name != key or key in (".", "..") for key in keys):
            raise RuntimeError("unsafe generation key")
        if out.is_symlink():
            raise RuntimeError("generation output may not be a symlink")
        if resume is not None:
            source = resume.resolve()
            target = out.resolve()
            if source == target or source in target.parents or target in source.parents:
                raise RuntimeError("resume and output directories must be disjoint")
            if not resume.is_dir() or resume.is_symlink():
                raise RuntimeError("resume source must be an ordinary directory")
            if not (resume / "generation-identity.json").is_file():
                raise RuntimeError("resume source has no generation identity")
            if out.exists() and (not out.is_dir() or any(out.iterdir())):
                raise RuntimeError("resume output must be empty")
            for item in resume.rglob("*"):
                if item.is_symlink() or not (item.is_file() or item.is_dir()):
                    raise RuntimeError(f"unsafe resume entry: {item}")
            shutil.copytree(resume, out, dirs_exist_ok=True)
        elif out.exists() and any(out.iterdir()):
            raise RuntimeError("existing output requires --resume into a new directory")
        out.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        encoded_identity = json_bytes(identity)
        self.identity_sha256 = hashlib.sha256(encoded_identity).hexdigest()
        write_immutable(out / "generation-identity.json", encoded_identity)
        allowed = {"generation-identity.json", "rows", terminal_name}
        for item in out.iterdir():
            custody = re.fullmatch(r"resume-artifact-custody-([0-9a-f]{64})\.json", item.name)
            if custody:
                if item.is_symlink() or not item.is_file() or sha256_file(item) != custody.group(1):
                    raise RuntimeError("resume artifact custody hash mismatch")
                continue
            if item.name not in allowed and not item.name.startswith(".pending-"):
                raise RuntimeError(f"unexpected generation evidence entry: {item}")
        rows_dir = out / "rows"
        if rows_dir.is_symlink():
            raise RuntimeError("checkpoint rows directory may not be a symlink")
        rows_dir.mkdir(exist_ok=True)
        paths = sorted(rows_dir.glob("*.json"))
        expected_names = [f"{i:03d}-{key}.json" for i, key in enumerate(keys, 1)]
        if [p.name for p in paths] != expected_names[:len(paths)]:
            raise RuntimeError("generation checkpoints are not a consecutive prefix")
        for item in rows_dir.iterdir():
            if item.is_symlink() or not item.is_file():
                raise RuntimeError(f"unsafe generation row: {item}")
            if item not in paths and not item.name.startswith(".pending-"):
                raise RuntimeError(f"unexpected generation row: {item}")
        self.rows: list[dict] = []
        self.previous_sha256 = self.identity_sha256
        for index, path in enumerate(paths):
            payload = json.loads(path.read_text())
            self._validate_payload(payload, index)
            self.rows.append(payload["row"])
            self.previous_sha256 = sha256_file(path)

    def _validate_payload(self, payload: dict, index: int) -> None:
        expected = {
            "schema": "village-001e-generation-row-v1",
            "identity_sha256": self.identity_sha256,
            "key": self.keys[index],
            "ordinal": index + 1,
            "previous_checkpoint_sha256": self.previous_sha256,
        }
        if not isinstance(payload, dict):
            raise RuntimeError("generation checkpoint must be an object")
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(f"generation checkpoint {key} mismatch")
        if not isinstance(payload.get("row"), dict):
            raise RuntimeError("generation checkpoint row must be an object")
        self.validate_row(self.keys[index], payload["row"], self.rows)

    def append(self, key: str, row: dict) -> None:
        index = len(self.rows)
        if index >= len(self.keys) or key != self.keys[index]:
            raise RuntimeError("generation checkpoint append is out of order")
        payload = {
            "schema": "village-001e-generation-row-v1",
            "identity_sha256": self.identity_sha256,
            "key": key,
            "ordinal": index + 1,
            "previous_checkpoint_sha256": self.previous_sha256,
            "row": row,
        }
        self._validate_payload(payload, index)
        path = self.out / "rows" / f"{index + 1:03d}-{key}.json"
        write_immutable(path, json_bytes(payload))
        self.rows.append(row)
        self.previous_sha256 = sha256_file(path)

    @property
    def complete(self) -> bool:
        return len(self.rows) == len(self.keys)

    def terminal(self, row_field: str) -> dict | None:
        path = self.out / self.terminal_name
        if not path.exists():
            return None
        result = json.loads(path.read_text())
        if not self.complete or result.get(row_field) != self.rows:
            raise RuntimeError("terminal result differs from generation checkpoints")
        if result.get("generation_identity_sha256") != self.identity_sha256:
            raise RuntimeError("terminal generation identity mismatch")
        if result.get("generation_checkpoint_tail_sha256") != self.previous_sha256:
            raise RuntimeError("terminal generation chain mismatch")
        return result

    def finish(self, result: dict, row_field: str) -> None:
        if not self.complete or result.get(row_field) != self.rows:
            raise RuntimeError("cannot finalize an incomplete or different generation")
        result["generation_identity_sha256"] = self.identity_sha256
        result["generation_checkpoint_tail_sha256"] = self.previous_sha256
        write_immutable(self.out / self.terminal_name, json_bytes(result))
