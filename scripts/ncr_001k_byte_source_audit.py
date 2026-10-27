#!/usr/bin/env python3
"""NCR-001K independent *static-only* audit of pinned void.1 tokenizer source.

This script downloads and parses source, never imports/executes it. A successful
result is evidence of what source exists, not approval to run it or train an NCA.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "experiments/ncr-001/source-pins.json"
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"
OUTPUT = ROOT / "runtime/ncr-001/void-byte-source-audit.json"

SOURCE_NAMES = (
    ("tokenization_byte.py", "tokenization_byte_py_sha256"),
    ("byte_vocab.json", "byte_vocab_sha256"),
    ("tokenizer_config.json", "tokenizer_config_sha256"),
    ("special_tokens_map.json", "special_tokens_map_sha256"),
)

REPORT = {
    "experiment": "NCR-001K",
    "stage": "independent_pinned_tokenizer_static_source_review",
    "status": "STARTED",
    "remote_python_executed": False,
    "tokenizer_runtime_qualified": False,
    "no_model_weights_downloaded": True,
    "model_training_performed": False,
    "training_authorized": False,
}


def digest(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def save() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(REPORT, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")


def imports(tree: ast.AST) -> list[dict]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                out.append({"line": node.lineno, "import": name.name, "alias": name.asname})
        if isinstance(node, ast.ImportFrom):
            out.append({"line": node.lineno, "from": node.module,
                        "names": [n.name for n in node.names]})
    return out


def call_name(n: ast.AST) -> str:
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.Attribute):
        return call_name(n.value) + "." + n.attr
    return type(n).__name__


def scan(tree: ast.Module) -> dict:
    class_nodes = [x for x in tree.body if isinstance(x, ast.ClassDef)]
    all_calls = [{"line": x.lineno, "call": call_name(x.func)}
                 for x in ast.walk(tree) if isinstance(x, ast.Call)]
    dangerous_tokens = (
        "eval", "exec", "compile", "open", "__import__", "os.", "sys.",
        "subprocess.", "socket.", "requests.", "urllib.", "pickle.",
        "shutil.", "pathlib.Path.", "runpy.", "importlib.", "ctypes.",
        "setattr", "getattr",
    )
    flagged = [call for call in all_calls if
               call["call"] in dangerous_tokens or
               any(call["call"].startswith(p) for p in dangerous_tokens if p.endswith("."))]
    top_level = []
    for node in tree.body:
        details = {
            "kind": type(node).__name__,
            "line": getattr(node, "lineno", None),
        }
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            details["name"] = node.name
        if isinstance(node, ast.Assign):
            details["targets"] = [
                call_name(t) for t in node.targets
            ]
        top_level.append(details)
    return {
        "imports": imports(tree),
        "top_level": top_level,
        "classes": [
            {"name": node.name,
             "bases": [call_name(b) for b in node.bases],
             "methods": [
                 {"name": child.name, "line": child.lineno}
                 for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
             ]}
            for node in class_nodes
        ],
        "flagged_calls_for_human_review": flagged,
        "all_call_sites": all_calls,
        "dynamic_execution_not_performed": True,
    }


def main() -> int:
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cfg = pins["models"]["void1"]
    m = protocol["models"]["void1"]
    assert protocol["train_authorized"] is False
    assert m["repo_id"] == cfg["repo_id"] == "dotlabs/void.1"
    assert m["revision"] == cfg["revision"] == "9c2be5087772867d45961abd11f6b35110617563"
    assert m["tokenizer_reference"]["train002b_source_sha256"] == cfg["train002b_tokenizer_source_sha256"]
    REPORT["source"] = {
        "repo_id": m["repo_id"],
        "revision": m["revision"],
        "source_pins_sha256": digest(PINS.read_bytes()),
        "protocol_sha256": digest(PROTOCOL.read_bytes()),
        "python": platform.python_version(),
    }
    REPORT["files"] = {}
    save()
    for filename, pin_key in SOURCE_NAMES:
        file = Path(hf_hub_download(m["repo_id"], filename=filename, revision=m["revision"]))
        data = file.read_bytes()
        actual = digest(data)
        expected = cfg[pin_key]
        if actual != expected:
            raise ValueError(
                f"Pinned tokenizer SOURCE SHA mismatch {filename}: {actual} != {expected}"
            )
        REPORT["files"][filename] = {
            "bytes": len(data), "sha256": actual,
            "source_pins_sha256": expected,
        }
        if filename == "tokenization_byte.py":
            assert actual == m["tokenizer_reference"]["train002b_source_sha256"]
            source = data.decode("utf-8")
            tree = ast.parse(source, filename=filename)
            REPORT["source_ast"] = scan(tree)
            # Human-readable exact source is public and fixed by hash.
            # Print, do not import. No executable custom tokenizer is loaded.
            print("BEGIN PINNED BYTE TOKENIZER STATIC SOURCE (NOT EXECUTED)")
            for lineno, line in enumerate(source.splitlines(), 1):
                print(f"{lineno:04}: {line}")
            print("END PINNED BYTE TOKENIZER STATIC SOURCE")
        elif filename.endswith(".json"):
            obj = json.loads(data)
            REPORT["files"][filename]["json_type"] = type(obj).__name__
            if filename == "tokenizer_config.json":
                REPORT["tokenizer_metadata"] = {
                    "tokenizer_class": obj.get("tokenizer_class"),
                    "auto_map": obj.get("auto_map"),
                    "bos_token": obj.get("bos_token"),
                    "eos_token": obj.get("eos_token"),
                    "pad_token": obj.get("pad_token"),
                    "add_bos_token": obj.get("add_bos_token"),
                    "add_eos_token": obj.get("add_eos_token"),
                }
        print(f"SOURCE VERIFIED (NOT EXECUTED): {filename} sha256={actual}")
        save()
    REPORT["status"] = "PASS_SOURCE_HASH_AND_AST_INVENTORY_ONLY__RUNTIME_BLOCKED"
    REPORT["manual_review_required"] = True
    REPORT["byte_tokenizer_behavior_qualified"] = False
    save()
    print("STATIC AUDIT SUMMARY", json.dumps({
        "status": REPORT["status"],
        "class_methods": REPORT["source_ast"]["classes"],
        "imports": REPORT["source_ast"]["imports"],
        "flagged": REPORT["source_ast"]["flagged_calls_for_human_review"],
        "tokenizer_meta": REPORT.get("tokenizer_metadata"),
        "remote_python_executed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        REPORT["status"] = "FAILED_STATIC_SOURCE_REVIEW"
        REPORT["error"] = f"{type(exc).__name__}: {exc}"
        REPORT["traceback"] = traceback.format_exc()[-4500:]
        save()
        raise
