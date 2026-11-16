"""VOID provisional DEV60 static prefix check. NOT a frozen benchmark."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

SYSTEM = ("Return only one valid CyberSand painter program JSON object. "
          "Follow the requested structure. Do not set host seeds, material permissions or focus.")
SYSTEM_SHA256 = "2beaa9dbcd9a006c6bdeb4b0f96fd56fdb50eb0cfc4e8056b7f64bb30a21661c"
HERE = Path(__file__).resolve().parent
# Exact reviewed source at qualified foundation 2a565df45efc568eb96a2166e0b77f299e289625.
PINNED_ADAPTER_BLOB_SHA1 = "5c1df9e90345090162485482de0bf258cd14292e"
PINNED_PROFILE_BLOB_SHA1 = "ae77df4a9f53c56d700163e66acf7d16263291aa"
FAMILIES = ("greebles", "pipework", "architecture")
FORMS = {f"F{i}" for i in range(5)}
KEYS = {"id", "family", "group_id", "formulation_id", "prompt", "prompt_sha256"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _pinned_file_bytes(path, expected_blob_sha1):
    """Reject source drift before executing the reviewed local adapter."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or symlinked VOID pinned source: " + path.name)
    raw = path.read_bytes()
    header = ("blob " + str(len(raw)) + "\0").encode("ascii")
    if hashlib.sha1(header + raw).hexdigest() != expected_blob_sha1:
        raise ValueError("unreviewed VOID pinned source bytes: " + path.name)
    return raw


def strict_json(raw):
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("BOM forbidden")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite JSON constant")
    return json.loads(raw.decode("utf-8", errors="strict"),
                      object_pairs_hook=unique, parse_constant=invalid)


def audit(rows, ceiling):
    if type(ceiling) is not int or ceiling < 1:
        raise ValueError("positive fixed integer ceiling required")
    if digest(SYSTEM.encode("utf-8")) != SYSTEM_SHA256:
        raise ValueError("VOID historical system instruction mismatch")
    _pinned_file_bytes(HERE / "adapter.py", PINNED_ADAPTER_BLOB_SHA1)
    profile = strict_json(_pinned_file_bytes(HERE / "profile.json", PINNED_PROFILE_BLOB_SHA1))
    # Bind execution to the exact reviewed local file, not a cached 'adapter'
    # module from sys.modules or an alternative import directory.
    spec = importlib.util.spec_from_file_location(
        "cybersand_void_pinned_dev60_adapter", HERE / "adapter.py")
    if spec is None or spec.loader is None:
        raise ValueError("cannot load pinned VOID adapter")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    if (profile["model_id"], profile["revision"], profile["max_sequence_tokens"],
        profile["vocabulary_size"], profile["special_token_ids"],
        profile["framing"]["id"]) != (adapter.MODEL_ID, adapter.REVISION, 2048,
                                     259, {"pad": 256, "bos": 257, "eos": 258},
                                     "cybersand-void-task-v1"):
        raise ValueError("VOID pinned model/encoder/framing changed")
    if type(rows) is not list or len(rows) != 60:
        raise ValueError("exactly 60 provisional requests required")
    tokenizer = adapter.CyberSandVoidByteTokenizer()
    ids, groups, report = set(), {}, []
    for row in rows:
        if type(row) is not dict or set(row) != KEYS:
            raise ValueError("unexpected/missing field: targets forbidden")
        rid, family, group, form, prompt = (row[k] for k in
            ("id", "family", "group_id", "formulation_id", "prompt"))
        if any(type(v) is not str for v in (rid, family, group, form, prompt, row["prompt_sha256"])):
            raise ValueError("request fields must be strings")
        if not rid or not group or not prompt.strip() or family not in FAMILIES or form not in FORMS:
            raise ValueError("invalid request identity")
        if rid in ids:
            raise ValueError("duplicate request ID")
        ids.add(rid)
        raw = prompt.encode("utf-8", errors="strict")
        if digest(raw) != row["prompt_sha256"]:
            raise ValueError("prompt digest mismatch")
        forms = groups.setdefault((family, group), set())
        if form in forms:
            raise ValueError("duplicate group form")
        forms.add(form)
        tokens = adapter.encode_prompt(tokenizer, prompt, SYSTEM)
        if any(type(t) is not int or not 0 <= t <= 255 for t in tokens):
            raise ValueError("unexpected prefix special token")
        if list(tokenizer.decode(tokens).encode("utf-8")) != tokens:
            raise ValueError("prefix UTF-8 roundtrip changed")
        remaining = 2048 - len(tokens) - ceiling
        if remaining < 0:
            raise ValueError("fixed context budget exceeded; no truncation")
        report.append({"id": rid, "family": family, "group_id": group,
                       "formulation_id": form, "prompt_sha256": digest(raw),
                       "prefix_tokens": len(tokens), "prefix_ids_sha256": digest(bytes(tokens)),
                       "headroom_after_ceiling": remaining})
    if any(sum(key[0] == family for key in groups) != 4 for family in FAMILIES):
        raise ValueError("must provide four groups per family")
    if any(forms != FORMS for forms in groups.values()):
        raise ValueError("must provide all five formulations per group")
    report.sort(key=lambda r: r["id"])
    canonical_rows = json.dumps(sorted(rows, key=lambda row: row["id"]),
                                sort_keys=True, ensure_ascii=False,
                                separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {"status": "PROVISIONAL_PREFIX_ONLY_NOT_DEV60_READY",
            "model_id": adapter.MODEL_ID, "revision": adapter.REVISION,
            "framing": adapter.FRAMING_ID, "automatic_bos": False,
            "terminal_eos_id": 258, "context_tokens": 2048,
            "system_sha256": SYSTEM_SHA256, "provisional_ceiling": ceiling,
            "adapter_git_blob_sha1": PINNED_ADAPTER_BLOB_SHA1,
            "profile_git_blob_sha1": PINNED_PROFILE_BLOB_SHA1,
            "request_set_sha256": digest(canonical_rows),
            "requests": 60, "groups": 12,
            "max_prefix_tokens": max(r["prefix_tokens"] for r in report),
            "min_headroom": min(r["headroom_after_ceiling"] for r in report),
            "prefixes": report, "target_representability_checked": False,
            "dev60_frozen": False, "model_inference": False,
            "new_native_operations": 0, "new_train_admissions": 0,
            "final_targets_opened": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-free-requests", type=Path, required=True)
    ap.add_argument("--provisional-ceiling", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("refusing to overwrite evidence")
    doc = strict_json(args.target_free_requests.read_bytes())
    if type(doc) is not dict or set(doc) != {"rows"}:
        raise ValueError("target-free provisional rows object required")
    result = audit(doc["rows"], args.provisional_ceiling)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
