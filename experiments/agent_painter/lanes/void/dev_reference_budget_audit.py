#!/usr/bin/env python3
"""Original native DEVELOPMENT teacher byte-budget audit for VOID, not model inference.

FINAL ZIP members are never opened or CRC-tested. Original source633 artifact
is SHA-256 checked as an opaque whole before reading named visible members.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile

HERE = Path(__file__).resolve().parent
NATIVE_SOURCE = "633920373a3d406a2ede61ce4137c324f76be23b"
ZIP_SHA = "50a9ad9b255b2a57180edf5558ca2b47dab9affd9ef9d2438f9011d648f81f39"
MANIFEST_SHA = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
DEV_SHA = "8c73edd82d9289a78721ef7968ab2f939f858b666a106731709bc2d80b347572"
ADAPTER_BLOB = "5c1df9e90345090162485482de0bf258cd14292e"
PROFILE_BLOB = "ae77df4a9f53c56d700163e66acf7d16263291aa"
REVISION = "9c2be5087772867d45961abd11f6b35110617563"
SYSTEM = ("Return only one valid CyberSand painter program JSON object. "
          "Follow the requested structure. Do not set host seeds, material permissions or focus.")
SYSTEM_SHA = "2beaa9dbcd9a006c6bdeb4b0f96fd56fdb50eb0cfc4e8056b7f64bb30a21661c"
FAMILIES = ("greebles", "pipework", "architecture")
SEEDS = {104729, 130363}

def need(condition, message):
    if not condition:
        raise ValueError(message)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def strict_json(raw):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            need(key not in obj, "duplicate JSON key")
            obj[key] = value
        return obj
    def invalid(value):
        raise ValueError("non-finite JSON constant")
    return json.loads(raw.decode("utf-8", errors="strict"),
                      object_pairs_hook=unique, parse_constant=invalid)

def canonical(program):
    return json.dumps(program, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)

def pinned(path, expected_sha1):
    need(path.is_file() and not path.is_symlink(), "missing/symlinked pinned source")
    raw = path.read_bytes()
    blob = ("blob " + str(len(raw)) + "\0").encode("ascii") + raw
    need(hashlib.sha1(blob).hexdigest() == expected_sha1,
         "unreviewed adapter/profile implementation")
    return raw

def member(archive, name):
    need(type(name) is str and name.startswith("corpus/")
         and "\\" not in name and "\0" not in name and not name.endswith("/"),
         "bad native archive member path")
    parts = name.split("/")
    need(not PurePosixPath(name).is_absolute()
         and not any(p in ("", ".", "..") or "final" in p.lower() for p in parts),
         "FINAL, traversal or unsafe member forbidden")
    matches = [x for x in archive.infolist() if x.filename == name]
    need(len(matches) == 1, "duplicate/missing visible member")
    entry = matches[0]
    need(not stat.S_ISLNK(entry.external_attr >> 16)
         and 0 <= entry.file_size <= 20_000_000
         and 0 <= entry.compress_size <= 20_000_000,
         "unexpected ZIP member type/size")
    return entry

def visible(archive, name):
    entry = member(archive, name)
    with archive.open(entry) as stream:
        raw = stream.read(20_000_001)
    need(len(raw) == entry.file_size, "visible member size/CRC changed")
    return raw

def group_rows(rows, manifest):
    need(type(rows) is list and len(rows) == 12, "12 native DEVELOPMENT views required")
    declared = set(manifest["structural_groups"]["DEVELOPMENT"])
    groups, ids = {}, set()
    for row in rows:
        need(row.get("split") == "DEVELOPMENT"
             and row.get("group_id") == row.get("structural_group")
             and row.get("group_id") in declared
             and row.get("family") in FAMILIES
             and row.get("seed") in SEEDS
             and type(row.get("prompt")) is str and row["prompt"].strip()
             and type(row.get("program")) is dict
             and row.get("id") not in ids, "invalid DEVELOPMENT structural row")
        ids.add(row["id"])
        groups.setdefault(row["group_id"], []).append(row)
    need(len(groups) == 6 and set(groups) == declared
         and all(sum(v[0]["family"] == fam for v in groups.values()) == 2 for fam in FAMILIES),
         "not two whole designs per family")
    for views in groups.values():
        need(len(views) == 2 and {v["seed"] for v in views} == SEEDS
             and views[0]["family"] == views[1]["family"]
             and views[0]["prompt"] == views[1]["prompt"]
             and views[0]["program"] == views[1]["program"],
             "seed-specific original teacher drift")
    return groups

def audit(z):
    m_raw = visible(z, "corpus/admitted-corpus.json")
    need(sha(m_raw) == MANIFEST_SHA, "original native corpus hash mismatch")
    m = strict_json(m_raw)
    splits = m.get("structural_groups", {})
    need(m.get("source_sha") == NATIVE_SOURCE and m.get("state") == "native-admitted"
         and set(splits) == {"TRAIN", "DEVELOPMENT", "FINAL"}
         and [len(splits[k]) for k in ("TRAIN", "DEVELOPMENT", "FINAL")] == [6, 6, 3]
         and len(set().union(*(set(v) for v in splits.values()))) == 15,
         "original native source/structural split mismatch")
    info = m["split_files"]["DEVELOPMENT"]
    need(info["sha256"] == DEV_SHA and info["count"] == 12
         and info["path"] == "development.jsonl", "native DEVELOPMENT split identity changed")
    d_raw = visible(z, "corpus/development.jsonl")
    need(sha(d_raw) == DEV_SHA, "original native DEVELOPMENT rows changed")
    rows = [strict_json(line) for line in d_raw.splitlines() if line.strip()]
    groups = group_rows(rows, m)
    pinned(HERE / "adapter.py", ADAPTER_BLOB)
    profile = strict_json(pinned(HERE / "profile.json", PROFILE_BLOB))
    need(profile["model_id"] == "dotlabs/void.1" and profile["revision"] == REVISION
         and profile["trust_remote_code"] is False
         and profile["max_sequence_tokens"] == 2048
         and profile["special_token_ids"] == {"pad":256, "bos":257, "eos":258}
         and profile["framing"]["id"] == "cybersand-void-task-v1"
         and sha(SYSTEM.encode("utf-8")) == SYSTEM_SHA,
         "reviewed VOID model/framing/system changed")
    spec = importlib.util.spec_from_file_location("void_pinned_byte_adapter", HERE / "adapter.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    need(adapter.REVISION == REVISION and adapter.FRAMING_ID == "cybersand-void-task-v1",
         "adapter model identity changed")
    tok = adapter.CyberSandVoidByteTokenizer()
    results, views_verified = [], 0
    for group, views in sorted(groups.items()):
        for row in views:
            adm = row["native_admission"]
            need(adm.get("state") == "PASS" and adm.get("native_execution") is True
                 and adm.get("verified") is True
                 and adm.get("source_sha") == NATIVE_SOURCE
                 and adm.get("runtime_sha256") == m["runtime_sha256"],
                 "not actual positive source-bound original native admission")
            for kind in ("receipt", "readback", "png"):
                data = visible(z, "corpus/" + adm["paths"][kind])
                need(sha(data) == adm[kind + "_sha256"],
                     "original native receipt/material/PNG digest differs")
                if kind == "receipt":
                    receipt = strict_json(data)
                    need(receipt.get("passed") is True and receipt.get("id") == row["id"]
                         and receipt.get("seed") == row["seed"]
                         and receipt.get("family") == row["family"]
                         and receipt.get("program_sha256") == adm["program_sha256"]
                         and receipt.get("operations_sha256") == adm["operations_sha256"]
                         and receipt.get("crop_i32le_sha256") == adm["readback_sha256"]
                         and receipt.get("deterministic_replay_checked") is True
                         and receipt.get("outside_area_cells") == 0
                         and receipt.get("unexpected_material_cells") == 0,
                         "original native receipt not bound to admitted visible data")
            need(sha(canonical(row["program"]).encode("utf-8")) == adm["program_sha256"],
                 "original native target program hash differs")
            views_verified += 1
        row = views[0]
        completion = canonical(row["program"])
        need("completion" not in row or row["completion"] == completion,
             "non-canonical original completion")
        seq = adapter.encode_row(tok, {"id":group, "prompt":row["prompt"],
                                      "completion":completion}, SYSTEM, 2048)
        target = seq["input_ids"][seq["boundary"]:]
        need(target and target[-1] == 258 and target.count(258) == 1
             and len(target) == len(completion.encode("utf-8")) + 1
             and len(target) == seq["target_tokens"]
             and seq["input_ids"][:seq["boundary"]] == seq["prefix_ids"]
             and len(seq["input_ids"]) <= 2048,
             "true terminal EOS/strict bytes/no-truncation contract failed")
        results.append({"group":group, "family":row["family"],
                        "prompt_sha256":sha(row["prompt"].encode("utf-8")),
                        "reference_sha256":sha(completion.encode("utf-8")),
                        "prefix_tokens":len(seq["prefix_ids"]),
                        "completion_tokens_including_eos258":len(target),
                        "full_reference_tokens":len(seq["input_ids"]),
                        "reference_fits_512":len(target) <= 512,
                        "reference_fits_768":len(target) <= 768,
                        "available_tokens_after_prefix":2048-len(seq["prefix_ids"])})
    need(views_verified == 12 and len(results) == 6, "incomplete original DEVELOPMENT")
    sizes = [x["completion_tokens_including_eos258"] for x in results]
    return {"schema":"cybersand.void.original-development-reference-budget/1",
            "status":"ORIGINAL_REFERENCE_ONLY_NOT_MODEL_EVALUATION",
            "original_source":NATIVE_SOURCE, "original_artifact_sha256":ZIP_SHA,
            "original_manifest_sha256":MANIFEST_SHA,
            "original_development_sha256":DEV_SHA, "model_revision":REVISION,
            "framing":"cybersand-void-task-v1", "terminal_eos_id":258,
            "independent_requests":6, "original_native_views_verified":views_verified,
            "completion_token_range":[min(sizes),max(sizes)],
            "all_references_fit_512":all(x["reference_fits_512"] for x in results),
            "all_references_fit_768":all(x["reference_fits_768"] for x in results),
            "groups":results,"model_weights_downloaded":False,
            "optimizer_updates":0,"model_generations":0,"native_model_paintings":0,
            "shared_train_admissions":0,"dev60_frozen":False,"final_opened":False}

def selftest():
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'\xff', b'\xef\xbb\xbf{}'):
        try:
            strict_json(raw)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid JSON or UTF-8")
    cases = [{"id":f+"-"+str(i)+"-"+str(seed),
              "split":"DEVELOPMENT", "group_id":f+str(i),
              "structural_group":f+str(i), "family":f, "seed":seed,
              "prompt":"test", "program":{"family":f,"version":2}}
             for f in FAMILIES for i in range(2) for seed in sorted(SEEDS)]
    groups = sorted({r["group_id"] for r in cases})
    need(len(group_rows(cases, {"structural_groups":{"DEVELOPMENT":groups}})) == 6,
         "positive original group test failed")
    cases[1]["prompt"] = "drift"
    try:
        group_rows(cases, {"structural_groups":{"DEVELOPMENT":groups}})
    except ValueError:
        pass
    else:
        raise AssertionError("seed drift accepted")
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
        path = Path(tmp)/"test.zip"
        with zipfile.ZipFile(path,"w") as z:
            z.writestr("corpus/development.jsonl","visible")
            z.writestr("corpus/final.jsonl","sealed")
        with zipfile.ZipFile(path) as z:
            need(visible(z,"corpus/development.jsonl") == b"visible", "visible read failed")
            for bad in ("corpus/final.jsonl", "corpus/../escape", "corpus/missing"):
                try:
                    member(z,bad)
                except ValueError:
                    pass
                else:
                    raise AssertionError("sealed or invalid member opened")
    print("VOID original DEVELOPMENT budget source self-tests PASS")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--zip",type=Path)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    if args.self_test:
        selftest()
        return
    need(args.zip and args.output and args.zip.is_file()
         and not args.zip.is_symlink() and not args.output.exists(),
         "fresh source artifact/output required")
    need(os.environ.get("GITHUB_ACTIONS") == "true"
         and os.environ.get("RUNNER_ENVIRONMENT") == "github-hosted"
         and os.environ.get("GITHUB_REPOSITORY") == "techrote/cybersand"
         and os.environ.get("GITHUB_REF") == "refs/heads/codex/agent-painter-void-568",
         "source-only original audit requires branch-hosted runner")
    need(file_sha(args.zip) == ZIP_SHA, "original source633 whole ZIP SHA256 differs")
    with zipfile.ZipFile(args.zip) as archive:
        result=audit(archive)
    result["lane_source_sha"]=os.environ["GITHUB_SHA"]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="groups"},sort_keys=True))

if __name__ == "__main__":
    main()
