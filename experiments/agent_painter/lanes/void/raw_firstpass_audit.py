#!/usr/bin/env python3
"""Re-audit immutable VOID step16/step120 DEVELOPMENT first-pass refusals.

Read-only, model-free, compiler-free. Never loads native compiler or teacher targets;
never opens FINAL. Reads whole original Actions ZIPs and preserves raw bytes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

AUDIT_ID = "void-original-development-firstpass-refusals-v2"
MODEL = "dotlabs/void.1"
REVISION = "9c2be5087772867d45961abd11f6b35110617563"
NATIVE_SOURCE = "633920373a3d406a2ede61ce4137c324f76be23b"
CORPUS_SHA = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
SEEDS = {104729, 130363}
SOURCES = {
    "step16": {"run": 43521427032, "artifact": 14084102379,
               "sha256": "004548689a264e7bb965bfba933846c5f6cc7e87dc78360e54511b1d293880a1",
               "producer": "e902646d13623aa49f4e0559b370c59bff27d5bd", "limit": 768},
    "step120": {"run": 43530053171, "artifact": 14088647201,
                "sha256": "46931f58574251a101128e62599febe4aed8d3b91af8c96e84bd6586b5348f88",
                "producer": "7d90bf5294d16a8644694a4c8a6a36a5227fa3c0", "limit": 1024},
}


def require(value, why):
    if not value:
        raise ValueError(why)


def independent_strict_json_valid(raw_text):
    """Match shared strict_json independently; reject duplicate keys and NaN/Infinity."""
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = item
        return result

    def nonfinite(value):
        raise ValueError("non-finite JSON constant: " + value)

    try:
        json.loads(raw_text, object_pairs_hook=unique, parse_constant=nonfinite)
    except (ValueError, TypeError):
        return False
    return True


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def category(output):
    if output["first_pass_accepted"]:
        return "compiler_accepted_not_semantically_scored"
    if output["json_valid"]:
        return "json_valid_compiler_rejected"
    if str(output.get("failure", "")).startswith("duplicate JSON key:"):
        return "duplicate_json_key_rejected"
    if not output["raw_text"].lstrip().startswith("{"):
        return "off_task_non_json_text"
    return "malformed_json_rejected"


def verify_raw(output, receipt, raw_bytes, batch_bytes, compiled_bytes=None):
    require(isinstance(output, dict) and isinstance(receipt, dict), "invalid receipt objects")
    require(type(output.get("raw_text")) is str and raw_bytes == output["raw_text"].encode("utf-8", "strict"),
            "original raw completion bytes differ")
    tokens = output.get("token_ids")
    require(isinstance(tokens, list) and tokens and
            all(type(x) is int and 0 <= x <= 258 for x in tokens), "bad original token IDs")
    require(256 not in tokens and 257 not in tokens and tokens.count(258) <= 1,
            "unexpected PAD/BOS or duplicate EOS in original completion")
    eos = tokens[-1] == 258
    require((258 in tokens) == eos and type(output.get("eos_terminated")) is bool
            and output["eos_terminated"] == eos, "true EOS identity differs")
    require(bytes(x for x in tokens if x < 256).decode("utf-8", "strict") == output["raw_text"],
            "strict UTF-8 original token round-trip differs")
    require(receipt.get("generated_tokens") == len(tokens)
            and all(type(output.get(k)) is bool and output[k] == receipt.get(k)
                    for k in ("eos_terminated", "json_valid", "program_valid", "first_pass_accepted")),
            "raw/receipt program or token decision differs")
    require(output["json_valid"] is independent_strict_json_valid(output["raw_text"]),
            "recorded JSON-valid flag contradicts independent strict parser")
    require(receipt.get("completion_sha256") == hashlib.sha256(raw_bytes).hexdigest(),
            "original receipt completion SHA-256 differs")
    require(type(receipt.get("compiler_accepted")) is bool and
            receipt["compiler_accepted"] == output["first_pass_accepted"],
            "compiled acceptance flag contradicts raw first pass")
    require(not output["first_pass_accepted"] or eos,
            "accepted first pass did not end in one true EOS")
    require(output["first_pass_accepted"] <= output["program_valid"] <= output["json_valid"],
            "acceptance chain inconsistent")
    require(receipt.get("native_execution") is False and receipt.get("native_acknowledged_cells") is None
            and receipt.get("visual_quality_claim") is False,
            "compiler evidence falsely promoted to native/art result")
    batch = json.loads(batch_bytes)
    if not output["first_pass_accepted"]:
        require(batch == [] and compiled_bytes is None and receipt.get("operations_sha256") is None,
                "rejected completion emitted or claimed compiled operations")
    else:
        require(compiled_bytes is not None and type(batch) is list and len(batch) > 0,
                "accepted completion missing compiled operations")
        compiled = json.loads(compiled_bytes)
        digest = canonical_digest(batch)
        require(type(compiled) is dict and compiled.get("operations") == batch and
                compiled.get("operations_sha256") == digest and
                receipt.get("operations_sha256") == digest,
                "accepted operations disagree with original compiled/receipt SHA-256")
    return {"category": category(output), "eos": eos, "json": output["json_valid"],
            "program": output["program_valid"], "compiler_accepted": output["first_pass_accepted"],
            "tokens": len(tokens), "failure": output.get("failure"),
            "raw_text_sha256": hashlib.sha256(raw_bytes).hexdigest()}


def audit_archive(path: Path, stage: str):
    contract = SOURCES[stage]
    require(path.is_file() and not path.is_symlink(), "missing/unsafe original archive")
    require(hashlib.sha256(path.read_bytes()).hexdigest() == contract["sha256"],
            "original whole Actions ZIP hash differs: " + stage)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        require(len(names) == len(set(names)) and z.testzip() is None,
                "original ZIP member names/CRC differ")
        for name in names:
            pp = PurePosixPath(name)
            require(not pp.is_absolute() and name == pp.as_posix() and
                    ".." not in pp.parts and not name.startswith("/"), "unsafe member path")
            require((name.startswith("evaluation/") or name.startswith("eval-env/"))
                    and "/FINAL/" not in name, "unexpected or sealed content")
        identity = json.loads(z.read("evaluation/identity.json"))
        summary = json.loads(z.read("evaluation/summary.json"))
        required = {"source_sha": contract["producer"], "native_source_sha": NATIVE_SOURCE,
                    "corpus_sha256": CORPUS_SHA, "model_id": MODEL,
                    "model_revision": REVISION, "split": "DEVELOPMENT",
                    "cases": 12, "max_new_tokens": contract["limit"],
                    "native_execution": False, "visual_quality_claim": False}
        require(all(identity.get(k) == v and type(identity.get(k)) is type(v)
                    for k, v in required.items()), "original evaluation source/model/split identity differs")
        require(summary.get("complete") is True and summary.get("cases") == 12
                and summary.get("native_execution") is False,
                "original evaluation incomplete or misleading")
        result = {"stage": stage, "source_sha": contract["producer"],
                  "original_run": contract["run"], "artifact_id": contract["artifact"],
                  "original_zip_sha256": contract["sha256"], "zip_members_crc_pass": len(names),
                  "generation_limit_tokens": contract["limit"], "arms": {}}
        for arm in ("base", "adapter"):
            dirs = sorted({name.rsplit("/", 1)[0] for name in names
                           if name.startswith(f"evaluation/{arm}/") and name.endswith("/receipt.json")})
            require(len(dirs) == 12, "original host-seed view count differs")
            groups = defaultdict(list)
            for directory in dirs:
                def read(filename):
                    return z.read(directory + "/" + filename)
                receipt = json.loads(read("receipt.json"))
                raw = json.loads(read("raw-output.json"))
                compiled_name = directory + "/compiled.json"
                compiled_bytes = read("compiled.json") if compiled_name in names else None
                case = verify_raw(raw, receipt, read("completion.txt"),
                                  read("art_batch.json"), compiled_bytes=compiled_bytes)
                require(case["tokens"] <= contract["limit"]
                        and receipt.get("arm") == arm and receipt.get("family") in
                        {"architecture", "greebles", "pipework"} and
                        type(receipt.get("host_seed")) is int and receipt["host_seed"] in SEEDS,
                        "original host case/limit/arm differs")
                groups[receipt["group"]].append((receipt, raw, case))
            require(len(groups) == 6, "distinct DEVELOPMENT structural-group count differs")
            cases = []
            for group, pair in sorted(groups.items()):
                require(len(pair) == 2 and {p[0]["host_seed"] for p in pair} == SEEDS,
                        "host-seed pair differs")
                require(pair[0][0]["original_prompt"] == pair[1][0]["original_prompt"] and
                        pair[0][0]["family"] == pair[1][0]["family"] and
                        pair[0][1] == pair[1][1],
                        "seed variants change the original first-pass language generation")
                if stage == "step120":
                    require(sum(p[0]["actual_model_generation"] for p in pair) == 1
                            and sum(p[0]["reused_first_pass_for_host_seed"] for p in pair) == 1,
                            "120-stage generated-versus-reused receipt differs")
                cases.append({"group": group, "family": pair[0][0]["family"], **pair[0][2]})
            require(Counter(c["family"] for c in cases) ==
                    {"architecture": 2, "greebles": 2, "pipework": 2},
                    "family denominator differs")
            counters = {"distinct_requests": 6, "host_seed_views": 12,
                        "eos_requests": sum(c["eos"] for c in cases),
                        "json_valid_requests": sum(c["json"] for c in cases),
                        "compiler_accepted_requests": sum(c["compiler_accepted"] for c in cases)}
            old = summary["arms"][arm]
            require(old["requests"] == 12 and
                    all(old[field] == 2 * counters[key]
                        for field, key in (("eos_terminated", "eos_requests"),
                                           ("json_valid", "json_valid_requests"),
                                           ("first_pass_accepted", "compiler_accepted_requests"))),
                    "published host-view counters differ from independently audited requests")
            result["arms"][arm] = {**counters,
                    "categories_per_distinct_request": dict(sorted(Counter(c["category"] for c in cases).items())),
                    "original_cases": cases}
        return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step16", required=True, type=Path)
    ap.add_argument("--step120", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    require(not args.output.exists(), "output receipt must be new, never overwrite")
    receipt = {"schema": AUDIT_ID,
               "sources": [audit_archive(args.step16, "step16"),
                           audit_archive(args.step120, "step120")],
               "new_optimizer_steps": 0, "new_native_operations": 0,
               "new_train_admissions": 0, "final_opened": False,
               "semantic_quality_evaluated_here": False,
               "caution": "A compiler-accepted first pass is not a request-conforming native painting."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ORIGINAL_RAW_FIRSTPASS_AUDIT_PASS",
                      "out": str(args.output), "stages": len(receipt["sources"])}, sort_keys=True))


if __name__ == "__main__":
    main()
