"""Freeze the 72 ORIGINAL Q8 native worm states for bounded shadow inference.

This contains *no* native operations or synthetic substitutions. Input must be
the original Github Actions artifact pairs from run 43678354515.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

SOURCE_SHA = "313b836b3c9ce2c60e4659e3763df140f84347a6"
RUN_ID = 43678354515
EXPECTED_DIGEST = "178aff09de061a1fe235b1039ace74665192a93651f826942b669ec9e7a78c15"
ARTIFACTS = {
    "free": {
        "id": 14147579972,
        "zip_sha256": "33083b5642c9ae6af04d7530147818453dd4513b9cf4e6f9576e4ef79e0d9aac",
    },
    "soft": {
        "id": 14147233975,
        "zip_sha256": "82d64e708e5706d4532900c60c52eaee7de04105d64888aaf436270136e7eca1",
    },
}

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()

def freeze_pair(root):
    root = Path(root)
    rows = []
    for mode in ("free", "soft"):
        source = root / mode
        summary_file = source / "execution/execution/summary.json"
        summary = json.loads(summary_file.read_text())
        if (summary.get("source_sha") != SOURCE_SHA or
            summary.get("model_decisions") != 36 or
            summary.get("independent_native_replay") is not True or
            summary.get("mode") != mode):
            raise ValueError("Original native Q8 worm source/replay mismatch: " + mode)
        model_dir = source / "execution/model"
        files = sorted(model_dir.glob("turn-*.json"))
        if [p.name for p in files] != [f"turn-{i:02d}.json" for i in range(36)]:
            raise ValueError("Expected exactly 36 original, in-order native decisions")
        for i, file in enumerate(files):
            original = json.loads(file.read_text())
            response = original["original_response"]
            ids = original["exact_candidate_ids_in_presented_order"]
            chosen = original["chosen_candidate"]
            if (original["index"] != i or original["mode"] != mode or
                response.get("status") != "ok" or
                chosen not in ids or len(ids) != len(set(ids)) or
                original["parsed_choice"]["candidate_id"] != chosen or
                response["raw_response"]["usage"]["output_tokens"] != 0):
                raise ValueError("Original Q8 decision/candidate identity inconsistent")
            rows.append({
                "mode": mode, "turn": i,
                "state": original["model_state"],
                "questions": original["questions"],
                "candidate_ids": ids, "q8_choice": chosen,
                "q8_probabilities": original["parsed_choice"]["probabilities"],
                "q8_latency_seconds": response["latency_seconds"],
                "host_legal_sha256": original["host_legal_candidate_sha256"],
                "presentation_sha256": original["presentation_sha256"],
            })
    digest = hashlib.sha256(canonical(rows)).hexdigest()
    if digest != EXPECTED_DIGEST:
        raise ValueError("Archived Q8 states do not match frozen source hash")
    return rows, digest

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rows, digest = freeze_pair(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(canonical({"schema":"wormbench-original-Q8/1",
                                  "source_sha":SOURCE_SHA, "run_id":RUN_ID,
                                  "digest":digest, "rows":rows}) + b"\n")
    print(json.dumps({"original_states":len(rows),"sha256":digest},sort_keys=True))

if __name__ == "__main__":
    main()
