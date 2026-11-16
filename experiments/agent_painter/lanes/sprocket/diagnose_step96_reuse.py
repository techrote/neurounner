#!/usr/bin/env python3
"""Immutable DEVELOPMENT first-pass geometry-reuse audit. Never reads FINAL."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import zipfile

PIN = {
    "early": ("686f67190766d71667b8e78c9a58bdcd121070eff1b75bad21327ac63f3d6b2a", "development/step16-raw-development.json", "development/comparison.json"),
    "late": ("991d2f2c0a578d935c839d5e5b4650ad8b9b9116348a54e873340f59f115ff23", "development96/step96-raw-development.json", "development96/comparison.json"),
}
NATIVE = "633920373a3d406a2ede61ce4137c324f76be23b"
CORPUS = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
MODEL = "kandivault/sprocket-500m"

def require(ok, reason):
    if not ok:
        raise ValueError(reason)

def source_archive(path: Path, stage: str) -> list:
    expected, raw_name, comp_name = PIN[stage]
    require(sha256(path.read_bytes()).hexdigest() == expected, "original archive hash mismatch: " + stage)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        require(len(names) == len(set(names)) and len(names) <= 30, "duplicate/oversized archive")
        require(set((raw_name, comp_name)).issubset(names) and not any("final" in n.lower() for n in names),
                "missing original DEVELOPMENT or unexpected sealed FINAL member")
        for i in z.infolist():
            p = Path(i.filename)
            require(not p.is_absolute() and ".." not in p.parts and i.file_size < 1000000, "unsafe archive member")
        # Deliberately read only two original DEVELOPMENT documents, not bulk extract/testzip.
        rows = json.loads(z.read(raw_name))
        comp = json.loads(z.read(comp_name))
    require(comp["model"] == MODEL and comp["native_source"] == NATIVE
            and comp["corpus_manifest_sha256"] == CORPUS
            and comp["max_new_tokens"] == 512 and comp["final_targets_opened"] is False,
            "original DEVELOPMENT lineage changed")
    require(len(rows) == 6 and len({r["structural_group"] for r in rows}) == 6,
            "six independent DEVELOPMENT groups required")
    require(Counter(r["family"] for r in rows) == Counter({"architecture": 2, "greebles": 2, "pipework": 2}),
            "family distribution changed")
    for r in rows:
        raw = r["raw"]
        ids = r["output_token_ids"]
        require(r["arm"] == ("step16" if stage == "early" else "step96"), "arm changed")
        require(isinstance(ids, list) and bool(ids) and len(ids) == r["output_tokens"]
                and ids == raw["token_ids"], "raw/token mismatch")
        require(raw["eos_terminated"] == (ids[-1] == 5 and 5 not in ids[:-1]), "EOS token mismatch")
        require(len(r["seed_results"]) == 2
                and all(x["native_execution"] is False for x in r["seed_results"]),
                "native claim drift")
    return rows

def parse_strict(raw):
    if not raw["json_valid"] or not raw["eos_terminated"]:
        return None
    def unique_pairs(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise ValueError("duplicate keys")
            d[k] = v
        return d
    return json.loads(raw["raw_text"], object_pairs_hook=unique_pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def analyze(early_rows, late_rows):
    early = {r["structural_group"]: r for r in early_rows}
    late = {r["structural_group"]: r for r in late_rows}
    require(early.keys() == late.keys(), "DEVELOPMENT group identity differs")
    require(all(early[k]["prompt"] == late[k]["prompt"]
                and early[k]["family"] == late[k]["family"] for k in early),
            "DEVELOPMENT original prompts differ")
    summary = {}
    for name, rows in (("step16", early_rows), ("step96", late_rows)):
        summary[name] = {
            "requests": 6,
            "terminal_eos": sum(r["raw"]["eos_terminated"] for r in rows),
            "strict_json": sum(r["raw"]["json_valid"] for r in rows),
            "valid_program": sum(r["raw"]["program_valid"] for r in rows),
            "compiler_accepted_requests": sum(r["admitted_seed_programs"] == 2 for r in rows),
            "native_paintings": 0,
        }
    boxes = defaultdict(list)
    for r in late_rows:
        obj = parse_strict(r["raw"])
        if not isinstance(obj, dict):
            continue
        parts = obj.get("components")
        if isinstance(parts, list) and len(parts) >= 2 and all(
                isinstance(c, dict) and isinstance(c.get("box"), list)
                and len(c["box"]) == 4 for c in parts[:2]):
            key = json.dumps([c["box"] for c in parts[:2]], separators=(",", ":"))
            boxes[key].append((r["structural_group"], r["family"]))
    cross = [{"first_two_boxes": json.loads(k), "groups": sorted(v),
              "request_count": len(v), "family_count": len({f for _, f in v})}
             for k, v in boxes.items() if len({f for _, f in v}) > 1]
    cross.sort(key=lambda x: (-x["request_count"], x["first_two_boxes"]))
    networks = []
    for r in late_rows:
        if r["family"] != "pipework":
            continue
        obj = parse_strict(r["raw"])
        require(isinstance(obj, dict) and len(obj["components"]) == 1,
                "original pipework structure changed")
        c = obj["components"][0]
        nodes, edges = c["nodes"], c["edges"]
        nonorth = [i for i, (a, b) in enumerate(edges)
                   if nodes[a][0] != nodes[b][0] and nodes[a][1] != nodes[b][1]]
        networks.append({"group": r["structural_group"], "nodes": nodes,
                         "edges": edges, "diameter": c["diameter"],
                         "declared_cycles": obj["requirements"]["cycles"],
                         "nonorthogonal_edge_indices": nonorth})
    require(len(networks) == 2, "two pipework requests required")
    return {
        "schema": "cybersand.sprocket-cross-family-first-pass-diagnostic/1",
        "status": "OBSERVATIONAL_FIRST_PASS_ONLY_NO_TRAIN_NO_NATIVE",
        "original_zip_sha256": {k: v[0] for k, v in PIN.items()},
        "native_source": NATIVE, "corpus_sha256": CORPUS, "model": MODEL,
        "summary": summary, "cross_family_exact_first_two_box_reuse": cross,
        "pipework_exact_nodes_edges_equal": networks[0]["nodes"] == networks[1]["nodes"]
            and networks[0]["edges"] == networks[1]["edges"],
        "pipework": networks,
        "new_optimizer_updates": 0, "new_model_completions": 0,
        "new_native_paintings": 0, "new_train_groups": 0, "final_opened": False,
        "caveat": "Exact reuse is observable; causation, learned generalization and native art are NOT established.",
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--early", type=Path, required=True)
    p.add_argument("--late", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    require(not args.output.exists(), "refuse overwrite")
    report = analyze(source_archive(args.early, "early"), source_archive(args.late, "late"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": report["summary"], "cross_family": report["cross_family_exact_first_two_box_reuse"],
                      "same_pipe_graph": report["pipework_exact_nodes_edges_equal"]}, sort_keys=True))

if __name__ == "__main__":
    main()
