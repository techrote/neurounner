"""Run a bounded, predeclared D-2 synthetic cohort; never launches native/model jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lab import packed, write, replay, VERSION

def run(catalogue_path: Path, output: Path):
    raw=catalogue_path.read_bytes()
    catalogue=json.loads(raw)
    assert catalogue["version"]=="d2.offline-corpus/1"
    assert catalogue["qualification"]=="DEVELOPMENT_SYNTHETIC_ONLY"
    assert 1<=len(catalogue["cases"])<=12
    results=[]
    names=set()
    for case in catalogue["cases"]:
        name=case["id"]
        assert isinstance(name,str) and name.isascii() and all(c.islower() or c.isdigit() or c=="-" for c in name)
        assert name not in names
        names.add(name)
        assert case["split"]=="development"
        assert case["recipe"]["version"]==VERSION
        target=output/name
        manifest=write(case["recipe"],target)
        assert replay(target)
        results.append({
            "id":name,"split":case["split"],"tags":case["tags"],
            "recipe_sha256":hashlib.sha256(packed(case["recipe"])).hexdigest(),
            "manifest_sha256":hashlib.sha256((target/"manifest.json").read_bytes()).hexdigest(),
            "summary":manifest["summary"],"verification":"OFFLINE_REPLAY_PASS"
        })
    index={"version":catalogue["version"],"catalogue_sha256":hashlib.sha256(raw).hexdigest(),
           "qualification":"DEVELOPMENT_SYNTHETIC_ONLY","native_replay_verified":False,
           "model_decisions":0,"cases":results}
    output.mkdir(parents=True,exist_ok=True)
    (output/"index.json").write_bytes(packed(index))
    return index

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--catalogue",default=str(Path(__file__).parent/"corpus/development-recipes.json"))
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    result=run(Path(args.catalogue),Path(args.out))
    for item in result["cases"]:
        print(json.dumps(item,sort_keys=True))
