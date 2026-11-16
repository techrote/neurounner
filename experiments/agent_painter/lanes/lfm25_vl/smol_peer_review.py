#!/usr/bin/env python3
"""LFM C1/C2 vs SMOL G06/P01 original source-only geometry; no TRAIN admission."""
import argparse
import hashlib
import json
from pathlib import Path

PINS = {
    "lfm": ("5fd5e33aad38296155c7bb47cf37a07f0ee07946",
            "226ce959c68a9e2649e88fe15c10713437cf82be"),
    "smol": ("6e412178c9b71cb0196d8e35408470077b50f884",
             "0b0adae3c02d1ccc9bda2cdca393f272ca7ecf63"),
}
IDS = {
    "lg": "lfm-c1-six-bay-service-chase",
    "lp": "lfm-c2-dual-cross-six-terminal-network",
    "sg": "smol_g06_stepped_absence",
    "sp": "smol_p01_nested_branch_tree",
}


def verify(path, blob):
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or symlinked source")
    raw = path.read_bytes()
    got = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if got != blob:
        raise ValueError("pinned Git blob mismatch")
    return raw


def get(data, ident, family):
    records = [r for r in data.get("proposals", []) if r.get("id") == ident]
    if len(records) != 1 or records[0].get("family") != family:
        raise ValueError("duplicate/missing/mislabelled proposal " + ident)
    p = records[0]["program"]
    if p.get("version") != 2 or p.get("family") != family:
        raise ValueError("unexpected program version/family")
    return p


def panels(p):
    result = []
    for item in p["components"]:
        b = item.get("box")
        if item.get("kind") != "panel" or not isinstance(b, list) or len(b) != 4:
            raise ValueError("invalid panel component")
        if any(type(v) is not int for v in b):
            raise ValueError("nonintegral panel")
        x,y,w,h = b
        if not (0 <= x < 96 and 0 <= y < 96 and 0 < w <= 96-x and 0 < h <= 96-y):
            raise ValueError("panel out of bounds")
        result.append((x,y,w,h))
    if not result:
        raise ValueError("no panels")
    return result


def filled(boxes):
    return {(x,y) for bx,by,w,h in boxes
            for x in range(bx,bx+w) for y in range(by,by+h)}


def tree(p):
    items = p["components"]
    if len(items) != 1 or items[0].get("kind") != "pipe_network":
        raise ValueError("not one pipe network")
    part = items[0]
    if part.get("diameter") != 3 or part.get("fittings") is not True:
        raise ValueError("unreviewed diameter/fittings")
    nodes, edges = part["nodes"],part["edges"]
    if not 2 <= len(nodes) <= 64 or len(edges) != len(nodes)-1:
        raise ValueError("not tree node/edge count")
    if len({tuple(n) for n in nodes}) != len(nodes):
        raise ValueError("repeated node coordinates")
    if any(not isinstance(n,list) or len(n)!=2 or
           any(type(v) is not int or not 0<=v<96 for v in n) for n in nodes):
        raise ValueError("bad node")
    adj = [set() for _ in nodes]
    seen = set()
    cells = set()
    edge_cells = []
    for edge in edges:
        if (not isinstance(edge,list) or len(edge)!=2 or
            any(type(v) is not int or not 0<=v<len(nodes) for v in edge)):
            raise ValueError("bad edge")
        a,b = edge
        key = tuple(sorted((a,b)))
        if a==b or key in seen:
            raise ValueError("duplicate/self edge")
        seen.add(key)
        x,y = nodes[a]
        u,v = nodes[b]
        if (x == u) == (y == v):
            raise ValueError("diagonal/zero edge")
        adj[a].add(b); adj[b].add(a)
        if x == u:
            segment={(x,k) for k in range(min(y,v),max(y,v)+1)}
        else:
            segment={(k,y) for k in range(min(x,u),max(x,u)+1)}
        # Every interior node/crossing must be explicit in the graph. A degree
        # multiset cannot detect branch relocation through an undeclared node.
        if any(tuple(node) in segment for i,node in enumerate(nodes) if i not in (a,b)):
            raise ValueError("undeclared pipe node on edge centerline")
        for (c,d), other in edge_cells:
            allowed={tuple(nodes[j]) for j in ({a,b} & {c,d})}
            if (segment & other) != allowed:
                raise ValueError("undeclared pipe crossing or overlapping edges")
        edge_cells.append(((a,b),segment))
        cells.update(segment)
    visited={0}; todo=[0]
    while todo:
        for v in adj[todo.pop()] - visited:
            visited.add(v);todo.append(v)
    if len(visited)!=len(nodes):
        raise ValueError("disconnected")
    return {"degrees":sorted(map(len,adj)), "cells":cells,
            "terminals":sum(len(x)==1 for x in adj)}


def overlap(a,b):
    inter=len(a&b); union=len(a|b)
    return {"intersect":inter, "union":union, "iou":round(inter/union,6)}


def compare(lfm,smol):
    if (lfm.get("status")!="PROPOSED_NOT_TRAIN_NOT_NATIVE_ADMITTED" or
        smol.get("status")!="PROPOSED_NOT_NATIVE_ADMITTED_NOT_TRAIN" or
        lfm.get("original_native_source")!=smol.get("original_native_source") or
        lfm.get("original_manifest_sha256")!=smol.get("original_manifest_sha256")):
        raise ValueError("research status / original native identity mismatch")
    lgb=panels(get(lfm,IDS["lg"],"greebles"))
    sgb=panels(get(smol,IDS["sg"],"greebles"))
    from c1_peer_review import paired_tiers
    lm,sm=filled(lgb),filled(sgb)
    if (len(lgb)!=6 or not paired_tiers(lgb) or len(sgb)!=3 or
        any(40<=x<56 for x,y in lm) or
        not any(40<=x<56 for x,y in sm)):
        raise ValueError("changed Greebles motif; manual re-review")
    lp=tree(get(lfm,IDS["lp"],"pipework"))
    sp=tree(get(smol,IDS["sp"],"pipework"))
    if (lp["degrees"]!=[1,1,1,1,1,1,4,4] or
        sp["degrees"]!=[1,1,1,1,1,3,4]):
        raise ValueError("changed pipe graph; manual re-review")
    return {
        "status":"PINNED_SOURCE_MOTIF_DIFFERENCE_REVIEW_ONLY",
        "greebles":{"lfm_panels":len(lgb),"smol_panels":len(sgb),
                    "lfm_box_cells":len(lm),"smol_box_cells":len(sm),
                    **overlap(lm,sm)},
        "pipework":{"lfm_degrees":lp["degrees"],"smol_degrees":sp["degrees"],
                    "lfm_terminals":lp["terminals"],"smol_terminals":sp["terminals"],
                    "lfm_centerline_cells":len(lp["cells"]),
                    "smol_centerline_cells":len(sp["cells"]),
                    **overlap(lp["cells"],sp["cells"])},
        "original_native_source":lfm["original_native_source"],
        "original_manifest_sha256":lfm["original_manifest_sha256"],
        "limitations":"Authored boxes/centerlines, not actual native pixels or global held-out semantics",
        "new_native_views":0,"train_admissions":0,"optimizer_steps":0,
        "model_calls":0,"final_opened":False,
    }


def main():
    parser=argparse.ArgumentParser()
    for name in ("lfm","smol","output"):
        parser.add_argument("--"+name,required=True,type=Path)
    a=parser.parse_args()
    if a.output.exists():
        raise ValueError("do not overwrite any receipt")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        sources={k:verify(getattr(a,k),v[1]) for k,v in PINS.items()}
        report=compare(*(json.loads(sources[k]) for k in ("lfm","smol")))
        report["source_pins"]={k:{"commit":PINS[k][0],"git_blob":PINS[k][1],
                                  "sha256":hashlib.sha256(raw).hexdigest()}
                               for k,raw in sources.items()}
    except Exception as ex:
        report={"status":"FAIL_NO_ADMISSION","error":repr(ex),"train_admissions":0,
                "final_opened":False}
        a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        raise
    a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":report["status"],"train":0}))


if __name__=="__main__":
    main()
