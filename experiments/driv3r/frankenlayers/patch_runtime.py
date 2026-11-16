#!/usr/bin/env python3
"""Patch the exact pinned llama.cpp LFM3 decision graph into three YOLO layer flows.

This changes only the layer execution order used by d1 System One. It does not
touch the Q8 donor weights, CyberSand native code, or completion-mode graphs.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

PINNED_LLAMA_COMMIT="8642541b255bfb5dc39cd351a178be09013a55d6"
FLOWS={
    "stock": tuple(range(24)),
    "greyhound": (0,1,2,3,4,6,8,10,14,18,20,22,23),
    "scrambled_eggs": (0,1,2,3,4,5,7,13,9,15,11,6,10,8,12,14,20,21,22,23),
    "branchwitch": (0,1,2,3,4,5,6,7,9,10,11,10,16,17,18,19,20,21,22,23),
    # M2 purposefully distinct, order-preserving layer-restoration and branch-bias tests.
    # Each changes the actually executed decision graph; no scramble/STOP-only arm.
    "greyhound_restore5": (0,1,2,3,4,5,6,8,10,14,18,20,22,23),
    "greyhound_restore7": (0,1,2,3,4,6,7,8,10,14,18,20,22,23),
    "greyhound_restore12": (0,1,2,3,4,6,8,10,12,14,18,20,22,23),
    "greyhound_restore16": (0,1,2,3,4,6,8,10,14,16,18,20,22,23),
    "branchwitch_no_repeat10": (0,1,2,3,4,5,6,7,9,10,11,16,17,18,19,20,21,22,23),
    "branchwitch_restore8": (0,1,2,3,4,5,6,7,8,9,10,11,10,16,17,18,19,20,21,22,23),
    "branchwitch_restore12": (0,1,2,3,4,5,6,7,9,10,11,10,12,16,17,18,19,20,21,22,23),
    "branchwitch_drop21": (0,1,2,3,4,5,6,7,9,10,11,10,16,17,18,19,20,22,23),
}
TARGET=Path("src/models/lfm3.cpp")
LOOP_MARKER='''    for (int il = 0; il < n_layer; ++il) {
        auto * prev_cur = cur;
'''
LAST_MARKER='''        if (il == n_layer - 1 && inp_out_ids) {
'''

def cpp_list(values):
    return "{" + ",".join(str(x) for x in values) + "}"

def patch(root:Path):
    path=root/TARGET
    original=path.read_text()
    if original.count(LOOP_MARKER)!=1 or original.count(LAST_MARKER)!=1:
        raise RuntimeError("Pinned LFM3 graph markers changed; refusing fuzzy patch")
    if '#include <cstdlib>' not in original:
        original=original.replace('#include "../llama-memory-hybrid-iswa.h"\n',
            '#include "../llama-memory-hybrid-iswa.h"\n#include <cstdlib>\n#include <string>\n#include <vector>\n')
    flow_code=f'''    // CyberSand #619 experimental System One layer surgery.
    // Only the decision graph is modified; normal completion/recurrent execution stays stock.
    std::vector<int> d1_flow;
    if constexpr (decision) {{
        const char * requested = std::getenv("CYBERSAND_D1_FRANKEN_FLOW");
        const std::string mode = requested ? requested : "stock";
        if (mode == "stock") {{
            d1_flow = {cpp_list(FLOWS["stock"])};
        }} else if (mode == "greyhound") {{
            d1_flow = {cpp_list(FLOWS["greyhound"])};
        }} else if (mode == "scrambled_eggs") {{
            d1_flow = {cpp_list(FLOWS["scrambled_eggs"])};
        }} else if (mode == "branchwitch") {{
            d1_flow = {cpp_list(FLOWS["branchwitch"])};
        }} else if (mode == "greyhound_restore5") {{
            d1_flow = {cpp_list(FLOWS["greyhound_restore5"])};
        }} else if (mode == "greyhound_restore7") {{
            d1_flow = {cpp_list(FLOWS["greyhound_restore7"])};
        }} else if (mode == "greyhound_restore12") {{
            d1_flow = {cpp_list(FLOWS["greyhound_restore12"])};
        }} else if (mode == "greyhound_restore16") {{
            d1_flow = {cpp_list(FLOWS["greyhound_restore16"])};
        }} else if (mode == "branchwitch_no_repeat10") {{
            d1_flow = {cpp_list(FLOWS["branchwitch_no_repeat10"])};
        }} else if (mode == "branchwitch_restore8") {{
            d1_flow = {cpp_list(FLOWS["branchwitch_restore8"])};
        }} else if (mode == "branchwitch_restore12") {{
            d1_flow = {cpp_list(FLOWS["branchwitch_restore12"])};
        }} else if (mode == "branchwitch_drop21") {{
            d1_flow = {cpp_list(FLOWS["branchwitch_drop21"])};
        }} else {{
            throw std::runtime_error("Unknown CYBERSAND_D1_FRANKEN_FLOW: " + mode);
        }}
    }} else {{
        for (int il = 0; il < n_layer; ++il) {{
            d1_flow.push_back(il);
        }}
    }}

    for (size_t d1_step = 0; d1_step < d1_flow.size(); ++d1_step) {{
        const int il = d1_flow[d1_step];
        auto * prev_cur = cur;
'''
    patched=original.replace(LOOP_MARKER,flow_code)
    patched=patched.replace(LAST_MARKER,
        '        if (d1_step + 1 == d1_flow.size() && inp_out_ids) {\n')
    path.write_text(patched)
    return {
        "schema":"cybersand-d1-frankenlayers-runtime-patch/1",
        "llama_commit":PINNED_LLAMA_COMMIT,
        "target":str(TARGET),
        "original_sha256":hashlib.sha256(original.encode()).hexdigest(),
        "patched_sha256":hashlib.sha256(patched.encode()).hexdigest(),
        "flows":{k:list(v) for k,v in FLOWS.items()},
        "flow_lengths":{k:len(v) for k,v in FLOWS.items()},
        "unique_layer_counts":{k:len(set(v)) for k,v in FLOWS.items()},
        "decision_graph_only":True,
        "q8_weights_modified":False,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--llama-root",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True)
    a=p.parse_args()
    receipt=patch(a.llama_root)
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps(receipt,sort_keys=True))

if __name__=="__main__":
    main()
