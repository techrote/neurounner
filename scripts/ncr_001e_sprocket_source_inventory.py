#!/usr/bin/env python3
"""NCR-001E: read-only pinned Sprocket ordinary-role/source-token inventory.

Downloads only public metadata/tokenizers/README at ONE fixed upstream commit.
No full checkpoint, no model generation or optimizer. Distinct variants cannot
inherit claims about the other's tool/memory behavior.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import transformers

from ncr_001e_sprocket_format import (
    SOURCE_REPO, SOURCE_REV, serialize_dialogue, validate_pinned_card,
    validate_pinned_tokenizer
)

ROOT=Path(__file__).resolve().parents[1]
PINS=ROOT/"experiments/ncr-001/source-pins.json"
PROTOCOL=ROOT/"experiments/ncr-001/protocol.json"
OUTPUT=ROOT/"runtime/ncr-001/sprocket-pinned-role-inventory.json"
RESULT={
    "experiment":"NCR-001E",
    "stage":"pinned_source_role_and_special_control_inventory_only",
    "status":"RUNNING",
    "training_performed":False,
    "model_weights_downloaded":False,
    "model_inference_performed":False,
    "tool_and_memory_behavior_qualified":False,
}


def sha(data: bytes)->str:
    return hashlib.sha256(data).hexdigest()


def save()->None:
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(RESULT,indent=2,sort_keys=True)+"\n",
                      encoding="utf-8")


def main()->int:
    protocol=json.loads(PROTOCOL.read_text())
    pins=json.loads(PINS.read_text())
    if protocol["train_authorized"] is not False:
        raise RuntimeError("NCR training gate unexpectedly enabled")
    wanted={"chat":"sprocket_chat","instruct":"sprocket_instruct"}
    RESULT.update({
        "source_id":SOURCE_REPO,"revision":SOURCE_REV,
        "protocol_sha256":sha(PROTOCOL.read_bytes()),
        "pin_inventory_sha256":sha(PINS.read_bytes()),
        "python":platform.python_version(),
        "transformers":transformers.__version__,
        "variants":{}
    })
    save()
    md=Path(hf_hub_download(SOURCE_REPO,"README.md",revision=SOURCE_REV))
    raw=md.read_bytes()
    validate_pinned_card(raw.decode("utf-8"))
    original_card=raw.decode("utf-8")
    normalized_card=" ".join(original_card.split())
    RESULT["model_card"]={
        "sha256":sha(raw),"bytes":len(raw),
        "ordinary_chat_format_source_verified":True,
        "pinned_variant_descriptive_claims":{
            "chat_examples_removed": "tool-calling and memory examples removed" in normalized_card,
            "instruct_tool_training_description": "Instruct build: the tool-calling one" in normalized_card,
            "chat_build_label": "Chat build: start here" in normalized_card,
        },
        "relevant_exact_source_lines":[
            line.strip()[:240] for line in original_card.splitlines()
            if any(term in line.lower() for term in ("chat build","instruct build",
                "memory examples removed", "tool-calling one", "chat format"))
        ][:16],
        "tool_runtime_behavior_independently_verified":False,
        "note":"Pinned README role example and tokens are source authority; variant prose presence is only an observed descriptor, not tool-call semantic qualification",
    }
    for variant,key in wanted.items():
        p=protocol["models"][key]
        pin=pins["models"][key]
        if p["repo_id"]!=SOURCE_REPO or p["revision"]!=SOURCE_REV or pin["revision"]!=SOURCE_REV:
            raise RuntimeError(f"{variant}: source/model/revision mismatch")
        if variant=="chat":
            files=[
                ("chat/config.json","chat_config_sha256"),
                ("chat/tokenizer.json","chat_tokenizer_sha256"),
                ("chat/tokenizer_config.json","chat_tokenizer_config_sha256"),
            ]
            subfolder="chat"
        else:
            files=[
                ("config.json","config_sha256"),
                ("tokenizer.json","tokenizer_sha256"),
                ("tokenizer_config.json","tokenizer_config_sha256"),
                ("special_tokens_map.json","special_tokens_map_sha256"),
            ]
            subfolder=None
        item={
            "repo_id":SOURCE_REPO,"revision":SOURCE_REV,
            "variant":variant,"weight_file":p["weight_file"],
            "source_files":{},"model_weights_downloaded":False,
            "tokenizer_metadata_has_template":False,
            "tool_and_memory_behavior_qualified":False
        }
        for path,field in files:
            content=Path(hf_hub_download(SOURCE_REPO,path,revision=SOURCE_REV)).read_bytes()
            actual=sha(content)
            if actual!=pin[field]:
                raise RuntimeError(f"{variant}: source SHA mismatch: {path}")
            item["source_files"][path]={"sha256":actual,"bytes":len(content)}
            if path.endswith("tokenizer_config.json"):
                cfg=json.loads(content)
                item["tokenizer_metadata_has_template"]=bool(cfg.get("chat_template"))
                if cfg.get("chat_template"):
                    raise RuntimeError("Chat template unexpectedly added at pinned revision")
        kw={"revision":SOURCE_REV}
        if subfolder:
            kw["subfolder"]=subfolder
        tok=AutoTokenizer.from_pretrained(SOURCE_REPO,**kw)
        marker_inventory=validate_pinned_tokenizer(tok,variant=variant)
        item["tokenizer"]={
            "class":tok.__class__.__name__,
            "vocab_size":tok.vocab_size,
            "eos_id":tok.eos_token_id,
            "pad_id":tok.pad_token_id,
            **marker_inventory,
        }
        rendered=serialize_dialogue(
            [{"role":"user","content":"your message"}],variant=variant)
        ids=tok.encode(rendered,add_special_tokens=False)
        role_start=marker_inventory["controls"]["<|user|>"]["token_ids"][0]
        role_end=marker_inventory["controls"]["<|assistant|>"]["token_ids"][0]
        sep=marker_inventory["controls"]["<|end|>"]["token_ids"][0]
        if not(ids[0]==role_start and ids[-1]==role_end and sep in ids):
            raise RuntimeError(f"{variant}: true tokenizer role boundaries differ")
        item["documented_example"]={
            "source_string":rendered,
            "tokens":len(ids),
            "prefix_user_id":ids[0],
            "end_id":sep,
            "trailing_assistant_id":ids[-1],
            "proven_role_marker_boundaries":True
        }
        RESULT["variants"][variant]=item
        print(f"NCR SOURCE FORMAT {variant} verified user/end/assistant role IDs; "
              "tool/memory behaviour NOT qualified",flush=True)
        save()
    RESULT["status"]="PASS_PINNED_SPROCKET_ROLE_FORMAT_AND_TOKEN_INVENTORY__BEHAVIOR_BLOCKED"
    RESULT["training_performed"]=False
    RESULT["tool_and_memory_behavior_qualified"]=False
    save()
    print(json.dumps({
        "status":RESULT["status"],
        "model_card":RESULT["model_card"],
        "variants":{k:{
            "role":v["documented_example"],
            "special_tokens":v["tokenizer"]["controls"],
            "chat_template":v["tokenizer_metadata_has_template"],
            "tool_behavior_qualified":False
        }for k,v in RESULT["variants"].items()}
    },indent=2),flush=True)
    return 0


if __name__=="__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        RESULT["status"]="BLOCKED_PINNED_SPROCKET_SOURCE_OR_FORMAT_MISMATCH"
        RESULT["error"]=f"{type(exc).__name__}: {exc}"
        RESULT["traceback"]=traceback.format_exc()[-5000:]
        save()
        raise
