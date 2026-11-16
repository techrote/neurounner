#!/usr/bin/env python3
"""SPROCKET G09 proposed language label: strict source / real tokenizer review ONLY."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PAINTER = ROOT / "experiments" / "agent_painter"
sys.path[:0] = [str(HERE), str(PAINTER / "training"), str(PAINTER)]
from compiler import canonical_json, parse_program
from threefamily_v3 import program_for

def must(x, message):
    if not x:
        raise ValueError(message)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def trainer_system():
    tree = ast.parse((PAINTER / "training" / "train.py").read_text())
    values = [n.value.value for n in tree.body
              if isinstance(n, ast.Assign) and
              any(isinstance(t, ast.Name) and t.id == "SYSTEM" for t in n.targets) and
              isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)]
    must(len(values) == 1, "trainer SYSTEM literal changed")
    return values[0]

def review(record=None, source_program=None, snapshot=None):
    record = read(HERE / "G09_PROPOSED_LABEL_REVIEW_V1.json") if record is None else record
    offer = read(HERE / "G09_SHARED_ADMISSION_REVIEW_OFFER.json")
    profile = read(HERE / "profile.json")
    must(record["schema"] == "cybersand.sprocket-g09-proposed-label/1" and
         record["status"] == "REVIEW_ONLY_NOT_SHARED_TRAIN", "not review-only")
    program = program_for("G09") if source_program is None else source_program
    completion = canonical_json(program)
    digest = sha(completion.encode())
    must(digest == record["program_sha256"] ==
         offer["authored_program"]["program_sha256"] ==
         "62fc07e8e27a67c020acd023a2285ddf21fc4f5f8701fbf8e52ae0943893ee51" and
         parse_program(completion) == program, "G09 canonical program identity mismatch")
    must(record["original_native_source"] ==
         offer["immutable_inputs"]["original_native_research_producer_sha"] ==
         "63ee8b2b561068a72e0652e54b6e4193cb47d5e6" and
         record["original_native_run"] == 43587600137, "original producer drift")
    must(record["model_id"] == profile["model_id"] == "kandivault/sprocket-500m" and
         record["model_revision"] == profile["revision"] ==
         "eb157d4ddddc4619e661d4a2426bb2dd000fc668" and
         (record["expected_bos_id"], record["expected_pad_id"], record["expected_eos_id"]) ==
         (0, 1, 5) and
         profile["framing"]["template_sha256"] ==
         "995184ace6367465228c6ae16dcc5cfbb36e13147ea6bdba956f6b0a077204fa",
         "root Instruct tokenizer pin changed")
    must(record["request"] ==
         "Draw three wide separated horizontal machine equipment bands descending down a flat 96 by 96 2D canvas. The top band is longest with ribs, the middle is narrower with vents, and the bottom is shortest with plates; shift each successive band farther right. Keep two continuous empty horizontal service channels and generous background negative space. No central spine, doors, rooms or piping." and
         record["system"] == trainer_system() and
         record["target"] == "canonical_json(program_for('G09'))",
         "proposed source label changed")
    must(record["global_group_id"] is None and record["split"] == "NOT_ADMITTED" and
         record["reviewer_and_shepherd_global_approval"] is False and
         record["shared_positive_native_train_receipt"] is None and
         record["final_opened"] is False and record["draft_hold"] is True and
         all(type(record[key]) is int and record[key] == 0 for key in
             ("new_train_rows", "new_native_views", "new_optimizer_steps",
              "model_generated_views")),
         "research cannot claim shared admission or optimizer/native result")
    result = {"schema": "cybersand.sprocket-g09-label-review-result/1",
              "status": "SOURCE_ONLY_TOKENIZER_NOT_RUN",
              "request": record["request"], "request_sha256": sha(record["request"].encode()),
              "completion": completion, "completion_sha256": digest,
              "completion_bytes": len(completion.encode()), "train_admitted": False,
              "new_native_views": 0, "new_optimizer_steps": 0, "final_opened": False}
    if snapshot is not None:
        from transformers import AutoTokenizer
        import adapter
        snap = Path(snapshot)
        for fn in ("config.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "chat_template.jinja"):
            f = snap / fn
            must(f.is_file() and sha(f.read_bytes()) == profile["required_hashes"][fn],
                 "pinned official tokenizer file hash mismatch: " + fn)
        must(not (snap / "model.safetensors").exists(), "no full base weights in this proof")
        tokenizer = AutoTokenizer.from_pretrained(snap, local_files_only=True,
                                                   trust_remote_code=False)
        adapter.validate_tokenizer(tokenizer)
        encoded = adapter.encode_row(tokenizer, {
            "id": "G09-review-only", "prompt": record["request"],
            "completion": completion}, record["system"], 2048)
        ids, labels, boundary = (encoded[k] for k in ("input_ids", "labels", "boundary"))
        must(0 < boundary < len(ids) <= 2048 and ids[0] == 2 and
             ids[-1] == labels[-1] == 5 and labels.count(5) == 1 and
             labels[:boundary] == [-100] * boundary and
             labels[boundary:] == ids[boundary:] and
             tokenizer.decode(ids[boundary:-1], skip_special_tokens=False) == completion,
             "unqualified EOS, completion or loss-mask")
        result.update({"status": "PINNED_UPSTREAM_ROOT_TOKENIZER_PASS_REVIEW_ONLY",
                       "total_tokens": len(ids), "masked_prefix_tokens": boundary,
                       "supervised_completion_tokens": len(ids)-boundary,
                       "actual_eos_id": 5, "input_token_ids": ids, "labels": labels})
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    must(not args.output.exists(), "immutable receipt exists")
    result = review(snapshot=args.snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print({k: v for k, v in result.items()
           if k not in ("request", "completion", "input_token_ids", "labels")})

if __name__ == "__main__":
    main()
