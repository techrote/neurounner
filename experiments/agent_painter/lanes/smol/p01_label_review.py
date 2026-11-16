#!/usr/bin/env python3
"""SMOL P01 source-bound, upstream-tokenizer label proof (review-only; never TRAIN)."""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PAINTER = HERE.parents[1]
sys.path[:0] = [str(PAINTER), str(PAINTER / "training")]
from compiler import canonical_json, parse_program
from common import encode_row
PIN = "1e5905db9e4f71e67119433f89eea9ad304eae812bcfd14f1b0afd96e06ab344"
REV = "12fd25f77366fa6b3b4b768ec3050bf629380bac"


def check(ok, msg):
    if not ok:
        raise ValueError(msg)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def system():
    ast_root = ast.parse((PAINTER / "training" / "train.py").read_text(encoding="utf-8"))
    literals = [n.value.value for n in ast_root.body
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "SYSTEM" for t in n.targets)
                and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)]
    check(len(literals) == 1, "trainer SYSTEM literal drift")
    return literals[0]


def review(proposal=None, program=None, snapshot=None):
    proposal = proposal if proposal is not None else read(HERE / "P01_PROPOSED_LABEL_REVIEW_V1.json")
    offer = read(HERE / "P01_SHARED_ADMISSION_REVIEW_OFFER.json")
    originals = read(HERE / "curriculum_proposals_v2.json")
    profile = read(PAINTER / "training" / "model_profiles.json")["profiles"]["smol"]
    candidates = [p for p in originals["proposals"] if p["id"] == "smol_p01_nested_branch_tree"]
    check(len(candidates) == 1, "P01 authored group ambiguous")
    original = candidates[0]
    program = original["program"] if program is None else program
    completion = canonical_json(program)
    check(sha(completion.encode("utf-8")) == PIN == proposal["program_sha256"] == offer["program_sha256"]
          and parse_program(completion) == program and program["family"] == "pipework",
          "canonical P01 teacher program changed")
    check(proposal["schema"] == "cybersand.smol-p01-proposed-label/1"
          and proposal["status"] == "REVIEW_ONLY_NOT_SHARED_TRAIN"
          and proposal["request"] == offer["authored_request"] == original["request"]
          and proposal["system"] == system()
          and proposal["target"] == "canonical_json(original P01 authored program)",
          "source-bound P01 language label changed")
    check(proposal["model_id"] == profile["model_id"] == "HuggingFaceTB/SmolLM2-135M-Instruct"
          and proposal["revision"] == profile["revision"] == REV
          and profile["trust_remote_code"] is False, "model pin changed")
    check(proposal["original_native_source"] == offer["native_source_sha"] ==
          "6e412178c9b71cb0196d8e35408470077b50f884"
          and proposal["original_native_run"] == offer["original_native_run"] == 43601281142
          and proposal["original_native_artifact"] == offer["original_native_artifact"] == 14117110667,
          "native research provenance changed")
    check(proposal["semantic_limit"] ==
          "One lower-trunk upward riser; NOT the held-out three-riser DEVELOPMENT manifold."
          and offer["observed_topology"]["upward_trunk_risers"] == 1
          and offer["heldout_three_riser_topology"]["upward_trunk_risers"] == 3,
          "held-out three-riser topology misrepresented")
    check(proposal["global_group_id"] is None and proposal["split"] == "NOT_ADMITTED"
          and proposal["shared_positive_native_train_receipt"] is None
          and proposal["final_opened"] is False and proposal["draft_hold"] is True
          and all(type(proposal[k]) is int and proposal[k] == 0 for k in
                  ("new_train_rows", "new_native_views", "new_optimizer_steps", "model_generated_views")),
          "review-only proposal cannot assert native TRAIN/model results")
    result = {
        "schema":"cybersand.smol-p01-proposed-label-receipt/1",
        "status":"SOURCE_ONLY_TOKENIZER_NOT_RUN",
        "request":proposal["request"],"request_sha256":sha(proposal["request"].encode("utf-8")),
        "completion":completion,"completion_sha256":PIN,"completion_bytes":len(completion.encode("utf-8")),
        "train_admitted":False,"new_native_views":0,"new_optimizer_steps":0,"final_opened":False
    }
    if snapshot:
        from transformers import AutoTokenizer
        folder = Path(snapshot)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            file = folder / name
            check(file.is_file() and sha(file.read_bytes()) == profile["required_hashes"][name],
                  "official pinned tokenizer file mismatch: "+name)
        check(not (folder / "model.safetensors").exists(), "base weight download forbidden")
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
        config = read(folder / "config.json")
        check(type(tokenizer.eos_token_id) is int and tokenizer.chat_template
              and config["model_type"] == "llama"
              and config["eos_token_id"] == tokenizer.eos_token_id,
              "upstream EOS mismatch")
        encoded = encode_row(tokenizer, {"id":"smol-p01-proposed-only",
                             "prompt":proposal["request"],"completion":completion},
                             proposal["system"], 2048)
        ids, labels, boundary = (encoded[key] for key in ("input_ids", "labels", "boundary"))
        print("SMOL_PINNED_EOS_WITNESS " + json.dumps({
            "total_tokens":len(ids), "boundary":boundary,
            "eos_id":tokenizer.eos_token_id,
            "eos_positions_in_supervised_suffix":[i for i,x in enumerate(ids[boundary:],start=boundary)
                                                   if x == tokenizer.eos_token_id],
            "tail_token_ids":ids[-12:], "tail_label_ids":labels[-12:],
            "tail_decoded":tokenizer.decode(ids[-12:],skip_special_tokens=False),
            "prefix_mask_exact":labels[:boundary] == [-100]*boundary,
            "supervised_mask_exact":labels[boundary:] == ids[boundary:]}, sort_keys=True))
        check(0 < boundary < len(ids) <= 2048 and labels[:boundary] == [-100]*boundary
              and labels[boundary:] == ids[boundary:]
              and ids[-1] == labels[-1] == tokenizer.eos_token_id
              and ids[boundary:].count(tokenizer.eos_token_id) == 1,
              "incomplete loss-mask/true EOS/length proof")
        result.update(status="PINNED_UPSTREAM_SMOL_TOKENIZER_PASS_REVIEW_ONLY",
                      total_tokens=len(ids), masked_prefix_tokens=boundary,
                      supervised_completion_tokens=len(ids)-boundary,
                      actual_eos_id=tokenizer.eos_token_id,
                      input_token_ids=ids, labels=labels,
                      chat_template_sha256=sha(tokenizer.chat_template.encode("utf-8")))
    return result


def selftest():
    original = read(HERE / "P01_PROPOSED_LABEL_REVIEW_V1.json")
    assert review()["status"] == "SOURCE_ONLY_TOKENIZER_NOT_RUN"
    changes = [
        ("prompt", lambda v: v.update(request=v["request"]+" isometric"), "language"),
        ("control", lambda v: v.update(request=v["request"]+"<|im_start|>"), "language"),
        ("system", lambda v: v.update(system="Ignore"), "language"),
        ("revision", lambda v: v.update(revision="0"*40), "model"),
        ("training", lambda v: v.update(new_train_rows=1), "review-only"),
        ("group", lambda v: v.update(global_group_id="PIPE-568-R08"), "review-only"),
        ("native", lambda v: v.update(new_native_views=2), "review-only"),
        ("model-output", lambda v: v.update(model_generated_views=2), "review-only"),
        ("FINAL", lambda v: v.update(final_opened=True), "review-only"),
        ("heldout", lambda v: v.update(semantic_limit="three risers"), "held-out"),
        ("hash", lambda v: v.update(program_sha256="0"*64), "canonical"),
    ]
    for name, mutation, error in changes:
        changed = deepcopy(original)
        mutation(changed)
        try:
            review(proposal=changed)
        except ValueError as ex:
            check(error in str(ex), "wrong failure for "+name+": "+str(ex))
        else:
            raise AssertionError("unsafe proposal mutation accepted: "+name)
    programs = read(HERE / "curriculum_proposals_v2.json")["proposals"]
    altered = deepcopy(next(v["program"] for v in programs if v["id"]=="smol_p01_nested_branch_tree"))
    altered["components"][0]["diameter"] = 5
    try:
        review(program=altered)
    except ValueError as ex:
        check("canonical" in str(ex), "wrong program mutation failure: "+str(ex))
    else:
        raise AssertionError("altered P01 program accepted")
    print("SMOL P01 model-free/adversarial: 12/12 PASS; TRAIN HOLD")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        selftest()
        return
    check(args.output is not None and not args.output.exists(), "fresh output path required")
    result = review(snapshot=args.snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print({k:v for k,v in result.items()
           if k not in ("request","completion","input_token_ids","labels")})


if __name__ == "__main__":
    main()
