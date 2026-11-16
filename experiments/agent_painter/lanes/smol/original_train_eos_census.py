#!/usr/bin/env python3
"""Read-only census of authentic SMOL EOS tails on native-admitted TRAIN and proposed P01.

This does not change the training encoder, normalize labels, run a model, or admit data.
FINAL payloads and DEVELOPMENT targets are never loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
PAINTER = HERE.parents[1]
sys.path[:0] = [str(PAINTER), str(PAINTER / "training"), str(HERE)]

from common import canonical, encode_row, load_training_corpus  # noqa: E402
from p01_label_review import review as review_p01, system as trainer_system  # noqa: E402

MODEL_REV = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
NATIVE_MANIFEST = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
NATIVE_SOURCE = "633920373a3d406a2ede61ce4137c324f76be23b"


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def assert_observed_tail(ids: list[int], labels: list[int], boundary: int,
                         eos_id: int, newline_id: int) -> None:
    """Fail closed: the *original*, unchanged encoder supervises exactly EOS then LF."""
    require(0 < boundary < len(ids) - 2, "invalid loss-mask boundary")
    require(labels == [-100] * boundary + ids[boundary:], "loss mask is not exact")
    require(ids[-2:] == [eos_id, newline_id], "unexpected supervised EOS presentation tail")
    require(ids[boundary:].count(eos_id) == 1, "missing or repeated genuine EOS")
    require(ids[-1] != eos_id and labels[-1] == newline_id,
            "historical encoder unexpectedly has terminal EOS")
    # EOS inside the masked system/user chat-prefix is legitimate upstream framing.


def self_test() -> None:
    original = [100, 101, 200, 2, 198]
    labels = [-100, -100, 200, 2, 198]
    assert_observed_tail(original, labels, 2, 2, 198)
    mutations = [
        (original[:-1], labels[:-1], 2, 2, 198),
        ([100, 101, 200, 2, 199], labels, 2, 2, 198),
        ([100, 101, 200, 2, 2, 198], [-100, -100, 200, 2, 2, 198], 2, 2, 198),
        ([100, 101, 200, 198, 198], [-100, -100, 200, 198, 198], 2, 2, 198),
        (original, [-100, -100, 200, -100, 198], 2, 2, 198),
        (original, labels, 0, 2, 198),
        (original, labels, 4, 2, 198),
        (original, [-100, -100, -100, 2, 198], 2, 2, 198),
    ]
    for index, args in enumerate(mutations):
        try:
            assert_observed_tail(*args)
        except ValueError:
            continue
        raise AssertionError(f"unsafe label-tail mutation {index} accepted")
    print("SMOL original EOS-tail census: 9/9 synthetic positive/negative controls PASS")


def row_witness(tokenizer, row: dict, system: str, eos_id: int, newline_id: int) -> dict:
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": row["prompt"]}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": row["completion"]}],
        tokenize=False, add_generation_prompt=False)
    require(full == prefix + row["completion"] + tokenizer.eos_token + "\n",
            "official template full JSON/prefix/EOS/newline differs: " + row["id"])
    encoded = encode_row(tokenizer, row, system, 2048)
    ids, labels, boundary = encoded["input_ids"], encoded["labels"], encoded["boundary"]
    assert_observed_tail(ids, labels, boundary, eos_id, newline_id)
    require(tokenizer.decode(ids[-2:], skip_special_tokens=False) == tokenizer.eos_token + "\n",
            "EOS presentation tail is not independently decodable")
    return {
        "id": row["id"],
        "prompt_sha256": sha_bytes(row["prompt"].encode("utf-8")),
        "completion_sha256": sha_bytes(row["completion"].encode("utf-8")),
        "total_tokens": len(ids), "masked_prefix_tokens": boundary,
        "supervised_tokens": len(ids) - boundary,
        "last_supervised_token_ids": ids[-2:],
        "eos_positions_in_supervised": [i for i in range(boundary, len(ids))
                                         if ids[i] == eos_id],
        "actual_terminal_supervised_eos": False,
    }


def run(corpus: Path, snapshot: Path, output: Path, source: str) -> None:
    require(re.fullmatch(r"[a-f0-9]{40}", source) is not None, "exact source missing")
    require(not output.exists(), "output must be new and immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_file = corpus / "admitted-corpus.json"
    require(manifest_file.is_file() and sha_bytes(manifest_file.read_bytes()) == NATIVE_MANIFEST,
            "original native-admitted manifest mismatch")
    rows, manifest = load_training_corpus(corpus)
    require(manifest["source_sha"] == NATIVE_SOURCE and len(rows) == 12
            and len(manifest["structural_groups"]["TRAIN"]) == 6
            and {r["family"] for r in rows} == {"greebles", "pipework", "architecture"},
            "original positive TRAIN corpus/splits changed")
    from transformers import AutoTokenizer
    profile = json.loads((PAINTER / "training" / "model_profiles.json").read_text())["profiles"]["smol"]
    require(profile["revision"] == MODEL_REV and profile["trust_remote_code"] is False,
            "official model identity changed")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        f = snapshot / name
        require(f.is_file() and sha_bytes(f.read_bytes()) == profile["required_hashes"][name],
                "upstream tokenizer/config checksum mismatch: " + name)
    require(not any(snapshot.glob("*.safetensors")), "model weights are forbidden in source-only census")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False, local_files_only=True)
    require(tokenizer.eos_token_id == 2 and tokenizer.eos_token == "<|im_end|>",
            "actual upstream SMOL EOS changed")
    require(tokenizer.encode("\n", add_special_tokens=False) == [198],
            "actual upstream LF token changed")
    sys_prompt = trainer_system()
    witnesses = []
    for row in rows:
        prepared = {"id": row["id"], "prompt": row["prompt"], "completion": canonical(row["program"])}
        witness = row_witness(tokenizer, prepared, sys_prompt, 2, 198)
        witness.update(family=row["family"], structural_group=row.get("group_id", row.get("structural_group")),
                       custody="ORIGINAL_POSITIVE_NATIVE_TRAIN")
        witnesses.append(witness)
    proposal = review_p01()
    require(proposal["status"] == "SOURCE_ONLY_TOKENIZER_NOT_RUN" and not proposal["train_admitted"],
            "P01 changed admission status")
    p01 = row_witness(tokenizer, {"id": "P01-REVIEW-ONLY", "prompt": proposal["request"],
                                 "completion": proposal["completion"]}, sys_prompt, 2, 198)
    require((p01["total_tokens"], p01["masked_prefix_tokens"], p01["supervised_tokens"])
            == (194, 71, 123), "original P01 witnessed tokenization changed")
    report = {
        "schema": "cybersand.smol-original-eos-tail-census/1",
        "finding": "ORIGINAL_SHARED_ENCODER_SUPERVISES_PRESENTATION_LF_AFTER_TRUE_EOS",
        "source_sha": source, "model_revision": MODEL_REV,
        "native_source_sha": NATIVE_SOURCE, "admitted_manifest_sha256": NATIVE_MANIFEST,
        "split_read": "TRAIN_ONLY", "train_rows": len(witnesses),
        "train_structural_groups": 6, "review_only_p01": p01,
        "original_train_witnesses": witnesses,
        "actual_eos_id": 2, "presentation_lf_id": 198,
        "shared_encoder_changed": False, "labels_normalized": False,
        "new_train_admissions": 0, "new_optimizer_updates": 0,
        "new_native_executions": 0, "model_generated_outputs": 0,
        "final_targets_accessed": False, "shared_label_fix_approved": False,
        "training_and_useful_freeze_authorized": False,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SMOL authentic historical TRAIN EOS-tail census completed: "
          f"{len(witnesses)}/12 actual positive TRAIN rows + 1 P01 review proposal")
    print("WARNING: this is source-only defect characterization; not corrected labels or TRAIN admission")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--snapshot", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--source", type=str)
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        require(all((args.corpus, args.snapshot, args.output, args.source)), "required arguments missing")
        run(args.corpus, args.snapshot, args.output, args.source)
