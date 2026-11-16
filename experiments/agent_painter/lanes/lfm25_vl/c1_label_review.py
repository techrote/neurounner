#!/usr/bin/env python3
"""Review-only LFM C1 teacher label: source custody and official-tokenizer proof.

NEVER a shared TRAIN admission, new native execution, optimizer step, or model
completion. FINAL is not opened. Optional --snapshot must contain only official,
SHA-verified small model files; full weights are neither needed nor substituted.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LABEL = HERE / "C1_LABEL_FRAMING_REVIEW_V1.json"
SOURCE = HERE / "contrastive_proposals_v1.json"
CUSTODY = HERE / "C1_SHARED_ADMISSION_REVIEW_OFFER.json"
TRAINER = ROOT / "experiments/agent_painter/training/train.py"


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def strict_json(path: Path):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            require(key not in obj, "duplicate JSON key: " + key)
            obj[key] = value
        return obj
    def forbid_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=unique, parse_constant=forbid_constant)


def trainer_system(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    matches = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SYSTEM"
            for target in node.targets
        ):
            require(isinstance(node.value, ast.Constant) and
                    isinstance(node.value.value, str),
                    "trainer SYSTEM must be an exact literal")
            matches.append(node.value.value)
    require(len(matches) == 1, "trainer SYSTEM missing or ambiguous")
    return matches[0]


def review(*, label: Path = LABEL, source: Path = SOURCE,
           custody: Path = CUSTODY, trainer: Path = TRAINER,
           snapshot: Path | None = None) -> dict:
    record = strict_json(label)
    proposal = strict_json(source)
    original = strict_json(custody)
    require(record["schema"] == "cybersand.lfm25_vl.c1-label-framing-review/1",
            "unreviewed label schema")
    require(record["status"] == "REVIEW_ONLY_NOT_SHARED_TRAIN_ADMISSION",
            "label may not imply admission")
    pin = record["source"]
    framing = record["framing"]
    limitations = record["restrictions"]

    require(sha(source.read_bytes()) == pin["proposal_file_sha256"] ==
            original["originals"]["original_proposal_file_sha256"],
            "original proposal file changed")
    require(proposal["schema"] == "cybersand.lfm25_vl.contrastive_research_proposals/1",
            "source proposal schema")
    matches = [p for p in proposal["proposals"] if p["id"] == pin["proposal_id"]]
    require(len(matches) == 1 and pin["proposal_id"] ==
            original["authored_program"]["proposal_id"],
            "C1 must be unique and match original custody")
    p = matches[0]
    require(p["family"] == "greebles" and p["program"]["family"] == "greebles"
            and p["program"]["version"] == 2, "wrong teacher family/version")
    completion = canonical(p["program"])
    require(sha(completion.encode("utf-8")) == pin["program_sha256"] ==
            original["authored_program"]["program_sha256"],
            "canonical teacher bytes changed")
    require(framing["prompt_exact"] == p["request"] ==
            original["authored_program"]["request"], "original prompt mismatch")
    system = trainer_system(trainer)
    require(system == framing["system_exact"] and
            framing["system_source"] ==
            "experiments/agent_painter/training/train.py:SYSTEM",
            "shared trainer SYSTEM changed")
    require(framing["canonical_json"] ==
            "sort_keys=True,separators=(',', ':'),ensure_ascii=False,allow_nan=False",
            "canonical target contract changed")
    require(framing["completion"] ==
            "canonical JSON of proposals[0].program; no markdown, repairs, or answer truncation",
            "completion contract changed")
    require(framing["model_id"] == "LiquidAI/LFM2.5-VL-450M" and
            framing["revision"] == "fc6221ca597f3315e4f82fc2df606783267b34ba"
            and framing["official_processor"] == "Lfm2VlProcessor" and
            framing["adapter_framing"] == "lfm25-vl-text-only-end-at-eos-v1",
            "model/processor/framing pin changed")
    require(framing["eos_token_id"] == 7 and
            framing["max_total_tokens"] == 2048 and
            framing["requires_actual_official_processor_measurement"] is True and
            framing["tokenizer_counts_claimed_without_hosted_measurement"] is False,
            "unqualified EOS/sequence budget claim")
    require(pin["original_native_run_conclusion"] == "failure" and
            original["originals"]["original_native_job_outcome"] ==
            "FAILED_POST_RUN_INPUT_CUSTODY_KEY" and
            pin["original_native_producer"] ==
            original["originals"]["original_native_research_producer_sha"] and
            pin["original_native_artifact_id"] == 14111581483,
            "failed original producer/custody changed")
    require(pin["original_corpus_manifest_sha256"] ==
            original["originals"]["original_corpus_manifest_sha256"] ==
            "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc",
            "original corpus identity changed")
    require(pin["independently_reviewed_seed_variants"] == [104729, 130363]
            and pin["independent_structural_groups"] == 1,
            "seed variants falsely counted as independent")
    require(original["selection"]["review_not_admission"] is True and
            original["selection"]["decision"] ==
            "CONDITIONAL_YES_FOR_SHARED_ADMISSION_REVIEW",
            "independent reviewer selection changed")
    require(limitations["global_group_assignment"] == "UNASSIGNED" and
            limitations["split"] == "NOT_ADMITTED",
            "unapproved global group or split")
    for k in ("original_failed_research_workflow_remains_failed",
              "new_successful_source_built_native_admission_required",
              "reviewer_global_selection_and_shepherd_write_lease_required"):
        require(limitations[k] is True, "fail-closed gate missing: " + k)
    require(limitations["final_targets_opened"] is False,
            "FINAL must remain sealed")
    for k in ("train_rows_admitted", "new_native_executions",
              "optimizer_updates", "model_generated_views", "useful_model_freezes"):
        require(type(limitations[k]) is int and limitations[k] == 0,
                "review packet claims unperformed work: " + k)

    result = {
        "schema": "cybersand.lfm25_vl.c1-label-framing-review-result/1",
        "status": "SOURCE_CUSTODY_PASS_OFFICIAL_PROCESSOR_UNMEASURED",
        "request": p["request"],
        "request_sha256": sha(p["request"].encode("utf-8")),
        "request_bytes": len(p["request"].encode("utf-8")),
        "system_sha256": sha(system.encode("utf-8")),
        "teacher_canonical_completion": completion,
        "completion_sha256": sha(completion.encode("utf-8")),
        "completion_bytes": len(completion.encode("utf-8")),
        "original_native_job": "FAILED_POST_RUN_INPUT_CUSTODY_KEY",
        "independent_structural_groups": 1,
        "global_split": "NOT_ADMITTED",
        "new_native_executions": 0,
        "optimizer_updates": 0,
        "model_generated_views": 0,
        "final_opened": False,
    }
    if snapshot is not None:
        # This imports the exact reviewed adapter, official processor and pinned
        # CPU dependencies. It does NOT load base weights or train any parameter.
        import adapter
        require(adapter.MODEL_ID == framing["model_id"] and
                adapter.REVISION == framing["revision"] and
                adapter.FRAMING_VERSION == framing["adapter_framing"],
                "loaded adapter identity differs")
        processor = adapter.load_processor(snapshot)
        encoded = adapter.encode_row(processor, {
            "id": pin["proposal_id"], "prompt": p["request"], "completion": completion,
        }, system, framing["max_total_tokens"])
        ids = encoded["input_ids"]
        labels = encoded["labels"]
        require(len(ids) == len(labels) and len(ids) <= 2048,
                "incomplete or over-budget labels")
        require(ids[-1] == labels[-1] == framing["eos_token_id"] and
                labels.count(framing["eos_token_id"]) == 1 and
                all(value == -100 for value in labels[:encoded["boundary"]]) and
                all(value >= 0 for value in labels[encoded["boundary"]:]),
                "completion labels must preserve all target tokens and unique EOS")
        require(encoded["completion_bytes"] == result["completion_bytes"] and
                encoded["prompt_bytes"] == result["request_bytes"],
                "actual processor dropped answer or request bytes")
        result.update({
            "status": "OFFICIAL_PROCESSOR_FRAMING_PASS_REVIEW_ONLY_NOT_TRAIN",
            "token_budget": 2048,
            "sequence_tokens": len(ids),
            "prefix_tokens_masked": encoded["boundary"],
            "completion_tokens_supervised_including_eos": encoded["target_tokens"],
            "true_eos_token_id": 7,
            "input_token_ids": ids,
            "training_labels": labels,
            "processor_qualified": True,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path,
                        help="downloaded official SHA-verified small-file snapshot")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    outcome = review(snapshot=args.snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outcome, sort_keys=True, indent=2,
                                      ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in outcome.items() if
                     k not in ("teacher_canonical_completion", "input_token_ids",
                               "training_labels")}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
