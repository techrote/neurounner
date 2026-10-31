#!/usr/bin/env python3
"""Independent original Sprocket Chat generate/right-gap receipt audit.

Standard-library only. Reads exact original Actions text evidence; does not
execute/download a model, open heldouts, compute gradients or authorize training.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import lzma
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
DIR=ROOT/"experiments/ncr-001/evidence-review/sprocket-chat-generate"
ARCHIVE=DIR/"original-text-evidence.tar.xz.b64"
PROVENANCE=DIR/"provenance.json"

ORIGINAL_RUN=40603226200
ORIGINAL_BRANCH_HEAD="82c77846f3bbcb2cb98c7420be9cd14b2ab435f1"
ORIGINAL_MERGE="ef025d5e6899288005966a94551c6441211705bd"
ORIGINAL_ZIP_SHA="b507c750a435880a290b4b3260a2de07cc20bf68f6ba17c68e1f502b89f7fff9"
ARCHIVE_XZ_SHA="bfbf783babfc12e12c09d86d2efa512c253c4b41f142b44e0ed5fdb445595ab3"
ARCHIVE_TAR_SHA="ad22df2f70116a25df4f515be2c9c3f847246862c44b56c0f813a5e10d3e59be"
REPORT_SHA="13a23dbe2ccb17bbbee0f168f1c518a85dc0b2b336a7700fa1320898e1b4149c"
CHAT_SHA="551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97"
INSTRUCT_SHA="1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864"
HOOK_SHA="83a56835a7476f17a17980d28dd1a7c14bb6f2f6"
FORMAT_BLOB="39cf015735e74a1d1e08682b623404bc1e600b9b"
SCRIPT_SHA="d9e5171e17cad7c9c6d60f0b9a3ff04115bed39d9b779a1bace6e9eb30a35e43"
GEN_IDS=[32,29,7220,277]
ATOL=RTOL=0.0005

EXPECTED={
    "sprocket-chat-generate-review.json":
        (13417,REPORT_SHA),
    "sprocket-chat-generate-prefetch.json":
        (1033,"61d984b22326c1699f090f74a578c010c7839caef280c13f9e809d16c5200501"),
    "sprocket-chat-generate-packages.txt":
        (1714,"506849ee72ac85b225d9f7ede1acb412b95df9233e5947c4eebab2d988bf6cb9"),
    "sprocket-chat-generate-source-commit.txt":
        (41,"df9952150a76dac284031131771ae22df26dc76ae5eba678a85574729e95838f"),
    "sprocket-chat-generate-source-hashes.sha256":
        (421,"6f186a306c0a621230f89be4cf539761e943e3704dda6f80b53f2585da4b57b0"),
}

PIN_FILES={
    "README.md":(None,8718),
    "chat/model.safetensors":("weight_declared_lfs_sha256_not_downloaded",1002207984),
    "chat/config.json":("chat_config_sha256",689),
    "chat/tokenizer.json":("chat_tokenizer_sha256",2269469),
    "chat/tokenizer_config.json":("chat_tokenizer_config_sha256",3325),
}


def require(ok,why):
    if not ok:
        raise RuntimeError("NCR original Chat generate audit REFUSED: "+why)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load_originals(archive_path=ARCHIVE,provenance_path=PROVENANCE):
    p=json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    encoded="".join(Path(archive_path).read_text(encoding="ascii").split())
    require(len(encoded)==5632 and encoded.isascii(),"base64 archive envelope")
    xz=base64.b64decode(encoded,validate=True)
    require(len(xz)==p["archive"]["xz_bytes"]==4224
            and digest(xz)==p["archive"]["xz_sha256"]==ARCHIVE_XZ_SHA,
            "original XZ bytes changed")
    raw=lzma.decompress(xz,format=lzma.FORMAT_XZ)
    require(len(raw)==p["archive"]["tar_bytes"]==30720
            and digest(raw)==p["archive"]["tar_sha256"]==ARCHIVE_TAR_SHA,
            "original USTAR bytes changed")
    out={}
    with tarfile.open(fileobj=io.BytesIO(raw),mode="r:") as tar:
        members=tar.getmembers()
        require([m.name for m in members]==sorted(EXPECTED),
                "missing/duplicate/reordered source members")
        for member in members:
            require(member.isfile() and member.name in EXPECTED
                    and member.mode==0o644 and member.uid==member.gid==member.mtime==0,
                    "noncanonical source member metadata")
            data=tar.extractfile(member).read()
            size,sha=EXPECTED[member.name]
            original=p["archive"]["originals"][member.name]
            require(member.size==len(data)==size==original["bytes"]
                    and digest(data)==sha==original["sha256"],
                    "original source bytes changed: "+member.name)
            out[member.name]=data
    return out,p


def verify_originals(files,p):
    actions=p["original_actions"]; source=p["source"]; runner=p["runner"]
    results=p["results"]; scope=p["scope"]
    report=json.loads(files["sprocket-chat-generate-review.json"])
    prefetch=json.loads(files["sprocket-chat-generate-prefetch.json"])

    require(p["schema_version"]==1 and p["source_issue"]==72
            and p["parent_issue"]==67 and p["reviewer_pr"]==123,
            "wrong experiment/PR identity")
    require(p["status"]==
            "PASS_CHAT_RIGHTGAP_AND_HF_GENERATE_PRETRAIN_ONLY__FINAL_72_BLOCKED"
            and report["status"]=="PASS_CHAT_RIGHTGAP_AND_HF_GENERATE_PRETRAIN_ONLY"
            and prefetch["status"]=="PASS_CHAT_FULL_SOURCE_PREFETCH",
            "source/model conclusion relabelled")
    require(actions["run_id"]==ORIGINAL_RUN and actions["run_attempt"]==1
            and actions["branch_head_during_execution"]==ORIGINAL_BRANCH_HEAD
            and actions["github_checkout_merge_sha"]==ORIGINAL_MERGE
            and actions["source_job_id"]==122536255569
            and actions["actual_model_job_id"]==122536314707
            and actions["source_result"]==actions["model_result"]=="success"
            and actions["artifact_id"]==12780115389
            and actions["original_zip_sha256"]==ORIGINAL_ZIP_SHA
            and actions["original_zip_bytes"]==5677
            and actions["original_zip_bytes_committed"] is False,
            "original Actions/ZIP custody changed")
    failure=p["preserved_pre_model_failure"]
    require(failure["run_id"]==40603167288 and failure["result"]=="failure"
            and failure["failed_stage"]=="source_only_synthetic_test"
            and failure["model_executed"] is False
            and failure["exact_head"]=="4df0c02120e39ff5c02127d968fd2553032eb249",
            "first source-only failed run removed/reclassified")
    require(files["sprocket-chat-generate-source-commit.txt"]==
            (ORIGINAL_MERGE+"\n").encode("ascii"),
            "original checkout source commit changed")

    hashes={}
    for line in files["sprocket-chat-generate-source-hashes.sha256"].decode("ascii").splitlines():
        require(len(line)>66 and line[64:66]=="  ","source SHA line format")
        sha,name=line[:64],line[66:]
        require(name not in hashes and len(sha)==64
                and all(c in "0123456789abcdef" for c in sha),
                "invalid/duplicate source digest")
        hashes[name]=sha
    require(hashes=={
        "scripts/ncr_001e_sprocket_chat_generate_review.py":SCRIPT_SHA,
        "scripts/ncr_001e_sprocket_format.py":
            "1ca6cfde8b4144e6f56fdfb61fb075e31fdc9b4ebc68ceeb68c57c01128c363b",
        "experiments/ncr-001/protocol.json":
            "a76d38f80a57012ca8e62fba183d1cf2e91fd0986032f1adcdfb0aef843bf0b4",
        "experiments/ncr-001/source-pins.json":
            "db794f53a49bb1bcbf8b23bd4a76d24430eb495da246639b506abdbbf11734b1",
    },"original source SHA inventory changed")
    require(digest((ROOT/"scripts/ncr_001e_sprocket_chat_generate_review.py").read_bytes())
            ==SCRIPT_SHA,"executed Chat reviewer source drift")
    protocol_path=ROOT/"experiments/ncr-001/protocol.json"
    pins_path=ROOT/"experiments/ncr-001/source-pins.json"
    require(digest(protocol_path.read_bytes())==source["protocol_sha256"]==
            report["source"]["protocol_sha256"]==hashes["experiments/ncr-001/protocol.json"],
            "protocol bytes differ from original run")
    require(digest(pins_path.read_bytes())==source["source_pins_sha256"]==
            report["source"]["source_pins_sha256"]==hashes["experiments/ncr-001/source-pins.json"],
            "source pin bytes differ from original run")
    protocol=json.loads(protocol_path.read_text());pins=json.loads(pins_path.read_text())
    require(protocol["train_authorized"] is False
            and set(protocol["models"])==set(pins["models"])==
            {"smollm2","sprocket_chat","sprocket_instruct","void1"}
            and protocol["conformance"]["full_vs_cached_prefill_atol"]==
            protocol["conformance"]["full_vs_cached_prefill_rtol"]==ATOL==RTOL,
            "four-model protocol/cache gate changed")
    pin=pins["models"]["sprocket_chat"]; root=pins["models"]["sprocket_instruct"]
    require(report["variant"]==prefetch["variant"]==source["variant"]=="chat"
            and source["checkpoint_file"]==report["source"]["checkpoint_file"]==
            pin["hf_weight_file"]=="chat/model.safetensors"
            and source["checkpoint_sha256"]==report["source"]["checkpoint_sha256"]==
            pin["weight_declared_lfs_sha256_not_downloaded"]==CHAT_SHA
            and source["distinct_instruct_checkpoint_sha256"]==
            report["source"]["distinct_root_instruct_sha256"]==
            root["weight_declared_lfs_sha256_not_downloaded"]==INSTRUCT_SHA
            and CHAT_SHA!=INSTRUCT_SHA,
            "Chat/root Instruct source substitution")
    require(source["original_hook_commit"]==report["source"]["original_hook_commit"]==
            HOOK_SHA and source["formatter_git_blob"]==
            report["source"]["formatter_git_blob"]==FORMAT_BLOB
            and source["model_card_sha256"]==report["source"]["model_card_sha256"]==
            "09aec879dc5b387bdc031dc3c74d77a9ba44362a55163f9a8b456761b68142c9"
            and source["ordinary_role_format_only"] is True
            and source["instruct_tool_memory_semantics_inherited"] is False
            and report["source"]["tool_memory_semantics_inherited_from_instruct"] is False,
            "ordinary Chat format/tool-memory scope changed")
    require(set(prefetch["files"])==set(report["actual_source_files"])==set(PIN_FILES),
            "complete original Chat source file inventory changed")
    for name,(pin_key,size) in PIN_FILES.items():
        a=prefetch["files"][name];b=report["actual_source_files"][name]
        expected=(source["model_card_sha256"] if pin_key is None else pin[pin_key])
        require(a["sha256"]==b["sha256"]==expected
                and a["bytes"]==b["bytes"]==size,
                "original source bytes changed: "+name)

    require(report["train_authorized"] is False
            and report["optimizer_steps"]==0 and report["heldout_evaluation"] is False
            and report["actual_training"] is False
            and report["tool_memory_semantics_qualified"] is False
            and report["final_four_model_72_conformance"] is False
            and prefetch["train_authorized"] is False and prefetch["optimizer_steps"]==0,
            "original no-training/semantic limits changed")
    for key in ("train_authorized","actual_training","heldout_evaluation",
                "tool_memory_semantics_qualified","final_four_model_72_conformance",
                "comparative_NCA_MLP_training","model_generated_tool_memory_competence"):
        require(scope[key] is False,"false final/training/semantic promotion: "+key)
    require(scope["optimizer_steps"]==0 and scope["all_parent_prs_draft"] is True,
            "zero optimizer/draft source limit changed")

    rr=report["runner"]
    require(rr["github_run_id"]==runner["github_run_id"]==str(ORIGINAL_RUN)
            and rr["github_sha"]==runner["github_merge_sha"]==ORIGINAL_MERGE
            and rr["runner_name"]==runner["runner_name"]=="GitHub Actions 1000045727"
            and rr["pid"]==runner["pid"]==2752
            and rr["cpu_count"]==runner["cpu_count"]==4
            and rr["python"]==runner["python"]=="3.12.3"
            and rr["torch"]==runner["torch"]=="2.14.1+cpu"
            and rr["transformers"]==runner["transformers"]=="4.57.6",
            "actual Chat runner/library/PID evidence changed")
    require(report["wire"]["long_chat"]["tokenizer_units"]==
            results["source_prompt_units"][0]==47
            and report["wire"]["short_chat"]["tokenizer_units"]==
            results["source_prompt_units"][1]==18
            and all(v["source_role_format_only"] is True for v in report["wire"].values()),
            "ordinary Chat source wire identity changed")
    gap=report["right_gap"]
    require(gap["prefill_width"]==results["right_gap_prefill_width"]==43
            and gap["valid_prefix_lengths"]==results["right_gap_valid_prefix_units"]==[43,14]
            and gap["tail_units"]==4 and gap["pad_id"]==1,
            "right-gap source geometry changed")
    require(report["frozen_base_parameters"]==results["frozen_base_parameters"]==501090560
            and report["adapter_parameters"]==results["adapter_parameters"]==658904
            and report["backbone_sha256_before"]==report["backbone_sha256_after"]==
            results["frozen_tensor_sha_before"]==results["frozen_tensor_sha_after"]==
            "232fdbdd370f72bdecf549952896970b75c56155b8c6b53387338ce38540813f",
            "frozen Chat base/source tensor evidence changed")
    checks=report["checks"]
    for name in ("frozen_zero_exact_long_chat","frozen_zero_exact_short_chat",
                 "masked_padding_control_id7_exact","post_generation_A_B_A_exact"):
        c=checks[name]
        require(c["pass"] is True and c["exact_required"] is True
                and c["max_abs"]==c["atol"]==c["rtol"]==0,
                "exact Chat source invariant failed: "+name)
    for name,expected in (
        ("rightgap_vs_unpadded_long_chat",results["rightgap_long_max_abs"]),
        ("rightgap_vs_unpadded_short_chat",results["rightgap_short_max_abs"]),
    ):
        c=checks[name]
        require(c["pass"] is True and c["exact_required"] is False
                and c["atol"]==c["rtol"]==ATOL and c["max_abs"]==expected<=ATOL,
                "right-gap cached Chat source mismatch: "+name)
    require(results["rightgap_long_max_abs"]==0.000028133392333984375
            and results["rightgap_short_max_abs"]==0.000027418136596679688,
            "original Chat right-gap numerical evidence changed")
    m=checks["manual_untrained_effect"]
    require(m["pass"] is True and m["trained"] is False
            and m["max_abs"]==results["manual_untrained_effect_max_abs"]==
            0.014346718788146973,"untrained Chat effect changed")
    bad=checks["unmasked_padding_negative_detected"]
    require(bad["pass"] is True and bad["expected_difference"] is True
            and bad["max_abs"]==results["invalid_unmasked_padding_max_abs"]==
            7.743958473205566,"invalid mask sentinel changed")

    generations=report["generations"]
    require(len(generations)==6 and results["generation_modes"]==
            ["disabled","zero","active_untrained"] and results["use_cache_modes"]==[False,True]
            and results["generated_ids"]==GEN_IDS,
            "actual Chat generation coverage/IDs changed")
    require([(g["mode"],g["use_cache"]) for g in generations]==[
        (mode,cache) for mode in ("disabled","zero","active_untrained")
        for cache in (False,True)],"Chat generation mode/cache order changed")
    score_map={(g["mode"],g["use_cache"]):g["score_sha256"] for g in generations}
    for g in generations:
        require(g["generated_ids"]==GEN_IDS and g["prompt_units"]==47
                and g["optimizer_steps"]==0
                and g["tool_memory_semantics_qualified"] is False
                and len(g["score_sha256"])==4
                and all(isinstance(h,str) and len(h)==64 for h in g["score_sha256"]),
                "raw Chat generation record changed")
    for cache in (False,True):
        require(score_map[("disabled",cache)]==score_map[("zero",cache)],
                "zero adapter original generation score hashes differ from frozen source")
    maxima=[]
    for cache in ("False","True"):
        for step in range(4):
            c=checks[f"generate_zero_exact_cache_{cache}_step_{step}"]
            require(c["pass"] is True and c["exact_required"] is True
                    and c["max_abs_finite"]==c["atol"]==c["rtol"]==0
                    and c["finite_logits"]==31999 and c["nonfinite_logits"]==1,
                    "zero Chat generate source parity changed")
    for mode in ("disabled","zero","active_untrained"):
        for step in range(4):
            c=checks[f"generate_full_vs_cache_{mode}_step_{step}"]
            require(c["pass"] is True and c["exact_required"] is False
                    and c["atol"]==c["rtol"]==ATOL
                    and c["finite_logits"]==31999 and c["nonfinite_logits"]==1
                    and 0<=c["max_abs_finite"]<=ATOL,
                    "Chat HF generate cache/no-cache score mismatch")
            maxima.append(c["max_abs_finite"])
    require(max(maxima)==results["max_generate_cache_on_off_abs"]==
            0.00002002716064453125,
            "Chat generate maximum source metric changed")


def main():
    files,p=load_originals()
    verify_originals(files,p)
    print(json.dumps({
        "status":"PASS_CHAT_ORIGINAL_GENERATE_ARCHIVE__FINAL_72_BLOCKED",
        "original_run_id":ORIGINAL_RUN,
        "original_model_json_sha256":REPORT_SHA,
        "original_zip_sha256":ORIGINAL_ZIP_SHA,
        "model_reexecuted_during_audit":False,
        "train_authorized":False,
        "optimizer_steps":0,
        "tool_memory_semantics_qualified":False,
    },sort_keys=True))


if __name__=="__main__":
    main()
