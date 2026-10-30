#!/usr/bin/env python3
"""Byte-exact audit of original BYTE-EOS-001 independent reviewer artifact.

Standard-library only: no Hugging Face runtime, no model, no optimizer.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DIR=ROOT/"experiments/ncr-001/byte-eos-001/independent-review"
ARCHIVE=DIR/"original-actions-a1.zip.b64"
PROVENANCE=DIR/"provenance.json"
PREREG=ROOT/"prereg/experiments/ncr-001/protocol.json"

ZIP_SHA="9cde28bc5a927c262a0eda907f445491f4937c0ed7283a1ea535d2b5b323b1bf"
REPORT_SHA="0400daf39c36e64707525856850e89720fd83742514520a9a27a019d1feae527"
CANDIDATE_SHA256="ab4dda9a8afeb98bfac9a31f9c8f0f6b651432aade30ab6eec16bc1148253256"
CANDIDATE_BLOB="724a5040d2fc9eb06ef0b17143d44daa36a4aa89"
REVIEW_SHA256="b2539fbc28dd28dc04efbb83a2d540cfd5a5eaf672373cadcfe49e55de95dc44"
TOKENIZER_SHA="279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03"
SOURCE_PINS_SHA="db794f53a49bb1bcbf8b23bd4a76d24430eb495da246639b506abdbbf11734b1"
BRANCH_PROTOCOL_SHA="2e294750a17b1bfb3eba448f8a5b55344c72aff700587351bd10da1af38885ab"
PREREG_PROTOCOL_SHA="a76d38f80a57012ca8e62fba183d1cf2e91fd0986032f1adcdfb0aef843bf0b4"
ORIGINAL_RECEIPT_SHA="6c611bdc0f021f6d9e2fa39d91b7e141d937844f2b849f9ae437151254572922"
ORIGINAL_PROVENANCE_SHA="b342c2665314d107a3be4c0390f4d674c286f3a52e38d90cc7dbfc40d8879c50"

EXPECTED={
    "byte-eos-001-independent-review.json":(4083,REPORT_SHA),
    "byte-eos-independent-source-commit.txt":
        (41,"2b5926b75f85c4e09d252b14b16b3fc8c736c84a2b630e7fe4289cfffe3c7009"),
    "byte-eos-independent-source-hashes.sha256":
        (648,"f435d05504b59f236a9ea4727042015c96b7cc208343594debac82e4dcdb4e5a"),
}


def require(ok,why):
    if not ok:
        raise RuntimeError("BYTE-EOS independent archive REFUSED: "+why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git_blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def load_originals(archive_path=ARCHIVE,provenance_path=PROVENANCE):
    p=json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    encoded="".join(Path(archive_path).read_text(encoding="ascii").split())
    raw=base64.b64decode(encoded,validate=True)
    require(len(raw)==p["original_actions"]["artifact_zip_bytes"]==2396
            and sha(raw)==p["original_actions"]["artifact_zip_sha256"]==ZIP_SHA,
            "original complete Actions ZIP changed")
    out={}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        require(z.testzip() is None,"ZIP CRC failure")
        infos=z.infolist()
        require([i.filename for i in infos]==list(EXPECTED),
                "unexpected ZIP member ordering/count")
        for info in infos:
            data=z.read(info.filename)
            size,digest=EXPECTED[info.filename]
            pfile=p["original_members"][info.filename]
            require(len(data)==info.file_size==size==pfile["bytes"]
                    and sha(data)==digest==pfile["sha256"],
                    "original member bytes changed: "+info.filename)
            out[info.filename]=data
    return out,p


def verify_originals(files,p):
    actions=p["original_actions"]; reviewed=p["reviewed_candidate"]
    source=p["source"]; result=p["result"]; boundary=p["boundaries"]
    report=json.loads(files["byte-eos-001-independent-review.json"])
    require(p["schema_version"]==1 and p["experiment"]=="BYTE-EOS-001"
            and p["issue"]==92 and p["candidate_pr"]==94
            and p["independent_reviewer_pr"]==126,
            "wrong issue/review identity")
    require(p["status"]==
            "PASS_INDEPENDENT_FUTURE_EOS_SERIALIZER__VOID_TRAINING_INTEGRATION_STILL_BLOCKED"
            and report["status"]==p["status"],
            "independent decision relabelled")
    require(actions["run_id"]==report["original_qualification"].get("review_run_id",actions["run_id"])==40604675320
            and actions["branch_head"]=="d2a57ec2b390c37345b967bef210291c3d0e953e"
            and actions["checkout_merge_sha"]==report["runner"]["github_sha"]==
            "24353315add62caef2361c35aee23a4ca4c15d26"
            and actions["source_job_id"]==122541088182
            and actions["runtime_job_id"]==122541140224
            and actions["source_result"]==actions["runtime_result"]=="success"
            and actions["artifact_id"]==12779882453
            and actions["artifact_zip_sha256"]==ZIP_SHA
            and actions["artifact_zip_bytes_committed_as_base64"] is True,
            "Actions source/ZIP custody changed")
    require(files["byte-eos-independent-source-commit.txt"]==
            b"24353315add62caef2361c35aee23a4ca4c15d26\n",
            "original Actions checkout SHA changed")

    hashes={}
    for line in files["byte-eos-independent-source-hashes.sha256"].decode("ascii").splitlines():
        require(len(line)>66 and line[64:66]=="  ","source hash line format")
        digest,name=line[:64],line[66:]
        require(name not in hashes and len(digest)==64
                and all(c in "0123456789abcdef" for c in digest),
                "invalid/duplicate source hash")
        hashes[name]=digest
    require(hashes=={
        "scripts/byte_eos_001_void_labels.py":CANDIDATE_SHA256,
        "scripts/byte_eos_001_independent_review.py":REVIEW_SHA256,
        "experiments/ncr-001/byte-eos-001/receipt.json":ORIGINAL_RECEIPT_SHA,
        "experiments/ncr-001/byte-eos-001/provenance.json":ORIGINAL_PROVENANCE_SHA,
        "experiments/ncr-001/source-pins.json":SOURCE_PINS_SHA,
        "prereg/experiments/ncr-001/protocol.json":PREREG_PROTOCOL_SHA,
    },"original source hash inventory changed")

    candidate=(ROOT/"scripts/byte_eos_001_void_labels.py").read_bytes()
    reviewer=(ROOT/"scripts/byte_eos_001_independent_review.py").read_bytes()
    require(sha(candidate)==reviewed["candidate_file_sha256"]==CANDIDATE_SHA256
            and git_blob(candidate)==reviewed["candidate_git_blob"]==
            report["candidate_blob"]==CANDIDATE_BLOB,
            "reviewed candidate source changed")
    require(sha(reviewer)==reviewed["independent_reviewer_file_sha256"]==REVIEW_SHA256,
            "executed independent reviewer source changed")
    receipt=ROOT/"experiments/ncr-001/byte-eos-001/receipt.json"
    oldprov=ROOT/"experiments/ncr-001/byte-eos-001/provenance.json"
    pins=ROOT/"experiments/ncr-001/source-pins.json"
    protocol=ROOT/"experiments/ncr-001/protocol.json"
    require(sha(receipt.read_bytes())==reviewed["original_byte_eos_receipt_sha256"]==
            report["original_qualification"]["receipt_sha256"]==ORIGINAL_RECEIPT_SHA,
            "original BYTE-EOS receipt changed")
    require(sha(oldprov.read_bytes())==reviewed["original_byte_eos_provenance_sha256"]==
            ORIGINAL_PROVENANCE_SHA,"original BYTE-EOS provenance changed")
    require(sha(pins.read_bytes())==source["source_pins_sha256"]==
            report["source"]["source_pins_sha256"]==SOURCE_PINS_SHA,
            "source-pins bytes changed")
    require(sha(protocol.read_bytes())==report["source"]["protocol_sha256"]==
            BRANCH_PROTOCOL_SHA,"candidate branch protocol changed")
    require(source["preregistration_commit"] ==
            report["source"]["preregistration_commit"] ==
            "a1b41c358d215974ea9ef2748baa13c3872b0fbb",
            "preregistration commit identity changed")
    require(PREREG.exists() and sha(PREREG.read_bytes())==
            source["preregistration_protocol_sha256"]==
            report["source"]["preregistration_protocol_sha256"]==
            PREREG_PROTOCOL_SHA,"exact preregistration protocol missing/changed")
    prereg=json.loads(PREREG.read_text())
    require(prereg["train_authorized"] is False
            and prereg["training"]["per_model_sequence_unit_limits"]["void1"]==1024
            and prereg["models"]["void1"]["revision"]==
            "9c2be5087772867d45961abd11f6b35110617563",
            "void preregistration admission/limit changed")

    require(source["tokenizer_source_sha256"]==report["source"]["tokenizer_source_sha256"]==
            TOKENIZER_SHA and report["source"]["ids"]==
            {"pad":256,"bos":257,"eos":258,"vocab":259}
            and report["source"]["max_units"]==source["max_units"]==1024,
            "ByteTokenizer source/special IDs changed")
    require(report["reviewed_pr94_head"]==reviewed["pr94_head"]==
            "b1c558f7d46f5d7c76b4e6336a37143886d16c08",
            "candidate PR #94 reviewed head changed")
    require(report["training_authorized"] is False and report["optimizer_steps"]==0
            and report["model_weights_loaded"] is False
            and report["historical_train002b_modified"] is False
            and report["historical_impact_measured"] is False
            and report["future_serializer_integrated_into_training"] is False,
            "report falsely claims training/history impact/integration")
    for key in ("model_weights_loaded","training_authorized","historical_train002b_modified",
                "historical_train002b_reinterpreted","heldout_evaluation"):
        require(boundary[key] is False,"false boundary: "+key)
    require(boundary["optimizer_steps"]==0
            and boundary["integration_requires_void72_and_explicit_training_authorization"] is True,
            "future integration gate altered")

    names=[c["name"] for c in report["cases"]]
    require(names==result["exact_oracle_cases"]
            and len(names)==7 and all(c["exact_oracle_match"] is True for c in report["cases"])
            and all(c["terminal_id"]==258 for c in report["cases"]),
            "independent oracle cases/result altered")
    require(report["cases"][-1]["name"]=="max_1024"
            and report["cases"][-1]["sequence_units"]==1024
            and report["cases"][-1]["supervised_units"]==1024
            and report["overflow_1025_refused"] is result["overflow_1025_refused"] is True
            and report["malformed_unicode_refused"] is result["malformed_unicode_refused"] is True,
            "EOS boundary/refusal evidence changed")
    hist=report["historical_identity"]
    require(hist["legacy_suffix_ids"]==result["historical_literal_eos_bytes"]==
            [60,101,111,115,62]
            and hist["corrected_suffix_ids"]==[258]
            and result["corrected_terminal_eos_id"]==258
            and hist["units_saved"]==result["units_saved_vs_historical_literal"]==4
            and hist["legacy_had_actual_eos"] is False
            and hist["corrected_has_actual_eos"] is True
            and hist["TRAIN002b_retroactive_modification"] is False,
            "historical EOS identity was rewritten")
    decision=report["decision"]
    require(decision["future_serializer_source_accepted"] is
            result["future_serializer_source_accepted"] is True
            and decision["integration_into_void_training_accepted"] is
            result["future_serializer_integrated_into_training"] is False
            and decision["historical_train002b_reinterpreted"] is False
            and decision["historical_generation_effect_size"]==
            result["historical_generation_effect_size"]=="UNMEASURED"
            and decision["requires_void72_and_explicit_training_authorization"] is True,
            "narrow serializer acceptance scope promoted")


def main():
    files,p=load_originals()
    verify_originals(files,p)
    print(json.dumps({
        "status":"PASS_BYTE_EOS_INDEPENDENT_ARCHIVE__INTEGRATION_BLOCKED",
        "original_review_run":40604675320,
        "original_zip_sha256":ZIP_SHA,
        "original_review_json_sha256":REPORT_SHA,
        "model_weights_loaded":False,
        "training_authorized":False,
        "optimizer_steps":0,
        "historical_effect_size":"UNMEASURED",
        "future_serializer_source_accepted":True,
        "training_integration_accepted":False,
    },sort_keys=True))


if __name__=="__main__":
    main()
