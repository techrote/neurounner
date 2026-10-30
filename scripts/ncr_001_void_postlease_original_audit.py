#!/usr/bin/env python3
"""NCR-001: offline audit of #167 original void.1 Actions ZIP bytes; no model/training."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

RUN = "40747035876"
HEAD = "9aed5ca0574a6885c5d60c07d16561e0f73d5043"
SERIALIZER = "9fbdee880697903aaf5b91cc29442c9e53479e50"
REV = "9c2be5087772867d45961abd11f6b35110617563"
WEIGHTS = "ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3"
ORIGINAL_ZIPS = {
    "source-original.zip": "f87e93a4c54a0a6fee7eda9c4d7db42a92006803134c554e69e9301a3e26646e",
    "qwen-original.zip": "ca581f11b0f4006170f6565322cff2f7eef72b5e6844e4c0e615961c05773144",
}
MEMBERS = {
    "consumer.json": "06d06d933b8faf7dc949014e92bc3119acb1f53bb1844532bfc91f6ba8d3593e",
    "mlp/adapter.pt": "78d27bb65304295b8913b435af7583af1110a7ce7c17e689784f9e8001eb7756",
    "consumer.log": "76e5fb8b4a9920a013358447e797468f3f9b30c73fe3516bc2bb119c1a071978",
    "mlp/metadata.json": "3a36cd91382cb68cc4b5f2689af8a15a3475caf93da0540c3acfd44e363c93cf",
    "nca/adapter.pt": "20138406697a68abae3e1b37e6510b797c4e82023537c23f1fc55902cf0ebfcb",
    "nca/metadata.json": "06739554a1b9072847ca7eb31cde1dc3fda7dffa678980edb9c318796e7f015c",
    "prefetch.log": "7987579a97faef580cc137ff2b1de303a9e44484d87849c6bded8580f62b7bec",
    "producer.json": "38a4159cb3ef0c5c5c757a4f517b36e2c94b360a53785157e75b352115631c7e",
    "producer.log": "13602b5e9f4fea140a0cb77f49fca7514651a4fed7b7fddf915f9b940154153a",
}
FROZEN = "7673957ba36e8eb5427e5ecc94dd52de71fbf9e6718bb0c2a94e4d7361b3f904"

def sha(data):
    return hashlib.sha256(data).hexdigest()

def require(cond, why):
    if not cond:
        raise ValueError("original void.1 evidence REFUSED: " + why)

def inspect_receipts(prod, cons):
    for phase, row in (("producer", prod), ("consumer", cons)):
        require(row["phase"] == phase and row["model"] == "dotlabs/void.1"
                and row["variant"] == "void1" and row["revision"] == REV,
                "lane/phase identity")
        require(row["runner"]["expected_head"] == row["source"]["main_commit"] == HEAD,
                "original head/source identity")
        require(row["source"]["serializer_commit"] == SERIALIZER, "EOS source revision")
        require(row["runner"]["run_id"] == RUN and row["runner"]["run_attempt"] == "1",
                "original Actions run provenance")
        require(row["training_authorized"] is False
                and row["optimizer_steps"] == 0
                and row["backward_executed"] is False
                and row["comparative_training_performed"] is False
                and row["heldout_opened"] is False
                and row["full_72_conformance_pass"] is False
                and row["independent_scientific_review"] is False,
                "false training/qualification admission")
        require(row["frozen_backbone_sha256_before"] ==
                row["frozen_backbone_sha256_after"] == FROZEN, "backbone mutation")
        require(row["packages"]["transformers"] == "4.57.6"
                and row["actual_source_files"]["model.safetensors"]["sha256"] == WEIGHTS
                and row["actual_source_files"]["model.safetensors"]["bytes"] == 360622744,
                "actual Qwen3 source or runtime mismatch")
    require(prod["status"] ==
            "PASS_ORIGINAL_VOID1_POSTLEASE_EOS_NCA_MLP_FORWARD_ONLY__72_BLOCKED"
            and cons["status"] ==
            "PASS_ORIGINAL_VOID1_POSTLEASE_COLD_RELOAD_ONLY__72_BLOCKED",
            "false final admission")
    require(prod["runner"]["pid"] != cons["runner"]["pid"]
            and prod["source"] == cons["source"]
            and prod["actual_source_files"] == cons["actual_source_files"],
            "cold Python process/source identity")
    expected = (
        ("empty_completion_one_byte_prefix", 1, 1, [258]),
        ("multilingual_utf8", 10, 21, [258]),
        ("literal_eos_bytes", 2, 6, [60,101,111,115,62,258]),
    )
    require(len(prod["cases"]) == len(expected), "three synthetic cases missing")
    checks = prod["checks"]
    for case, (name, prefix, length, tail) in zip(prod["cases"], expected):
        require(case["name"] == name and case["real_eos_id"] == 258
                and case["prompt_bytes"] == prefix
                and case["supervised_after_shift"] == length
                and case["labels"][:prefix] == [-100] * prefix
                and case["labels"][prefix:] == case["ids"][prefix:]
                and case["ids"][-len(tail):] == tail
                and case["labels"][-1] == 258, "shifted byte/EOS supervision")
        require(abs(case["native_frozen_loss"] - case["manual_shifted_ce"]) <= 1e-5,
                "native versus independently shifted CE")
        for kind in ("nca", "mlp"):
            for mode in ("zero", "disabled"):
                record = checks[name + "_" + kind + "_" + mode]
                require(record["exact"] is True and record["sha256"] ==
                        case["logits_sha256"], "zero/disabled reference parity")
    require(checks["cross_adapter_lease"] == "disabled_reference_refused",
            "contaminated frozen-reference call")
    for kind in ("nca", "mlp"):
        effect = checks[kind + "_manual_untrained_effect"]
        require(effect["trained"] is False and effect["max_abs"] > 0,
                "nonzero untrained adapter effect")
        for mode in ("A_B_A_isolation", "causal_future_suffix"):
            require(checks[kind + "_" + mode]["exact"] is True,
                    "request isolation or future-token leak")
        for n in (32, 128):
            cache = checks[kind + "_cache_" + str(n)]
            require(0 <= cache["max_abs"] <= 0.0005
                    and cache["tolerance_atol"] == cache["tolerance_rtol"] == 0.0005,
                    "cache numerical gate drift")
        require(cons["checks"][kind + "_separate_process_cold_reload"]["exact"] is True
                and cons["checks"][kind + "_separate_process_cold_reload"]["logits_sha256"]
                   == checks[kind + "_cold_logits_sha256"], "cold adapter replay mismatch")
        require(checks[kind + "_state_sha256"] == MEMBERS[kind + "/adapter.pt"]
                and checks[kind + "_metadata_sha256"] == MEMBERS[kind + "/metadata.json"],
                "original adapter/metadata blob hash drift")

def audit(src_bytes, host_bytes):
    require(sha(src_bytes) == ORIGINAL_ZIPS["source-original.zip"], "original source ZIP drift")
    require(sha(host_bytes) == ORIGINAL_ZIPS["qwen-original.zip"], "original host ZIP drift")
    with zipfile.ZipFile(io.BytesIO(src_bytes)) as z:
        require(z.namelist() == ["source-tests.log"], "source member set")
        require(sha(z.read("source-tests.log")) ==
                "e7eac731a92f292e183141f374bc4fdd1b7a455f38f8bb1d65d782741a3b9436",
                "original source test log bytes")
        require(b"Ran 9 tests" in z.read("source-tests.log")
                and b"OK" in z.read("source-tests.log"), "source-negative tests")
    with zipfile.ZipFile(io.BytesIO(host_bytes)) as z:
        require(set(z.namelist()) == set(MEMBERS), "original host artifact members")
        for name, digest in MEMBERS.items():
            require(sha(z.read(name)) == digest, "original artifact member " + name)
        prod, cons = json.loads(z.read("producer.json")), json.loads(z.read("consumer.json"))
    inspect_receipts(prod, cons)
    return prod, cons

def adversarial(src, host):
    prod, cons = audit(src, host)
    faults = [
        ("authorization", lambda a,b: a.__setitem__("training_authorized", True)),
        ("optimizer", lambda a,b: a.__setitem__("optimizer_steps", 1)),
        ("source head", lambda a,b: a["source"].__setitem__("main_commit", "0"*40)),
        ("serializer", lambda a,b: a["source"].__setitem__("serializer_commit", "0"*40)),
        ("EOS", lambda a,b: a["cases"][0].__setitem__("real_eos_id", 62)),
        ("prompt leak", lambda a,b: a["cases"][1]["labels"].__setitem__(0, 42)),
        ("cache tolerance", lambda a,b: a["checks"]["nca_cache_128"].__setitem__("max_abs", .1)),
        ("false gate", lambda a,b: a.__setitem__("full_72_conformance_pass", True)),
        ("cold mismatch", lambda a,b: b["checks"]["mlp_separate_process_cold_reload"].__setitem__("exact", False)),
        ("backbone", lambda a,b: b.__setitem__("frozen_backbone_sha256_after", "0"*64)),
    ]
    for label, mutate in faults:
        a, b = copy.deepcopy(prod), copy.deepcopy(cons)
        mutate(a, b)
        try:
            inspect_receipts(a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("falsified original evidence accepted: " + label)
    try:
        audit(src, host[:-1] + bytes([host[-1] ^ 1]))
    except ValueError:
        pass
    else:
        raise AssertionError("mutated original artifact ZIP accepted")
    print(json.dumps({"status": "ORIGINAL_SOURCE_BOUND_EVIDENCE_ONLY__72_BLOCKED",
                     "original_run": RUN, "source_head": HEAD,
                     "source_zip_sha256": sha(src), "host_zip_sha256": sha(host),
                     "artifact_members": 10, "negative_tests": len(faults)+1,
                     "train_authorized": False}, sort_keys=True))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: ... source-original.zip qwen-original.zip")
    adversarial(Path(sys.argv[1]).read_bytes(), Path(sys.argv[2]).read_bytes())
