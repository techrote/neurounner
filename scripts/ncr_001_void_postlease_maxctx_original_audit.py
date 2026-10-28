#!/usr/bin/env python3
"""Offline custody audit for NCR-001 void.1 post-lease 2048 evidence.

Standard-library only: no model/tokenizer import, network, optimizer, held-out
fixture, or training path. The source/host ZIPs are lossless base64 encodings
of the original GitHub Actions artifacts from run 40755894558.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/ncr-001/controls-evidence/void-postlease-maxctx-20261028"
RUN = "40755894558"
HEAD = "1438f45efc0dab8ed669a814b4897fc7a839302f"
SERIALIZER = "9fbdee880697903aaf5b91cc29442c9e53479e50"
REV = "9c2be5087772867d45961abd11f6b35110617563"
WEIGHTS = "ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3"
TOKENIZER = "279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03"
FROZEN = "7673957ba36e8eb5427e5ecc94dd52de71fbf9e6718bb0c2a94e4d7361b3f904"
ZIP_SHA = {
    "source": "37babf3139f3d74deac90b64a8927db4bf697de9a70cbd5633798676ef79a427",
    "qwen": "eb209e2be9ab73615db23b60e5d75cfbe0bb14f9aa05ba81373a2d9231bb9026",
}
MEMBERS = {
    "source-tests.log": "be45dd6567ebad0f711329df7e7e9441d272bfa04106bb00f21844db64dc89b7",
    "maxctx.json": "df950b891a8b1a3e49adcd854d33611983224f4ca33180e6892f55ba063f82d3",
    "maxctx.log": "efa1151c25a2c9fbccd6cc569136c85335724f42e951fc2b8d86a06d6fb7eee3",
    "prefetch.log": "7987579a97faef580cc137ff2b1de303a9e44484d87849c6bded8580f62b7bec",
    "runner.txt": "a7082d6b282f4f35e2da366c8c40a3fe952c785b06b420dd9f43282b6a125b29",
}
TOL = 0.0005


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError("void.1 post-lease maxctx evidence REFUSED: " + why)


def decode_original(name: str) -> bytes:
    raw = (EVIDENCE / f"{name}-original.zip.b64").read_text(encoding="ascii")
    compact = "".join(raw.split())
    return base64.b64decode(compact, validate=True)


def inspect_receipt(row: dict) -> None:
    require(row["schema"] == "ncr001-void-postlease-maxctx-v1"
            and row["phase"] == "maxctx" and row["variant"] == "void1",
            "schema/lane/phase")
    require(row["status"] ==
            "PASS_VOID1_POSTLEASE_512_1024_2048_AND_GENERATE__72_BLOCKED",
            "status or final-gate boundary")
    require(row["runner"]["expected_head"] == row["source"]["main_commit"] == HEAD
            and row["runner"]["run_id"] == RUN
            and row["runner"]["run_attempt"] == "1",
            "original Actions/source head")
    require(row["source"]["serializer_commit"] == SERIALIZER
            and row["source"]["corrected_serializer_sha256"] ==
            "a2f575fa536792aaf8cc283a329b23eb8e98c5a07b7a18e4b1c622d5a1435118",
            "corrected EOS serializer identity")
    require(row["source"]["postlease_nca_source_sha256"] ==
            "b85516da0b2020985d03b12a498e51d883433c335858c6d93cce8fda7f63748a"
            and row["source"]["postlease_mlp_source_sha256"] ==
            "f68d72092658319147d571e70dff682bdb501333e94275f3b3099b71c11a92d5",
            "post-lease wrapper identity")
    require(row["source"]["protocol_sha256"] ==
            "a76d38f80a57012ca8e62fba183d1cf2e91fd0986032f1adcdfb0aef843bf0b4"
            and row["source"]["pins_sha256"] ==
            "db794f53a49bb1bcbf8b23bd4a76d24430eb495da246639b506abdbbf11734b1",
            "protocol/source-pin identity")
    require(row["actual_source_files"]["model.safetensors"]["sha256"] == WEIGHTS
            and row["actual_source_files"]["model.safetensors"]["bytes"] == 360622744
            and row["actual_source_files"]["tokenization_byte.py"]["sha256"] == TOKENIZER,
            "actual Qwen3/ByteTokenizer bytes")
    require(row["packages"] == {"torch": "2.14.1+cpu", "transformers": "4.57.6"},
            "runtime package identity")
    require(row["training_authorized"] is False
            and row["optimizer_steps"] == 0
            and row["backward_executed"] is False
            and row["comparative_training_performed"] is False
            and row["heldout_opened"] is False
            and row["full_72_conformance_pass"] is False
            and row["independent_scientific_review"] is False
            and row["historical_byte_eos_effect_measured"] is False,
            "false training/scientific admission")
    require(row["frozen_backbone_sha256_before"] ==
            row["frozen_backbone_sha256_after"] == FROZEN,
            "frozen backbone mutation")
    require(row["parameters"] == {"mlp": 329472, "nca": 329944},
            "matched-control parameter identity")
    source_text = row["source_text"]
    require(source_text["available_tokenizer_units"] == source_text["utf8_bytes"] == 4256
            and source_text["selected_units"] == 2048
            and source_text["characters"] == 3808
            and source_text["ordinary_byte_ids_only"] is True
            and source_text["supervised_training_case"] is False,
            "max-context diagnostic source accounting")

    checks = row["checks"]
    require(len(checks) == 32, "expected 32 physical checks")
    for kind in ("nca", "mlp"):
        for length in (512, 1024, 2048):
            disabled = checks[f"{kind}_raw_disabled_{length}"]
            zero = checks[f"{kind}_raw_zero_{length}"]
            require(disabled["exact"] is True and zero["exact"] is True
                    and disabled["sha256"] == zero["sha256"]
                    and disabled["shape"] == zero["shape"] == [1, length, 259],
                    f"{kind} {length} frozen/zero exact parity")
            effect = checks[f"{kind}_untrained_{length}"]
            require(effect["trained"] is False and effect["max_abs"] > 0,
                    f"{kind} {length} manual untrained effect")
            cache = checks[f"{kind}_maxctx_cache_{length}"]
            require(cache["pass"] is True and 0 <= cache["max_abs"] <= TOL
                    and cache["atol"] == cache["rtol"] == TOL,
                    f"{kind} {length} incremental cache gate")
        suffix = checks[f"{kind}_2048_strict_future_suffix"]
        require(suffix["exact"] is True and suffix["shape"] == [1, 2047, 259],
                f"{kind} strict 2048 future-suffix causality")
        for mode in ("disabled", "zero", "active_untrained"):
            gen = checks[f"{kind}_generate_{mode}_cache_parity"]
            require(gen["pass"] is True and gen["atol"] == gen["rtol"] == TOL
                    and len(gen["steps"]) == 4,
                    f"{kind} {mode} GenerationMixin cache gate")
            for step in gen["steps"]:
                require(step["pass"] is True and step["same_nonfinite_mask"] is True
                        and step["finite_values"] == 258
                        and 0 <= step["max_abs_finite"] <= TOL,
                        f"{kind} {mode} generation score step")

    generations = row["generations"]
    require(len(generations) == 12, "twelve generation records")
    combos = set()
    for g in generations:
        combos.add((g["adapter"], g["mode"], g["use_cache"]))
        require(g["ids"] == [101, 110, 32, 109]
                and g["actual_eos_id_emitted"] is False
                and g["learned_eos_claim"] is False
                and g["trained"] is False
                and len(g["score_sha256"]) == 4,
                "generation result or EOS boundary")
    require(combos == {
        (adapter, mode, cached)
        for adapter in ("nca", "mlp")
        for mode in ("disabled", "zero", "active_untrained")
        for cached in (False, True)
    }, "generation mode coverage")


def audit(source_zip: bytes, qwen_zip: bytes) -> dict:
    require(sha(source_zip) == ZIP_SHA["source"], "original source ZIP bytes")
    require(sha(qwen_zip) == ZIP_SHA["qwen"], "original Qwen ZIP bytes")
    with zipfile.ZipFile(io.BytesIO(source_zip)) as z:
        require(z.namelist() == ["source-tests.log"], "source artifact member set")
        source_log = z.read("source-tests.log")
        require(sha(source_log) == MEMBERS["source-tests.log"]
                and b"Ran 8 tests" in source_log and b"OK" in source_log,
                "eight source/negative tests")
    with zipfile.ZipFile(io.BytesIO(qwen_zip)) as z:
        require(set(z.namelist()) == {"maxctx.json", "maxctx.log", "prefetch.log", "runner.txt"},
                "Qwen artifact member set")
        for name in z.namelist():
            require(sha(z.read(name)) == MEMBERS[name], "artifact member " + name)
        receipt = json.loads(z.read("maxctx.json"))
        require(b'"check_count": 32' in z.read("maxctx.log")
                and b"PASS_VOID1_POSTLEASE_512_1024_2048_AND_GENERATE__72_BLOCKED"
                    in z.read("maxctx.log"),
                "physical summary log")
        require(b"SOURCE_SHA256 model.safetensors " + WEIGHTS.encode()
                    in z.read("prefetch.log")
                and b"SOURCE_SHA256 tokenization_byte.py " + TOKENIZER.encode()
                    in z.read("prefetch.log"),
                "source prefetch hashes")
        require(b"AMD EPYC 7763 64-Core Processor" in z.read("runner.txt"),
                "runner CPU identity")
    inspect_receipt(receipt)
    return receipt


def adversarial(source_zip: bytes, qwen_zip: bytes) -> dict:
    receipt = audit(source_zip, qwen_zip)
    faults = [
        ("training authorization", lambda r: r.__setitem__("training_authorized", True)),
        ("optimizer", lambda r: r.__setitem__("optimizer_steps", 1)),
        ("backward", lambda r: r.__setitem__("backward_executed", True)),
        ("false final gate", lambda r: r.__setitem__("full_72_conformance_pass", True)),
        ("source head", lambda r: r["source"].__setitem__("main_commit", "0" * 40)),
        ("serializer", lambda r: r["source"].__setitem__("serializer_commit", "0" * 40)),
        ("tokenizer", lambda r: r["actual_source_files"]["tokenization_byte.py"].__setitem__("sha256", "0" * 64)),
        ("backbone", lambda r: r.__setitem__("frozen_backbone_sha256_after", "0" * 64)),
        ("cache failure", lambda r: r["checks"]["nca_maxctx_cache_2048"].__setitem__("max_abs", .1)),
        ("causal leak", lambda r: r["checks"]["mlp_2048_strict_future_suffix"].__setitem__("exact", False)),
        ("zero mismatch", lambda r: r["checks"]["nca_raw_zero_2048"].__setitem__("sha256", "0" * 64)),
        ("EOS invented", lambda r: r["generations"][0].__setitem__("actual_eos_id_emitted", True)),
        ("learned EOS claim", lambda r: r["generations"][1].__setitem__("learned_eos_claim", True)),
        ("generation score failure", lambda r: r["checks"]["mlp_generate_active_untrained_cache_parity"]["steps"][2].__setitem__("pass", False)),
        ("context relabel", lambda r: r["source_text"].__setitem__("selected_units", 4096)),
    ]
    for label, mutate in faults:
        row = copy.deepcopy(receipt)
        mutate(row)
        try:
            inspect_receipt(row)
        except ValueError:
            pass
        else:
            raise AssertionError("falsified maxctx evidence accepted: " + label)
    corrupted = bytearray(qwen_zip)
    corrupted[-1] ^= 1
    try:
        audit(source_zip, bytes(corrupted))
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted original Qwen ZIP accepted")
    return {
        "status": "PASS_ORIGINAL_VOID1_POSTLEASE_MAXCTX_EVIDENCE_READBACK__72_BLOCKED",
        "original_run": int(RUN),
        "source_head": HEAD,
        "source_zip_sha256": sha(source_zip),
        "qwen_zip_sha256": sha(qwen_zip),
        "source_tests": 8,
        "physical_checks": 32,
        "negative_mutations_rejected": len(faults) + 1,
        "train_authorized": False,
        "final_72_conformance_pass": False,
    }


def main() -> None:
    source_zip = decode_original("source")
    qwen_zip = decode_original("qwen")
    print(json.dumps(adversarial(source_zip, qwen_zip), sort_keys=True))


if __name__ == "__main__":
    main()
