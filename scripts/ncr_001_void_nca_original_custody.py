#!/usr/bin/env python3
"""Independent stdlib-only original Actions ZIP/member and diagnostic readback.

No model import, tokenizer, optimizer, network, heldouts or comparative training.
Exact original raw artifact ZIP bytes, not regenerated JSON summaries, are the
authority. Separate CI tests fresh API downloads against the Git custody files.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/ncr-001/controls-evidence/void-nca-learning-20261028"
MANIFEST = EVIDENCE / "manifest.json"
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"
PINS = ROOT / "experiments/ncr-001/source-pins.json"
RUN = "40765025332"
HEAD = "2810e800ef890f0c99f9ea44adca29f879b5d0ea"
FROZEN = "7673957ba36e8eb5427e5ecc94dd52de71fbf9e6718bb0c2a94e4d7361b3f904"
WEIGHT = "ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3"
TOKENIZER = "279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03"
SHAS = {
    "source": "b153d177497f15227545bfe6b717a6430219a662f2b0b868e7a68b89682a90d0",
    "host": "8e9152042fb909f18b13589e6e21552c40f3104c4cc292cf9a643c600fbf08b5",
}
GRADS = ("readout.weight", "observe.weight", "perception.weight",
         "rule.weight", "input_norm.weight")


def refuse(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError("original void1 NCA diagnostic evidence REFUSED: " + why)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def original_zip(kind: str) -> bytes:
    text = (EVIDENCE / (kind + "-original.zip.b64")).read_text("ascii")
    return base64.b64decode("".join(text.split()), validate=True)


def inspect_receipt(row: dict) -> None:
    refuse(row["experiment"] == "NCR-001E" and row["lane"] == "void1"
           and row["source_parent"] ==
           "1738ac2b6bc6105d57d6a5b4c35f4beaad94ff86"
           and row["status"] ==
           "PASS_VOID1_ONE_STEP_NCA_DIAGNOSTIC__72_SCIENTIFIC_ADMISSION_BLOCKED",
           "experiment/lane/final admission boundary")
    refuse(row["runner"]["expected_head"] == row["source"]["main_commit"] == HEAD
           and row["runner"]["run_id"] == RUN
           and row["runner"]["run_attempt"] == "1"
           and row["runner"]["pid"] == 2554
           and row["runner"]["torch"] == "2.14.1+cpu"
           and row["runner"]["python"] == "3.12.3"
           and row["runner"]["threads"] == 2
           and row["packages"] == {
               "torch": "2.14.1+cpu", "transformers": "4.57.6"},
           "actual frozen host/run identity")
    refuse(row["train_authorized"] is False
           and row["full_72_conformance_pass"] is False
           and row["independent_scientific_review"] is False
           and row["comparative_training_performed"] is False
           and row["model_training_experiment"] is False
           and row["heldout_opened"] is False
           and row["diagnostic_optimizer_steps"] == 1,
           "false learning/admission claim")
    refuse(row["actual_source_files"]["model.safetensors"]["sha256"] == WEIGHT
           and row["actual_source_files"]["model.safetensors"]["bytes"] == 360622744
           and row["actual_source_files"]["tokenization_byte.py"]["sha256"] == TOKENIZER
           and row["source"]["serializer_commit"] ==
           "9fbdee880697903aaf5b91cc29442c9e53479e50"
           and row["source"]["corrected_serializer_sha256"] ==
           "a2f575fa536792aaf8cc283a329b23eb8e98c5a07b7a18e4b1c622d5a1435118"
           and row["source"]["postlease_nca_source_sha256"] ==
           "b85516da0b2020985d03b12a498e51d883433c335858c6d93cce8fda7f63748a"
           and row["source"]["postlease_mlp_source_sha256"] ==
           "f68d72092658319147d571e70dff682bdb501333e94275f3b3099b71c11a92d5"
           and row["source_sha256"]["learning_runner"] ==
           "189a354c6057b15d3052df2ff9bd200878b95d872923d52ac090e1bee146b694",
           "actual pinned weight/tokenizer/serializer/hook/code source")
    refuse(row["source"]["protocol_sha256"] ==
           "a76d38f80a57012ca8e62fba183d1cf2e91fd0986032f1adcdfb0aef843bf0b4"
           and row["source"]["pins_sha256"] ==
           "db794f53a49bb1bcbf8b23bd4a76d24430eb495da246639b506abdbbf11734b1",
           "preregistered source changed")
    sample = row["synthetic_sample"]
    refuse(sample["sample_kind"] == "synthetic_public_diagnostic_only"
           and sample["heldout"] is False
           and sample["prompt_bytes"] == 10
           and sample["total_units"] == 31
           and sample["source_text_utf8_bytes"] == 30
           and sample["supervised_after_shift"] == 21
           and sample["real_eos_id"] == sample["supervised_eos_id"] == 258
           and sample["ids_sha256"] ==
           "ee6fa2a893301b858b7fdac7fe2b9190cd1fc786ddff1ffdf906da757488c09e"
           and sample["labels_sha256"] ==
           "260dec3e7622aeada51e0483e005ad9d59cbae8f186d41432c381f2c35835241",
           "synthetic byte/EOS/assistant-only mask source")
    refuse(row["initial_zero_readout_parity_exact"] is True
           and row["initial_source_logits_sha256"] ==
           "3855c1e6fcbefe0642932a72c293f23b7f22790358ee8cff6c35dd31cc820a52"
           and row["backbone_before_sha256"] ==
           row["backbone_after_sha256"] == FROZEN,
           "frozen base parity or post-step mutation")
    refuse(set(row["checks"]) == {
        "backbone_unchanged_after_one_step", "disabled_source_logits_unchanged",
        "initial_zero_adapter_parity", "native_shifted_byte_eos_loss",
        "no_cross_model_comparison", "post_step_upstream_gradients",
        "readout_gradient_first_backward", "three_distinct_depth_states",
    } and all(v is True for v in row["checks"].values()),
           "missing post-warmup/frozen-source diagnostic checks")
    warm = row["warmup"]
    post = row["post_warmup"]
    refuse(warm["optimizer"] == "NCA-only SGD"
           and warm["learning_rate"] == 0.005
           and warm["readout_before_sha256"] != warm["readout_after_sha256"]
           and math.isfinite(warm["native_causal_eos_loss"])
           and warm["native_causal_eos_loss"] > 0
           and math.isfinite(post["native_causal_eos_loss"])
           and post["native_causal_eos_loss"] > 0,
           "single-step update or native supervised losses")
    for name, value in warm["gradient_l1"].items():
        refuse(math.isfinite(value) and
               (value > 0 if name == "readout.weight" else value == 0),
               "zero-readout first-pass gradient routing " + name)
    refuse(set(GRADS).issubset(post["gradient_l1"]), "post-warmup gradients omitted")
    for name, value in post["gradient_l1"].items():
        refuse(math.isfinite(value) and value > 0,
               "nonfinite/zero post-warmup gradient " + name)
    fields = post["ordered_field_trace"]
    refuse(len(fields) == 3
           and [f["tap_index"] for f in fields] == [0, 1, 2]
           and [f["next_tap"] for f in fields] == [1, 2, 3]
           and all(f["shape"] == [1, 31, 4, 8, 8] for f in fields)
           and len({f["field_sha256"] for f in fields}) == 3,
           "not three distinct ordered native depth fields")


def audit() -> dict:
    protocol = json.loads(PROTOCOL.read_text("utf-8"))
    pins = json.loads(PINS.read_text("utf-8"))
    manifest = json.loads(MANIFEST.read_text("utf-8"))
    refuse(protocol["train_authorized"] is False
           and set(protocol["models"]) == set(pins["models"]) ==
           {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"},
           "four-lane protocol/training gate changed")
    refuse(sha(PROTOCOL.read_bytes()) == manifest["source"]["protocol_sha256"]
           and sha(PINS.read_bytes()) == manifest["source"]["source_pins_sha256"],
           "protocol/pins physical SHA mismatch")
    refuse(manifest["original_run"] == int(RUN)
           and manifest["original_head"] == HEAD
           and manifest["original_jobs"] == {
               "source": 123086735654, "host": 123087080911}
           and manifest["original_artifacts"] == {
               "source": 12850046593, "host": 12850675205}
           and manifest["boundary"]["train_authorized"] is False
           and manifest["boundary"]["full_72_conformance_pass"] is False
           and manifest["boundary"]["independent_scientific_review"] is False
           and manifest["boundary"]["comparative_training_performed"] is False,
           "original Actions identity or admission state")
    original = {}
    for kind in ("source", "host"):
        archive = original_zip(kind)
        refuse(sha(archive) == SHAS[kind]
               == manifest["original_artifact_zip_sha256"][kind]
               and len(archive) == manifest["original_artifact_zip_bytes"][kind],
               kind + " original ZIP bytes changed")
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            expected = manifest["artifact_members"][kind]
            refuse(set(z.namelist()) == set(expected), "original member set " + kind)
            for name, desc in expected.items():
                raw = z.read(name)
                refuse(len(raw) == desc["bytes"] and sha(raw) == desc["sha256"],
                       kind + " member drift " + name)
            original[kind] = {
                name: z.read(name) for name in expected}
    tests = original["source"]["tests.log"]
    refuse(b"Ran 11 tests" in tests and b"OK" in tests
           and b"test_text_eos_bytes_rejected" in tests,
           "original source adversarial tests did not pass")
    receipt = json.loads(original["host"]["ncr001-void-nca-learning/receipt.json"])
    inspect_receipt(receipt)
    refuse(b"PASS_VOID1_ONE_STEP_NCA_DIAGNOSTIC__72_SCIENTIFIC_ADMISSION_BLOCKED"
           in original["host"]["ncr001-void-nca-learning/host.log"]
           and b"torch_dtype" in original["host"]["ncr001-void-nca-learning/host.log"],
           "original host transcript / deprecation warning lost")
    frozen = manifest["physical_result"]
    refuse(frozen["status"] == receipt["status"]
           and frozen["source_backbone_sha256"] == FROZEN
           and frozen["diagnostic_optimizer_steps"] == 1
           and frozen["initial_native_loss"] == receipt["warmup"]["native_causal_eos_loss"]
           and frozen["post_warmup_native_loss"] == receipt["post_warmup"]["native_causal_eos_loss"],
           "manifest numerical promotion or omission")
    for kind in ("source", "host"):
        log = (EVIDENCE / (kind + "-original-job.log")).read_text("utf-8-sig")
        refuse(HEAD in log and f"/runs/{RUN}" not in log,
               "decoded log/head mismatch " + kind)
    host_log = (EVIDENCE / "host-original-job.log").read_text("utf-8-sig")
    refuse(WEIGHT in host_log and TOKENIZER in host_log
           and receipt["status"] in host_log
           and "Artifact ID 12850675205" in host_log,
           "original host job log/source provenance missing")
    return receipt


def adversarial() -> dict:
    receipt = audit()
    mutations = [
        ("false gate", lambda r: r.__setitem__("full_72_conformance_pass", True)),
        ("trained", lambda r: r.__setitem__("model_training_experiment", True)),
        ("optimizer count", lambda r: r.__setitem__("diagnostic_optimizer_steps", 2)),
        ("heldout", lambda r: r.__setitem__("heldout_opened", True)),
        ("wrong head", lambda r: r["runner"].__setitem__("expected_head", "0" * 40)),
        ("wrong weight", lambda r: r["actual_source_files"]["model.safetensors"].__setitem__("sha256", "0" * 64)),
        ("serializer", lambda r: r["source"].__setitem__("serializer_commit", "0" * 40)),
        ("literal EOS", lambda r: r["synthetic_sample"].__setitem__("real_eos_id", 62)),
        ("backbone drift", lambda r: r.__setitem__("backbone_after_sha256", "0" * 64)),
        ("first upstream leaked", lambda r: r["warmup"]["gradient_l1"].__setitem__("rule.weight", 1.0)),
        ("post upstream missing", lambda r: r["post_warmup"]["gradient_l1"].__setitem__("observe.weight", 0.0)),
        ("nan loss", lambda r: r["post_warmup"].__setitem__("native_causal_eos_loss", float("nan"))),
        ("tap order", lambda r: r["post_warmup"]["ordered_field_trace"][1].__setitem__("tap_index", 0)),
        ("false parity", lambda r: r["checks"].__setitem__("initial_zero_adapter_parity", False)),
    ]
    for title, mutate in mutations:
        bad = copy.deepcopy(receipt)
        mutate(bad)
        try:
            inspect_receipt(bad)
        except ValueError:
            continue
        raise AssertionError("falsified original Qwen3 diagnostic accepted: " + title)
    return {
        "status": "PASS_ORIGINAL_VOID1_NCA_ONLY_DIAGNOSTIC_CUSTODY__72_REVIEW_BLOCKED",
        "original_run": int(RUN), "source_head": HEAD,
        "source_zip_sha256": SHAS["source"],
        "host_zip_sha256": SHAS["host"],
        "source_tests": 11, "negative_mutations_refused": len(mutations),
        "diagnostic_optimizer_steps": 1,
        "comparative_training_performed": False,
        "train_authorized": False,
        "full_72_conformance_pass": False,
        "independent_scientific_review": False,
    }


if __name__ == "__main__":
    print(json.dumps(adversarial(), sort_keys=True))
