#!/usr/bin/env python3
"""NCR-001: independent offline readback of original source-matched Sprocket mask receipts.

This audits physical original GitHub Actions evidence, not a current-host rerun.
No torch, model, tokenizer, optimizer, network, or held-out fixture is used.
"""
from __future__ import annotations
import base64
import copy
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "experiments/ncr-001/controls-evidence/sprocket-explicit-mask-20261028"
HEAD = "8275063544b90c83d2740f0d356288460b58ef6b"
RUN = 40698955980
BUNDLE_SHA = "719c95e11020c9c904f203ccad1d07494af2fa252c0b6f4abd1ab9fadcd84247"
ZIPS = {
    "chat": "3065c5d4d86b70bb2d5bf658f8db1506a46ed3fca422e18cf96c179877070d63",
    "instruct": "c400f4bdcd2a679f6b8472a3d7bb349a78583d6073f4b433b5658e4fc684d0b0",
    "source": "d750b5a95e3c9a7d2d187583ad216a739dcc57cef4dfa9cdf508091faaadfe9e",
}
WEIGHTS = {
    "chat": ("sprocket_chat", "chat/model.safetensors",
             "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97"),
    "instruct": ("sprocket_instruct", "model.safetensors",
                 "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864"),
}
SOURCE_SHA = {
    "experiments/ncr-001/protocol.json": "a76d38f80a57012ca8e62fba183d1cf2e91fd0986032f1adcdfb0aef843bf0b4",
    "experiments/ncr-001/source-pins.json": "db794f53a49bb1bcbf8b23bd4a76d24430eb495da246639b506abdbbf11734b1",
    "scripts/ncr_001_backbone_lease.py": "e9e43b9b80962298081d327a37290677fb7d4bd877df4dc6d513b4fefb738468",
    "scripts/ncr_001_postlease_sprocket.py": "f25b5a0aae02206bd0f7523483cdf9000ad672e74f56bf3ea2d9390b5023a897",
    "scripts/ncr_001_sprocket_explicit_mask.py": "e3f6f3bd2a9e405d27cf5285963c449f10f6676a6eb388dfba856078ac31df17",
}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def ok(cond, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def load_original() -> tuple[dict, dict[str, bytes]]:
    m = json.loads((E / "manifest.json").read_text(encoding="utf-8"))
    encoded = (E / "original-receipts.tar.gz.b64").read_text(encoding="ascii").strip()
    b = base64.b64decode(encoded, validate=True)
    ok(len(b) == m["receipt_bundle_bytes"] and
       sha(b) == BUNDLE_SHA == m["receipt_bundle_sha256"], "Gzip original archive SHA/size")
    raw = gzip.decompress(b)
    ok(sha(raw) == m["uncompressed_tar_sha256"] and
       len(raw) == m["uncompressed_tar_bytes"], "Original TAR SHA/size")
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as t:
        for item in t:
            ok(item.isfile() and item.name and
               item.name.split("/")[0] in ("chat", "instruct", "source") and
               all(component not in ("..", ".", "") for component in item.name.split("/")) and
               item.name not in files, "Unsafe, duplicate or unexpected TAR member")
            files[item.name] = t.extractfile(item).read()
    verify_original_files(m, files)
    return m, files


def verify_original_files(m: dict, files: dict[str, bytes]) -> None:
    ok(set(files) == set(m["original_files"]) ==
       {"chat/host.log", "chat/receipt.json", "instruct/host.log",
        "instruct/receipt.json", "source/shared-lease-tests.log",
        "source/source-tests.log"}, "Original ZIP member set changed")
    for name, b in files.items():
        detail = m["original_files"][name]
        ok(sha(b) == detail["sha256"] and len(b) == detail["size"],
           "Corrupted original member: " + name)


def audit(m: dict, files: dict[str, bytes]) -> dict:
    ok(m["original_source_head"] == HEAD and m["original_run_id"] == RUN and
       m["original_run_attempt"] == 1 and
       m["original_run_conclusion"] == "success", "Wrong source run")
    ok(m["qualification"] ==
       "SOURCE_MATCHED_EXPLICIT_MASK_GENERATION_PREPASS_PASS__FINAL_72_BLOCKED",
       "Wrong experimental scope")
    ok(set(m["variants"]) ==
       {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"}, "Collapsed model lanes")
    for flag in ("training_authorized", "heldout_access",
                 "independent_scientific_review", "tool_memory_semantics_qualified",
                 "final_72_conformance_pass", "learning_update_loss_gates_qualified"):
        ok(m[flag] is False, "False admission: " + flag)
    ok(m["optimizer_steps"] == 0 and m["positive_source_tests"] == 6 and
       m["positive_shared_lease_tests"] == 10 and
       m["positive_actual_host_checks_per_lane"] == 21, "Test/optimizer miscount")
    ok(m["context_mask_shape"] == [2, 26] and
       m["left_synthetic_pad_units"] == 19 and
       m["generated_units"] == 3, "Mask context/report shape")
    for k, expected in ZIPS.items():
        ok(m["original_actions_zip_artifacts"][k]["sha256"] == expected,
           "Wrong original ZIP SHA: " + k)
    ok([x["run_id"] for x in m["negative_previous_workflow_runs"]] ==
       [40698646280, 40698793257] and
       all(x["physical_jobs_ran"] is False for x in
           m["negative_previous_workflow_runs"]), "Precorrected syntax failures erased")
    ok(b"Ran 6 tests" in files["source/source-tests.log"] and
       b"\nOK\n" in files["source/source-tests.log"] and
       b"Ran 10 tests" in files["source/shared-lease-tests.log"] and
       b"\nOK\n" in files["source/shared-lease-tests.log"], "Original source tests missing")
    result = {}
    for variant, (lane, path, weight_sha) in WEIGHTS.items():
        r = json.loads(files[variant + "/receipt.json"])
        pin = m["model_pins"][variant]
        ok(r["status"] == "PASS_EXPLICIT_SOURCE_MASK_GENERATION_WITNESS" and
           r["original_frozen_model_executed"] is True and r["lane"] == lane,
           "Original physical host status/lane")
        for flag in ("train_authorized", "heldout_opened", "training_performed",
                     "comparative_training_performed", "tool_memory_semantics_qualified",
                     "full_72_conformance_pass"):
            ok(r[flag] is False, "False host-science claim: " + flag)
        ok(r["optimizer_steps"] == 0, "Forbidden optimizer steps")
        si = r["source_identity"]
        ok(si["weight_sha256"] == weight_sha and si["weight_path"] == path and
           si["model_id"] == "kandivault/sprocket-500m" and
           si["revision"] == m["model_revision"] ==
           "eb157d4ddddc4619e661d4a2426bb2dd000fc668" and
           si["taps"] == m["model_taps"] == [7, 16, 25] and
           si["width"] == 1280 and si["layers"] == 26 and
           si["model_type"] == "llama", "Original checkpoint source substitution")
        actual_weight = r["source_files"][path]
        ok(actual_weight["sha256"] == weight_sha and
           actual_weight["bytes"] == 1002207984 and
           r["source_files"]["README.md"]["sha256"] ==
           "09aec879dc5b387bdc031dc3c74d77a9ba44362a55163f9a8b456761b68142c9",
           "Original complete downloaded weights/source identity")
        ok(r["source_sha256"] == SOURCE_SHA, "Script/protocol/source-pin drift")
        for name, key in (("protocol", "experiments/ncr-001/protocol.json"),
                          ("source_pins", "experiments/ncr-001/source-pins.json"),
                          ("lease", "scripts/ncr_001_backbone_lease.py"),
                          ("previous_host", "scripts/ncr_001_postlease_sprocket.py"),
                          ("mask_host", "scripts/ncr_001_sprocket_explicit_mask.py")):
            ok(m["source_hashes"][name] == SOURCE_SHA[key],
               "Manifest source hash mismatch " + name)
        ri = r["runner"]
        ok(ri["run_id"] == str(RUN) and ri["head_sha"] == HEAD and
           ri["attempt"] == "1" and ri["threads"] == 4 and
           ri["torch"] == "2.14.1+cpu" and ri["transformers"] == "4.57.6" and
           ri["runner_os"] == "Linux" and
           ri["runner_arch"] == "X64", "Original source-host/runner mismatch")
        ok(ri["host"] == pin["runner_host"] and
           ri["pid"] == pin["runner_pid"] and
           ri["runner_name"] == pin["runner_name"], "Runner identity changed")
        ok(r["in_memory_frozen_before_sha256"] ==
           r["in_memory_frozen_after_sha256"] == pin["backbone_sha256"],
           "Original model backbone mutated")
        ok(r["peak_rss_kib"] == pin["rss_kib"], "CPU peak RSS evidence changed")
        checks = r["checks"]
        ok(len(checks) == 21 and m["positive_actual_host_checks_per_lane"] == 21,
           "Missing source-matched host checks")
        masks = checks["tokenizer_source_mask"]
        ok(masks["pass"] is True and masks["batch_shape"] == [2, 26] and
           masks["naive_id_mask_refused"] is True and
           masks["pad_equals_end_marker"] is True and
           len(masks["per_row"]) == 2 and
           masks["per_row"][0]["synthetic_pad"] == 19 and
           masks["per_row"][1]["synthetic_pad"] == 0 and
           all(x["real_end_marker_id_5_visible"] is True for x in
               masks["per_row"]), "Genuine role marker masked as padding")
        assert set(checks) == {"tokenizer_source_mask"} | {
            adapter + "_" + mode + "_" + part
            for adapter in ("nca", "mlp")
            for mode in ("disabled", "enabled_zero", "enabled_manual")
            for part in ("batch_vs_singles_cached", "batch_vs_singles_uncached",
                         "cache_on_off")
        } | {adapter + "_zero_readout_vs_disabled" for adapter in ("nca", "mlp")}
        for adapter in ("nca", "mlp"):
            for mode in ("disabled", "enabled_zero", "enabled_manual"):
                pairs = []
                for suffix in ("uncached", "cached"):
                    key = adapter + "_" + mode + "_batch_vs_singles_" + suffix
                    item = checks[key]
                    ok(item["pass"] is True and item["explicit_mask"] is True and
                       item["new_tokens"] == item["single_tokens"] and
                       len(item["new_tokens"]) == 2 and
                       all(len(seq) == 3 for seq in item["new_tokens"]),
                       "Mismatch masked batched and source single generation: " + key)
                    pairs.append(item["new_tokens"])
                cache = checks[adapter + "_" + mode + "_cache_on_off"]
                ok(cache["pass"] is True and cache["exact_tokens"] is True and
                   pairs[0] == pairs[1], "Cached/uncached generation mismatch")
            zero = checks[adapter + "_zero_readout_vs_disabled"]
            ok(zero["pass"] is True and zero["exact_tokens"] is True and
               checks[adapter + "_disabled_batch_vs_singles_cached"]["new_tokens"] ==
               checks[adapter + "_enabled_zero_batch_vs_singles_cached"]["new_tokens"],
               "Frozen baseline and zero-readout disagree")
        ok(len(files[variant + "/host.log"]) == m["original_files"]
           [variant + "/host.log"]["size"] and
           b"PASS_EXPLICIT_SOURCE_MASK_GENERATION_WITNESS" in
           files[variant + "/host.log"], "Original host execution log failed")
        result[lane] = {
            "checks": len(checks),
            "masked_batch_shape": masks["batch_shape"],
            "left_pad_units": masks["per_row"][0]["synthetic_pad"],
            "all_generation_checks_pass": True,
            "original_frozen_model_executed": True,
            "greedy_new_tokens": checks["nca_enabled_manual_batch_vs_singles_cached"]["new_tokens"],
            "peak_rss_kib": r["peak_rss_kib"],
        }
    ok(result["sprocket_chat"]["greedy_new_tokens"] !=
       result["sprocket_instruct"]["greedy_new_tokens"],
       "Distinct frozen variants accidentally collapsed")
    return result


def negative(m: dict, files: dict[str, bytes]) -> int:
    mutations = [
        ("chat", lambda r: r.update(train_authorized=True)),
        ("instruct", lambda r: r["source_identity"].update(weight_path="chat/model.safetensors")),
        ("chat", lambda r: r["checks"]["tokenizer_source_mask"].update(naive_id_mask_refused=False)),
        ("instruct", lambda r: r["checks"]["mlp_enabled_manual_cache_on_off"].update({"pass": False})),
        ("chat", lambda r: r["checks"]["nca_enabled_manual_batch_vs_singles_cached"]["new_tokens"][0].__setitem__(0,999)),
        ("instruct", lambda r: r["checks"]["nca_enabled_zero_batch_vs_singles_uncached"].update(explicit_mask=False)),
        ("chat", lambda r: r["checks"]["nca_enabled_zero_batch_vs_singles_cached"]["new_tokens"].__setitem__(0,[9,9,9])),
        ("chat", lambda r: r.update(full_72_conformance_pass=True)),
        ("instruct", lambda r: r.update(tool_memory_semantics_qualified=True)),
        ("chat", lambda r: r.update(in_memory_frozen_after_sha256="0"*64)),
    ]
    for variant, transform in mutations:
        copied = dict(files)
        target = variant + "/receipt.json"
        original = json.loads(files[target])
        transform(original)
        copied[target] = json.dumps(original,sort_keys=True).encode()
        try:
            audit(m,copied)
        except AssertionError:
            continue
        raise AssertionError("Semantic negative sentinel not detected: " + target)
    n = copy.deepcopy(m)
    n["original_actions_zip_artifacts"]["chat"]["sha256"] = "0"*64
    try:
        audit(n, files)
    except AssertionError:
        pass
    else:
        raise AssertionError("Artifact ZIP tampering not detected")
    n = copy.deepcopy(m)
    n["independent_scientific_review"] = True
    try:
        audit(n, files)
    except AssertionError:
        pass
    else:
        raise AssertionError("False scientific acceptance not detected")
    n = copy.deepcopy(m)
    n["original_files"]["chat/receipt.json"]["sha256"] = "0"*64
    try:
        verify_original_files(n, files)
    except AssertionError:
        pass
    else:
        raise AssertionError("Raw receipt SHA tampering not detected")
    return len(mutations) + 3


def main() -> None:
    m, files = load_original()
    result = audit(m, files)
    print(json.dumps({
        "classification": m["qualification"],
        "original_run_id": RUN,
        "source_head": HEAD,
        "original_members": len(files),
        "original_archive_sha256": BUNDLE_SHA,
        "four_model_lanes_preserved": True,
        "source_negative_tests": m["positive_source_tests"],
        "cross_decoder_lease_tests": m["positive_shared_lease_tests"],
        "original_hosts": result,
        "adversarial_receipt_mutations_caught": negative(m, files),
        "independent_scientific_review": False,
        "train_authorized": False,
        "optimizer_steps": 0,
        "final_72_conformance_pass": False,
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
