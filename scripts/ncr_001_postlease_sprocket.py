#!/usr/bin/env python3
"""NCR-001: source-matched post-lease Sprocket NCA/MLP original-host probe.

Only pinned ordinary user/assistant roles and raw control-ID transport.
No optimizer, backward, train data, held-out evaluation or training admission.
Producer and consumer execute in separate Python interpreters on each lane.
Always persist partial results, including refusals and exceptions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import traceback

import torch
import transformers
import huggingface_hub
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from ncr_001_sprocket_chat import (
    FrozenSprocketChatDepthNCA, PINNED_CHAT_SAFETENSORS_SHA256,
)
from ncr_001_sprocket_instruct import (
    FrozenSprocketInstructDepthNCA, PINNED_ROOT_SAFETENSORS_SHA256,
)
from ncr_001f_frozen_hooks import FrozenDepthGatedMLP, PINNED

ROOT = Path(__file__).resolve().parents[1]
FORMAT_SOURCE = ROOT / "format-authority"
FORMAT_COMMIT = "2271218d625e1462b2cf218445da975efe17fe32"
CARD_SHA256 = "09aec879dc5b387bdc031dc3c74d77a9ba44362a55163f9a8b456761b68142c9"
MODEL_REVISION = "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
LANE_WEIGHTS = {
    "chat": ("chat/model.safetensors", PINNED_CHAT_SAFETENSORS_SHA256),
    "instruct": ("model.safetensors", PINNED_ROOT_SAFETENSORS_SHA256),
}
SOURCE_FILES = {
    "chat": ("chat/config.json", "chat/tokenizer.json", "chat/tokenizer_config.json"),
    "instruct": ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json"),
}
CONTROL_IDS = {"<|tool_call|>": 6, "<|tool_result|>": 7,
               "<|memory_read|>": 10, "<|memory_write|>": 11}
REQUIRED_LANES = {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"}
REPORT: dict = {}


def require(condition, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_contract(variant: str, protocol: dict, pins: dict) -> dict:
    """Refuse wrong variant or an altered source/training protocol before load."""
    require(variant in LANE_WEIGHTS, "Unknown Sprocket variant")
    require(protocol.get("train_authorized") is False, "Training gate opened")
    require(set(protocol.get("models", {})) == REQUIRED_LANES,
            "Four frozen-model identities altered")
    require(set(pins.get("models", {})) == REQUIRED_LANES, "Source pins altered")
    lane = "sprocket_" + variant
    p, m, spec = pins["models"][lane], protocol["models"][lane], PINNED[lane]
    weight, verified_sha = LANE_WEIGHTS[variant]
    sibling = pins["models"]["sprocket_instruct" if variant == "chat"
                             else "sprocket_chat"]
    require(all((
        spec["model_id"] == p["repo_id"] == m["repo_id"] == "kandivault/sprocket-500m",
        spec["revision"] == p["revision"] == m["revision"] == MODEL_REVISION,
        p["variant"] == m["variant"] == variant,
        p["hf_weight_file"] == m["weight_file"] == spec["weight_path"] == weight,
        p["weight_declared_lfs_sha256_not_downloaded"]
        == spec["weight_sha256"] == verified_sha,
        sibling["weight_declared_lfs_sha256_not_downloaded"] != verified_sha,
        p["weight_bytes"] == 1002207984,
        spec["model_type"] == "llama",
        spec["width"] == m["expected_config"]["hidden_size"] == 1280,
        spec["layers"] == m["expected_config"]["num_hidden_layers"] == 26,
        tuple(spec["taps"]) == tuple(m["taps_after_block_zero_based"])
        == tuple(p["tap_indices_post_block_zero_based"]) == (7, 16, 25),
    )), "Pinned Chat/Instruct checkpoint, geometry or source contract changed")
    return spec


def archive_path(variant: str, phase: str) -> Path:
    return ROOT / "runtime/ncr001-postlease-sprocket" / variant / (phase + ".json")


def save(variant: str, phase: str) -> None:
    path = archive_path(variant, phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    REPORT["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    path.write_text(json.dumps(REPORT, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def record(variant: str, phase: str, key: str, value) -> None:
    REPORT["checks"][key] = value
    save(variant, phase)


def check_logits(variant: str, phase: str, key: str, left, right,
                 *, atol: float, rtol: float, exact: bool = False) -> None:
    require(left.shape == right.shape, key + ": output shape disagreement")
    delta = (left - right).abs()
    actual = float(delta.max())
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    good = bool(torch.allclose(left, right, atol=atol, rtol=rtol))
    record(variant, phase, key, {
        "max_abs": actual, "shape": list(left.shape), "finite": finite,
        "atol": atol, "rtol": rtol, "exact_required": exact,
        "pass": finite and good and (not exact or actual == 0),
    })
    require(finite and good and (not exact or actual == 0),
            key + ": frozen parity, causal or cache gate failed")


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().contiguous().cpu().numpy().tobytes()
    ).hexdigest()


def fingerprint_backbone(base) -> str:
    h = hashlib.sha256()
    for name, param in base.named_parameters():
        h.update(name.encode("utf-8"))
        h.update(str(tuple(param.shape)).encode("utf-8"))
        h.update(str(param.dtype).encode("ascii"))
        h.update(param.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def load_formatter():
    actual = subprocess.check_output(
        ["git", "-C", str(FORMAT_SOURCE), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    require(actual == FORMAT_COMMIT, "Independent Sprocket role source changed")
    sys.path.insert(0, str(FORMAT_SOURCE / "scripts"))
    from ncr_001e_sprocket_format import (
        serialize_dialogue, validate_pinned_card, validate_pinned_tokenizer,
    )
    return serialize_dialogue, validate_pinned_card, validate_pinned_tokenizer


def load_actual(variant: str, spec: dict, pin: dict, formatter):
    """Validate full cached source bytes before ever constructing the host."""
    serialize, validate_card, validate_tokenizer = formatter
    weight, _ = LANE_WEIGHTS[variant]
    paths = ["README.md", *SOURCE_FILES[variant], weight]
    hashes = {}
    for name in paths:
        path = Path(hf_hub_download(spec["model_id"], name,
                                    revision=spec["revision"],
                                    local_files_only=True))
        actual = digest(path)
        key = name.rsplit("/", 1)[-1].replace(".", "_")
        if name == "README.md":
            expected = CARD_SHA256
        elif name == weight:
            expected = spec["weight_sha256"]
        elif variant == "chat":
            expected = pin["chat_" + key + "_sha256"]
        else:
            expected = pin[key + "_sha256"]
        hashes[name] = {"sha256": actual, "bytes": path.stat().st_size}
        require(actual == expected, "Pinned HF file hash mismatch: " + name)
        if name == weight:
            require(path.stat().st_size == 1002207984,
                    "Incomplete Sprocket model weight file")
    card = Path(hf_hub_download(spec["model_id"], "README.md",
                                revision=spec["revision"],
                                local_files_only=True)).read_text(encoding="utf-8")
    validate_card(card)
    options = {"revision": spec["revision"], "local_files_only": True,
               "trust_remote_code": False}
    if variant == "chat":
        options["subfolder"] = "chat"
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], **options)
    controls = validate_tokenizer(tokenizer, variant=variant)["controls"]
    for marker, expected in {
        "<|user|>": 3, "<|assistant|>": 4, "<|end|>": 5, **CONTROL_IDS
    }.items():
        require(controls[marker]["token_ids"] == [expected]
                and controls[marker]["dedicated"],
                "Pinned source control-token ID mismatch: " + marker)
    tokenizer.pad_token_id = controls["<|end|>"]["token_ids"][0]
    model = AutoModelForCausalLM.from_pretrained(
        spec["model_id"], **options, torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    if variant == "chat":
        nca = FrozenSprocketChatDepthNCA(
            model, verified_weight_sha256=spec["weight_sha256"]).eval()
    else:
        nca = FrozenSprocketInstructDepthNCA(
            model, verified_weight_sha256=spec["weight_sha256"]).eval()
    mlp = FrozenDepthGatedMLP(
        model, lane="sprocket_" + variant,
        verified_weight_sha256=spec["weight_sha256"],
        source_revision=spec["revision"]).eval()
    require(nca._backbone_lock is mlp._backbone_lock,
            "NCA and MLP do not share the decoder-root lease")
    require(not any(p.requires_grad for p in model.parameters()),
            "Actual frozen base parameters trainable")
    return model, tokenizer, nca, mlp, hashes, serialize


def ids_for(tokenizer, serialize, variant: str, content: str):
    text = serialize([{"role": "user", "content": content}], variant=variant)
    return tokenizer(text, add_special_tokens=False,
                     return_tensors="pt")["input_ids"]


def cache_probe(variant: str, phase: str, kind: str, wrapper, tokens,
                *, atol: float, rtol: float) -> None:
    """Nonzero untrained adapter: full 2048 versus prefill+four KV steps."""
    for n in (512, 2048):
        x = tokens[:, :n]
        full = wrapper(input_ids=x, use_cache=False).logits[:, -4:]
        prefill = wrapper(input_ids=x[:, :-4], use_cache=True)
        past = prefill.past_key_values
        pieces = []
        for i in range(n - 4, n):
            one = wrapper(input_ids=x[:, i:i + 1],
                          past_key_values=past, use_cache=True)
            pieces.append(one.logits[:, -1:])
            past = one.past_key_values
        cached = torch.cat(pieces, dim=1)
        check_logits(variant, phase, kind + "_full_vs_cached_" + str(n),
                     full, cached, atol=atol, rtol=rtol)
        del full, prefill, past, pieces, cached


def producer(variant: str, spec: dict, pin: dict, protocol: dict, formatter):
    model, tok, nca, mlp, hashes, serialize = load_actual(
        variant, spec, pin, formatter)
    REPORT["original_frozen_model_executed"] = True
    REPORT["verified_source_files"] = hashes
    REPORT["original_backbone_parameters"] = sum(p.numel() for p in model.parameters())
    REPORT["nca_parameters"] = nca.nca.trainable_parameters()
    REPORT["mlp_parameters"] = mlp.control.trainable_parameters()
    require(REPORT["nca_parameters"] == 658904 and
            REPORT["mlp_parameters"] == 658432,
            "Matched adapter parameter count changed")
    before = fingerprint_backbone(model)
    REPORT["in_memory_frozen_before_sha256"] = before
    save(variant, "producer")
    text = ("Inspect the bearing, verify the rail and check the drive. "
            "Record reversible observations before a shutdown. ") * 180
    tokens = ids_for(tok, serialize, variant, text)
    require(tokens.shape[1] >= 2048, "Pinned ordinary-role input too short for 2048")
    tokens = tokens[:, :2048]
    REPORT["ordinary_role_input_sha256"] = tensor_hash(tokens)
    REPORT["context_units"] = [512, 2048]
    atol = float(protocol["conformance"]["full_vs_cached_prefill_atol"])
    rtol = float(protocol["conformance"]["full_vs_cached_prefill_rtol"])
    require(atol == rtol == 0.0005, "Preregistered cache tolerance changed")
    with torch.inference_mode():
        for n in (16, 128):
            x = tokens[:, :n]
            raw = model(input_ids=x, use_cache=False).logits
            for kind, wrapper in (("nca", nca), ("mlp", mlp)):
                check_logits(variant, "producer", kind + "_zero_noop_" + str(n),
                             raw, wrapper(input_ids=x, use_cache=False).logits,
                             atol=0, rtol=0, exact=True)
                check_logits(variant, "producer", kind + "_disabled_" + str(n),
                             raw, wrapper(input_ids=x, enabled=False,
                                          use_cache=False).logits,
                             atol=0, rtol=0, exact=True)
        for side in ("left", "right"):
            tok.padding_side = side
            batch = tok([
                serialize([{"role": "user", "content": "Check the drive."}],
                          variant=variant),
                serialize([{"role": "user",
                            "content": "Inspect the bearing temperature first."}],
                          variant=variant),
            ], add_special_tokens=False, padding=True, return_tensors="pt")
            raw = model(**batch, use_cache=False).logits
            for kind, wrapper in (("nca", nca), ("mlp", mlp)):
                check_logits(variant, "producer", kind + "_" + side + "_masked",
                             raw, wrapper(**batch, use_cache=False).logits,
                             atol=0, rtol=0, exact=True)
        prompt = tokens[:, :16]
        for kind, wrapper in (("nca", nca), ("mlp", mlp)):
            gen = {"max_new_tokens": 4, "min_new_tokens": 4,
                   "do_sample": False, "use_cache": True,
                   "pad_token_id": tok.pad_token_id}
            plain = wrapper.generate(prompt, enabled=False, **gen)
            zero = wrapper.generate(prompt, **gen)
            require(torch.equal(plain, zero),
                    kind + ": zero-initialized source generation mismatch")
            record(variant, "producer", kind + "_hf_generate_zero_cache", {
                "exact": True, "new_units": int(zero.shape[1] - prompt.shape[1])})
        for kind, owner, foreign in (
            ("nca_blocks_mlp", nca, mlp), ("mlp_blocks_nca", mlp, nca)
        ):
            with owner._instrument():
                rejected = False
                try:
                    foreign(input_ids=prompt, enabled=False, use_cache=False)
                except RuntimeError as exc:
                    rejected = "Shared frozen" in str(exc)
                require(rejected, "Actual host leaked disabled reference through " + kind)
            require(not model.model._forward_hooks and
                    not model.model._forward_pre_hooks,
                    "A post-lease reference refusal leaked hooks")
            record(variant, "producer", kind, {"foreign_disabled_refused": True})
        with torch.no_grad():
            nca.nca.readout.weight[0, 0] = 0.03
            mlp.control.down.weight[0, 0] = 0.05
        for kind, wrapper in (("nca", nca), ("mlp", mlp)):
            x = tokens[:, :64]
            frozen = wrapper(input_ids=x, enabled=False,
                             use_cache=False).logits
            active = wrapper(input_ids=x, use_cache=False).logits
            effect = float((active - frozen).abs().max())
            require(effect > 0 and bool(torch.isfinite(active).all()),
                    kind + ": untrained manual adapter intervention ineffective")
            record(variant, "producer", kind + "_manual_untrained_effect",
                   {"max_abs": effect, "learned": False})
            cache_probe(variant, "producer", kind, wrapper, tokens,
                        atol=atol, rtol=rtol)
            x = tokens[:, :32]
            a = wrapper(input_ids=x, use_cache=False).logits
            wrapper(input_ids=tokens[:, 32:64], use_cache=False)
            again = wrapper(input_ids=x, use_cache=False).logits
            check_logits(variant, "producer", kind + "_A_B_A_reset",
                         a, again, atol=0, rtol=0, exact=True)
            x = tokens[:, :64].clone()
            old = wrapper(input_ids=x, use_cache=False).logits
            x[0, -1] = (x[0, -1] + 7) % 32000
            changed = wrapper(input_ids=x, use_cache=False).logits
            check_logits(variant, "producer", kind + "_suffix_invariance",
                         old[:, :-1], changed[:, :-1], atol=0, rtol=0, exact=True)
            REPORT["checks"][kind + "_cold_target_sha256"] = tensor_hash(active)
        raw_ids = tokens[:, :64].clone()
        raw_ids[0, -4:] = torch.tensor(list(CONTROL_IDS.values()))
        raw = nca(input_ids=raw_ids, use_cache=False).logits[:, -4:]
        pre = nca(input_ids=raw_ids[:, :-4], use_cache=True)
        past, outputs = pre.past_key_values, []
        for i in range(60, 64):
            step = nca(input_ids=raw_ids[:, i:i + 1],
                       past_key_values=past, use_cache=True)
            outputs.append(step.logits[:, -1:])
            past = step.past_key_values
        check_logits(variant, "producer", "raw_control_ids_cache_transport_only",
                     raw, torch.cat(outputs, dim=1), atol=atol, rtol=rtol)
    for kind, wrapper in (("nca", nca), ("mlp", mlp)):
        folder = archive_path(variant, "producer").parent / kind
        wrapper.save_adapter(folder)
        REPORT["checks"][kind + "_adapter_metadata_sha256"] = digest(
            folder / "metadata.json")
        REPORT["checks"][kind + "_adapter_state_sha256"] = digest(
            folder / "adapter.pt")
    after = fingerprint_backbone(model)
    REPORT["in_memory_frozen_after_sha256"] = after
    require(before == after, "Original frozen backbone changed despite no optimizer")
    require(all(p.grad is None for p in model.parameters()),
            "Original frozen backbone received gradients")
    REPORT["status"] = "PASS_SOURCE_MATCHED_POSTLEASE_PRODUCER"


def consumer(variant: str, spec: dict, pin: dict, protocol: dict, formatter):
    producer_report = json.loads(archive_path(variant, "producer").read_text())
    require(producer_report["status"] == "PASS_SOURCE_MATCHED_POSTLEASE_PRODUCER",
            "Producer receipt incomplete or unsuccessful")
    require(producer_report["runner"]["head_sha"] == REPORT["runner"]["head_sha"],
            "Cold replay uses different implementation head")
    require(producer_report["runner"]["run_id"] == REPORT["runner"]["run_id"],
            "Producer and consumer are not from same workflow run")
    require(producer_report["runner"]["pid"] != os.getpid(),
            "Cold replay did not use a distinct Python interpreter")
    model, tok, nca, mlp, hashes, serialize = load_actual(
        variant, spec, pin, formatter)
    REPORT["verified_source_files"] = hashes
    require(hashes == producer_report["verified_source_files"],
            "Consumer loaded different actual frozen source bytes")
    require(not any(p.requires_grad for p in model.parameters()),
            "Consumer base not frozen")
    for kind, wrapper in (("nca", nca), ("mlp", mlp)):
        folder = archive_path(variant, "producer").parent / kind
        for name, key in (("metadata.json", kind + "_adapter_metadata_sha256"),
                          ("adapter.pt", kind + "_adapter_state_sha256")):
            require(digest(folder / name) == producer_report["checks"][key],
                    "Cold adapter file bytes changed: " + name)
        wrapper.load_adapter(folder)
    text = ("Inspect the bearing, verify the rail and check the drive. "
            "Record reversible observations before a shutdown. ") * 180
    tokens = ids_for(tok, serialize, variant, text)
    require(tokens.shape[1] >= 2048, "Cold source roles lost long context")
    require(tensor_hash(tokens[:, :2048]) ==
            producer_report["ordinary_role_input_sha256"],
            "Cold tokenizer produced different ordinary role IDs")
    REPORT["original_frozen_model_executed"] = True
    with torch.inference_mode():
        for kind, wrapper in (("nca", nca), ("mlp", mlp)):
            output = wrapper(input_ids=tokens[:, :64], use_cache=False).logits
            observed = tensor_hash(output)
            expected = producer_report["checks"][kind + "_cold_target_sha256"]
            record(variant, "consumer", kind + "_separate_python_reload", {
                "sha256": observed, "expected_sha256": expected,
                "byte_exact": observed == expected})
            require(observed == expected,
                    "Separate-process " + kind + " adapter/logits replay mismatch")
    REPORT["producer_pid"] = producer_report["runner"]["pid"]
    REPORT["status"] = "PASS_SOURCE_MATCHED_POSTLEASE_CONSUMER"


def main(variant: str, phase: str) -> None:
    require(os.environ.get("HF_HUB_OFFLINE") == "1" and
            os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "Original frozen model must run offline after verified prefetch")
    require(os.environ.get("GITHUB_EVENT_NAME") == "pull_request",
            "This source-matched workload must run on a draft PR")
    protocol_path = ROOT / "experiments/ncr-001/protocol.json"
    pin_path = ROOT / "experiments/ncr-001/source-pins.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    pins = json.loads(pin_path.read_text(encoding="utf-8"))
    spec = verify_contract(variant, protocol, pins)
    expected_head = os.environ.get("NCR_EXPECTED_HEAD")
    actual_head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    require(bool(expected_head) and actual_head == expected_head,
            "Expected source HEAD differs from checked-out PR head")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    torch.manual_seed(20261028)
    REPORT.update({
        "experiment": "NCR-001", "lane": "sprocket_" + variant, "phase": phase,
        "status": "STARTED", "checks": {}, "train_authorized": False,
        "optimizer_steps": 0, "comparative_training_performed": False,
        "heldout_opened": False, "training_performed": False,
        "actual_host_final_admission": False, "tool_memory_semantics_qualified": False,
        "required_learning_update_loss_checks_pass": False,
        "source_identity": spec,
        "runner": {
            "pid": os.getpid(), "run_id": os.getenv("GITHUB_RUN_ID"),
            "attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
            "runner_name": os.getenv("RUNNER_NAME"),
            "runner_os": os.getenv("RUNNER_OS"),
            "runner_arch": os.getenv("RUNNER_ARCH"),
            "host": platform.node(), "platform": platform.platform(),
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "head_sha": actual_head, "threads": torch.get_num_threads(),
            "cpu_model": next(
                (line.split(":", 1)[1].strip()
                 for line in Path("/proc/cpuinfo").read_text().splitlines()
                 if line.startswith("model name")), "unknown"
            ) if Path("/proc/cpuinfo").exists() else "unknown",
        },
        "source_sha256": {
            p: digest(ROOT / p) for p in (
                "scripts/ncr_001_postlease_sprocket.py",
                "scripts/ncr_001_backbone_lease.py",
                "scripts/ncr_001_sprocket_chat.py",
                "scripts/ncr_001_sprocket_instruct.py",
                "scripts/ncr_001f_frozen_hooks.py",
                "scripts/ncr_001_core.py",
                "scripts/ncr_001f_controls.py",
                "experiments/ncr-001/protocol.json",
                "experiments/ncr-001/source-pins.json",
            )
        },
    })
    save(variant, phase)
    formatter = load_formatter()
    REPORT["formatter_authority"] = {
        "commit": FORMAT_COMMIT,
        "sha256": digest(FORMAT_SOURCE / "scripts/ncr_001e_sprocket_format.py"),
        "scope": "source_documented_ordinary_roles_only",
    }
    save(variant, phase)
    if phase == "producer":
        producer(variant, spec, pins["models"]["sprocket_" + variant],
                 protocol, formatter)
    else:
        consumer(variant, spec, pins["models"]["sprocket_" + variant],
                 protocol, formatter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("chat", "instruct"), required=True)
    parser.add_argument("--phase", choices=("producer", "consumer"), required=True)
    args = parser.parse_args()
    try:
        main(args.variant, args.phase)
    except BaseException as exc:
        REPORT["status"] = "FAIL_SOURCE_MATCHED_POSTLEASE_" + args.phase.upper()
        REPORT["exception_type"] = type(exc).__name__
        REPORT["failure_traceback"] = traceback.format_exc()
        raise
    finally:
        save(args.variant, args.phase)
        print(json.dumps({
            "lane": "sprocket_" + args.variant, "phase": args.phase,
            "status": REPORT.get("status"), "run_id": os.getenv("GITHUB_RUN_ID"),
            "head_sha": REPORT.get("runner", {}).get("head_sha"),
            "optimizer_steps": 0, "train_authorized": False,
        }, sort_keys=True), flush=True)
