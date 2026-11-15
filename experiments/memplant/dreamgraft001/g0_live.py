"""Run DREAMGRAFT G0 on *real* frozen model weights (no simulated success).

G0 tests two distinct capabilities:
  I: can frozen B consume/learn a small native soft prefix?
  T: can B perform the micro-world task given the full source text?
Only then profile five gradient steps from actual donor V-state into B.
No G1/G2 training or scientific memory-transfer claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback

import torch

from experiments.memplant.dreamgraft001.contract import (
    DONOR_ID, DONOR_REV, RECEIVER_ID, RECEIVER_REV, SCHEMA_VERSION,
    FIXED_QUESTION, FIXTURES, metadata, write_json)
from experiments.memplant.dreamgraft001.bridge import (
    DreamPrefix, source_fingerprint, format_prompt, answer_loss,
    generation_from_prefix)


def measure_rss():
    import psutil
    return round(psutil.Process(os.getpid()).memory_info().rss / 1024**2, 2)


def digest_file(path: str | Path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda: f.read(2**20), b""):
            h.update(c)
    return h.hexdigest()


def load_pinned(model_id, rev):
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    info = HfApi().model_info(model_id, revision=rev)
    if info.sha != rev:
        raise RuntimeError(f"Model source moved: {model_id}: {info.sha} != {rev}")
    t0 = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=rev, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=rev, trust_remote_code=False, use_safetensors=True,
        torch_dtype=torch.float32).to("cpu").eval()
    weight_file = hf_hub_download(model_id, filename="model.safetensors", revision=rev)
    tokenizer_file = hf_hub_download(model_id, filename="tokenizer.json", revision=rev)
    return model, tokenizer, {
        "id": model_id, "revision": rev, "model_type": model.config.model_type,
        "layers": int(model.config.num_hidden_layers),
        "hidden_size": int(model.config.hidden_size),
        "num_attention_heads": int(model.config.num_attention_heads),
        "num_key_value_heads": int(model.config.num_key_value_heads),
        "parameters": sum(p.numel() for p in model.parameters()),
        "weight_sha256": digest_file(weight_file),
        "tokenizer_sha256": digest_file(tokenizer_file),
        "load_seconds_including_download": round(time.monotonic() - t0, 3),
        "rss_mib": measure_rss(),
    }


def target_ids(tokenizer, text):
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if ids.shape[1] == 0:
        raise RuntimeError("answer tokenized to empty")
    return ids


def full_text_reference(receiver, tokenizer):
    result = []
    for mem in FIXTURES[:2]:
        prompt = format_prompt(tokenizer, "Source document:\n" + mem.donor_text + "\n\n" + FIXED_QUESTION)
        correct = target_ids(tokenizer, " " + mem.answer)
        opposite = target_ids(tokenizer, " " + ("softens" if mem.answer == "hardens" else "hardens"))
        with torch.no_grad():
            c_loss = answer_loss(receiver, prompt, correct)
            n_loss = answer_loss(receiver, prompt, opposite)
        result.append({"fixture_id": mem.name, "correct_nll": round(c_loss.item(), 5),
                       "opposite_nll": round(n_loss.item(), 5),
                       "prefers_correct": bool(c_loss.item() < n_loss.item()),
                       "full_text_input_only": True})
    return {"case_results": result,
            "passed_cases": sum(x["prefers_correct"] for x in result),
            "total_cases": len(result),
            "status": "PASS" if all(x["prefers_correct"] for x in result) else "NOT_YET_PASS"}


def optimize_native_prefix(receiver, tokenizer, *, steps=8):
    torch.manual_seed(221)
    prompt = format_prompt(tokenizer, FIXED_QUESTION)
    answer = target_ids(tokenizer, " " + FIXTURES[0].answer)
    width = int(receiver.config.hidden_size)
    scale = max(float(receiver.get_input_embeddings().weight.detach().std().item()), .002)
    prefix = torch.nn.Parameter(torch.randn(1, 8, width) * scale)
    opt = torch.optim.Adam([prefix], lr=0.08)
    with torch.no_grad():
        initial = float(answer_loss(receiver, prompt, answer, prefix).item())
    history, t0 = [], time.monotonic()
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = answer_loss(receiver, prompt, answer, prefix)
        loss.backward()
        norm = float(prefix.grad.norm().item())
        if not (torch.isfinite(loss).item() and torch.isfinite(prefix.grad).all().item() and norm > 0):
            raise RuntimeError(f"native prefix oracle gradient invalid at {i}")
        opt.step()
        history.append({"update": i + 1, "loss_before_update": round(float(loss.item()), 6),
                        "gradient_norm": round(norm, 6), "rss_mib": measure_rss()})
    with torch.no_grad():
        last = float(answer_loss(receiver, prompt, answer, prefix).item())
    status = "PASS" if last < initial - 1e-4 else "NOT_YET_PASS"
    return prefix.detach(), {"status": status, "initial_nll": round(initial, 6),
                              "final_nll": round(last, 6), "steps": steps,
                              "train_seconds": round(time.monotonic()-t0, 3),
                              "history": history, "prefix_slots": 8}


def adapter_five_step(receiver, tokenizer, source_features, *, steps=5):
    torch.manual_seed(321)
    width = int(receiver.config.hidden_size)
    emb_std = max(float(receiver.get_input_embeddings().weight.detach().std().item()), .002)
    adapter = DreamPrefix(source_features[0].numel(), width, slots=8,
                          bottleneck=64, scale=emb_std)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.01)
    assert all(not p.requires_grad for p in receiver.parameters())
    targets = [(format_prompt(tokenizer, FIXED_QUESTION), target_ids(tokenizer, " "+x.answer))
               for x in FIXTURES[:2]]
    results = []
    for i in range(steps):
        m = i % 2
        t0 = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        prefix = adapter(source_features[m].unsqueeze(0))
        loss = answer_loss(receiver, targets[m][0], targets[m][1], prefix)
        loss.backward()
        grads = [p.grad for p in adapter.parameters()]
        if any(g is None or not torch.isfinite(g).all() for g in grads):
            raise RuntimeError(f"adapter gradient missing/nonfinite, step {i+1}")
        grad_norm = float(sum(g.square().sum() for g in grads).sqrt().item())
        if grad_norm <= 0:
            raise RuntimeError(f"adapter gradient zero at step {i+1}")
        optimizer.step()
        results.append({"step": i+1, "fixture": FIXTURES[m].name,
                        "nll_before_update": round(float(loss.item()), 6),
                        "gradient_norm": round(grad_norm, 6),
                        "seconds": round(time.monotonic()-t0, 3),
                        "rss_mib": measure_rss()})
    if any(p.grad is not None for p in receiver.parameters()):
        raise RuntimeError("frozen recipient accumulated weight gradients")
    return adapter, {"steps": results,
                     "adapter_trainable_parameters": sum(p.numel() for p in adapter.parameters()),
                     "total_train_seconds": round(sum(r["seconds"] for r in results), 3),
                     "all_finite_gradients": True,
                     "five_step_probe_not_a_trained_transplant": True}


def run(output: str):
    import transformers
    import psutil
    from huggingface_hub import __version__ as hf_hub_version
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(111)
    out = {"schema": SCHEMA_VERSION, "mission": "G0 instrumentation only",
           "runtime": {"utc": datetime.now(timezone.utc).isoformat(),
                       "platform": platform.platform(), "python": platform.python_version(),
                       "torch": torch.__version__, "transformers": transformers.__version__,
                       "huggingface_hub": hf_hub_version, "psutil": psutil.__version__,
                       "cpu_threads": torch.get_num_threads(), "start_rss_mib": measure_rss()},
           "fixtures": metadata(), "input_audit": {
               "receiver_query_fixed": True,
               "receiver_never_receives_donor_text_in_numerical_arms": True,
               "query_sha256": hashlib.sha256(FIXED_QUESTION.encode()).hexdigest()}}
    dest = Path(output)
    try:
        donor, donor_tokenizer, d_info = load_pinned(DONOR_ID, DONOR_REV)
        out["donor"] = d_info
        features, audits = [], []
        for mem in FIXTURES:
            src = donor_tokenizer(mem.donor_text, return_tensors="pt", add_special_tokens=True)
            with torch.no_grad():
                observed = donor(**src, use_cache=True, return_dict=True)
            feature, meta = source_fingerprint(observed.past_key_values)
            if feature.numel() != 2048 or meta["cache_layers"] != 24:
                raise RuntimeError("Qwen KV geometry changed from pinned contract")
            meta["fixture_id"] = mem.name
            features.append(feature)
            audits.append(meta)
            del observed, src
        out["source_cache"] = audits
        out["runtime"]["rss_after_donor_mib"] = measure_rss()
        del donor, donor_tokenizer
        gc.collect()
        out["runtime"]["rss_after_donor_unload_mib"] = measure_rss()

        receiver, tokenizer, b_info = load_pinned(RECEIVER_ID, RECEIVER_REV)
        out["receiver"] = b_info
        if b_info["hidden_size"] != 576 or b_info["layers"] != 30:
            raise RuntimeError("pinned SmolLM2 geometry unexpectedly changed")
        for p in receiver.parameters():
            p.requires_grad_(False)
        receiver.eval()

        out["task_capability_oracle"] = full_text_reference(receiver, tokenizer)
        native_prefix, out["native_prefix_oracle"] = optimize_native_prefix(receiver, tokenizer)
        adapter, out["adapter_five_step"] = adapter_five_step(receiver, tokenizer, features)

        prompt = format_prompt(tokenizer, FIXED_QUESTION)
        with torch.no_grad():
            real_prefix = adapter(features[0].unsqueeze(0))
            zero_prefix = torch.zeros_like(real_prefix)
            other_prefix = adapter(features[1].unsqueeze(0))
            real_a, ids_real_a = generation_from_prefix(receiver, tokenizer, prompt, real_prefix, max_tokens=10)
            real_b, ids_real_b = generation_from_prefix(receiver, tokenizer, prompt, real_prefix, max_tokens=10)
            zero, ids_zero = generation_from_prefix(receiver, tokenizer, prompt, zero_prefix, max_tokens=10)
            swapped, ids_swap = generation_from_prefix(receiver, tokenizer, prompt, other_prefix, max_tokens=10)
            fitted, ids_native = generation_from_prefix(receiver, tokenizer, prompt, native_prefix, max_tokens=10)
        if ids_real_a != ids_real_b:
            raise RuntimeError("native prefix continuation not repeatable at zero temperature")
        out["prefix_continuation"] = {
            "real_donor": real_a, "zero": zero, "swapped_donor": swapped,
            "native_direct_prefix": fitted, "ids_real": ids_real_a,
            "ids_zero": ids_zero, "ids_swapped": ids_swap, "ids_native": ids_native,
            "repeat_exact": True, "real_vs_swapped_distinct": ids_real_a != ids_swap,
            "evidence_status": "exploratory; no generalization; not a verified transplant"}

        # G0 creative artefact: all raw postcard conditions are retained, not curated.
        creative_tasks = (
            "Invent an impossible material from the world you inherited. In two sentences, explain its strange behavior.",
            "Design a small machine that could work in the world you inherited. Give it a name and an unexpected failure.",
            "Describe a peculiar ecosystem that might arise under the laws of the world you inherited.",
        )
        postcards = []
        with torch.no_grad():
            for task_i, task in enumerate(creative_tasks):
                p = format_prompt(tokenizer, task)
                for cond, pref in (("donor-A", real_prefix), ("donor-B-swapped", other_prefix),
                                   ("zero-prefix", zero_prefix)):
                    raw, ids = generation_from_prefix(receiver, tokenizer, p, pref, max_tokens=32)
                    postcards.append({"task_id": task_i, "task": task,
                                      "condition": cond, "raw": raw,
                                      "generated_token_ids": ids})
        out["creative_postcards"] = {
            "status": "unscored G0 exploratory raw outputs; NOT transplant evidence",
            "count": len(postcards), "records": postcards,
            "no_editor_used": True, "no_gallery_selection": True}

        out["runtime"]["end_rss_mib"] = measure_rss()
        out["qualifications"] = {
            "g0_instrumentation": "PASS" if len(audits) == 3 and out["native_prefix_oracle"]["status"] == "PASS" and out["adapter_five_step"]["all_finite_gradients"] else "NOT_YET_PASS",
            "full_text_task_oracle": out["task_capability_oracle"]["status"],
            "g1_authorised": False, "g1_executed": False,
            "kv_specific_semantic_transfer_proven": False,
            "live_deep_kv_transplant_executed": False,
        }
        write_json(dest, out)
        return out
    except Exception as exc:
        out["error"] = {"kind": type(exc).__name__, "message": str(exc),
                        "trace": traceback.format_exc()[-3500:]}
        out["qualifications"] = {"g0_instrumentation": "BLOCKED", "g1_executed": False}
        write_json(dest, out)
        raise


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", default="runtime/memplant/g0/report.json")
    args=ap.parse_args()
    result=run(args.out)
    print(json.dumps({"model_ids": [DONOR_ID, RECEIVER_ID],
                      "source_cache_shapes": [x["feature_dim"] for x in result["source_cache"]],
                      "task_oracle": result["task_capability_oracle"]["status"],
                      "native_prefix": result["native_prefix_oracle"]["status"],
                      "adapter_steps": len(result["adapter_five_step"]["steps"]),
                      "g0": result["qualifications"]["g0_instrumentation"]}, indent=2))

if __name__ == "__main__":
    main()
