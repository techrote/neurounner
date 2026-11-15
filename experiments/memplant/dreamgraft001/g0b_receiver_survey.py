"""G0B: investigate weak recipient task competence, NOT a G1 training run."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import platform
import traceback

import torch
from .bridge import answer_loss, format_prompt
from .contract import FIXTURES, FIXED_QUESTION, write_json
from .g0_live import load_pinned, measure_rss, target_ids, optimize_native_prefix

CANDIDATE = "HuggingFaceTB/SmolLM2-360M-Instruct"
REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"
ALT_QUESTION = FIXED_QUESTION.replace("hardens or softens", "softens or hardens")


def survey(output):
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(619)
    info = {"mission": "G0B recipient task-capability diagnostic; NOT memory transplant",
            "utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(), "task_questions": [FIXED_QUESTION, ALT_QUESTION],
            "g1_executed": False, "real_source_KV_not_used_in_this_survey": True}
    try:
        receiver, tokenizer, provenance = load_pinned(CANDIDATE, REVISION)
        info["receiver"] = provenance
        for p in receiver.parameters():
            p.requires_grad_(False)
        receiver.eval()
        rows = []
        for order, question in enumerate((FIXED_QUESTION, ALT_QUESTION)):
            for mem in FIXTURES[:2]:
                prompt = format_prompt(tokenizer, "Source document:\n" + mem.donor_text + "\n\n" + question)
                hard = target_ids(tokenizer, " hardens")
                soft = target_ids(tokenizer, " softens")
                with torch.no_grad():
                    hl = float(answer_loss(receiver, prompt, hard).item())
                    sl = float(answer_loss(receiver, prompt, soft).item())
                rows.append({"order": order, "world": mem.name, "expected": mem.answer,
                             "hardens_nll_per_token": round(hl, 6),
                             "softens_nll_per_token": round(sl, 6),
                             "log_odds_hard_vs_soft": round(sl - hl, 6),
                             "prefers_correct": bool(hl < sl if mem.answer == "hardens" else sl < hl)})
        per_order = []
        for k in (0, 1):
            pair = [r for r in rows if r["order"] == k]
            per_order.append({"order": k,
                              "correct": sum(r["prefers_correct"] for r in pair),
                              "paired_source_contrast": round(
                                  pair[0]["log_odds_hard_vs_soft"] - pair[1]["log_odds_hard_vs_soft"], 6)})
        info["task_capability"] = {
            "case_results": rows, "contrasts": per_order,
            "both_worlds_correct_in_any_order": any(x["correct"] == 2 for x in per_order),
            "source_specific_contrast_positive_both_orders": all(
                x["paired_source_contrast"] > 0 for x in per_order)}
        native, oracle = optimize_native_prefix(receiver, tokenizer, steps=6)
        info["native_prefix_oracle"] = oracle
        info["rss_after_mib"] = measure_rss()
        info["interpretation"] = "Diagnostic only: model capability, not evidence of cross-model KV transfer"
        write_json(output, info)
        return info
    except Exception as e:
        info["error"] = {"kind": type(e).__name__, "message": str(e),
                         "trace": traceback.format_exc()[-3000:]}
        write_json(output, info)
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runtime/memplant/g0b/receiver_360m.json")
    a = p.parse_args()
    result = survey(a.out)
    print("G0B source-aware diagnostic:", result["task_capability"]["contrasts"])
    print("G0B native prefix oracle:", result["native_prefix_oracle"]["status"])


if __name__ == "__main__":
    main()
