#!/usr/bin/env python3
"""DETECTIVE-004A Stage A: deterministic source-native scripter curriculum and sealed-final recipe.

Offline only. No weights, inference, optimizer, provider calls or relevance labels.
The future trainer MUST use the exact publisher tokenizer and verify a 512-token
bound before any optimizer step; character spans here document loss ownership,
not a substitute for token-level publisher SFT masking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import detective_001b3_sprocket as native  # noqa: E402
from scripts.detective_001_contract import ContractError, parse_program  # noqa: E402

CONFIG = ROOT / "experiments/detective-004/specialization.json"
DATASET = ROOT / "datasets/detective-004a"
ROLE = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>", "tool": "<|tool_result|>"}
OPS = ("read_asset", "correlate_assets", "probe_terminal")
EXPECTED_FAMILIES = ("amber-quay", "cinder-span", "ferrite-yard", "opal-canal", "signal-orchard", "violet-works")
EXPECTED_FINAL_FAMILIES = ("ravel-gate", "lumen-vault", "boreal-bridge")
TRAINING = {"stage_a_optimizer_steps": 0, "stage_b_approved_only_after_merge": True,
            "adapter_count": 1, "rank": 8, "alpha": 16, "dropout": 0,
            "target_modules": ["q_proj", "v_proj"], "epochs": 1,
            "optimizer_steps": 36, "learning_rate": 0.0002,
            "optimizer": "AdamW", "max_length": 512, "gradient_clip": 1,
            "threads": 4, "max_minutes": 35, "paid_api_calls": 0,
            "gpu": False, "weight_frozen": True}
GATE = {"tool_selection_min": 7, "per_tool_min": 2, "restraint_min": 1,
        "result_use_min": 2, "improvement_over_base_min": 4,
        "required_tool_total": 9, "required_restraint_total": 1,
        "required_result_total": 2, "no_out_of_view_ids": True}


class ProtocolError(ValueError):
    pass


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return sha256_bytes(path.read_bytes())


def git_blob_sha(path):
    b = path.read_bytes()
    return hashlib.sha1(f"blob {len(b)}\0".encode("ascii") + b).hexdigest()


def pretty(obj):
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def assert_exact(cond, why):
    if not cond:
        raise ProtocolError(why)


def load_contract():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert_exact((cfg.get("schema"), cfg.get("experiment"), cfg.get("issue"), cfg.get("parent_issue")) ==
                 ("detective-004a-stage-a-v1", "DETECTIVE-004A", 148, 103), "identity drift")
    b3, _ = native.load_spec()
    pins = cfg["authority"]
    for entry in ("b3_spec", "b3_runner", "bridge_config"):
        path = ROOT / pins[entry]["path"]
        assert_exact(path.is_file() and git_blob_sha(path) == pins[entry]["git_blob_sha1"],
                     "source Git blob drift: " + entry)
    assert_exact(pins["b3_spec"] == {"path": "experiments/detective-001/sprocket-scripter.json",
                                    "git_blob_sha1": "b2a4eaff9194160d2afd5455c6d658d9434d90b0"}, "B3 spec pin drift")
    assert_exact(pins["b3_runner"] == {"path": "scripts/detective_001b3_sprocket.py",
                                      "git_blob_sha1": "214c16301075113b51123db4afeca550b4226fb9"}, "B3 wire pin drift")
    assert_exact(pins["bridge_config"] == {"path": "experiments/detective-003/scripter-bridge.json",
                                          "git_blob_sha1": "31af5060cc66230c2f4740af5f5c35400d7e99cc"}, "003C pin drift")
    assert_exact(pins["model"] == {k: b3["model"][k] for k in pins["model"]} and
                 set(pins["model"]) == {"repo", "revision", "variant", "weight_file", "weight_bytes",
                                        "weight_sha256", "chat_checkpoint_forbidden"}, "root model pin drift")
    assert_exact(pins["publisher"] == {k: b3["publisher"][k] for k in pins["publisher"]}, "publisher source drift")
    assert_exact(pins["special_token_ids"] == b3["special_token_ids"], "token ID drift")
    assert_exact(cfg["corpus"] == {
        "dataset_dir": "datasets/detective-004a", "num_examples": 36,
        "training_families": list(EXPECTED_FAMILIES), "tool_selection_per_family": 4,
        "restraint_per_family": 1, "result_per_family": 1,
        "tool_selection_counts": {op: 8 for op in OPS},
        "result_counts": {"success": 3, "denied": 3},
        "case_namespace": "fiction-only-d004a", "seed": 14836}, "training corpus contract drift")
    assert_exact(cfg["future_training"] == TRAINING, "training budget drift")
    assert_exact(cfg["gate"] == GATE, "gate drift")
    assert_exact(cfg["final_recipe"] == {
        "schema": "detective-004a-posthash-recipe-v1",
        "family_slugs": list(EXPECTED_FINAL_FAMILIES),
        "fixed_salt": "DETECTIVE-004A-sealed-v1-20261028",
        "tasks": {"read_asset": 3, "correlate_assets": 3, "probe_terminal": 3,
                  "restraint": 1, "result_success": 1, "result_denied": 1},
        "adapter_directory_sha256_required": True,
        "training_manifest_sha256_required": True, "materialize_pretraining": False}, "final recipe drift")
    assert_exact(cfg["invariants"] == {
        "training_case_from_bay_seventeen": False, "final_case_family_overlap": False,
        "hidden_relevance_access": False, "learner_final_access": False,
        "model_generation": False, "provider_calls": False, "host_execution": False,
        "body_selection": False, "population_expansion": False}, "invariants drift")
    return cfg, b3


def safe_host(text):
    assert_exact(isinstance(text, str) and bool(text.strip()) and len(text) <= 4000,
                 "invalid host text")
    assert_exact(not native.CONTROL_RX.search(text), "host control-token injection")
    assert_exact("\r" not in text and "\x00" not in text, "host newline injection")


def render_with_mask(turns, b3):
    """Return source-native text and trained CHARACTER spans for auditing.

    The publisher's actual tokenizer is required for future TOKEN targets.
    Assistant content + closing <|end|> trained; ALL headers/host turns masked.
    """
    assert_exact(isinstance(turns, list) and len(turns) in (3, 5), "bad turn count")
    expected_roles = ("system", "user", "assistant") if len(turns) == 3 else (
        "system", "user", "assistant", "tool", "assistant")
    assert_exact(tuple(t.get("role") for t in turns) == expected_roles, "role sequence invalid")
    out, spans = "", []
    for turn in turns:
        assert_exact(set(turn) == {"role", "content"}, "unexpected turn fields")
        role, body = turn["role"], turn["content"]
        if role != "assistant":
            safe_host(body)
        elif body.startswith("<|tool_call|>"):
            assert_exact(body.count("<|tool_call|>") == 1, "multiple native calls")
            try:
                call = native.parse_native_tool_call(body)
                native.tool_call_to_program(call, b3)
            except ContractError as exc:
                raise ProtocolError("invalid assistant tool call") from exc
        else:
            safe_host(body)
        out += ROLE[role]
        begin = len(out)
        out += body + "<|end|>"
        if role == "assistant":
            spans.append({"start": begin, "end": len(out)})
    assert_exact(bool(spans) and all(s["end"] > s["start"] for s in spans), "missing targets")
    assert_exact(sum(len(out[s["start"]:s["end"]]) for s in spans) < len(out),
                 "host text accidentally trained")
    return out, spans


def family_data(slug, seed=""):
    assert_exact(re.fullmatch(r"[a-z]+(?:-[a-z]+)*", slug) is not None, "invalid family")
    suffix = ("-" + seed[:10]) if seed else ""
    key = slug + suffix
    assets = [key + "-ledger", key + "-plate", key + "-trace"]
    return {
        "case_id": "fiction-only-d004a-" + key,
        "assets": [{"asset_id": a, "excerpt": "Synthetic inventory label for " + a + "."} for a in assets],
        "terminal_id": key + "-relay",
        "credential": "PASS-" + hashlib.sha256(key.encode()).hexdigest()[:8].upper(),
    }


def make_view(info, request, *, assets=None, terminals=False):
    visible = list(info["assets"] if assets is None else assets)
    view = {"schema": "detective-scripter-view-v1",
            "case_id": info["case_id"], "player_request": request,
            "visible_assets": visible,
            "disclosed_terminals": ([{"terminal_id": info["terminal_id"],
                                     "player_known_credential": info["credential"]}]
                                    if terminals else [])}
    return view


def tool_request(info, op, index):
    a = info["assets"]
    if op == "read_asset":
        chosen = a[index % 3]
        return ("Read the listed evidence asset " + chosen["asset_id"] + " using the game tool.",
                {"name": op, "arguments": {"asset_id": chosen["asset_id"]}},
                [chosen], False)
    if op == "correlate_assets":
        pair = [a[index % 3], a[(index + 1) % 3]]
        return ("Correlate the two visible records " + pair[0]["asset_id"] + " and " +
                pair[1]["asset_id"] + " using the game tool.",
                {"name": op, "arguments": {"left_id": pair[0]["asset_id"],
                                            "right_id": pair[1]["asset_id"]}},
                pair, False)
    if op == "probe_terminal":
        return ("Probe the disclosed " + info["terminal_id"] +
                " using only the player-supplied credential.",
                {"name": op, "arguments": {"terminal_id": info["terminal_id"],
                                            "credential": info["credential"]}},
                [], True)
    raise ProtocolError("unknown operation")


def make_example(family, case, kind, index, op=None):
    if kind == "tool_selection":
        request, call, visible, show_terminal = tool_request(case, op, index)
        view = make_view(case, request, assets=visible, terminals=show_terminal)
        content = native.tool_call_text(call)
        turns = [{"role": "system", "content": native.manifest()},
                 {"role": "user", "content": "Player request:\n" + request +
                  "\nCurrent fictional game view:\n" + canonical(view)},
                 {"role": "assistant", "content": content}]
        target = {"tool": op, "arguments": call["arguments"]}
    elif kind == "restraint":
        request = "Define a chain of custody in one sentence. Do not look up case records."
        view = make_view(case, request, assets=[])
        turns = [{"role": "system", "content": native.manifest()},
                 {"role": "user", "content": "Player request:\n" + request +
                  "\nCurrent fictional game view:\n" + canonical(view)},
                 {"role": "assistant", "content": "A chain of custody records who handled evidence and when."}]
        target = {"no_tool": True}
    elif kind in ("result_success", "result_denied"):
        if kind == "result_success":
            request, call, visible, show_terminal = tool_request(case, "read_asset", index)
            marker = "confirmed:" + family + ":" + str(index)
            result = {"status": "OK", "marker": marker, "excerpt": "Synthetic receipt " + marker}
            reply = "The returned evidence reports " + marker + "."
            expected, forbidden = [marker], ["classified", "unknown-package"]
        else:
            request, call, visible, show_terminal = tool_request(case, "probe_terminal", index)
            # The game returns denial, even for a disclosed attempted credential.
            result = {"status": "ACCESS_DENIED", "evidence": None}
            reply = "Access denied; no evidence was returned."
            expected, forbidden = ["access denied"], ["classified", "unknown-package"]
        view = make_view(case, request, assets=visible, terminals=show_terminal)
        turns = [{"role": "system", "content": native.manifest()},
                 {"role": "user", "content": "Player request:\n" + request +
                  "\nCurrent fictional game view:\n" + canonical(view)},
                 {"role": "assistant", "content": native.tool_call_text(call)},
                 {"role": "tool", "content": canonical(result)},
                 {"role": "assistant", "content": reply}]
        target = {"expected_markers": expected, "forbidden_markers": forbidden}
    else:
        raise ProtocolError("invalid task kind")
    return {"id": family + "-" + kind + "-" + str(index).zfill(2),
            "case_family": family, "kind": kind, "view": view,
            "turns": turns, "target": target}


def validate_example(row, b3):
    assert_exact(set(row) == {"id", "case_family", "kind", "view", "turns", "target",
                             "rendered", "loss_char_spans"}, "row field drift")
    assert_exact(row["view"]["schema"] == "detective-scripter-view-v1", "view schema drift")
    assert_exact(set(row["view"]) == {"schema", "case_id", "player_request",
                                     "visible_assets", "disclosed_terminals"}, "view widened")
    view = row["view"]
    assert_exact("fact_ids" not in canonical(view) and
                 "canonical_truth" not in canonical(view), "hidden authority in view")
    visible = {x["asset_id"] for x in view["visible_assets"]}
    terminal_creds = {x["terminal_id"]: x["player_known_credential"]
                      for x in view["disclosed_terminals"]}
    assert_exact(len(visible) == len(view["visible_assets"]) and
                 len(terminal_creds) == len(view["disclosed_terminals"]), "duplicate view IDs")
    for asset in view["visible_assets"]:
        assert_exact(set(asset) == {"asset_id", "excerpt"}, "asset view leak")
    for term in view["disclosed_terminals"]:
        assert_exact(set(term) == {"terminal_id", "player_known_credential"}, "terminal view leak")
    rendered, spans = render_with_mask(row["turns"], b3)
    assert_exact((rendered, spans) == (row["rendered"], row["loss_char_spans"]), "render/mask drift")
    if row["kind"] == "tool_selection":
        target = row["target"]
        call = native.parse_native_tool_call(row["turns"][-1]["content"])
        assert_exact(call == {"name": target["tool"], "arguments": target["arguments"]},
                     "train action label mismatch")
        args = call["arguments"]
        if call["name"] == "read_asset":
            assert_exact(args["asset_id"] in visible, "read invisible asset")
        if call["name"] == "correlate_assets":
            assert_exact(args["left_id"] in visible and args["right_id"] in visible and
                         args["left_id"] != args["right_id"], "correlate invisible/duplicate asset")
        if call["name"] == "probe_terminal":
            assert_exact(terminal_creds.get(args["terminal_id"]) == args["credential"],
                         "invented terminal credential")
    elif row["kind"] == "restraint":
        assert_exact(row["target"] == {"no_tool": True}, "restraint label drift")
    else:
        assert_exact(row["kind"] in ("result_success", "result_denied"), "result kind drift")
        assert_exact(row["turns"][3]["role"] == "tool", "host tool-result missing")


def build_training(cfg, b3):
    rows = []
    for j, name in enumerate(cfg["corpus"]["training_families"]):
        info = family_data(name)
        for k in range(4):
            op = OPS[(4*j+k) % 3]
            rows.append(make_example(name, info, "tool_selection", k, op))
        rows.append(make_example(name, info, "restraint", 4))
        rows.append(make_example(name, info, "result_success" if j < 3 else "result_denied", 5))
    for row in rows:
        row["rendered"], row["loss_char_spans"] = render_with_mask(row["turns"], b3)
        validate_example(row, b3)
    c = Counter(x["kind"] for x in rows)
    ops = Counter(x["target"]["tool"] for x in rows if x["kind"] == "tool_selection")
    assert_exact(len(rows) == 36 and len({x["id"] for x in rows}) == 36, "corpus size/uniqueness")
    assert_exact(ops == Counter({x: 8 for x in OPS}), "unequal tool counts")
    assert_exact(c == Counter({"tool_selection": 24, "restraint": 6,
                              "result_success": 3, "result_denied": 3}), "kind counts")
    return rows


def encode_jsonl(rows):
    return "".join(canonical(row) + "\n" for row in rows).encode("utf-8")


def training_archive(cfg, b3):
    rows = build_training(cfg, b3)
    payload = encode_jsonl(rows)
    m = {"schema": "detective-004a-corpus-manifest-v1", "experiment": "DETECTIVE-004A",
         "source_config_git_blob_sha1": git_blob_sha(CONFIG), "source_config_sha256": file_sha(CONFIG),
         "source_authority": cfg["authority"],
         "generator_git_blob_sha1": git_blob_sha(Path(__file__)),
         "training_jsonl_sha256": sha256_bytes(payload), "training_jsonl_bytes": len(payload),
         "case_families": list(EXPECTED_FAMILIES), "counts": dict(sorted(Counter(x["kind"] for x in rows).items())),
         "tool_counts": {op: 8 for op in OPS}, "example_ids": [row["id"] for row in rows],
         "generated_by_model": False, "provider_calls": 0, "optimizer_steps": 0,
         "learner_final_material": False, "repair_history": []}
    return payload, pretty(m).encode("utf-8")


def write_archive(folder, cfg, b3):
    payload, manifest = training_archive(cfg, b3)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "training.jsonl").write_bytes(payload)
    (folder / "manifest.json").write_bytes(manifest)
    return {"rows": 36, "training_sha256": sha256_bytes(payload),
            "manifest_sha256": sha256_bytes(manifest)}


def verify_archive(folder, cfg, b3):
    payload, manifest = training_archive(cfg, b3)
    try:
        train_bytes = (folder / "training.jsonl").read_bytes()
        manifest_bytes = (folder / "manifest.json").read_bytes()
    except OSError as exc:
        raise ProtocolError("training archive absent or unreadable") from exc
    assert_exact(train_bytes == payload, "training archive drift")
    assert_exact(manifest_bytes == manifest, "training manifest drift")
    return sha256_bytes(manifest)


def adapter_dir_sha256(folder):
    assert_exact(folder.is_dir() and not folder.is_symlink(), "adapter directory not found")
    paths = sorted(p for p in folder.rglob("*") if p.is_file())
    assert_exact(all(not p.is_symlink() for p in paths), "adapter symlink disallowed")
    names = {p.relative_to(folder).as_posix() for p in paths}
    assert_exact({"adapter_model.safetensors", "adapter_config.json"} <= names, "no real adapter files")
    assert_exact(len(paths) <= 12 and all(p.stat().st_size <= 2_000_000_000 for p in paths),
                 "adapter archive bound")
    record = [(p.relative_to(folder).as_posix(), file_sha(p), p.stat().st_size) for p in paths]
    return sha256_bytes(canonical(record).encode("utf-8"))


def final_recipe_sha(cfg):
    return sha256_bytes(canonical(cfg["final_recipe"]).encode("utf-8"))


def build_final_in_memory(cfg, b3, adapter_dir, expected_manifest_sha256, train_dir=DATASET):
    """Only callable with an actual existing adapter and matching committed training manifest.

    Never called by Stage-A CI. Outputs remain in memory unless future Stage B
    explicitly materializes them after training and independently reviews hashes.
    """
    assert_exact(re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256 or "") is not None,
                 "missing training manifest SHA")
    actual_manifest = verify_archive(train_dir, cfg, b3)
    assert_exact(actual_manifest == expected_manifest_sha256, "training manifest SHA mismatch")
    adapter_sha = adapter_dir_sha256(Path(adapter_dir))
    seed = sha256_bytes(canonical({"recipe_sha": final_recipe_sha(cfg),
                                  "manifest_sha": actual_manifest,
                                  "adapter_dir_sha": adapter_sha,
                                  "salt": cfg["final_recipe"]["fixed_salt"]}).encode("utf-8"))
    rows = []
    for i, family in enumerate(EXPECTED_FINAL_FAMILIES):
        info = family_data(family, seed)
        for j, op in enumerate(OPS):
            rows.append(make_example(family, info, "tool_selection", j, op))
        if i == 0:
            rows.append(make_example(family, info, "result_success", 3))
        elif i == 1:
            rows.append(make_example(family, info, "result_denied", 3))
        else:
            rows.append(make_example(family, info, "restraint", 3))
    assert_exact(len(rows) == 12 and not ({r["case_family"] for r in rows} &
                 set(EXPECTED_FAMILIES)), "final collision")
    for row in rows:
        row["rendered"], row["loss_char_spans"] = render_with_mask(row["turns"], b3)
        validate_example(row, b3)
    return {"recipe_sha256": final_recipe_sha(cfg),
            "training_manifest_sha256": actual_manifest, "adapter_dir_sha256": adapter_sha,
            "post_hash_seed_sha256": seed, "tasks": rows}


def score_output(task, raw, b3):
    """Strict no-repair task-level scoring on raw assistant output."""
    assert_exact(isinstance(raw, str), "raw output must be text")
    kind = task["kind"]
    if kind == "tool_selection":
        if not raw.startswith("<|tool_call|>"):
            return False
        try:
            call = native.parse_native_tool_call(raw)
            native.tool_call_to_program(call, b3)
            return call == {"name": task["target"]["tool"],
                            "arguments": task["target"]["arguments"]}
        except (ContractError, ValueError):
            return False
    text = raw[:-len("<|end|>")] if raw.endswith("<|end|>") else raw
    markers = ("<|tool_call|>", "<|memory_read|>", "<|memory_write|>",
               "<|tool_result|>", "<|assistant|>", "<|system|>", "<|user|>",
               "<think>", "</think>", "<|end|>")
    if not text.strip() or any(x in text for x in markers):
        return False
    if kind == "restraint":
        return True
    target = task["target"]
    lower = text.casefold()
    return (all(x.casefold() in lower for x in target["expected_markers"]) and
            all(x.casefold() not in lower for x in target["forbidden_markers"]))


def admission_gate(final_tasks, adapted_outputs, base_outputs, b3):
    assert_exact(len(final_tasks) == 12 and
                 set(adapted_outputs) == {t["id"] for t in final_tasks} ==
                 set(base_outputs), "final outputs incomplete")
    good = {t["id"]: score_output(t, adapted_outputs[t["id"]], b3) for t in final_tasks}
    control = {t["id"]: score_output(t, base_outputs[t["id"]], b3) for t in final_tasks}
    tool = [t for t in final_tasks if t["kind"] == "tool_selection"]
    restraint = [t for t in final_tasks if t["kind"] == "restraint"]
    result = [t for t in final_tasks if t["kind"].startswith("result_")]
    assert_exact((len(tool), len(restraint), len(result)) == (9, 1, 2),
                 "final threshold denominator drift")
    op_success = {op: sum(good[t["id"]] for t in tool if t["target"]["tool"] == op)
                  for op in OPS}
    n_tool, n_base = sum(good[t["id"]] for t in tool), sum(control[t["id"]] for t in tool)
    passed = (n_tool >= 7 and min(op_success.values()) >= 2 and
              all(good[t["id"]] for t in restraint + result) and
              n_tool - n_base >= 4)
    return {"status": "research_candidate" if passed else "fail",
            "trained_tool_selection": n_tool, "base_tool_selection": n_base,
            "tool_by_family": op_success,
            "trained_restraint": sum(good[t["id"]] for t in restraint),
            "trained_result_use": sum(good[t["id"]] for t in result),
            "tool_improvement": n_tool - n_base,
            "production_admission": False, "population_expansion_admitted": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-training", type=Path, help="export deterministic training archive")
    ap.add_argument("--verify-training", type=Path, help="compare a committed archive byte-for-byte")
    args = ap.parse_args()
    cfg, b3 = load_contract()
    rows = build_training(cfg, b3)
    print(f"DETECTIVE-004A Stage A: {len(rows)} examples; no training/inference")
    if args.write_training is not None:
        print(pretty(write_archive(args.write_training, cfg, b3)).strip())
    if args.verify_training is not None:
        print("Archive manifest SHA-256:", verify_archive(args.verify_training, cfg, b3))


if __name__ == "__main__":
    main()
