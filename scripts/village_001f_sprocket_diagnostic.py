#!/usr/bin/env python3
"""Fixed 36-generation Sprocket prompt diagnostic; no training or cloud calls.

The default mode only freezes a plan. --execute uses the existing pinned local
llama-server; --offline reconstructs completed evidence without starting a model.
No Stage E final-suite path is read by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import statistics
import struct
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

from babble_002_expand import packet_for
from village_001_manifest import build_manifest
from village_001d_sprocket_feedback import (
    SPECIALIZATION_GUIDANCE, feedback_system, feedback_user, load_env, parse_feedback,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/village-001/diagnostic-20261119/sprocket-config.json"
STOPWORDS = set("a an and are as at be been but by do for from has have i in is it its not of on or so that the their them then there these they this to was we were what when which who will with you your one two keep mutate failure guess genericness low medium high response situation concrete behavior element change".split())


def canonical(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def immutable(path: Path, data: bytes) -> None:
    if path.is_symlink():
        raise RuntimeError(f"symlink evidence path: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise RuntimeError(f"immutable evidence mismatch: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as fh:
        temp = Path(fh.name)
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path: Path, value) -> None:
    immutable(path, canonical(value))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def source_path(relative: str) -> Path:
    path = ROOT / relative
    require(not path.is_symlink() and ROOT.resolve() in path.resolve().parents, "source must be a repository regular file")
    require("final-eval" not in relative and "/final-" not in relative and "/runtime/" not in relative, "final-suite source is outside this diagnostic")
    return path


def compact_system(member: dict, packet: dict) -> str:
    return '''Return exactly one compact JSON object, no prose or Markdown:
{"keep":["one specific behavior"],"mutate":["one concrete change"],"failure_guess":"one likely failure","genericness":"medium"}

Critique the supplied villager RESPONSE. Replace the example phrases with its actual behavior. Use short phrases, one keep item and one mutate item. Choose genericness low, medium or high. Do not rewrite the answer or invent events.
Preserve local oddness instead of generic politeness.
Temperament: ''' + packet["persona_summary"] + "\nSibling pressure: " + SPECIALIZATION_GUIDANCE[member["sibling"]]


def build_plan(config_path: Path) -> tuple[dict, dict]:
    require(not config_path.is_symlink(), 'diagnostic config may not be a symlink')
    config_path = config_path.resolve()
    require(ROOT.resolve() in config_path.parents, 'diagnostic config must be in the repository')
    cfg = json.loads(config_path.read_text())
    require(cfg.get("generation_limit") == 36 and cfg.get("optimizer_steps") == 0 and cfg.get("cloud_requests") == 0 and cfg.get("final_suite_access") is False, "diagnostic scope drift")
    require(cfg.get("workers") == [1, 2, 3] and cfg.get("arms") == ["original", "compact_schema_first"], "diagnostic arm/worker drift")
    require(cfg.get("sampling") == {"max_tokens": 160, "temperature": 1.05, "top_p": 0.97}, "sampling contract drift")
    require(len(cfg.get("cases", [])) == 6, "exactly six fixed cases required")
    campaign_path, preference_path = source_path(cfg["campaign_config"]), source_path(cfg["preference_config"])
    campaign, preference = json.loads(campaign_path.read_text()), json.loads(preference_path.read_text())
    manifest = build_manifest(campaign)
    members = {row["member_id"]: row for row in manifest["members"]}
    pins_path = source_path(cfg["pins_path"])
    pins = load_env(pins_path)
    for key, expected in {"LLAMA_TAG": cfg["runtime"]["tag"], "LLAMA_COMMIT": cfg["runtime"]["commit"], "LLAMA_SHA256": cfg["runtime"]["archive_sha256"], "SPROCKET_REV": cfg["model"]["revision"], "SPROCKET_SHA256": cfg["model"]["sha256"], "SPROCKET_FILE": cfg["model"]["filename"]}.items():
        require(pins.get(key) == expected, f"pinned runtime/model mismatch: {key}")
    require(campaign["sprocket"]["source_sha256"] == cfg["model"]["sha256"] and campaign["sprocket"]["source_revision"] == cfg["model"]["revision"], "campaign Sprocket identity mismatch")
    for key, expected in cfg["sampling"].items():
        require(preference["feedback"][key] == expected, f"historical sampling mismatch: {key}")
    require(cfg["runtime"]["server_context_requested"] == preference["feedback"]["server_context"] == 4096, "server context contract drift")
    require(cfg["runtime"]["server_threads"] == preference["feedback"]["server_threads"] == 4, "server thread contract drift")
    sources = {str(config_path.relative_to(ROOT)): file_sha(config_path), cfg["campaign_config"]: file_sha(campaign_path), cfg["preference_config"]: file_sha(preference_path), cfg["pins_path"]: file_sha(pins_path)}
    for name in (Path(__file__).name, "village_001d_sprocket_feedback.py", "village_001_manifest.py", "babble_002_expand.py"):
        sources[f"scripts/{name}"] = file_sha(ROOT / "scripts" / name)
    cases, requests, seen = [], [], set()
    for case_index, selection in enumerate(cfg["cases"]):
        mid, sid = selection["member_id"], selection["scenario_id"]
        require(re.fullmatch(r"[TP][0-9]{2}", sid) is not None, "only existing T/P inputs are permitted")
        require((mid, sid) not in seen, "duplicate selected case")
        seen.add((mid, sid))
        member = members[mid]
        sft_path, feedback_path = source_path(selection["sft_result_path"]), source_path(selection["feedback_path"])
        require(file_sha(sft_path) == selection["sft_result_sha256"], "SFT source bytes changed")
        require(file_sha(feedback_path) == selection["feedback_sha256"], "historical feedback source bytes changed")
        sources[selection["sft_result_path"]] = selection["sft_result_sha256"]
        sources[selection["feedback_path"]] = selection["feedback_sha256"]
        sft = json.loads(sft_path.read_text())
        require(sft["member_id"] == mid and sft["member_contract_sha256"] == manifest["member_contract_sha256"], "SFT member identity mismatch")
        scenario_rows = sft["evaluation"]["responses"] + sft["member_probe_evaluation"]["responses"]
        matches = [row for row in scenario_rows if row["scenario_id"] == sid]
        require(len(matches) == 1, "selected source scenario not unique")
        scenario, packet = matches[0], packet_for(member["lineage"])
        packet_path = source_path(member['sources']['packet'])
        sources[member['sources']['packet']] = file_sha(packet_path)
        historical = [json.loads(line) for line in feedback_path.read_text().splitlines() if line.strip()]
        historical = sorted([r for r in historical if r["scenario_id"] == sid], key=lambda row: row["sprocket_worker"])
        require([r["sprocket_worker"] for r in historical] == [1, 2, 3], "historical worker set incomplete")
        require([r["request"]["seed"] for r in historical] == selection["worker_seeds"], "historical seed drift")
        case_id = f"{mid}-{sid}"
        cases.append({"case_id": case_id, "member_id": mid, "scenario_id": sid, "lineage": member["lineage"], "sibling": member["sibling"], "prompt": scenario["prompt"], "response": scenario["response"], "sft_result_sha256": selection["sft_result_sha256"], "parent_adapter_sha256": sft["adapter"]["sha256"]})
        for worker, old in enumerate(historical, 1):
            seed = (int(campaign["population"]["campaign_seed"]) + int(member["seed"]) + int(sid[1:]) * 101 + worker * 10007) % 2147483647
            original = {"model": "local", "messages": [{"role": "system", "content": feedback_system(member, packet)}, {"role": "user", "content": feedback_user(scenario)}], **cfg["sampling"], "seed": seed, "stream": False}
            require(old["request"] == original, "historical request is not reconstructible from frozen source")
            require(old["member_id"] == mid and old["parent_adapter_sha256"] == sft["adapter"]["sha256"], "historical feedback parent mismatch")
            arms = cfg["arms"] if (case_index * 3 + worker - 1) % 2 == 0 else list(reversed(cfg["arms"]))
            for arm in arms:
                payload = json.loads(json.dumps(original))
                if arm == "compact_schema_first":
                    payload["messages"][0]["content"] = compact_system(member, packet)
                # These are observational fields in pinned llama.cpp source. No
                # grammar, response_format, stop, sampler or cache override.
                payload.update({"verbose": True, "return_tokens": True})
                key = f"{len(requests) + 1:03d}-{case_id}-W{worker}-{arm}"
                requests.append({"key": key, "case_id": case_id, "member_id": mid, "scenario_id": sid, "worker": worker, "arm": arm, "seed": seed, "payload": payload, "historical_feedback_id": old["feedback_id"], "historical_raw_response": old["raw_response"], "historical_completion_tokens": old["completion_tokens"], "historical_finish_reason": old["finish_reason"]})
    require(len(requests) == 36, "generation cardinality mismatch")
    plan = {"schema": cfg["schema"], "stage": "sprocket-feedback-prompt-diagnostic", "sources": sources, "member_contract_sha256": manifest["member_contract_sha256"], "model": cfg["model"], "runtime": cfg["runtime"], "sampling": cfg["sampling"], "wire": cfg["wire"], "generation_limit": 36, "optimizer_steps": 0, "cloud_requests": 0, "final_suite_access": False, "cases": cases, "requests": requests}
    return cfg, plan


def prepare_output(out: Path, resume: Path | None, plan: dict) -> None:
    require(not out.is_symlink(), "output may not be a symlink")
    require(not out.exists() or (out.is_dir() and not any(out.iterdir())), "output must be empty")
    if resume:
        a, b = resume.resolve(), out.resolve()
        require(a != b and a not in b.parents and b not in a.parents, "resume and output must be disjoint")
        require(resume.is_dir() and not resume.is_symlink(), "resume must be a regular directory")
        for path in resume.rglob("*"):
            require(not path.is_symlink() and (path.is_file() or path.is_dir()), "resume contains an unsafe entry")
        require((resume / "plan.json").read_bytes() == canonical(plan), "resume plan/source identity changed")
        shutil.copytree(resume, out, dirs_exist_ok=True)
    else:
        out.mkdir(parents=True, exist_ok=True)
    write_json(out / "plan.json", plan)


def gguf_metadata(path: Path) -> dict:
    """Read metadata directly from the hash-verified GGUF; never execute its text."""
    scalar_formats = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
    with path.open('rb') as fh:
        def unpack(fmt):
            size = struct.calcsize('<' + fmt)
            data = fh.read(size)
            require(len(data) == size, 'truncated GGUF metadata')
            return struct.unpack('<' + fmt, data)[0]
        def string():
            size = unpack('Q')
            require(size <= 16 * 1024 * 1024, 'unbounded GGUF metadata string')
            data = fh.read(size)
            require(len(data) == size, 'truncated GGUF string')
            return data.decode('utf-8')
        def value(kind):
            if kind in scalar_formats:
                return unpack(scalar_formats[kind])
            if kind == 8:
                return string()
            if kind == 9:
                inner, count = unpack('I'), unpack('Q')
                require(count <= 1000000 and inner != 9, 'unsupported GGUF array')
                return [value(inner) for _ in range(count)]
            raise RuntimeError(f'unknown GGUF metadata type {kind}')
        require(fh.read(4) == b'GGUF', 'model is not GGUF')
        version, tensors, count = unpack('I'), unpack('Q'), unpack('Q')
        require(version in (2, 3) and count < 10000, 'unsupported GGUF metadata header')
        metadata, token_samples = {}, {}
        for _ in range(count):
            key, kind = string(), unpack('I')
            start = fh.tell()
            item = value(kind)
            end = fh.tell()
            require(key not in metadata, 'duplicate GGUF metadata key')
            if isinstance(item, list):
                fh.seek(start)
                raw_sha = digest(fh.read(end - start))
                fh.seek(end)
                metadata[key] = {'array_length': len(item), 'encoded_value_sha256': raw_sha}
                if key == 'tokenizer.ggml.tokens':
                    token_samples = {str(i): token for i, token in enumerate(item) if isinstance(token, str) and len(token) < 80 and ('<|' in token or token in ('<s>', '</s>', '<unk>'))}
            else:
                metadata[key] = item
        return {'version': version, 'tensor_count': tensors, 'metadata_end_offset': fh.tell(), 'metadata': metadata, 'control_token_strings_by_id': token_samples}


def verify_runtime(cache: Path, cfg: dict) -> tuple[Path, Path, dict]:
    require(not any(os.environ.get(k) for k in ('LD_PRELOAD', 'LD_AUDIT', 'LD_LIBRARY_PATH')), 'dynamic-loader environment overrides are outside the frozen diagnostic')
    require(not any(k.startswith('LLAMA_ARG_') for k in os.environ), 'LLAMA_ARG environment overrides are outside the frozen diagnostic')
    env = load_env(cache / 'paths.env')
    server, model = Path(env['LLAMA_SERVER']), Path(env['SPROCKET_MODEL'])
    pins = load_env(source_path(cfg['pins_path']))
    archive = cache / 'downloads' / pins['LLAMA_ARCHIVE']
    require(file_sha(model) == cfg['model']['sha256'], 'pinned Sprocket GGUF hash mismatch')
    require(file_sha(archive) == cfg['runtime']['archive_sha256'], 'pinned llama.cpp archive hash mismatch')
    checked = {}
    with tarfile.open(archive, 'r:gz') as tar:
        files = [entry for entry in tar.getmembers() if entry.isfile()]
        targets = [server, *sorted(server.parent.glob('*.so*'))]
        for path in targets:
            if not path.is_file():
                continue
            resolved = path.resolve()
            require(server.parent.resolve() in resolved.parents, 'runtime library escapes pinned extraction')
            matches = [entry for entry in files if Path(entry.name).name == resolved.name]
            require(len(matches) == 1, f'cannot identify runtime file in pinned archive: {path.name}')
            expected = digest(tar.extractfile(matches[0]).read())
            require(file_sha(path) == expected, f'runtime extraction differs from pinned archive: {path.name}')
            checked[path.name] = expected
    version_result = subprocess.run([str(server), '--version'], capture_output=True, text=True, check=True)
    version = version_result.stdout + version_result.stderr
    require(cfg['runtime']['commit'][:7] in version and '11852' in version, 'llama-server build identity mismatch')
    gguf = gguf_metadata(model)
    metadata = gguf['metadata']
    require(digest(metadata['tokenizer.chat_template'].encode()) == cfg['wire']['gguf_chat_template_sha256'], 'native GGUF template drift')
    for ident, token in cfg['wire']['expected_control_tokens'].items():
        require(gguf['control_token_strings_by_id'].get(ident) == token, 'native GGUF role-token identity drift')
    for name, expected in [('bos_token_id', 0), ('eos_token_id', 5), ('padding_token_id', 1)]:
        require(metadata.get('tokenizer.ggml.' + name) == expected, 'native GGUF special-token metadata drift')
    return server, model, {'server_version': version, 'runtime_archive_sha256': file_sha(archive), 'runtime_files_sha256': checked, 'model_sha256': file_sha(model), 'gguf': gguf}


class LocalHTTP:
    def __init__(self, port: int, out: Path):
        self.base, self.out = f'http://127.0.0.1:{port}', out

    def request(self, label: str, endpoint: str, payload=None, *, generation=False):
        base = self.out / 'http' / label
        request_path, response_path = base.with_suffix('.request.json'), base.with_suffix('.response.json')
        record = {'endpoint': endpoint, 'payload': payload, 'generation': generation}
        if request_path.exists():
            require(request_path.read_bytes() == canonical(record), 'archived local request differs from plan')
            if response_path.exists():
                return archived_exchange(self.out, label, endpoint, payload, generation=generation)
            require(not generation, 'generation request has no durable response; outcome is ambiguous, do not reissue it')
        require(not response_path.exists(), 'orphaned local response has no request; do not reissue it')
        write_json(request_path, record)
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
        request = urllib.request.Request(self.base + endpoint, data=data, headers={'Content-Type': 'application/json'}, method='GET' if payload is None else 'POST')
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=240) as response:
            raw, status = response.read(), response.status
        immutable(response_path, raw)
        write_json(base.with_suffix('.exchange.json'), {'http_status': status, 'elapsed_seconds': time.monotonic() - started, 'response_sha256': digest(raw)})
        return json.loads(raw)


def archived_exchange(out: Path, label: str, endpoint: str, payload=None, *, generation=False) -> dict:
    base = out / 'http' / label
    request_path, response_path = base.with_suffix('.request.json'), base.with_suffix('.response.json')
    record = {'endpoint': endpoint, 'payload': payload, 'generation': generation}
    require(request_path.is_file() and response_path.is_file(), 'offline local exchange is incomplete')
    require(request_path.read_bytes() == canonical(record), 'archived local request differs from plan')
    raw = response_path.read_bytes()
    exchange_path = base.with_suffix('.exchange.json')
    # The response is committed before the timing receipt. A crash in between
    # leaves a recoverable completed response, never a reason to call again.
    if exchange_path.exists():
        exchange = json.loads(exchange_path.read_text())
        require(exchange.get('http_status') == 200 and exchange.get('response_sha256') == digest(raw), 'local response receipt/hash mismatch')
    value = json.loads(raw)
    require(isinstance(value, dict), 'local response must be a JSON object')
    return value


def validate_wire(request: dict, rendered: str, tokenization: dict, counted: dict, context: int) -> list[int]:
    require(isinstance(rendered, str) and rendered, 'native chat rendering missing')
    messages = request['payload']['messages']
    require(messages[0]['content'].strip() in rendered and messages[1]['content'].strip() in rendered, 'native template lost system or user content')
    require([m['role'] for m in messages] == ['system', 'user'], 'unexpected native chat roles')
    expected_rendering = '<|system|>' + messages[0]['content'] + '<|end|><|user|>' + messages[1]['content'] + '<|end|><|assistant|>'
    require(rendered == expected_rendering, 'native rendering differs from exact pinned GGUF template')
    require(rendered.find(messages[0]['content'].strip()) < rendered.find(messages[1]['content'].strip()), 'native template reversed system/user order')
    ids = tokenization.get('tokens')
    require(isinstance(ids, list) and ids and all(type(x) is int and x >= 0 for x in ids), 'native input token IDs missing')
    require((ids[:1] == [2] or ids[:2] == [0, 2]) and ids[-1:] == [4] and ids.count(2) == ids.count(3) == ids.count(4) == 1 and ids.count(5) == 2, 'native role-token boundaries differ from pinned template')
    require(counted.get('input_tokens') == len(ids), 'native chat/tokenizer input counts disagree')
    require(len(ids) + 160 <= context, 'diagnostic prompt exceeds the effective model context')
    return ids


def wire_record(request: dict, template: dict, tokenization: dict, counted: dict, context: int, template_sha: str) -> dict:
    ids = validate_wire(request, template['prompt'], tokenization, counted, context)
    return {'rendered_prompt': template['prompt'], 'input_token_ids': ids, 'input_token_ids_sha256': digest(canonical(ids)), 'actual_template_sha256': template_sha, 'system_content_retained': True, 'user_content_retained': True, 'input_count_matches_native_chat': True, 'effective_context': context}


def reconstruct_wire(out: Path, request: dict, cfg: dict) -> dict:
    key = request['key']
    template = archived_exchange(out, key + '-template', '/apply-template', request['payload'])
    tokenization = archived_exchange(out, key + '-tokenize', '/tokenize', {'content': template['prompt'], 'add_special': True, 'parse_special': True})
    counted = archived_exchange(out, key + '-count', '/v1/chat/completions/input_tokens', request['payload'])
    wire = wire_record(request, template, tokenization, counted, cfg['wire']['expected_training_context'], cfg['wire']['gguf_chat_template_sha256'])
    require((out / 'wire' / f'{key}.json').read_bytes() == canonical(wire), 'offline native wire reconstruction mismatch')
    return wire


def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.casefold())


def lexical_metrics(raw: str, case: dict, instruction: str) -> dict:
    tokens = words(raw)
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    counts = Counter(trigrams)
    source_words = (set(words(case['prompt'])) | set(words(case['response']))) - STOPWORDS
    output_words = set(tokens) - STOPWORDS
    echo5 = list(zip(*[tokens[i:] for i in range(5)]))
    system_words = words(instruction)
    instruction5 = set(zip(*[system_words[i:] for i in range(5)]))
    return {'descriptive_proxy_only': True, 'source_lexical_anchors': sorted(output_words & source_words), 'unique_content_word_count': len(output_words), 'duplicate_trigram_fraction': sum(n - 1 for n in counts.values()) / len(trigrams) if trigrams else 0, 'instruction_fivegram_echo_fraction': sum(x in instruction5 for x in echo5) / len(echo5) if echo5 else 0}


def analyze_response(request: dict, case: dict, wire: dict, response: dict, actual_prompt_tokens: dict) -> dict:
    require(isinstance(response.get('__verbose'), dict), 'native verbose receipt missing')
    verbose, choice = response['__verbose'], response['choices'][0]
    raw = choice['message'].get('content')
    raw = '' if raw is None else raw
    require(isinstance(raw, str), 'completion content is not text')
    ids = verbose.get('tokens')
    require(isinstance(ids, list) and all(type(x) is int and x >= 0 for x in ids), 'native output token IDs missing')
    count = response['usage']['completion_tokens']
    require(type(count) is int and count == len(ids) and count <= 160, 'native completion token custody/count mismatch')
    require(response['usage']['prompt_tokens'] == len(wire['input_token_ids']), 'actual native prompt count differs from wire preflight')
    require(verbose.get('truncated') is False, 'native context truncated during diagnostic')
    require(actual_prompt_tokens.get('tokens') == wire['input_token_ids'], 'actual native prompt token IDs differ from preflight')
    settings = verbose.get('generation_settings') or {}
    require(settings.get('seed') == request['seed'] and settings.get('n_predict') == 160, 'native seed/token budget changed')
    require(abs(float(settings.get('temperature', -1)) - 1.05) < 0.00001 and abs(float(settings.get('top_p', -1)) - 0.97) < 0.00001, 'native sampling settings changed')
    parsed = parse_feedback(raw)
    try:
        strict_object = json.loads(raw)
        strict_valid = isinstance(strict_object, dict) and parsed is not None
    except ValueError:
        strict_valid = False
    return {'key': request['key'], 'case_id': request['case_id'], 'worker': request['worker'], 'arm': request['arm'], 'seed': request['seed'], 'request_sha256': digest(canonical(request['payload'])), 'native_rendered_prompt': wire['rendered_prompt'], 'input_token_ids': wire['input_token_ids'], 'native_detokenized_prompt': verbose.get('prompt'), 'native_prompt_token_ids_verified': True, 'generated_token_ids': ids, 'raw_response': raw, 'completion_tokens': count, 'finish_reason': choice['finish_reason'], 'native_stop_type': verbose.get('stop_type'), 'generation_settings': verbose.get('generation_settings'), 'legacy_schema_valid': parsed is not None, 'strict_json_schema_valid': strict_valid, 'parsed': parsed, 'empty_response': not raw.strip(), 'historical_raw_exact_match': raw.strip() == request['historical_raw_response'], 'historical_tokens_match': count == request['historical_completion_tokens'], 'historical_finish_match': choice['finish_reason'] == request['historical_finish_reason'], 'lexical_metrics': lexical_metrics(raw, case, request['payload']['messages'][0]['content']), 'semantic_grounding_score': None, 'semantic_grounding_status': 'pending_blinded_review'}


def summarize(out: Path, plan: dict, rows: list[dict]) -> dict:
    require([r['key'] for r in rows] == [r['key'] for r in plan['requests']], 'diagnostic rows incomplete or reordered')
    summary = {'schema': 'village-001f-sprocket-diagnostic-summary-v1', 'plan_sha256': file_sha(out / 'plan.json'), 'local_generations': len(rows), 'optimizer_steps': 0, 'cloud_requests': 0, 'final_suite_access': False, 'arms': {}, 'semantic_grounding_status': 'pending_blinded_review', 'interpretation': 'A fixed exploratory prompt pilot on existing T/P inputs. Syntax validity alone does not establish useful feedback or a causal contribution to learning. Native cache defaults and source seeds are retained; historical byte equality is measured, not assumed.'}
    for arm in ['original', 'compact_schema_first']:
        selected = [row for row in rows if row['arm'] == arm]
        summary['arms'][arm] = {'count': len(selected), 'legacy_schema_valid': sum(r['legacy_schema_valid'] for r in selected), 'strict_json_schema_valid': sum(r['strict_json_schema_valid'] for r in selected), 'nonempty': sum(not r['empty_response'] for r in selected), 'finish_reasons': dict(Counter(r['finish_reason'] for r in selected)), 'completion_tokens_equal_160': sum(r['completion_tokens'] == 160 for r in selected), 'mean_completion_tokens': statistics.mean(r['completion_tokens'] for r in selected), 'mean_duplicate_trigram_fraction': statistics.mean(r['lexical_metrics']['duplicate_trigram_fraction'] for r in selected), 'historical_raw_exact_matches': sum(r['historical_raw_exact_match'] for r in selected) if arm == 'original' else None}
    cases = {case['case_id']: case for case in plan['cases']}
    review, key = [], {}
    for row in rows:
        rid = 'B-' + digest(row['key'].encode())[:14]
        case = cases[row['case_id']]
        key[rid] = {'key': row['key'], 'arm': row['arm'], 'case_id': row['case_id'], 'worker': row['worker']}
        review.append({'review_id': rid, 'row_sha256': digest(canonical(row)), 'situation': case['prompt'], 'villager_response': case['response'], 'feedback': row['raw_response'], 'response_grounding_0_to_2': None, 'scenario_grounding_0_to_2': None, 'corrective_usefulness_0_to_2': None, 'unsupported_claims': None, 'source_quote': None, 'reason': None})
    review.sort(key=lambda item: item['review_id'])
    write_json(out / 'summary.json', summary)
    write_json(out / 'blinded-grounding-review.json', {'schema': 'village-001f-blinded-grounding-review-v1', 'ratings_are_unfilled': True, 'rubric': {'response_grounding': '0: no identifiable response behavior; 1: refers to an actual behavior but vague or mixed with invention; 2: concrete faithful observation supported by an exact source quote.', 'scenario_grounding': '0: ignores or contradicts situation; 1: weakly relevant; 2: specifically fits the stated situation without invented facts.', 'corrective_usefulness': '0: no actionable relevant change; 1: plausible but underspecified; 2: specific change tied to a visible response failure.', 'unsupported_claims': 'True when feedback asserts events or behavior absent from the situation/response. No syntax-only pass or lexical-overlap shortcut.'}, 'items': review})
    write_json(out / 'blinded-grounding-key.json', key)
    inventory = {p.relative_to(out).as_posix(): {'sha256': file_sha(p), 'bytes': p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'evidence-hashes.json' and not p.name.startswith('.pending-')}
    write_json(out / 'evidence-hashes.json', {'schema_version': 1, 'files': inventory})
    return summary


def execute(out: Path, cfg: dict, plan: dict, cache: Path, port: int, *, offline: bool) -> dict:
    cases = {row['case_id']: row for row in plan['cases']}
    rows = []
    # A completed resume is a reconstruction, even if --execute was supplied.
    if all((out / 'rows' / f"{request['key']}.json").is_file() for request in plan['requests']):
        offline = True
    if offline:
        for request in plan['requests']:
            key = request['key']
            wire = reconstruct_wire(out, request, cfg)
            response = archived_exchange(out, key + '-generate', '/v1/chat/completions', request['payload'], generation=True)
            native_prompt = (response.get('__verbose') or {}).get('prompt')
            require(isinstance(native_prompt, str) and native_prompt, 'actual native detokenized prompt missing')
            actual_prompt_tokens = archived_exchange(out, key + '-actual-tokenize', '/tokenize', {'content': native_prompt, 'add_special': False, 'parse_special': True})
            row = analyze_response(request, cases[request['case_id']], wire, response, actual_prompt_tokens)
            require((out / 'rows' / f'{key}.json').read_bytes() == canonical(row), 'offline diagnostic reconstruction mismatch')
            rows.append(row)
        return summarize(out, plan, rows)
    server, model, runtime = verify_runtime(cache, cfg)
    with socket.socket() as probe:
        require(probe.connect_ex(('127.0.0.1', port)) != 0, 'diagnostic port is already occupied')
    epoch = 1
    while (out / f'runtime-{epoch:03d}.json').exists():
        epoch += 1
    log_path = out / f'server-{epoch:03d}.log'
    command = [str(server), '-m', str(model), '--host', '127.0.0.1', '--port', str(port), '-t', str(cfg['runtime']['server_threads']), '-c', str(cfg['runtime']['server_context_requested']), '--log-prompts-dir', str(out / f'native-prompts-{epoch:03d}')]
    http = LocalHTTP(port, out)
    with log_path.open('x') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                require(proc.poll() is None, 'pinned llama-server exited before health')
                try:
                    with urllib.request.urlopen(http.base + '/health', timeout=1) as r:
                        if r.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(0.2)
            else:
                raise RuntimeError('pinned llama-server startup timed out')
            props = http.request(f'runtime-{epoch:03d}-props', '/props')
            require(Path(props.get('model_path', '')).resolve() == model.resolve(), 'server model path mismatch')
            require('11852' in props.get('build_info', '') and cfg['runtime']['commit'][:7] in props.get('build_info', ''), 'server build_info mismatch')
            metadata = runtime['gguf']['metadata']
            require(digest(props['chat_template'].encode()) == cfg['wire']['gguf_chat_template_sha256'], 'server active template differs from pinned GGUF template')
            context = min(int(props['default_generation_settings']['n_ctx']), int(metadata[f"{metadata['general.architecture']}.context_length"]))
            require(context == 2048, 'effective Sprocket context changed')
            runtime.update({'command': command, 'pid': proc.pid, 'props': props, 'effective_context': context, 'actual_template_sha256': digest(props['chat_template'].encode())})
            write_json(out / f'runtime-{epoch:03d}.json', runtime)
            for request in plan['requests']:
                key = request['key']
                template = http.request(key + '-template', '/apply-template', request['payload'])
                tokenization = http.request(key + '-tokenize', '/tokenize', {'content': template['prompt'], 'add_special': True, 'parse_special': True})
                counted = http.request(key + '-count', '/v1/chat/completions/input_tokens', request['payload'])
                wire = wire_record(request, template, tokenization, counted, context, runtime['actual_template_sha256'])
                write_json(out / 'wire' / f'{key}.json', wire)
            # Every prompt is qualified before the first stochastic generation.
            for request in plan['requests']:
                key = request['key']
                wire = json.loads((out / 'wire' / f'{key}.json').read_text())
                print(json.dumps({'diagnostic': key, 'phase': 'generate_or_replay'}), flush=True)
                response = http.request(key + '-generate', '/v1/chat/completions', request['payload'], generation=True)
                native_prompt = (response.get('__verbose') or {}).get('prompt')
                require(isinstance(native_prompt, str) and native_prompt, 'actual native detokenized prompt missing')
                actual_prompt_tokens = http.request(key + '-actual-tokenize', '/tokenize', {'content': native_prompt, 'add_special': False, 'parse_special': True})
                row = analyze_response(request, cases[request['case_id']], wire, response, actual_prompt_tokens)
                write_json(out / 'rows' / f'{key}.json', row)
                rows.append(row)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(5)
    return summarize(out, plan, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--cache', type=Path, default=Path(os.environ.get('NEURO_CACHE', ROOT / '.cache/neurounner')))
    parser.add_argument('--port', type=int, default=18950)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    cfg, plan = build_plan(args.config)
    prepare_output(args.out, args.resume, plan)
    if not args.execute and not args.offline:
        print(json.dumps({'status': 'PLAN_ONLY', 'planned_generations': 36, 'executed_generations': 0, 'cloud_requests': 0, 'optimizer_steps': 0, 'final_suite_access': False}))
        return
    result = execute(args.out, cfg, plan, args.cache, args.port, offline=args.offline)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
