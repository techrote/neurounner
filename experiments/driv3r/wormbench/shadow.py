"""Source-fenced CPU-only Q4 and SLMworm shadow inference.

The host-approved candidate IDs come from the original verified 36+36 Q8
native worm decisions; this job NEVER changes World state or issues a PAINT.
d1 Q4 retains System One; normal SLMs receive a separate generative prompt.
"""
from __future__ import annotations
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import tempfile
import time
import urllib.request

from experiments.driv3r.backend import (
    PINS, ResidentDecisionBackend, parse_choice_response, sha256_file,
    _install_bundle,
)
from .reference import canonical, EXPECTED_DIGEST, SOURCE_SHA

MODELS = {
    "d1-q4": {"kind":"systemone", "model":PINS["optional_q4"]},
    "lfm350": {"kind":"chat", "repository":"LiquidAI/LFM2.5-350M-GGUF",
               "filename":"LFM2.5-350M-Q4_K_M.gguf",
               "expected_sha256":"7e6f72643caafc9a68256686638c4d7916f2cec76d1df478d4c3ddcd95a6aed4"},
    "smol135": {"kind":"chat", "repository":"QuantFactory/SmolLM2-135M-Instruct-GGUF",
                "filename":"SmolLM2-135M-Instruct.Q4_K_M.gguf"},
    "smol135-grammar": {"kind":"chat_grammar", "repository":"QuantFactory/SmolLM2-135M-Instruct-GGUF",
                "filename":"SmolLM2-135M-Instruct.Q4_K_M.gguf"},
    "sprocket500": {"kind":"chat", "repository":"kandivault/sprocket-500m",
                    "filename":"sprocket-500m-q4_k_m.gguf"},
    "sprocket500-grammar":{"kind":"chat_grammar",
               "repository":"kandivault/sprocket-500m",
               "filename":"sprocket-500m-q4_k_m.gguf"},
}
MAX_ROWS = 72
REQUEST_DEADLINE = 65
MAX_RESPONSE_BYTES = 100000
ROOT = Path(__file__).resolve().parents[3]

def verify_request(request, model, source_sha, rows_doc):
    if set(request.get("eligible_models", [])) != set(MODELS):
        raise ValueError("Model list changed from owner-bounded request")
    if (request.get("schema") != "driv3r-slmworm-shadow/1" or
        request.get("allow_training") is not False or
        request.get("allow_native") is not False or
        request.get("allow_paid_api") is not False or
        request.get("max_shadow_states_per_model") != MAX_ROWS or
        request.get("original_q8_artifact_digest") != EXPECTED_DIGEST or
        request.get("owner_authorization") != "Try d1 Q4 performance and bring in SLMworms" or
        model not in MODELS or not source_sha):
        raise ValueError("Missing or different bounded owner-approved shadow request")
    if (rows_doc.get("digest") != EXPECTED_DIGEST or
        rows_doc.get("source_sha") != SOURCE_SHA or
        len(rows_doc.get("rows", [])) != MAX_ROWS or
        hashlib.sha256(canonical(rows_doc["rows"])).hexdigest() != EXPECTED_DIGEST):
        raise ValueError("Original native Q8 snapshot custody mismatch")
    return rows_doc["rows"]

def acquire_runtime(cache):
    p = PINS["runtime"]
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / p["archive"]
    if not archive.is_file() or archive.stat().st_size != p["archive_bytes"] or sha256_file(archive) != p["archive_sha256"]:
        archive.unlink(missing_ok=True)
        url = f'https://github.com/{p["repository"]}/releases/download/{p["tag"]}/{p["archive"]}'
        temp = archive.with_suffix(archive.suffix+".part")
        try:
            with urllib.request.urlopen(url, timeout=40) as incoming, temp.open("wb") as out:
                while block := incoming.read(1024*1024):
                    out.write(block)
            if temp.stat().st_size != p["archive_bytes"] or sha256_file(temp) != p["archive_sha256"]:
                raise ValueError("Pinned llama.cpp binary archive SHA/bytes mismatch")
            temp.replace(archive)
        finally:
            temp.unlink(missing_ok=True)
    bundle, reused = _install_bundle(archive, cache, p)
    binary = bundle / "llama-server"
    if not binary.is_file() or sha256_file(binary) != p["binary_sha256"]:
        raise ValueError("Pinned llama-server binary bytes mismatch")
    return binary, {"revision":p["revision"], "binary_sha256":p["binary_sha256"],"reused":reused}

def acquire_model(cache, name):
    pin = MODELS[name]
    if name == "d1-q4":
        desc = PINS["optional_q4"]
        repo, revision, filename = desc["repository"], desc["revision"], desc["filename"]
        expected = desc["sha256"]
        expected_bytes = desc["bytes"]
    else:
        repo, filename = pin["repository"], pin["filename"]
        from huggingface_hub import HfApi
        # Resolve 'main' ONCE, before download. Download exclusively from the
        # immutable revision; retain revision and actual SHA even on failures.
        info = HfApi().model_info(repo, files_metadata=True)
        revision = info.sha
        matches = [s for s in info.siblings if s.rfilename == filename]
        if len(matches) != 1:
            raise ValueError("Required exact SLM GGUF filename absent: " + filename)
        sibling = matches[0]
        expected = pin.get("expected_sha256") or (sibling.lfs or {}).get("sha256")
        expected_bytes = sibling.size
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(repo_id=repo, revision=revision,
                                filename=filename, local_dir=cache / name))
    actual = sha256_file(path)
    if expected and actual != expected:
        raise ValueError("Pinned model bytes not equal repository/external published SHA")
    if expected_bytes and path.stat().st_size != expected_bytes:
        raise ValueError("Pinned model size mismatch")
    return path, {"repository":repo, "revision":revision, "filename":filename,
                  "sha256":actual, "bytes":path.stat().st_size,
                  "quantization":"Q4_K_M","content_verified":bool(expected)}

def summarize(rows, results, identity, runtime, startup_seconds, total_seconds):
    good = [x for x in results if x["status"] == "valid"]
    times = [x["latency_seconds"] for x in results if x.get("latency_seconds") is not None]
    q8_times = [r["q8_latency_seconds"] for r in rows]
    by_arm = {}
    for arm in ("free", "soft"):
        pair = [x for x in results if x["mode"] == arm]
        valid = [x for x in pair if x["status"] == "valid"]
        by_arm[arm] = {"states":len(pair), "valid_choices":len(valid),
            "identical_q8_choices":sum(x["choice"] == x["q8_choice"] for x in valid),
            "latency_median_seconds":statistics.median(x["latency_seconds"] for x in pair if x.get("latency_seconds") is not None) if pair else None}
    tvd = [r.get("q8_tvd") for r in results if r.get("q8_tvd") is not None]
    return {
        "schema":"driv3r-slmworm-shadow-result/1",
        "not_native":True, "not_artistic_preference":True,
        "original_q8_artifact_digest":EXPECTED_DIGEST, "original_source_sha":SOURCE_SHA,
        "source_commit":os.environ.get("GITHUB_SHA"), "model":identity,
        "runtime":runtime, "sample_count":len(results),
        "valid_choice_count":len(good), "invalid_or_failed_count":len(results)-len(good),
        "q8_original_median_decision_seconds":statistics.median(q8_times),
        "shadow_median_decision_seconds":statistics.median(times) if times else None,
        "shadow_mean_decision_seconds":statistics.mean(times) if times else None,
        "mean_q8_distribution_tvd":statistics.mean(tvd) if tvd else None,
        "model_load_seconds":startup_seconds,"total_wall_seconds":total_seconds,
        "arms":by_arm,
    }

class Q4Backend(ResidentDecisionBackend):
    """Experimental alternate fixed weight SHA; frozen default Q8 PINS are intact."""
    def __init__(self, binary, model, evidence_dir, **kwargs):
        super().__init__(binary, model, evidence_dir, **kwargs)
        # Explicit separate Q4 pin, checked by ResidentDecisionBackend.start().
        self.model_identity = dict(PINS["model"], **PINS["optional_q4"])
        self.model_identity["quantization"] = "Q4_K_M"

def q4_shadow(binary, model, rows, out):
    results = []
    startup = time.monotonic()
    backend = Q4Backend(binary, model, out/"resident", threads=2,
                        ctx_size=4096, batch_size=2048)
    with backend:
        startup_elapsed = time.monotonic()-startup
        for index, row in enumerate(rows):
            started = time.monotonic()
            reply = backend.decide(canonical(row["state"]).decode(),
                                   row["questions"], timeout_seconds=REQUEST_DEADLINE)
            data = {"mode":row["mode"],"turn":row["turn"],
                    "q8_choice":row["q8_choice"],"status":"error",
                    "latency_seconds":round(time.monotonic()-started,6)}
            if reply["status"] == "ok":
                try:
                    choice = parse_choice_response(reply["raw_response"],"worm_move",
                                                   row["candidate_ids"])
                    data.update({"choice":choice["candidate_id"],"status":"valid",
                                 "probabilities":choice["probabilities"],
                                 "q8_tvd":round(sum(abs(choice["probabilities"][k]-
                                      row["q8_probabilities"][k])
                                      for k in row["candidate_ids"])/2,8)})
                except Exception as exc:
                    data["error"] = "Response validation: "+str(exc)[:400]
            else:
                data["error"] = str(reply.get("error",reply["status"]))[:400]
            results.append(data)
            (out/"decisions.jsonl").open("a").write(json.dumps(data,sort_keys=True)+"\n")
        return results, startup_elapsed

def choose_chat_prompt(row):
    q = row["questions"]["worm_move"]
    description = "\n".join(f"{key}: {q['criteria'][key]}" for key in row["candidate_ids"])
    state = row["state"]
    compact = {key:state[key] for key in (
        "position","heading","recent_positions","current_impulse","worm_turn",
        "candidate_hints","scent_rule","mission") if key in state}
    return ("You control a tiny eccentric drawing worm, not a pathfinder. "
            "Choose ONE offered legal action ID to paint an interesting continuing line, "
            "with occasional odd bends. The scent is optional. Backtrack/relocate "
            "raise the brush; STOP ends this drawing.\n"
            "Return ONLY one exact offered action ID, with no explanation.\n"
            "State: "+json.dumps(compact,separators=(",",":"),sort_keys=True)+
            "\nAvailable actions:\n"+description)

def parse_chat_choice(text, ids):
    if type(text) is not str or len(text) > 256:
        return None
    stripped = text.strip()
    if stripped in ids:
        return stripped
    try:
        parsed = json.loads(stripped)
        if type(parsed) is str and parsed in ids:
            return parsed
    except (ValueError,TypeError):
        pass
    return None

def legal_id_grammar(ids):
    """GGUF grammar restricts decoding to one EXACT source-offered legal ID."""
    if (not isinstance(ids,list) or len(ids)<2 or len(ids)>32 or
        any(not isinstance(x,str) or not x or
            any(not (c.isalnum() and c.isascii() or c=="_") for c in x) for x in ids) or
        len(set(ids)) != len(ids)):
        raise ValueError("Invalid source-offered legal catalog for constrained decoding")
    return "root ::= " + " | ".join(json.dumps(x) for x in ids)

def _http_request(port, key, method, path, data=None, timeout=15):
    conn = http.client.HTTPConnection("127.0.0.1",port,timeout=timeout)
    try:
        conn.request(method,path,body=json.dumps(data).encode() if data is not None else None,
                     headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
        r = conn.getresponse()
        raw = r.read(MAX_RESPONSE_BYTES+1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("Response exceeded 100k bytes")
        if r.status != 200:
            raise ValueError("HTTP "+str(r.status)+" "+raw[:200].decode(errors="replace"))
        return json.loads(raw)
    finally:
        conn.close()

def slm_shadow(binary, model, rows, out, *, grammar_constrained=False):
    results = []
    with socket.socket() as s:
        s.bind(("127.0.0.1",0))
        port = s.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="wormbench-auth-") as tmp:
        key = hashlib.sha256(os.urandom(32)).hexdigest()
        keyfile = Path(tmp)/"secret"
        keyfile.write_text(key+"\n")
        keyfile.chmod(0o600)
        command = [str(binary),"-m",str(model),"--host","127.0.0.1",
                   "--port",str(port),"--api-key-file",str(keyfile),
                   "-t","2","-tb","2","-ngl","0","-np","1",
                   "-c","4096","-b","2048","-ub","2048",
                   "--no-mmproj"]
        clean_env = {"PATH":os.getenv("PATH","/usr/bin:/bin"),
                     "LANG":"C.UTF-8","LC_ALL":"C.UTF-8","OMP_NUM_THREADS":"2",
                     "LD_LIBRARY_PATH":str(binary.parent)}
        started = time.monotonic()
        with (out/"server.log").open("wb") as log:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,
                                  stderr=subprocess.STDOUT,start_new_session=True,
                                  cwd=binary.parent,env=clean_env)
            try:
                while time.monotonic()-started < 150:
                    if proc.poll() is not None:
                        raise RuntimeError("Generative model exited during startup")
                    try:
                        _http_request(port,key,"GET","/health",timeout=2)
                        break
                    except (OSError,ValueError):
                        time.sleep(0.25)
                else:
                    raise TimeoutError("Generative model startup deadline exceeded")
                startup_elapsed=time.monotonic()-started
                for row in rows:
                    prompt=choose_chat_prompt(row)
                    begin=time.monotonic()
                    d={"mode":row["mode"],"turn":row["turn"],
                       "q8_choice":row["q8_choice"],"status":"error"}
                    try:
                        payload = {"messages":[{"role":"system","content":
                            "You are a quirky drawing worm choosing from a finite list."},
                            {"role":"user","content":prompt}],
                           "temperature":0,"max_tokens":24,"seed":570,
                           "stream":False}
                        if grammar_constrained:
                            payload["grammar"] = legal_id_grammar(row["candidate_ids"])
                        response=_http_request(port,key,"POST","/v1/chat/completions",
                                               payload, timeout=REQUEST_DEADLINE)
                        msg=response["choices"][0]["message"]["content"]
                        choice=parse_chat_choice(msg,row["candidate_ids"])
                        d.update({"status":"valid" if choice else "invalid",
                                  "choice":choice,"raw_text":str(msg)[:256],
                                  "usage":response.get("usage")})
                    except Exception as exc:
                        d["error"]=f"{type(exc).__name__}: {exc}"[:400]
                    d["latency_seconds"]=round(time.monotonic()-begin,6)
                    results.append(d)
                    with (out/"decisions.jsonl").open("a") as f:
                        f.write(json.dumps(d,sort_keys=True)+"\n")
                return results,startup_elapsed
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:
                        proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL)
                        proc.wait(timeout=10)
                (out/"process-exit.json").write_text(json.dumps(
                    {"pid":proc.pid,"exit_code":proc.returncode,"reaped":proc.poll() is not None})+"\n")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",choices=MODELS,required=True)
    parser.add_argument("--snapshots",type=Path,required=True)
    parser.add_argument("--request",type=Path,required=True)
    parser.add_argument("--source-sha",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite original benchmark evidence")
    rows_doc=json.loads(args.snapshots.read_text())
    request=json.loads(args.request.read_text())
    rows=verify_request(request,args.model,args.source_sha,rows_doc)
    if args.source_sha != subprocess.check_output(
       ["git","rev-parse","HEAD"],text=True).strip():
        raise ValueError("Checked out Git source differs from requested exact HEAD")
    args.output.mkdir(parents=True)
    started=time.monotonic()
    report={"status":"incomplete","source_sha":args.source_sha,"model_label":args.model,
            "original_q8_digest":EXPECTED_DIGEST}
    try:
        cache=Path(os.environ.get("RUNNER_TEMP","/tmp"))/"wormbench-models"
        binary,runtime=acquire_runtime(cache)
        model,identity=acquire_model(cache,args.model)
        (args.output/"model-identity.json").write_text(json.dumps(identity,indent=2)+"\n")
        if args.model=="d1-q4":
            result,startup=q4_shadow(binary,model,rows,args.output)
        else:
            result,startup=slm_shadow(binary,model,rows,args.output,
                                       grammar_constrained=MODELS[args.model]["kind"]=="chat_grammar")
        report=summarize(rows,result,identity,runtime,startup,time.monotonic()-started)
        report["output_format_policy"]=("gbnf_exact_legal_id" if MODELS[args.model]["kind"]=="chat_grammar"
                                         else "strict_literal_legal_id_only")
        report["status"]="complete_shadow_inference"
        return_code=0
    except Exception as exc:
        report.update({"status":"failed","error":f"{type(exc).__name__}: {exc}",
                       "elapsed_seconds":round(time.monotonic()-started,3)})
        return_code=1
    finally:
        (args.output/"report.json").write_text(json.dumps(
            report,sort_keys=True,indent=2,allow_nan=False)+"\n")
        print(json.dumps(report,sort_keys=True),flush=True)
    return return_code

if __name__=="__main__":
    raise SystemExit(main())
