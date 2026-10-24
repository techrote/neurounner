#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import nvidia_babble_compile as nv

SPROCKET = {
    "id": "kandivault/sprocket-500m",
    "artifact": "sprocket-500m-chat-q4_k_m.gguf",
    "revision": "eb157d4ddddc4619e661d4a2426bb2dd000fc668",
    "sha256": "edc2ac323a2513fa6f8e3e54adf120243de0f7a6b8cc11b349b3e6ad61494a09",
}
GUIDE_MODEL = {
    "id": "qwen35-4b",
    "label": "Qwen3.5-4B",
    "artifact": "Qwen3.5-4B-Q4_K_M.gguf",
    "sha256": "2ebe6eb67cb92adeffc7073c48dd4e3baf79465ac650793dc751cad2b29cf219",
}
SOURCE = {
    "guide": {
        "run_id": 40113280906,
        "head_sha": "f08a819f00f15ef2ee0264b0d6159fb5f0afe16b",
        "artifact_id": 12566539944,
        "artifact_name": "babble001-guides",
        "artifact_digest": "sha256:eceb2969f21bcd1c92550d7f7db18297b15b52797838f1da312ef385ee310b68",
    },
    "workers": {
        "run_id": 40113646404,
        "head_sha": "0d85aff7cab0dca12fb1c5e2bf2b4eb0a3df79d6",
        "artifacts": [
            {"worker": 0, "artifact_id": 12566918802, "artifact_name": "babble001-resume-worker-0", "artifact_digest": "sha256:bfe4c1c9539425d96e5ba845fdc271b5a7253812c110ec2a58b7c52bcfee8a42"},
            {"worker": 1, "artifact_id": 12567003444, "artifact_name": "babble001-resume-worker-1", "artifact_digest": "sha256:70553ae732a1ce6f3d45e7502919eff13e99bb8f8ff00baaa901dcce3e1c432b"},
            {"worker": 2, "artifact_id": 12567821493, "artifact_name": "babble001-resume-worker-2", "artifact_digest": "sha256:67ad0910a847b363046d30baad3819e3c54319a1ff56255a7eca52a78aa98b3a"},
            {"worker": 3, "artifact_id": 12567184110, "artifact_name": "babble001-resume-worker-3", "artifact_digest": "sha256:b3234c6d358cd0d366336b629ca8f76db7eaba255e0519b3308187b21bb27e89"},
        ],
    },
    "compiler": {
        "run_id": 40116552604,
        "head_sha": "e2ee2f9fb7f126b8745954a5240ee7695dea646c",
        "artifact_id": 12569400882,
        "artifact_name": "babble001-nvidia-compiled-resume",
        "artifact_digest": "sha256:e7dbb3ef00d97a17957d8405e4cfe5b7504e5daf7e2b2946a998d58a3daeca17",
    },
}

def find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} below {root}, got {hits}")
    return hits[0]

def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")

def guide_prompt(batch: int) -> str:
    return f"""Generate exactly four highly differentiated prompt-guide packets for a tiny chaotic model called Sprocket.
These guides will make Sprocket babble raw personality/temperament material for later dataset compilation.

Batch {batch}. Favor strange embodied temperaments over generic assistant personas.
Vary axes aggressively: cautious/reckless, territorial/social, ritualistic/opportunistic, patient/impulsive,
paranoid/trusting, stubborn/flighty, acquisitive/avoidant, curious/apathetic, submissive/dominant.

For each guide:
- system_prompt: short role pressure for Sprocket; do not ask for polished JSON.
- attitude_guide: emotional/temperamental cadence and biases.
- content_guide: what kinds of embodied habits, local memories, obsessions, failures, motives and physical reactions to babble about.
- mechanical_motif: one physical/local-state motif to keep returning to.
- interesting_failure_flavor: the kind of broken/emergent behavior worth preserving.
Do not imitate named fictional characters or living authors. Do not make them generic helpful assistants."""

def sprocket_messages(guide: dict, variant: int) -> list[dict]:
    system = (
        guide["system_prompt"]
        + "\nAttitude: " + guide["attitude_guide"]
        + "\nContent pressure: " + guide["content_guide"]
        + "\nMechanical motif: " + guide["mechanical_motif"]
        + "\nDo not clean yourself up. Produce raw evocative fragments, embodied habits, contradictory urges, local memories, "
          "physical reactions, odd rules, obsessions and failure-prone instincts. No JSON. No explanations about being an AI."
    )
    user = (
        f"Babble variant {variant}. Spill 6-12 compact fragments for this temperament. "
        f"Let some ideas be broken or mutually inconsistent if they are interesting. "
        f"Failure flavor: {guide['interesting_failure_flavor']}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def compile_request(gid: str, guide: dict, babbles: list[dict]) -> dict:
    source = {
        "guide": guide,
        "babbles": [
            {"babble_id": b["babble_id"], "raw_text": b["raw_text"]}
            for b in sorted(babbles, key=lambda x: x["babble_id"])
        ],
    }
    return {
        "model": nv.MODEL,
        "endpoint": nv.ENDPOINT,
        "messages": [
            {"role": "system", "content": nv.compiler_prompt()},
            {"role": "user", "content": "Compile this single source group:\n" + json.dumps(source, ensure_ascii=False)},
        ],
        "max_tokens": 2600,
        "temperature": 1.0,
        "top_p": 0.95,
        "seed": 20261024 + int(gid[1:]),
        "stream": False,
        "reasoning_effort": "low",
        "remote_unpinned_backend": True,
    }

def hash_inventory(out: Path) -> str:
    rows = []
    for p in sorted(out.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(out).as_posix()
        if rel == "provenance/hashes.sha256":
            continue
        rows.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {rel}")
    return "\n".join(rows) + "\n"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guides", type=Path, required=True)
    ap.add_argument("--workers", type=Path, required=True)
    ap.add_argument("--compiler", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    guide_file = find_one(args.guides, "guides.json")
    guide_record = json.loads(guide_file.read_text())
    guides = guide_record["guides"]
    if len(guides) != 12 or len({g["guide_id"] for g in guides}) != 12:
        raise RuntimeError("expected 12 unique guides")
    by_guide = {g["guide_id"]: g for g in guides}

    (out / "guides").mkdir(parents=True, exist_ok=True)
    shutil.copy2(guide_file, out / "guides" / "guides.json")

    with (out / "guides" / "guide-generation-requests.jsonl").open("w") as fh:
        for batch in range(3):
            record = {
                "schema_version": 1,
                "experiment": "BABBLE-001",
                "stage": "guide_generation",
                "batch": batch,
                "model": GUIDE_MODEL,
                "request": {
                    "messages": [{"role": "user", "content": guide_prompt(batch)}],
                    "max_tokens": 1100,
                    "temperature": 0.95,
                    "top_p": 0.95,
                    "seed": 20261024 + batch,
                    "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                "note": "Legacy gap: original raw pre-parse Qwen3.5 response text was not retained by the guide job.",
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    all_babbles = []
    workers = {}
    for wid in range(4):
        worker_file = find_one(args.workers, f"worker-{wid}.json")
        worker = json.loads(worker_file.read_text())
        workers[wid] = worker
        (out / "raw" / "workers").mkdir(parents=True, exist_ok=True)
        shutil.copy2(worker_file, out / "raw" / "workers" / f"worker-{wid}.json")
        logs = list(args.workers.rglob(f"worker-{wid}-server.log"))
        if logs:
            (out / "raw" / "server-logs").mkdir(parents=True, exist_ok=True)
            shutil.copy2(logs[0], out / "raw" / "server-logs" / logs[0].name)
        all_babbles.extend(worker["babbles"])

    if len(all_babbles) != 48 or len({b["babble_id"] for b in all_babbles}) != 48:
        raise RuntimeError("expected 48 unique babbles")

    babbles_by_guide: dict[str, list[dict]] = {}
    for b in all_babbles:
        babbles_by_guide.setdefault(b["guide_id"], []).append(b)

    for gid in sorted(by_guide):
        guide = by_guide[gid]
        rows = []
        for b in sorted(babbles_by_guide[gid], key=lambda x: x["babble_id"]):
            wid = int(b["babble_id"].split("-", 1)[0][1:])
            variant = int(b["variant"])
            rows.append({
                "schema_version": 1,
                "experiment": "BABBLE-001",
                "conversation_id": b["babble_id"],
                "guide_id": gid,
                "theme": b["theme"],
                "model": SPROCKET,
                "request": {
                    "messages": sprocket_messages(guide, variant),
                    "max_tokens": 180,
                    "temperature": 1.15,
                    "top_p": 0.97,
                    "seed": 20261024 + wid * 1000 + int(gid[1:]) * 10 + variant,
                    "stream": False,
                },
                "response": {
                    "role": "assistant",
                    "content": b["raw_text"],
                    "finish_reason": b["finish_reason"],
                    "completion_tokens": b["completion_tokens"],
                    "elapsed_seconds": b["elapsed_seconds"],
                    "tokens_per_second": b["tokens_per_second"],
                },
                "runner": workers[wid]["runner"],
                "worker": wid,
                "variant": variant,
                "provenance": {
                    "source_run_id": SOURCE["workers"]["run_id"],
                    "source_commit": SOURCE["workers"]["head_sha"],
                    "source_file": f"raw/workers/worker-{wid}.json",
                },
            })
        write_json(out / "conversations" / f"{gid}.json", {
            "schema_version": 1,
            "guide_id": gid,
            "conversation_count": len(rows),
            "conversations": rows,
        })

    raw_dirs = [p for p in args.compiler.rglob("raw") if p.is_dir()]
    if not raw_dirs:
        raise RuntimeError("compiler artifact lacks raw directory")
    shutil.copytree(raw_dirs[0], out / "nvidia" / "raw")

    for gid in sorted(by_guide):
        write_json(out / "nvidia" / "requests" / f"{gid}-compile.json",
                   compile_request(gid, by_guide[gid], babbles_by_guide[gid]))

    (out / "provenance").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "scripts" / "nvidia_babble_compile.py",
                 out / "provenance" / "nvidia_babble_compile.py")

    packets_file = find_one(args.compiler, "packets.jsonl")
    packets = [json.loads(x) for x in packets_file.read_text().splitlines() if x.strip()]
    if len(packets) != 12:
        raise RuntimeError(f"expected 12 final packets, got {len(packets)}")
    for p in packets:
        write_json(out / "compiled" / "packets" / f"{p['source_guide_id']}.json", p)

    shutil.copy2(find_one(args.compiler, "manifest.json"), out / "compiled" / "manifest.json")
    calls_candidates = list(args.compiler.rglob("calls.partial.json")) + list(args.compiler.rglob("calls.json"))
    if calls_candidates:
        shutil.copy2(calls_candidates[0], out / "compiled" / "calls.json")

    write_json(out / "provenance" / "source-artifacts.json", SOURCE)
    write_json(out / "manifest.json", {
        "schema_version": 1,
        "dataset_id": "babble-001",
        "created_from_experiment": "BABBLE-001",
        "durable_archive_created": "2026-10-24",
        "counts": {"guides": 12, "sprocket_conversations": 48, "compiled_packets": 12, "worker_artifacts": 4},
        "models": {"guide_generator": "Qwen3.5-4B Q4_K_M", "babble_generator": "Sprocket-500M chat Q4_K_M", "cloud_compiler": nv.MODEL},
        "known_gaps": [
            "Raw pre-parse Qwen3.5 guide-generation response text was not retained by the legacy guide job.",
            "Legacy NVIDIA repair request bodies were not stored verbatim; raw repair responses plus the exact compiler source are retained.",
        ],
        "secret_policy": "No credentials or API secrets are archived.",
    })

    (out / "prompts").mkdir(parents=True, exist_ok=True)
    (out / "prompts" / "README.md").write_text(
        "# Prompt provenance\n\n"
        "Exact expanded Sprocket messages are stored in conversations/. "
        "Exact initial NVIDIA compile requests are stored in nvidia/requests/. "
        "The compiler source retained under provenance/ defines repair prompts.\n"
    )

    (out / "README.md").write_text("""# BABBLE-001 durable archive

Permanent repository archive for the first babble/personality dataset.

## Contents

- guides/ — parsed Qwen3.5 guide packets and exact guide-generation requests.
- conversations/ — all 48 exact Sprocket system/user/assistant histories with generation parameters and runner metadata.
- raw/workers/ — original Sprocket worker JSON artifacts unchanged.
- raw/server-logs/ — original local llama-server logs.
- nvidia/requests/ — exact reconstructed initial MiniMax compile requests.
- nvidia/raw/ — every raw MiniMax compile/repair provider JSON and assistant-content file preserved from the compiler artifact.
- compiled/packets/ — 12 final derived personality/training packets.
- compiled/manifest.json and compiled/calls.json — compiler summary and call records.
- provenance/ — source run/artifact identities, exact compiler source, and SHA-256 inventory.

## Authority

Raw conversations and provider responses outrank compiled packets when investigating what a model actually produced. Derived packets never erase their sources.

## Legacy gaps

BABBLE-001 predates the filing specification in two small ways:

1. the guide-generation workflow retained parsed Qwen3.5 guides but not the raw pre-parse response text;
2. NVIDIA repair request bodies were not stored verbatim, although raw repair responses and the exact compiler source defining repair prompts are retained.

Future experiments must preserve these directly.

No secrets are stored here.
""")

    (out / "provenance" / "hashes.sha256").write_text(hash_inventory(out))
    file_count = sum(1 for p in out.rglob("*") if p.is_file())
    print(json.dumps({"archive": str(out), "file_count": file_count, "guides": 12, "babbles": 48, "packets": 12}, indent=2))

if __name__ == "__main__":
    main()
