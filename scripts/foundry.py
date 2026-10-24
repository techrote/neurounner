#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import json
import time
from pathlib import Path

from experiment import CACHE, Server, load_kv, runner_info, save, server_meta


def extract_json(text: str):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def main() -> int:
    paths = load_kv(CACHE / "paths.env")
    server_bin = Path(paths["LLAMA_SERVER"])
    sprocket = Path(paths["SPROCKET_MODEL"])
    lfm_model = Path(paths["LFM_MODEL"])

    sprockets = [
        Server("sprocket-inverter", server_bin, sprocket, 18201, 1),
        Server("sprocket-memory", server_bin, sprocket, 18202, 1),
        Server("sprocket-contact", server_bin, sprocket, 18203, 1),
    ]
    selector = Server("lfm-selector", server_bin, lfm_model, 18210, 1)
    servers = sprockets + [selector]
    seed = (
        "A small ground actor remembers the direction from which it was unexpectedly shoved. "
        "The memory should alter later movement but remain local, bounded and deterministic."
    )
    prompts = [
        "Mutate by INVERTING or cross-wiring one response. Return only the concrete trigger, memory and action in at most two short sentences.\n\n" + seed,
        "Mutate by adding one tiny temporal memory: decay, hysteresis, refractory time, delay or accumulated irritation. Return only the mechanism in at most two short sentences.\n\n" + seed,
        "Mutate using contact geometry: support loss, surface normal, obstacle contact, relative velocity, pushing or turning. Return only the mechanism in at most two short sentences.\n\n" + seed,
    ]

    try:
        for server in servers:
            server.start()
        metadata = server_meta(servers)

        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(server.chat, prompt, max_tokens=48, temperature=1.0, seed=1201 + i)
                for i, (server, prompt) in enumerate(zip(sprockets, prompts))
            ]
            mutations = [future.result() for future in futures]
        mutation_wall = time.monotonic() - started

        candidates = "\n\n".join(
            f"CANDIDATE {i}: {mutation['content']}" for i, mutation in enumerate(mutations, 1)
        )
        synthesis_prompt = f"""Select, repair and recombine the three noisy behaviour mutations below into ONE deterministic physics-NPC mechanism.

SEED:
{seed}

{candidates}

Use only local observations, at most three bounded persistent state values, and movement, turning, jumping, pushing or waiting. Discard repetition and incoherence.

Return JSON only with exactly these keys:
"name": short snake_case string
"trigger": concrete local trigger
"state": array of one to three short state descriptions
"update": deterministic state update/decay
"action": visible action
"interesting_failure": plausible emergent failure mode
"parents_used": array containing candidate numbers 1, 2 and/or 3
"""
        synthesis = selector.chat(synthesis_prompt, max_tokens=220, temperature=0.2, seed=4242)
        parsed = extract_json(synthesis["content"])

        print("=== parallel Sprocket mutations ===")
        for i, mutation in enumerate(mutations, 1):
            print(f"\n[{i}] {mutation['speaker']} — {mutation['completion_tokens']} tokens / {mutation['elapsed_seconds']:.3f}s")
            print(mutation["content"])
        print("\n=== LFM selection/recombination ===")
        print(synthesis["content"])
        print(f"parsed_json={parsed is not None}")

        payload = {
            "experiment": "foundry",
            "runner": runner_info(),
            "servers": metadata,
            "seed_mechanism": seed,
            "mutation_wall_seconds": mutation_wall,
            "mutations": mutations,
            "synthesis": synthesis,
            "synthesis_json": parsed,
            "total_inference_wall_seconds": mutation_wall + synthesis["elapsed_seconds"],
        }
        md = ["# Foundry mutation/select experiment", "", f"Three parallel Sprocket mutations: **{mutation_wall:.3f}s**.", ""]
        for i, mutation in enumerate(mutations, 1):
            md += [f"## Mutation {i}", "", mutation["content"], ""]
        md += ["## LFM selected/recombined kernel", "", "~~~json", json.dumps(parsed, indent=2) if parsed else synthesis["content"], "~~~", ""]
        save("foundry", payload, "\n".join(md))
        return 0
    finally:
        for server in servers:
            server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
