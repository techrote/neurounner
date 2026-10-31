# DETECTIVE-001E — playable offline Bay Seventeen case (candidate)

Parent: issue #103. Companion: #176.

Status: source-pinned, dependency-free, model-free terminal adapter over the accepted DETECTIVE-001A offline case, DETECTIVE-003A interview/session ledger and DETECTIVE-003C typed fictional scripter bridge.

This is a real player-facing terminal route for the existing fixture. It does NOT demonstrate new model competence, an actual trained cast, a production game engine or the Sprocket specialization held in #148.

## Run

On standard Python 3.11+:

    python3 scripts/detective_001e_playable.py --smoke
    python3 scripts/detective_001e_playable.py

Interactive case commands:

    status
    ask V-G02-A Why is the clock unreliable?
    ask V-G09-C Do you remember the rail scan?
    inspect dock-clock-ledger
    inspect transfer-card
    status
    present V-G09-C dock-clock-ledger

These two inspected public documents recover both mandatory case leads even if every witness lies or refuses to speak. Camera-cache is private and cannot be freely inspected.

### Typed fictional scripter

Only an already disclosed fictional terminal and game-inspected assets are placed in the script request. A human player may enter:

    script - {"schema":"detective-script-v1","steps":[{"op":"read_asset","asset_id":"dock-clock-ledger"}]}

The second script argument may instead be an *explicit player-known fictional credential*. Only the exact strict script grammar is accepted; no model-authored host shell, real filesystem or network execution exists. A valid proposal creates a transaction without changing the case or interview session. A separate human command is required:

    apply V-G09-C camera-cache

This applies only if the exact source-bound, current-session fictional transaction grants that particular private evidence. Wrong credentials, invented IDs, extra fields, unknown operations, stale receipts and rejected scripts fail closed.

## Replay and authorities

- The accepted 003C bridge verifies exact frozen Git-blob identities of the case, 001A contract and 003A session code. No modifications to those authorities are made.
- The game state stores a case hash, original 003A event chain and a separate append-only hash-linked public-inspection journal. Replay validates both. These hashes detect accidental/casual corruption; an attacker with full state write access could recompute an unkeyed hash.
- Only publicly inspected game-owned evidence satisfies a mandatory lead. Dialogue never qualifies leads; no claims are auto-annotated from fixture/model speech.
- 003A projects witness-specific admitted memory, shown evidence and redacted same-witness history without leaking private authority tokens.
- Scripter proposals are type checked and executed only in 003C's fictional case graph. Execution and evidence presentation are two distinct API calls.
- This is a **single-process offline fixture**, not a durable multi-process session store, graphical game bridge, full progression graph, model inference provider, production personality admission or general fictional hacking competence qualification. Only its own two public case routes are established as playable.

## Verification

    python3 -m unittest tests.test_detective_001_contract tests.test_detective_003a_session tests.test_detective_003c_scripter_bridge tests.test_detective_001e_playable -v

Path-scoped GitHub Actions runs this source regression + no-model command-line smoke. No paid APIs, optimizer steps, sealed final access or foreign repo integration.

Keep #176 separate: its 144 real SFT generations have original evidence, but human 0–4 ratings and witnessed freeze must precede unblinding/shortlisting. Keep #148 separate: its Stage-A independent source acceptance and explicit optimizer permission are not inferred from this offline integration.

**Outcome:** an offline playable integration candidate only. Independent source review, exact-head checks and the still-unmet external/production gates remain necessary before either parent epic can be closed.
