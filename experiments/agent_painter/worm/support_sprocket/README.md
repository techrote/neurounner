# WORM-001 SPROCKET support — deterministic local decisions, v0

**Scope:** experimental, source-faithful **Python W0** support only, not learned policy, native Godot execution, complete-Art v2 output or useful-model freeze. Authority: [WORM #612](https://github.com/techrote/cybersand/issues/612) and [owner pivot](https://github.com/techrote/cybersand/issues/568#issuecomment-6891678794). Keep this and SPROCKET primary #576 Draft/HOLD; do not merge. Historical SMOL #573 untouched.

The support worker imports, never forks or edits the shepherd-owned W0 implementation at ../common/worm.py, contract cybersand.worm.observation-action/0-experimental. Exact action tokens: STRAIGHT, LEFT, RIGHT, BRANCH, RETRACE, STOP. Each input contains W0 fields x, y, heading (E/S/W/N=0/1/2/3), local_5x5 (@ current, # wall, + visited, . fresh), legal, last_actions, family and tick. The host alone determines legal moves, native permission/focus, Stop/Undo/history and materials.

## Execute (stdlib Python, no model weights)

From repository root:

    python3 -m unittest discover -s experiments/agent_painter/worm/support_sprocket -p 'test_curriculum.py' -v
    python3 experiments/agent_painter/worm/support_sprocket/curriculum.py --output /tmp/worm-sprocket-support

Outputs: synthetic dataset/train.jsonl and dataset/development.jsonl with canonical hashes and versioned manifest; **24 actual 384×384 PNGs and 24 full exact W0 replay JSON receipts** at two gallery world seeds × three families × four policies. Policies are W0 random, wall, fidget, plus SPROCKET-support synthetic teacher. The four policies share world seed, 96×96 grid, obstacles and maximum 128-step budget. Teacher is selected from actual W0 legal observations after each replay prefix, never by a duplicated local physics simulator. This costs O(steps squared) but is capped at 128 steps. Gallery manifest exposes all seeds, metrics, hashes and first-pass choices; poor outputs are retained.

## Dataset / split custody

- TRAIN world seeds: 101, 107, 113, 127, 131, 137. DEVELOPMENT: 1009, 1013, 1019. GALLERY: 2111, 2129. A world seed is the independent obstacle-structure group; many steps, policies and family variants are correlated views.
- Real W0 random/wall/fidget observations sampled every third tick receive deterministic rule-derived, legal local targets. Four extra clearly constructed rare-action fixtures per family cover junction BRANCH, blocked RETRACE, terminal STOP and deadline STOP, tagged constructed-not-native and TRAIN-only.
- Exact-observation SHA-256 dedupe removes repeated input states including collisions across splits; DEVELOPMENT world-seed set remains disjoint. Neither property proves semantic generalization.
- Each JSONL row contains input (compact serialized observation prompt ending ACTION=), target (one bare uppercase legal action), observation, group, world_seed, source, observation_sha256 and version. The SPROCKET primary may apply its verified pinned model-specific chat/EOS framing in a separately sourced WORM recipe.
- The corpus is synthetic WORM action-selection data, **not original native-admitted Art v2 TRAIN**, not frozen DEV60 and not sealed FINAL. Never use it to overwrite legacy checkpoints.

## Limitations / handoff

The teacher is intentionally simple: Greebles favours frequent turns, Pipework longer straight tracks, Architecture rhythmic orthogonal segments. It can BRANCH, RETRACE and STOP, but cannot plan entire buildings, guarantee loops, predict playability or establish art quality. W0 output is a source-faithful PNG grid raster, not actual native Godot material cells. Train a SPROCKET WORM model only under the primary's explicit unique training lease; this support lane never runs an optimizer.

The primary should compare genuine model first-pass choices and rollouts against all paired same-seed baselines, tracking illegal decisions, EOS, blank/uninteresting traces, branches, retraces, revisits, coverage and visual interest. No hidden action repairs or cherry-picked only-success images. Publish exact source commit, TRAIN/DEVELOPMENT hashes, action counts and gallery receipt hashes in issue #612, cross-link #568, and request independent visual review before any useful-version claim.


## V1: six-action anti-collapse probes and replayed RIGHT-only stress

To generate independent, **support-only** v1 sidecars (without altering the v0 data):
\`python3 experiments/agent_painter/worm/support_sprocket/probes_v1.py --output /tmp/sprocket-v1\`

The first genuinely trained SPROCKET WORM policies learned legal EOS framing but collapsed overwhelmingly to STRAIGHT (W1) and then RIGHT (W2 static probes); the primary also measured literal model-prompt overlap across its source TRAIN/EVAL worlds. V1 therefore supplies **54 constructed TRAIN local-motif action examples** (3 variants × 6 labels × 3 themes), **36 constructed DEVELOPMENT probes** (2 different variants × 6 × 3), and a **54-row TRAIN-only family/action-balanced candidate sampling view** (3 per pair, preferring up to two genuine W0-observed TRAIN rows). The validator fails closed on literal input overlap across **all original support v0 and new v1** TRAIN/DEV rows, and on conflicting labels for identical prompts. All masks/targets are legal according to the imported W0 contract.

These LOCAL counterfactuals are *not* executed world states, independent structures, native Art admissions, or actual learned decisions. Their coordinates/variants do not prove semantic novelty; they are diagnostic failure probes. Never optimize against DEVELOPMENT; the primary owns all eventual model-training leases/recipe revisions. The unchanged original v0 corpus remains separate.

Six additional genuine **W0 source-faithful, model-free** 384×384 PNGs and paired JSON receipts replay a deliberately constant RIGHT raw controller on the same two gallery world seeds × three themes. Raw policy choices and host substitutions are recorded separately. This does **not** purport to reproduce the trained W2 model, whose own original online evaluation is authoritative. The support Actions workflow runs all tests and retains versioned JSONL/PNG/source SHA-256 evidence.

## S1 independent frozen-score counterfactual (15 November 2026)

Original S1: [hosted #43726271494, artifact #14168225018](https://github.com/techrote/cybersand/actions/runs/43726271494/artifacts/14168225018). Script `audit_s1_crn.py` pins its immutable original ZIP SHA-256 and exact shepherd W0 source SHA-256 from commit `4bb9d6e1d0d0d78183f87d1e3c29648edafacf61`. It verifies all 69 CRCs, 68 internal SHA-256 checks and 12 original W0 pixel-exact S1 PNGs.

    python3 experiments/agent_painter/worm/support_sprocket/audit_s1_crn.py \
        /path/to/original-S1.zip --trusted-w0 /path/to/immutable-w0.py --output /tmp/s1-crn

With ZERO optimizer, checkpoint edits or model downloads, the audit compares **real first-step trained six-token logits** on genuinely identical host observations except family, or with identical prompt/logits across temperatures. This isolates first-step score sensitivity from the host RNG confound. Two obstacle seeds are two independent worlds, not six.

It then executes **24 new bounded W0 host replays** (three families × two worlds × four counterfactual policies; ≤24 decisions each). The four variants apply the frozen original T1 per-tick six-logit tape at T1/T3, its time-mean prior at T3, or legal-action-uniform logits at T3. The host draws the same deterministic SHA-256 uniform for a given world seed and tick across family, temperature and ablations; six original-pixel contact sheets include the genuine first-party S1 T1/T3 gallery plus the four new control PNGs.

**Scientific restriction:** after the first trajectory divergence the original logit tape is *off-policy*: those logits were not computed for the new observation. This experiment diagnoses host sampling and prior effects, NOT fresh live learned model-in-the-loop responses or model-conditioned closed-loop steering. It is W0 Python, not native Godot Art.

Use `s1-crn-audit-report.json` for exact source custody, per-policy action histograms, STOP/survival denominators, bbox/radius and first-step logit/softmax total-variation measurements. Early STOP reduces executed actions below the 24-step allocation; do not compare raw coverage without those denominators. Send only substantive results to #612, cross-link #568; do not modify the shepherd-owned dashboard body. Keep PR Draft/HOLD.


## Four-stream fleet handshake (15 Nov 2026; W0-only, NOT live model fleet)

The owner [fleet coordination charter](https://github.com/techrote/cybersand/issues/570#issuecomment-6901564046) requires >=4 **real** simultaneous worm inference instances per wrangler. This support rehearsal deliberately proves **only four concurrent synthetic Python W0 OS processes**, never four resident SPROCKET models or four native episodes. Use `fleet_four_v1.py --output /new/empty/dir` and `python3 -m unittest discover -s experiments/agent_painter/worm/support_sprocket -p test_fleet_four_v1.py -v`.

- Independently verified exact source `bfc024a307831fb2c5981cb38ffa005364a37368`, [completed workflow #43765251866](https://github.com/techrote/cybersand/actions/runs/43765251866), [immutable artifact #14182221819](https://github.com/techrote/cybersand/actions/runs/43765251866/artifacts/14182221819), ZIP SHA-256 `9bf6abe2803dfe739f9e6bba3b5298578e7ca6f43420b72a686586ea4230bb85`; 15/15 ZIP entries CRC-clean. Three actual tests PASS, including process residency, exact replay, corrupted PNG, wrong seed, tampered legality and refusal to overwrite existing evidence.
- Four distinct live W0 PIDs (2239/2240/2241/2242) all reported `loaded_w0`, `entered_host_run`, `completed_w0` and **9,886,003 ns genuine common execution-window overlap**. They performed four separately seeded 96-turn original W0 runs with 4 independently hashed original 384×384 PNGs and 4 JSON traces, four legal-action and full-state+pixel exact replay checks; 0 optimizer updates. These PID values are evidence from that one host run, not stable identities for later runs.
- Cohort: Greebles/fidget seed313 (83 distinct cells, 13 BRANCH /12 RETRACE, bbox 20×13); Pipework/random seed337 (74 cells, 10/12, 8×18); Architecture/wall seed347 (97 cells, 0/0, **10×10 tiny spiral**); Pipework/fidget seed359 (88 cells, 11/6, 30×16). Wall-nearby ticks: 48/46/0/0. **Actual complete dead ends: 0/4 streams**; do not fabricate dead-end encounters. Small curled paths and weak layouts are accepted negative art evidence.
- Each independent identity includes wrangler/lease/seed/family/controller, W0 SHA-256 `32ec272c6eff61bbf15956c12f5c52f4209908c75ea87320ce9ebebeb73487d3`, no model/checkpoint/adapter, strict six-action legality, finite Stop, turn budget, state/PNG SHA-256 and branch/retrace/wall/dead-end/streak/reach signatures. Runtime/OS PID, CPU, peak *per-process* RSS, and phase-transition evidence are in `runtime.json`, separate from reproducible model-free `manifest.json`. A sum of per-process maximum RSS is **not** a measurement of simultaneous total resident memory.
- Model-fleet readiness **NOT established**: no resident model was loaded; no model ID or adapter; no 1-vs-2 models-per-4-vCPU RAM/throughput comparison; no 2×2 runner acceptance, no native material PNG, no 128-turn native horizon and no genuine independent learned-model rollout. The Commander must carry out those distinct authorized gates before advertising four inference worms. Do not confuse one Python worker process with one model resident.
- Four **different** obstacle worlds demonstrate distinct identities but do **not** furnish same-world matched model-versus-control inference. A future model comparison must run *each* eligible model and random/wall/fidget/corridor controls on identical world seed, action budget, host STOP policy and renderer; different post-decision observations remain causally diverged. Preserve failed/puny drawings, dead ends and early STOPs.

Handshake recommendation to primary #576 and #570 Commander: grant each genuine model resident a unique `wrangler_id / lease_id / model-revision+adapter digest / process PID / run+job / seed / W0-or-native-source / last-decision receipt`, with explicit QUEUED → STARTING → LOADED → ACTIVELY_DECIDING → COMPLETED/FAILED state transitions, host-authoritative legality, replayable RNG, finite Stop/deadline, per-runner actual RSS and decision-versus-render overhead. Only count `ACTIVELY_DECIDING` when backed by a real loaded model and measured inference, never by this synthetic W0 rehearsal. First measure one versus two residents on the same four-vCPU class; choose 2×2 only if resource/latency evidence supports it. Keep both primary optimizer leases separate and untouched.
