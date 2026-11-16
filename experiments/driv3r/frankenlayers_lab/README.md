# Frankenlayers Lab — 6/4-layer game-policy carcasses (research only)

This is a **stacked** experiment on #626, not a replacement for its accepted
8-layer source/evidence. It changes only the experiment's *source-built llama.cpp
System One decision graph*, using the immutable d1-2B Q8 donor and the exact
phase-2 stock-vs-mutant W0 paired evaluator.

Four new speculative sparse skeletons are tried: `skele6`, `aero6`,
`skele4`, `aero4`. All preserve known phase-2 attention ballast candidates.
A graph-time metadata check requires both full and sliding attention paths to
be used; missing paths are a documented **runtime rejection**, not valid model
inference. No claim is made that the flows will work before their actual jobs.

First wave: source tests, pinned runtime build, pinned Q8 donor acquisition,
three W0 game-like families × up to 16 decisions each, each mutant paired with
stock on the same runner. Preserve HTTP decision responses, exact choices,
legal-action lists, p50/p95 latency, PNGs, hashes, failed server logs and CPU
metadata. One full CI evidence archive per mutant, including failures.

Only after a viable 6/4-layer candidate is evidenced should follow-up waves
try personality mutants (reorder/duplicate/weight surgery), alternate seeds,
Q4/IQ3/IQ2 or bounded repair training. This staged choice avoids comparing
quantized mutations against unqualified skeletons.

**Game-target boundary:** models are for CyberSand-style painting/navigation,
but this lane never edits the CyberSand game implementation or content. No
native/Workbench runs, paid API/GPU, production provider changes or merge.
Draft/HOLD. This PR stacks on `research/619-d1-frankenlayers-yolo`.

## Wave 1 observed; wave 2 staged

Wave 1 real outcomes and source/artifact references: [results-2026-11-16.md](results-2026-11-16.md).
AERO6 survived 42 legal game decisions (36 BRANCH, 5 RETRACE, 1 STOP) at ~4.27x
paired stock p50. SKELE6 made 48 STRAIGHT choices at ~4.27x. SKELE4
immediately STOPed three times; its 7.98x p50 is not a valid nonterminal
speed claim. AERO4 failed the intended attention-ballast initialization guard;
its original flow identity is retained unchanged.

Wave 2 introduces *new* named tail variants, rather than retrospectively
rewriting failed AERO4. It also builds a reproducible ephemeral Q4_K_M from
the pinned Q8 donor and records a same-runner Q8-stock/Q4-stock/Q4-AERO6
triad. The new quantised model is a **requantisation** and is not equivalent to
an F16-source Q4. Large generated model files are kept out of Git history.
All model transforms and hashes appear in original Actions evidence.

## Wave 3 findings and wave 4 strategy

Source-fenced [results](results-2026-11-16.md) establish that five-layer
AERO5_NO23 is alive with 48 legal, nonterminal BRANCH/RETRACE/RIGHT actions;
four-layer TAILFLIP4 also survives, but gives a nearly world-invariant BRANCH
personality. Four-execution ECHO4 survives and has some world-conditioned LEFT
turns. Wave 4 tries Q4_K_M requantisation of AERO5_NO23 and TAILFLIP4, plus
seed-571 replication of AERO5_NO23 and ECHO4. Quantised checkpoints remain
out of Git history; Q4 is made from immutable Q8 (not FP16). Game code is
read-only and all W0 projections are synthetic, not native acceptance.

## Wave 4 verified, wave 5 dispatched

Four original Wave-4 archives verified 150/150 content hashes: five-layer AERO5
survives Q4 (48 nonterminal decisions, 29 BRANCH/11 RETRACE/8 RIGHT), whereas
TAILFLIP4 Q4 ends after five alternating branch/retrace moves in each world.
The Q8 seed-571 AERO5 and ECHO4 remain nonterminal, though broader responsiveness
remains unproven. [Details and archived evidence](results-2026-11-16.md).

Wave 5 checks a new four-layer deletion, a five-layer reordered tail,
ECHO4-Q4, and AERO5-Q4 at another seed. New modes are speculative and run
against the unchanged W0 legal-action host with pinned same-runner stock controls.
No CyberSand game/native/Workbench modifications or production promotion.

## Wave 5 verified, wave 6: four-layer Q4 frontier

The original Wave-5 archives pass 150/150 individual SHA-256 file checks.
AERO4_NO10 `0,2,4,20` survives 48/48 Q8 choices, with 25 BRANCH, 6
RETRACE and 17 RIGHT at 0.5850s p50 (7.35x same-runner Q8 stock).
Greebles and architecture produce identical choices, while pipework differs:
there is limited input-conditioning, not proven strategy. AERO5_TAILFLIP
produced 48 RIGHT (a fixed four-cell circle). ECHO4-Q4 produced 48 STRAIGHT,
losing its earlier Q8 LEFT/STRAIGHT variation. AERO5-Q4 seed 571 survived
48/48 choices (27 BRANCH, 10 RETRACE, 11 RIGHT) at 0.5012s p50, or 7.19x
same-runner Q8 stock.

Wave 6 tests Q4_K_M compression and second-seed reproducibility of the newly
viable four-layer AERO4_NO10, plus the speculative distinct AERO4_NO2 deletion.
No model has passed held-out spatial recovery, native Workbench, or learned
strategy gates; training remains a separate model-only next phase.

## Wave 11 — calibrated IQ3_XS remediation, bounded single-specimen pilot

[Wave-10 inference run 43876518503](https://github.com/techrote/cybersand/actions/runs/43876518503) finished with all three model jobs failing **before inference**. All three original failure archives were independently downloaded, and **27/27 content files matched SHA256SUMS**. The source-built pinned llama.cpp quantizer rejected tensor `blk.2.attn_k.weight` (`iq3_xxs`) because the IQ3_XS recipe used no importance matrix. This is a confirmed **quantization prerequisite**, not evidence of a model's game behaviour. Preserve Wave 10's failing request and source unchanged.

Wave 11 explicitly generates deterministic, calibration-only CyberSand-like legal-action prompts (96 distinct rows; seeds 101/103/107/109, disjoint from W0 test seeds 570/571); executes pinned `llama-imatrix` on immutable Q8 stock for 8 chunks; then passes the matrix into the pinned `llama-quantize --allow-requantize IQ3_XS`. Retain calibration text, matrix, logs, individual SHA-256s and conversion receipt; a model-only triad compares Q8 stock, IQ3 stock and IQ3 sparse AERO4_NO10 on the same CPU runner. **Only one pilot is dispatched** to avoid repeating three expensive acquisitions before this quantization path is proven. No IQ3 performance, learned policy or calibrated quality claim is made before original evidence qualification.
