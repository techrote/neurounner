# DRIV3R Wormfleet — source-pinned inference lease desk (Draft/HOLD)

Owner-approved #570 research, stacked on the exact qualified native d1-2B Q8 worm source in Draft #611. No changes to #611's historical source or evidence, #612 learned-model lanes, the frozen D-1A/B/D-2 research, production providers or native Art.

Implemented under this exclusive directory:
- Deterministic plan for exactly FOUR independently identified inference episodes, arranged as two hosted runners with two models per runner. Immutable source, host, model, checkpoint, seeds, names and budgets; no automatic expansion or training.
- Refusal of paid APIs, model-owned native permissions, new optimizer jobs, unqualified native >36 decisions or >36 edits. Native M3 logical-history's separate 64-step ceiling is unchanged.
- Completed-episode original-byte checks for PNGs, raw model choices, host legal receipts, source/lease/model+revision, PID+job, RSS/wall, response and action counts, native versus honestly labeled Python W0, and reported overlap of four model-forward intervals.
- Explicitly DOES NOT make a four-live claim: local JSON/PNG consistency and self-reported PID/timestamps are not independently verified resident processes. Source-qualified actual GitHub job/runner evidence is a separate requirement. The audit's independently_verified_current_active_models field is always 0.
- All tests use MOCK fixtures: source-only tests do not involve resident models or real native material.

Source-only tests from repository root:

    python3 -m unittest discover -s experiments/driv3r/wormbrain/fleet -p 'test_*.py' -v

CLI planning (requires separate qualified JSON under schema driv3r.wormfleet.request/1):

    python3 experiments/driv3r/wormbrain/fleet/ledger.py plan --request /private/request.json --output /private/plan.json

Auditing four complete ORIGINAL result directories under schema driv3r.wormfleet.episode/1:

    python3 experiments/driv3r/wormbrain/fleet/ledger.py audit --request /private/request.json --plan /private/plan.json --results /private/results --output /private/audit.json

Planner is intentionally incapable of dispatch; preserves fixed two-runner lease budgets until Commander inspects overall account Actions activity, model-specific 1-vs-2 resident throughput/RSS, and source/host authority. Output paths must be fresh; existing evidence cannot be overwritten.

## Important existing four-model evidence — NOT DRIV3R native

On 15 November 2026, SPROCKET WORM-001 primary already successfully ran an independently audited real four-model inference pilot: https://github.com/techrote/cybersand/actions/runs/43766586563

Two independent standard Linux VMs each ran TWO separately resident root-Instruct W6 model processes, each 24 Python W0 model-scored decisions with seed-distinct, original PNGs/choices/receipts. The separate audit job reported **161.919 seconds of actual four-model-forward interval overlap**. Each runner reported sampled peak total RSS about 4.94 million KiB. Zero optimizer updates. The W6 step24 checkpoint was salvaged from the historical FAILED original producer; do not reclassify it as successfully trained/useful artist. The original data is source-faithful PYTHON W0, NOT native Godot PAINT. This is relevant proof 2×2 can work with that exact model/recipe, not a claim of present-live worms, new native support, universal model suitability or artistic quality.

A separate genuine 1-vs-2 model benchmark on full LFM2.5-VL 450M (different learned model lane) succeeded: https://github.com/techrote/cybersand/actions/runs/43765506673 ; its job printed a measured 1.974 speedup and peak sample RSS 4,572,604 KiB. Neither result authorises retraining or reassignment of #612's leases.

Next: source-pinned adapters for untouched pretrained d1-2B, LFM2.5-350M, SmolLM2-135M-Instruct, and root Sprocket Instruct; measured per-model serial-vs-dual residency before requesting a mixed four-worm campaign. Do NOT duplicate the already-proven learned-Sprocket F2 pilot. Native 128-turns require separate host/source/Stop/Undo and M3 cursor-bound qualification. Ethologist and Body Lab may consume ledger outputs, not edit the Commander-owned fleet directory. Draft/HOLD; no merge or Ready.

## Untuned SmolLM2 135M actual residency bench (opt-in)

`smol_residency.py` reuses frozen W0 byte content at `4bb9d6e1` and the exact `HuggingFaceTB/SmolLM2-135M-Instruct@12fd25f7` base. A dedicated one-runner workflow is **push-gated solely by a new request file**: it first runs two source-faithful W0 model-scored 8-turn episodes serially, then reruns the same seeds with TWO genuinely separate resident SmolLM2 processes. It measures forward count, wall throughput, aggregate RSS, actual process overlap, raw unmasked digit-logit action preference, host-constrained legal choices, per-episode original PNG/JSON, and reproducible independent W0 replay. The language model cannot write native materials or issue arbitrary coordinates. A model-scored digit-logit action is **not** an unmodified free-form raw completion. All evidence is Python W0, NOT Godot/PAINT; no LoRA/adapters, optimizer, new provider, or production operation. Successful serial-vs-dual evidence is capacity for **two** models on one VM only, not four simultaneous mixed-model DRIV3R worms. A separate later two-runner qualification is required. The workflow never launches from ordinary source pushes, preserving explicit owner budget/lease control. No request is generated by the test suite itself.

### Original SmolLM2 untuned residency result — 15 November 2026

**Actual first inference, not a mock or learned checkpoint:** [Actions #43773365647](https://github.com/techrote/cybersand/actions/runs/43773365647) **SUCCESS**, source/request `d0413bae5637c17c3d34ea9e0dbcb397b3d565cf`, original [artifact #14184528588](https://github.com/techrote/cybersand/actions/runs/43773365647/artifacts/14184528588) ZIP SHA256 `b6af2ac421ec41f1d4b5a4fb199e6fdd394abc9862151e3004236f95e63096f2`. Original archive independently inspected: 40 ZIP members CRC PASS, **39/39 SHA256SUMS file digests PASS**, four original PNG hashes match their completed records, original serial/dual matched-seed PNG pairs byte-identical; all 32 independently checked model-score selections are inside the host's action offers and match W0 original events. Model untouched **HuggingFaceTB/SmolLM2-135M-Instruct@12fd25f77366fa6b3b4b768ec3050bf629380bac**, no adapter/optimizer/training.

On ONE ordinary 4-vCPU runner, serial **16 true model forward decisions** (8+8) took **16.22378 s** end-to-end after environment/model preparation; dual **16** same-seed decisions in separate simultaneously loaded PID **2363/2364** took **8.6155 s**, **1.8831× end-to-end phase throughput**, with **4.042018 s** overlap of *inference-episode intervals* and sampled combined resident RSS **1,946,612 KiB (~1.86 GiB)**. Original CPU model scored legal single-digit indices; the unmasked first preference was STRAIGHT and later seven each LEFT. Both seeds produced **4 distinct trail cells**, effectively tiny turning loops, **not useful art**. These are *source-faithful synthetic Python W0*, not actual Godot native pixels. Same-seed serial/dual original PNGs are identical. A single short benchmark is not a throughput distribution or proof of other model families.

**Disposition:** PASS for two-real-resident untuned Smol capacity on one runner, HOLD for four mixed DRIV3R models, creativity and 128 native actions. Next: source-verify other untouched model adapters and benchmark the untuned LFM2.5-350M before a bounded 2×2 mixed-family fleet; no re-spending learned SPROCKET/LFM four-worm qualification from #612. The result doesn't weaken the strict 36-PAINT native cap or transfer training leases.


## Untuned LFM2.5-350M dual-residency candidate — source qualification only

The additive `lfm350_residency.py` entrypoint reuses the successful, pinned Smol model-process supervisor, genuine independent two-PID serial-versus-dual test, raw token-logit action scoring, host-legal W0 and PNG byte replay. It substitutes **untuned** `LiquidAI/LFM2.5-350M@9921cd4d82c0e3986887aa4b376f9446729ef8a3` (the already qualified #567 model identity), with seeds 4201/4219 and Architecture/Greebles cues. Its child-process entrypoint is explicit rather than silently falling back to Smol. Original pinned W0 source/blob is unchanged. The new opt-in workflow has a single ordinary CPU runner with hard timeouts, exact Hugging Face revision resolution, local safetensors/config digests, scoped signal/kill, output hashes, original PNGs/JSONL and failure-preserving artifact upload. A push to the LFM350 **request file** is the sole expensive-run trigger; ordinary source/test commits cannot dispatch it.

**No LFM350 inference, 1-vs-2 speedup, model-load memory, four-mixed-worm concurrency or artistic quality is established merely by source qualification.** Treat until an exact-source completed workflow and independently audited original artifact exists as `PREPARED / NOT RUN`. Native PAINT, training, adapters and sealed FINAL remain prohibited. Do not alter already-qualified original Smol output or rewrite its historical receipts.


### Preserved original preflight failure and corrected bounded retry

[Attempt #43779108118](https://github.com/techrote/cybersand/actions/runs/43779108118) reached the source/provider/scheduling checks (14/14 tests pass) but **FAILED before model download or inference**: dynamic preflight import could not resolve sibling `smol_residency` when loaded via `SourceFileLoader`. Its `if: always()` failure collector also lacked a pre-created directory. This is a **real failed run**, not a benchmark observation. The corrected wrapper prepends its own source directory for import and the workflow creates its evidence directory before validation and cleanup. An explicitly reserialized, semantically identical attempt-1 request re-triggers once after these repairs; unique GitHub run/source lease prevents substituting original evidence. Its actual result must be independently checked, not assumed.


## Critical inter-runner clock distinction (16 November 2026)

An independent code review found the original planner/auditor compared `time.monotonic_ns()` values from **different GitHub virtual machines** to compute alleged four-way simultaneous inference. That arithmetic is invalid because those monotonic clocks have unrelated epochs. The source-only auditor now checks two actual-model interval claims **within each distinct runner job**, which share a monotonic clock, and explicitly returns `cross_runner_wall_clock_overlap_verified=false` and `claimed_four_forward_overlap_seconds=null`. Never market the old derived global overlap value as a verified metric. Cross-VM four-way overlap requires independently grounded common-time evidence (e.g. GitHub job timestamps and source-linked UTC process/forward receipts with bounded clock uncertainty); mere local JSON stamps cannot provide it. This correction does not invalidate original two-resident-on-one-runner evidence, successful per-runner inference, or other workers' separately proven producer receipts. It prevents this Commander's source-only ledger from falsely upgrading two local overlaps to a four-resident proof.
