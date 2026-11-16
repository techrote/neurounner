# WORM LFM fleet — original four-episode evidence interface (v1)

**Support-only, Draft/HOLD.** This is NOT a model scheduler or loader. No new
optimizer or inference lease is dispatched. Source-only W0 runs are not
native Workbench, and a queued Actions workflow is not a resident model.

LFM primary PR #575 owns trained checkpoint inference; Commander #570 owns
its *separate untuned* fleet. Shared W0 #613 exact baseline is
4bb9d6e1d0d0d78183f87d1e3c29648edafacf61. Neither primary's two
optimizer VM leases is transferred to support. VOID #574, sealed FINAL,
model pins, native host permissions and production provider are untouched.

## Producer manifest and original evidence

The LFM primary/wrangler must supply at least four **different obstacle seeds**
and four distinct process identities and leases, one common 96x96 W0 decision
budget (2–16 short preflight; target 128+ only once qualified), original PNGs,
original W0 traces and model-forward choices for every decision.

Version: cybersand.worm.lfm-fleet-original-inference/1. A manifest begins
with the following, with at least four independently populated episodes:

    {
      "schema": "cybersand.worm.lfm-fleet-original-inference/1",
      "campaign_id": "real-unique-campaign",
      "decision_budget": 16,
      "episodes": [
        {
          "episode_id": "instance-a", "lease_id": "distinct-lease-a",
          "kind": "trained_lfm", "status": "completed",
          "runner_id": "actual-runner-name", "workflow_run_id": 123456,
          "pid": 1234, "pid_start_ticks": 1234567,
          "seed": 3101, "family": "greebles",
          "model_id": "LiquidAI/LFM2.5-VL-450M",
          "model_revision": "fc6221ca597f3315e4f82fc2df606783267b34ba",
          "w0_source_sha": "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61",
          "source_sha": "<actual 40-character Git commit SHA>",
          "checkpoint_sha256": "<actual 64-character SHA256>",
          "original_producer_zip_sha256": "<actual 64-character SHA256>",
          "native_execution": false, "optimizer_updates": 0,
          "trace": "episodes/a/trace.json", "trace_sha256": "<actual SHA256>",
          "png": "episodes/a/original.png", "png_sha256": "<actual SHA256>",
          "inference": "episodes/a/inference.jsonl",
          "inference_sha256": "<actual SHA256>",
          "pulses": "episodes/a/pulses.jsonl", "pulses_sha256": "<actual SHA256>"
        }
      ]
    }

The row above is merely a documentation template, **not model evidence**.
Do not use its placeholder IDs, PID or synthetic model claims. The artifact
must contain a matching original model Actions run with verified checkpoint
and immutable producer ZIP. The auditor validates Git SHA format, source
pin and file hashes, but GitHub Actions run provenance and actual model
weights must also be independently checked outside the parser.

Every trace is a full W0 baseline-contract JSON record produced by the
original genuine model selection run. For each event, inference.jsonl must
contain one actual recorded model-forward decision with tick, raw_choice,
producer_run_id, model_receipt_sha256, inference_mode
(raw_first_pass or six_logits_host_sample), and positive latency_ms.
A score-conditioned host sampler is not a raw EOS7 greedy prediction:
preserve and distinguish its actual first-pass tokens/logits, host RNG,
raw preference and legal sampling in the producing artifact.

The auditor independently replays original raw actions through shared W0
host legality, obstacles, branching, positions, per-step receipts and PNG
bytes. Matched seed/family/budget random and deterministic wall controls
are recalculated without modifying the original model evidence.
Reports separate invalid raw choices, novelty, coverage, branches,
revisits, recovery, repetition and source PNG hashes. Visual eccentricity
is judged separately from statistically improved steering.

## Capture two true process/action observations per active inference worker

On the actual four-vCPU Linux runner owning each model PID, capture
once at coordinated sampling slot a and again at slot b AFTER the
original inference log advances. Example:

    python3 experiments/agent_painter/worm/support_lfm/capture_fleet_pulse.py \
      --pid "$MODEL_WORKER_PID" --inference-jsonl episodes/a/inference.jsonl \
      --episode-id "$EPISODE_ID" --runner-id "$RUNNER_NAME" \
      --slot a --output episodes/a/pulses.jsonl

This capture checks Linux /proc PID existence, start-tick identity,
VmRSS, four-vCPU affinity, true model-forward JSONL rows and the
observed exact log-prefix digest. Slot output is append-only and
refuses duplicate slots. Four simultaneous action-progress episodes
require both slot-a and slot-b timestamps across four distinct model
processes to fit their respective 10-second windows, with increasing
source-recorded actions per process. A queue state, idle model, duplicate
process, duplicated world seed, altered PNG, or unverified log is refused.

    python3 experiments/agent_painter/worm/support_lfm/fleet_episode_audit.py \
      --manifest original-artifact/manifest.json \
      --output original-artifact/four-episode-audit.json

Successful SOURCE audit does not independently rerun neural weights;
therefore original model checkpoint, original Actions producer runs and
actual PID receipts require outside verification. Test-only fabricated
fixtures in this directory explicitly are NOT resident LFM models.
Never claim four live workers merely from passing synthetic tests.

## Four-vCPU 1x4 versus 2x2 capacity check

Acquire first a genuine four-runner/one-model-per-runner experiment,
then, only if CPU/RSS/leases permit, a separate two-runner/two-model
per-runner experiment with the same checkpoint, source, seeds and
budget. Do not assume a 2x2 speedup or silently start four runs.

    python3 experiments/agent_painter/worm/support_lfm/compare_fleet_capacity.py \
      --one-report four-runners-one-model.json \
      --two-report two-runners-two-model.json \
      --output genuine-capacity-comparison.json

That script fails on non-matched or non-qualified reports and compares
actual inference latency, measured action progress per second, runner
process occupancy and combined peak RSS. The fleet Commander and
LFM primary must still assess Actions queue time, quotas, RAM headroom,
native host authority and future 128+ decision suitability.

**Current gate:** no four-independent-running LFM campaign has been verified
by support; this tooling is a bounded short-run instrument, not success
evidence. W10 genuine legal EOS formatting improvement and failed
theme-conditioned model steering remain historical negatives.
