# LFM2.5-VL provisional DEV60 processor-prefix qualification v1

**15 November 2026 · source-only / synthetic-only · NOT DEV60, NOT TRAIN**

Purpose: verify the *actual official* LiquidAI LFM2.5-VL-450M processor,
frozen text-only upstream template, system instruction, EOS7 and byte-level
source custody against a proposed 3 families × 4 whole groups × 5 formulations
static benchmark shape, without running model inference or touching developmental
teacher outputs. The 60 synthetic requests are **all deliberately fake**.
They are *neither* candidate selected prompts *nor* six new native-admitted
DEVELOPMENT groups and must **never** become benchmark results or training data.

The probe is lane-owned and restricted to the existing single #575 Draft/HOLD
branch. It refuses local adapter, profile, common framing and training source
blob drift against qualified source 2a565df45efc568eb96a2166e0b77f299e289625
*before import*. The source-verified adapter is now loaded by exact file path
rather than an ambient `import adapter`: a pre-cached `sys.modules["adapter"]`
impostor cannot replace the qualified on-disk adapter. A regression checks that
boundary without downloading model files. This is local process-custody hardening,
not a claim to defend against a compromised runner. It downloads only original pinned small model/processor
files, checks their existing reviewed SHA-256 hashes, model-card LFM 1.0
attribution, then loads Lfm2VlProcessor through the unchanged adapter.
The complete 450M VL weights are **not** downloaded for this static task;
this does not substitute the official full-model requirement for **training
or inference**. The training profile, 92 LoRA paths, 2,998,272 trainables,
vision/projector freeze and numerical settings are unchanged.

The model-specific 512 generated-token reserve is **provisional capacity
testing only**; it is not approved for DEV60, and is not asserted sufficient
for any teacher JSON. The actual common DEV60 needs a distinct shared
manifest version, six further positively native-admitted independent
DEVELOPMENT whole groups, all 60 literal UTF-8 prompts, independently approved
five-form semantic equivalence, offline teacher-target representability
checks, and per-model frozen output budgets. Do not use this synthetic report
as evidence of any of those gates. No original DEVELOPMENT or sealed FINAL
targets are read by the probe. Its target_representability_checked=false
and dev60_frozen=false are intentional hard labels.

## Evidence / reproducibility

Entry point: dev60_prefix_probe_v1.py. Model-free adversarial tests:

    python3 -m unittest discover -s experiments/agent_painter/lanes/lfm25_vl -p test_dev60_prefix_probe_v1.py -v

The dedicated opt-in/changed-source-only hosted Actions workflow
agent-painter-lfm25_vl-dev60-prefix.yml installs the original pinned
profile-only CPU dependency set (including torchvision==0.29.2+cpu,
Pillow==13.0.0), downloads the original small processor files on its own
ordinary hosted CPU VM, runs the exact Lfm2VlProcessor 60-prefix check,
and uploads original receipts even on failure. No model generation,
optimizer steps, native application, image conditioning, external
paid compute, recurring training or FINAL file access.

For each synthetic request the receipt contains immutable literal request
SHA-256, original official encoded prefix token count, and a canonical
token-ID digest. The order-independent synthetic request-set SHA is distinct
from any future common DEV60 manifest digest. Recorded bounds are
prefix_tokens + provisional_reserve <= 2048; no truncation or repair.
A successful processor check is an *infrastructure* pass, not painter quality.

## Required later work

The shepherd owns shared allocation, positive native admissions, and the
authoritative frozen DEV60 corpus/manifest. Only after its reviewer-approved
freeze may this lane add a **new versioned manifest consumer** using the exact
shared request bytes and independent teacher-side reference representability.
The original full-model, raw-untuned 3/3 negative first-pass baseline remains
the valid scientific control. No additional fixed-six-structure training is
warranted just because this source probe passes.
