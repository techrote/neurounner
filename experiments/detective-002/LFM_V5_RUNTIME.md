# DETECTIVE-002C — LFM2.5-350M Transformers-v5 runtime compatibility

Parent: [DETECTIVE-001 #103](https://github.com/techrote/neurounner/issues/103). Tracking: [#133](https://github.com/techrote/neurounner/issues/133). Historical control: completed DETECTIVE-002B #130 / PR #132.

This experiment answers one narrow question:

> Does the exact frozen LFM2.5-350M checkpoint that failed tokenizer construction under Transformers 4.57.6 complete the same four-prompt runtime smoke under a preregistered Transformers 5.20.1 stack?

It does **not** rewrite the 002B failure, select a model, train anything, clear the LFM license, or admit population expansion.

## Frozen authority

Three merged inputs are immutable and Git-blob-bound:

1. DETECTIVE-002A source archive — exact LFM model revision/files.
2. DETECTIVE-002B runtime config — exact system message and B01–B04 semantic battery.
3. DETECTIVE-002B raw LFM result — historical 4.57.6 failure.

Exact model:

- repo: `LiquidAI/LFM2.5-350M`;
- revision: `9e6c6ccf47cd318696e137d381a7ded8fe4df09f`;
- architecture: `Lfm2ForCausalLM`;
- parameter count: 354,483,968;
- weight: `model.safetensors`;
- weight SHA-256: `1c9c77a4471a7f590f85240f74ed1fc26df7fbde88c3006724e2f93ca993ea4e`;
- weight bytes: 708,984,464;
- tokenizer-config SHA-256: `3701f370c70034d06e28947ff51c1063983393319f429e3ce58f06a45fae67cc`;
- chat-template SHA-256: `ba551d58630afa3190b1be3602e28301f3d2e9bbac978dfc49d6d825171648b6`;
- license: `lfm1.0`, production/game-distribution gate **retained**.

## Historical control

DETECTIVE-002B run 40608478076 used Transformers 4.57.6. Exact source/weight verification passed. Tokenizer construction then failed before model/dialogue:

```
ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported.
```

That record remains authoritative for 4.57.6.

## Preregistered runtime change

Only the Transformers tokenizer/runtime generation changes:

- Python 3.12;
- Torch 2.14.1+cpu;
- Transformers **5.20.1**;
- exact PyPI wheel SHA-256 `5f51ce444193e5d0a751e3fa489ddbc2224ff12f6675a5cfefd4044b09757c19`;
- four CPU threads;
- float32;
- seed 260213;
- greedy/no sampling;
- max 512 input tokens;
- max 64 generated tokens per prompt / 256 total.

The workflow downloads the 5.20.1 wheel without dependencies, verifies that wheel hash, and only then installs the stack. Runtime evidence records the actually resolved HF/tokenizers/safetensors versions.

## Stage A — tokenizer only

The model weight is not downloaded.

1. download only the frozen small config/tokenizer/chat-template files;
2. verify their Stage-A hashes/byte counts;
3. verify the frozen tokenizer config declares `tokenizer_class: TokenizersBackend` and `backend: tokenizers`;
4. instantiate the tokenizer under exactly Transformers 5.20.1;
5. require runtime class `TokenizersBackend`;
6. render the exact merged 002B B01–B04 messages;
7. retain input token counts/digests and require <=512 tokens.

If any tokenizer/rendering step fails, the experiment records `tokenizer_incompatible`, proves `weight_downloaded: false`, and stops.

## Stage B — model only after tokenizer PASS

Only after Stage A succeeds:

1. download the exact frozen `model.safetensors`;
2. verify SHA-256 and byte count before local staging;
3. load with native float32 CPU inference;
4. run the exact B01–B04 messages produced by Stage A;
5. greedy decode <=64 tokens/task, <=256 total;
6. retain raw output text, special-token text, output token IDs, load/RSS, prompt prefill and completion timing, truncation and the same literal observations as 002B.

Allowed outcome:

- `runtime_compatibility_pass` — all four prompts complete;
- `model_runtime_incompatible` — tokenizer PASS, exact weight verified, model stage cannot complete;
- `tokenizer_incompatible` — Stage A cannot construct/render the frozen tokenizer.

Any source/config/weight/wheel identity mismatch is a hard CI failure, never a model outcome.

## Evidence

Durable archive target: `datasets/detective-002c/`:

- `result.json`;
- `manifest.json`;
- `README.md`;
- source/run provenance;
- all-file hash inventory.

A separate standard-library verifier proves:

- frozen Stage-A/battery/historical Git blobs;
- historical 4.57.6 failure unchanged;
- exact 5.20.1 runtime identity;
- tokenizer-before-weight ordering;
- exact frozen weight identity for model-stage results;
- model prompts equal tokenizer-only prompt digests;
- raw token/observation replay;
- no training/relevance/license/admission authority.

After evidence is committed, later CI enters a verification-only fast path and skips live inference.

## Decision boundary

A v5 PASS means only that the exact checkpoint is runtime-compatible with this frozen v5 stack. It does not erase the v4.57.6 failure and does not imply:

- better dialogue than the six 002B bodies;
- detective-NPC suitability;
- permission to train;
- production/game license clearance;
- population expansion.

Any later body decision still requires independent relevance #121 plus prior policy evidence.


## Executed outcome — 2026-10-28

Authoritative run: [Actions 40609927745](https://github.com/techrote/neurounner/actions/runs/40609927745), exact pre-evidence source `053ddd90dcae646a647806fc65ca851ae8d85232`.

**Outcome: `tokenizer_incompatible`.**

- exact Transformers 5.20.1 wheel SHA-256 verification PASS;
- resolved runtime: Python 3.12.15, Torch 2.14.1+cpu, Transformers 5.20.1, huggingface_hub 2.3.0, tokenizers 0.23.3, safetensors 0.8.0;
- exact frozen config/tokenizer-config/chat-template identities PASS;
- v5 successfully constructs the frozen tokenizer as **`TokenizersBackend`** with frozen `backend: tokenizers`;
- rendering the exact merged B01–B04 chat messages then raises **`AttributeError`** with an empty message;
- rendered prompts: 0;
- **model weight downloaded: false**;
- model stage: not entered;
- optimizer steps/API calls/relevance access: 0/0/false;
- LFM production/game license gate remains true.

Thus Transformers v5 resolves the original 4.57.6 class-availability failure but does **not** qualify the exact LFM2.5-350M checkpoint/template for the common detective chat runtime. See [REPORT_002C.md](REPORT_002C.md).

This evidence-ready branch must now run verification-only CI; live requalification must be skipped.
