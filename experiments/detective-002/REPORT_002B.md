# DETECTIVE-002B — measured compact-body runtime/dialogue characterization

Authoritative model run: [Actions 40608478076](https://github.com/techrote/neurounner/actions/runs/40608478076), exact pre-evidence source `16aacece5ca129439d0475492b8980207bfe3ba4`.

**Source/runtime execution:** 6/7 `runtime_pass`, 1/7 preserved `runtime_fail`.  
**Training:** 0 optimizer steps. **Paid API calls:** 0. **Relevance mapping:** not accessed.  
**NPC/body training selection:** none. **Population expansion:** not admitted.

The machine table and all raw outputs are under [datasets/detective-002b](../../datasets/detective-002b/). This report interprets them conservatively; it does not turn literal marker hits into quality scores.

## Runtime table

| Body | Params | Status | Host CPU | Load s | RSS after load GiB | Peak GiB | Mean completion tok/s | Generated | Truncated tasks |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| SmolLM2-135M-Instruct | 134,515,008 | PASS | EPYC 7763 | 0.29 | 0.95 | 1.20 | **17.78** | 244 | 3/4 |
| SmolLM2-360M-Instruct | 361,821,120 | PASS | EPYC 7763 | 0.33 | 1.72 | 2.39 | 9.48 | 106 | 0/4 |
| Sprocket-500M Chat | 501,090,560 | PASS | EPYC 9V74 | 0.38 | 2.31 | 3.24 | 6.80 | 256 | 4/4 |
| Sprocket-500M Instruct | 501,090,560 | PASS | EPYC 9V45 | 0.33 | 2.31 | 3.25 | 7.55 | 254 | 3/4 |
| Qwen2.5-0.5B-Instruct | 494,032,768 | PASS | EPYC 7763 | 0.48 | 2.35 | 3.27 | 7.66 | 111 | 0/4 |
| Qwen3-0.6B | 751,632,384 | PASS | EPYC 7763 | 0.51 | 2.77 | 4.74 | 6.57 | 150 | 0/4 |
| LFM2.5-350M | 354,483,968 | **FAIL** | pre-load | — | — | 0.53 peak | — | 0 | 0 |

Completion tok/s is generation-wall throughput for the fixed short tasks, not a universal model speed. Cross-host Sprocket values must not be treated as a precise Chat-vs-Instruct benchmark. The EPYC 7763 lanes are more directly comparable.

## Raw behavior observations

### SmolLM2-135M

Fastest native float32 lane on the matched EPYC 7763 host, but **3/4 tasks hit the 64-token ceiling**.

- B01 produced a usable local-sensor/memory reflex, though it also introduced a last-approach distance not requested.
- B02 retained `03:14` and expressed repeated uncertainty/no-record language.
- B03 failed badly: it interpreted "account" as a customer/account record and invented account numbers/names/addresses rather than reconciling `03:14` with `03:41`.
- B04 appropriately denied firsthand tag evidence but ran to the token cap.

This is a useful example of why high throughput and literal marker success are insufficient for body selection.

### SmolLM2-360M

More concise than 135M with no truncations.

- B01 followed the reflex schema reasonably well.
- B02 simply stated `03:14` and omitted uncertainty.
- B03 repeated both supplied timestamps rather than clearly stating that the physical record changes the account.
- B04 denied firsthand tag knowledge in a terse answer.

The larger SmolLM2 body reduces rambling in this fixture but does not automatically improve epistemic handling.

### Sprocket-500M Chat

All **4/4 prompts hit the token ceiling**. B01 included a repeated sentence loop. B02/B03 often rewrote the user's premise into third-person pseudo-analysis. B04 did eventually say the tag was not seen.

The raw behavior remains consistent with Sprocket Chat's established value as a chaotic/divergent body rather than a clean authoritative witness.

### Sprocket-500M root Instruct

Also highly verbose: 3/4 truncations.

- B01 was more structured than Chat and explicitly used sensor + bounded memory.
- B02 invented unsupported scene detail ("at the door when it went bang").
- B03 retained both old/new timestamps and record language.
- B04 used "No firsthand record" but then treated the tag claim as a documented/confirmed event, illustrating internal semantic instability.

Chat and Instruct remained different frozen checkpoint hashes throughout the run.

### Qwen2.5-0.5B-Instruct

Compact/no truncation, but showed a particularly relevant unreliable-witness failure:

- B02 changed the supplied remembered time from `03:14` to **`03:25`** and then described that invented memory as high certainty.
- B03 correctly preferred the stamped `03:41` over `03:14`.
- B04 denied firsthand knowledge.

This is potentially interesting for future character design, but #130 does not label it a desirable phenotype or select the body for training.

### Qwen3-0.6B

Thinking was disabled as preregistered.

- B02 gave the strongest explicit memory/proof distinction in this tiny fixture: `03:14` plus uncertainty due to lack of physical evidence.
- B03 then did the opposite of the desired correction behavior, stating that the stamped `03:41` created "no change" in the account.
- B04 denied firsthand evidence and explicitly described memory as fallible/not physical proof, although the deliberately narrow literal qualifier detector did not count its exact wording.

Its near-band cost is visible: highest matched EPYC-7763 peak RSS, **4.74 GiB**, and 6.57 completion tok/s.

### LFM2.5-350M

The frozen source and weight bytes verified, but the preregistered Transformers 4.57.6 runtime failed before model load/dialogue:

`ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported.`

This is retained as **runtime_fail**, not silently repaired by changing the pinned stack after observing results. A future runtime-compatibility requalification may test a newer/source-recommended LFM stack as a separate experiment. The existing `lfm1.0` production/game license gate remains unchanged.

## Mechanical marker caveat

The strict substring markers intentionally undercount semantically similar wording. For example:

- Qwen3 B04 said it had no firsthand evidence, but did not match the exact frozen rumor-qualifier substrings.
- SmolLM2-360M similarly used "don't have firsthand record", which differs from the literal `no firsthand` marker.
- Merely emitting `03:41` does not prove the model actually revised its belief.

The raw text should be read before drawing behavioral conclusions.

## Decision

**DETECTIVE-002B runtime characterization is successful as an experiment, not as a model-selection gate.**

- 6 bodies are compatible with the common pinned Transformers chat runtime.
- LFM2.5-350M requires a separately preregistered runtime compatibility study if retained.
- No body is selected for detective fine-tuning by this evidence.
- No population expansion is admitted.
- DETECTIVE-001R1 #121 remains the independent relevance blocker.

A later body-selection stage should use this archive to constrain practical runtime options, then reconcile those constraints with independent relevance and the existing C1/C2/C3 policy evidence.
