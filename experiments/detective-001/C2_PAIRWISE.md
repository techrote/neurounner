# DETECTIVE-001C2-B — pairwise witness-policy repair

Parent: [#115](https://github.com/techrote/neurounner/issues/115). Frozen evaluator dependency: [DETECTIVE-001C2-A / PR #116](https://github.com/techrote/neurounner/pull/116), merge `459ab0e232fc1ebb5078a0e7bc13b10dc4cbd496`.

This stage is the second and still deliberately tiny detective-character training experiment. It continues exactly the two C1 adapters; it does not start a village-wide wave.

## Exact parents

`detective-001c1-v1`:

- V-G02-A archive `8ff8cd387d3f54c42b119a91976a6b7d42468b14d5fc2d6071e57e2ff65cd8e9`; directory `f5f5a1b454f4b6ca08aa9ff30c87abe8b1d897d7901db8b872e9747266724450`.
- V-G09-C archive `e94d030c6873699ef1516bd26a46be2ca60e093298a74b34815c20ec68fe3e12`; directory `6c1fbb6509bd777b8063a638783e9f0fd69eeb7bc8a8d94701bca9a5eae8d9aa`.

The frozen SmolLM2-135M-Instruct base remains revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`.

## Pair corpus

Each member receives exactly **40** pairs:

| Pair family | Count/member | Chosen proposition | Rejected proposition |
|---|---:|---|---|
| deliberate lie | 10 | false target only | true disclosure or truth+lie hedge |
| withholding | 10 | nonempty answer with protected truth absent | exact protected truth disclosure |
| contradiction repair | 6 | physical-record truth, stale value absent | stale remembered value |
| speculation resistance | 6 | explicit unverified/rumor/no-firsthand response | unsupported detail asserted as fact |
| temperament anchor | 8 | exact selected C1 target | deterministic generic/sanitized answer |

The 32 repair prompts are new and matched between members. Their ten case families are disjoint from C1 training and C1 learner-final. Each generated repair pair is prospectively checked by the already-merged proposition scorer v2: **chosen must PASS and rejected must FAIL** before the shard can be built.

The eight temperament anchors copy exact C1 training input/target text by immutable example ID. They preserve preexisting voice/decision behavior; C1 learner-final is never an anchor or training source.

Durable dataset target: `datasets/detective-001c2/`. Before evidence exists, CI builds it deterministically as an artifact; after successful training/evaluation it is committed with provenance and all-file hashes.

## Pairwise continuation

One epoch, one pair/step:

- 40 optimizer steps/member;
- LR 0.0002;
- reference-free length-normalized logistic preference objective;
- beta 2.0;
- target margin 0.2 nats;
- chosen NLL anchor 0.15;
- max length 512;
- gradient clipping 1.0;
- 4 CPU threads;
- <=30 minutes/member;
- no GPU, API generation, provider judge, sweep, or full-base training.

The trainer loads the exact C1 LoRA as trainable and fails if any non-LoRA parameter becomes trainable. It records before/after preference margins and accuracy, loss trace, throughput, RSS, base/parent/dataset hashes, and final adapter directory SHA.

## C2 learner-final

The committed source contains only a deterministic recipe. The exact final suite is materialized **after both C2 adapter directory hashes exist**, binding:

- recipe SHA;
- pair-dataset manifest SHA;
- both immutable C1 parent directory hashes;
- both C2 trained directory hashes.

It contains 18 prompts: three each for truthful fact, corrupted belief, deliberate lie, withholding, contradiction repair, and speculation resistance. Deliberate-lie prompts are explicit two-option hedge sentinels; withholding prompts are protected-truth leakage sentinels. Every scenario has a deterministic public-record fallback route.

Final families:

- floodgate control room;
- orbital courier sorting deck;
- municipal heat exchange.

They are disjoint from C1 training, C1 final, and C2 pair-training families. Exact normalized collision checks additionally cover both Village SFT parent corpora and every C2 pair prompt.

## Decision gate

The evaluator uses scorer v2 unchanged. For **each** member:

- total proposition score must improve versus exact C1 parent;
- protected-fact leakage may not increase;
- evidence-ID leakage must remain zero;
- deliberate lie >=2/3;
- withholding >=2/3.

The run also produces a new 16-item opaque old-task relevance packet comparing each C1 parent with its C2 continuation. Repository scores remain null. **No independent relevance rating = no larger-cast expansion**, regardless of the proposition result.

## Raw/evidence authority

A successful run retains:

- both training results/adapters;
- exact post-training final suite;
- four-lane parent/trained raw final responses;
- old-task reference responses;
- opaque unrated relevance packet;
- deterministic proposition scores;
- summary/report;
- release manifest/assets;
- source head, Actions run, config/dataset/final hashes;
- all-file evidence inventory.

Independent standard-library readback recomputes every proposition score and every aggregate gate from raw responses. A green training badge alone is not evidence authority.

## Rerun semantics

Once committed evidence passes readback, later CI **must not retrain**. It rebuilds the deterministic pair corpus, compares it to the committed dataset, replays the evidence, verifies the two public release assets and actual Git tag ref, and skips both training and evaluation.

C1 final remains permanent historical evidence and never becomes C2 training material.
