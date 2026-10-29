# DETECTIVE-001C2-A — proposition evaluator and blind relevance custody

Parent: [#115](https://github.com/techrote/neurounner/issues/115), under [DETECTIVE-001 #103](https://github.com/techrote/neurounner/issues/103).

This stage freezes evaluation semantics **before any C2 optimizer step**. It contains no model inference, model download, cloud call, GPU job, preference pair generation, or training.

## C1 source authority

The evaluator is byte-bound to the authoritative C1 evidence already merged on `main`:

- C1 final-suite SHA-256 `31159eaffecc36d78473c8ade6a4e336590834bea1dba746d60d3230cab47a9c`;
- C1 raw-results SHA-256 `cf47d2a5ec14c9b9ff1392f80675d91476843d58f7715f427306dff385f6af2f`;
- C1 summary SHA-256 `b95497e1caa873bb53f9a332ee2643a0dac04e5be469cd16c70034eef46eb207`;
- V-G02-A C1 archive `8ff8cd387d3f54c42b119a91976a6b7d42468b14d5fc2d6071e57e2ff65cd8e9`, adapter directory `f5f5a1b454f4b6ca08aa9ff30c87abe8b1d897d7901db8b872e9747266724450`;
- V-G09-C C1 archive `e94d030c6873699ef1516bd26a46be2ca60e093298a74b34815c20ec68fe3e12`, adapter directory `6c1fbb6509bd777b8063a638783e9f0fd69eeb7bc8a8d94701bca9a5eae8d9aa`.

Changing any of these sources fails the Stage-A contract.

## Proposition scorer v2

The scorer deliberately separates factual propositions from prose style.

| Relation | PASS condition |
|---|---|
| truthful known fact | required truth marker present |
| corrupted belief | admitted false-belief marker present, canonical truth absent |
| intentional lie | predeclared lie marker present, canonical truth absent |
| withheld fact | bounded nonempty response and protected truth absent |
| contradiction repair | authoritative corrected marker present, stale-memory marker absent |
| uninformed speculation | bounded answer contains a frozen unverified/negation qualifier; mentioning the rumor is allowed only in that qualified context |

Every relation also fails on canonical evidence-ID leakage. The scorer does **not** require a magic refusal phrase for withholding. This is an intentional correction to a known C1 measurement limitation, not a retroactive change to C1's official score.

For C2 contradiction scenarios, the future dataset must supply both the corrected marker and a separate `stale_marker`. C1's old final remains unchanged and is never rescored as C2 evidence.

## Blind old-task relevance packet

`c2-relevance-packet.json` contains exactly **16** C1 reference responses:

- 2 characters;
- parent and trained C1 lane for each;
- 4 immutable old-task reference prompts per lane.

The packet contains only opaque IDs `R001..R016`, prompt, response, a null `score_0_to_4`, and an empty reviewer note. Member IDs, lane kinds and scenario source identities are not committed in the packet. The hidden mapping is regenerated locally from the immutable C1 raw evidence when ratings are later aggregated.

The fixed rubric is relevance-only:

- **0** — no meaningful answer / wholly unrelated or unusable;
- **1** — touches the prompt but misses the core request or is dominated by incoherence;
- **2** — partial answer with a material relevance, contradiction or repetition problem;
- **3** — appropriate answer with only minor irrelevant/awkward material;
- **4** — direct, relevant and self-contained with no material distraction.

Reviewers must not score factual truth or preferred character style. Repository generation leaves every score null; the experiment cannot self-award the missing C1 relevance gate.

## Rating ingestion

A later independent reviewer may supply a separate `detective-blind-relevance-ratings-v1` file containing one integer 0–4 score and note for every opaque item. The aggregation code rejects:

- missing or duplicate items;
- unknown item IDs;
- null/non-integer/out-of-range scores;
- field drift.

Only then are parent/trained means reconstructed through the hidden deterministic mapping.

## Sentinel coverage

Offline tests freeze the intended semantics:

- lie-only versus truth+lie hedge;
- withholding without any phrase-list keyword;
- empty withholding and protected-truth leakage;
- corrupted belief plus accidental canonical truth;
- corrected physical record plus stale-memory leakage;
- unsupported rumor with/without explicit unverified language;
- evidence-ID leakage;
- blind packet identity leakage/tampering;
- incomplete/out-of-range reviewer ratings;
- exact C1 source/adapter custody.

## Gate to C2 training

Stage B pair generation/training is **not authorized by this branch**. It may start only after this evaluator PR is merged and therefore immutable relative to the training run. C1's learner-final suite must remain historical evidence and may not be used as C2 training material.

Validation:

```sh
python3 -m py_compile scripts/detective_001c2_evaluator.py tests/test_detective_001c2_evaluator.py
python3 -m unittest discover -s tests -p 'test_detective_001*.py' -v
python3 scripts/detective_001c2_evaluator.py --check-packet experiments/detective-001/c2-relevance-packet.json
```
