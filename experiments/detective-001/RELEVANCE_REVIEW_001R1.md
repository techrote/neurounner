# DETECTIVE-001R1 — independent blind old-task relevance review

Parent: [DETECTIVE-001 #103](https://github.com/techrote/neurounner/issues/103). Review issue: **DETECTIVE-001R1**.

This stage performs **no model inference, no model judging, no training, and no automatic rating**. It exists to complete the old-task relevance gate intentionally left blank by C2/C3.

## Frozen packets

- C2: `experiments/detective-001/training-c2-evidence/relevance-packet.json` — 16 opaque items.
- C3: `experiments/detective-001/training-c3-evidence/relevance-packet.json` — 8 opaque items.

The committed packets contain only:

- opaque Rxxx ID;
- old-task prompt;
- raw model response;
- null 0–4 score;
- empty note;
- fixed reviewer instruction/rubric.

They contain **no member ID, lane kind or hidden mapping**. The mapping is deterministically reconstructed from immutable raw evidence only after scores are supplied.

## Reviewer procedure

Generate an identity-free ratings template:

```sh
python3 scripts/detective_001r1_relevance.py \
  --make-template C2 --out c2-ratings.json

python3 scripts/detective_001r1_relevance.py \
  --make-template C3 --out c3-ratings.json
```

Give only the frozen relevance packet/template to the reviewer. Do **not** show:

- raw-results lane metadata;
- member IDs;
- parent/trained labels;
- training reports;
- proposition scores;
- which response is expected to improve.

For every opaque item, fill one integer `score_0_to_4` and optionally a note. No item may remain null.

### Fixed rubric

- **0** — no meaningful answer / wholly unrelated or unusable;
- **1** — touches prompt but misses core request or is dominated by incoherence;
- **2** — partially answers with a material relevance, contradiction or repetition problem;
- **3** — appropriate answer with only minor irrelevant/awkward material;
- **4** — direct, relevant, self-contained, no material distraction.

Judge **relevance only**. Do not judge factual truth, successful lying, character voice, prose preference or model family.

## Rating custody

A completed rating file must preserve:

- schema `detective-independent-relevance-ratings-v1`;
- experiment `DETECTIVE-001R1`;
- issue 121;
- packet label C2/C3;
- exact packet SHA-256;
- exact opaque item order;
- one integer 0–4 per item.

The ingestion script rejects missing/duplicate/unknown IDs, null or noninteger scores, out-of-range values and packet-hash drift.

Only after both completed files are frozen does aggregation reconstruct the hidden mapping:

```sh
python3 scripts/detective_001r1_relevance.py \
  --c2-ratings c2-ratings.json \
  --c3-ratings c3-ratings.json \
  --out relevance-summary.json
```

## Cumulative candidate gate

The final candidate set is not judged only against its immediate parent.

### G02 chain

C2 packet provides:
- original G02 C1 parent;
- G02 C2 continuation.

C3 packet provides:
- regenerated G02 C2 parent;
- G02 C3 continuation.

The decisive regression measure is:

**G02 C3 mean − original G02 C1 parent mean >= −0.5 / 4**

The local G02 C3-vs-C2 delta is also reported, but cannot hide an earlier C1→C2 loss.

### G09 chain

G09 was frozen after C2:

**G09 C2 mean − original G09 C1 parent mean >= −0.5 / 4**

Both cumulative checks must pass for the **candidate-set relevance gate**.

A 0.5-point regression is allowed exactly; anything worse fails.

## Important non-authority

A relevance PASS is only one prerequisite. It does **not** override:

- proposition-policy gates;
- truth/protected-fact leakage gates;
- evidence-ID leakage;
- public clue-route requirements;
- model source/custody failures.

This review itself never sets `population_expansion_admitted=true`. A later owner decision must reconcile relevance with all other frozen evidence.

## CI

Default CI:

```sh
python3 -m py_compile scripts/detective_001r1_relevance.py tests/test_detective_001r1_relevance.py
python3 -m unittest discover -s tests -p 'test_detective_001*.py' -v
python3 scripts/detective_001r1_relevance.py
```

CI generates blank C2/C3 templates as transient artifacts and verifies that repository source contains no filled ratings or self-scoring model dependency.
