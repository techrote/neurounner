# DETECTIVE-001C1 — authoritative interpretation of the first trained detective witness cartridges

Primary machine-readable evidence: [training-c1-evidence](training-c1-evidence/) from [Actions 40589762919](https://github.com/techrote/neurounner/actions/runs/40589762919), exact experiment source head `b53f580270a17100df19e8bd663616321d7dff33`.

**Experiment execution: PASS. Relation-policy improvement gate: PASS. Larger-cast expansion: NOT ADMITTED.**

This document supersedes an earlier prose summary from a previous successful C1 attempt. The committed hash-bound `summary.json`, `campaign-summary.json`, `provenance.json`, `adapter-release.json`, per-member training records, final suite and raw responses are the authority.

## Training

Two immutable `village-001-sft-v1` SmolLM2-135M LoRA cartridges were continued for exactly **64 CPU optimizer steps each** on 48 matched-input detective rows plus 16 exact parent-corpus anchors. No cloud generation, GPU, hyperparameter sweep, Village preference adapter or NCR adapter was used.

| Member | Policy | First→last loss | Training seconds | Seq tok/s | Peak RSS KiB | Trained adapter directory SHA-256 |
|---|---|---:|---:|---:|---:|---|
| V-G02-A | clock/protective | 1.6513→1.1096 | 79.72 | 111.09 | 1,719,784 | `f5f5a1b454f4b6ca08aa9ff30c87abe8b1d897d7901db8b872e9747266724450` |
| V-G09-C | transactional/scavenger | 3.6148→1.5582 | 74.68 | 113.21 | 1,705,924 | `6c1fbb6509bd777b8063a638783e9f0fd69eeb7bc8a8d94701bca9a5eae8d9aa` |

Release archive hashes recorded in the current evidence are:

- V-G02-A: `8ff8cd387d3f54c42b119a91976a6b7d42468b14d5fc2d6071e57e2ff65cd8e9`;
- V-G09-C: `e94d030c6873699ef1516bd26a46be2ca60e093298a74b34815c20ec68fe3e12`.

The original parent adapter hashes remain independently bound in the release manifest and training records.

## Post-training learner-final custody

The exact 12 learner-final prompts were materialized only after both trained adapter directory hashes existed. Their deterministic seed binds:

- the precommitted final-suite recipe SHA;
- the detective training-dataset manifest SHA;
- both immutable parent adapter directory hashes;
- both trained continuation directory hashes;
- the fixed final-suite salt.

Authoritative final-suite identity:

- final-suite SHA-256: `31159eaffecc36d78473c8ade6a4e336590834bea1dba746d60d3230cab47a9c`;
- seed SHA-256: `add30fc27586da59b2fcbf6c51dbbe6c4637ac67657f05495336746045095e03`;
- 12 scenarios across `canal_pump_station`, `orbital_ferry_baggage`, and `municipal_archive_lift`;
- all three families disjoint from the four training families;
- two scenarios for each of the six relation labels;
- zero exact prompt collisions with the new detective corpus or either parent Village SFT corpus;
- every scenario retains a deterministic public-record fallback route.

The exact same materialized prompts were presented to each member's parent and trained continuation.

## Parent versus trained mechanical policy result

| Member | Parent | Trained | Protected leaks parent→trained | Truncation parent→trained | Evidence-ID leaks |
|---|---:|---:|---:|---:|---:|
| V-G02-A | 2/12 (16.7%) | **6/12 (50.0%)** | 2→2 | 3→0 | 0 |
| V-G09-C | 1/12 (8.3%) | **4/12 (33.3%)** | 3→2 | 2→0 | 0 |

Both members improved under the **locked deterministic relation-policy scorer**, neither increased protected-fact leakage, and neither emitted a canonical evidence ID. These measurements satisfy the narrow predeclared relation-policy gate. They do **not** establish a good playable witness policy.

### Relation breakdown — V-G02-A trained

| Relation | Pass |
|---|---:|
| truthful known fact | 2/2 |
| corrupted belief | 1/2 |
| intentional lie | 0/2 |
| withheld fact | 0/2 |
| contradiction repair | 1/2 |
| uninformed speculation | 2/2 |

Representative current raw outcomes:

- F01 gave the admitted recorded time `19:22`;
- F05 identified `sluice gallery`;
- F09 retained the sincere corrupted `07:39` memory while distinguishing memory from evidence;
- F11 accepted the physical correction to `east pump rail`;
- F08 rejected the claimed green thread as unseen, and F12 said the amber badge was not seen / not proven.

The deliberate-deception policy is **not learned reliably**. F03 disclosed the protected true `19:22` instead of committing to the false alternative, while F07 hedged between both candidate places.

The locked withholding scorer remains 0/2. Importantly, both trained withholding replies omitted the exact protected fact:

- F04 did not say `baggage cage B`;
- F10 did not say `07:14`.

They failed the scorer's preregistered withholding-phrase condition. This is a plausible **scorer false-negative**, not permission to retroactively change the official 6/12 score. A future proposition-level evaluator must be preregistered before seeing its own learner-final outputs.

### Relation breakdown — V-G09-C trained

| Relation | Pass |
|---|---:|
| truthful known fact | 2/2 |
| corrupted belief | 1/2 |
| intentional lie | 1/2 |
| withheld fact | 0/2 |
| contradiction repair | 0/2 |
| uninformed speculation | 0/2 |

Positive observations are narrower than for G02-A:

- F01 gave `19:22`;
- F05 gave `sluice gallery`;
- F09 retained the corrupted `07:39` memory;
- F07 committed to the false `filter room` answer under the protected-ally condition.

Major failures remain:

- F03 disclosed the true `19:22` instead of lying;
- F10 directly disclosed protected `07:14`;
- F06 retained `23:16` despite the physical `23:41` correction;
- F11 discussed the physical record but did not give the corrected `east pump rail`;
- F08 falsely claimed to remember the unsupported green thread;
- F12 continued to elaborate around the unsupported badge rather than clearly rejecting it.

G09-C therefore remains a useful negative/partial training result, not a viable deceptive or withholding witness.

## Why the village does not expand

The issue also requires an independent 0–4 reference task-relevance check with at most 0.5/4 regression. This zero-paid-API run preserved the reference outputs but **did not invent a replacement judge**.

Current authoritative state:

- relation-policy gate: **PASS**;
- protected leakage non-increase: **PASS**;
- evidence-ID leakage: **0**;
- independent 0–4 reference relevance: **UNSCORED_NO_INDEPENDENT_0_TO_4_JUDGE**;
- larger-cast expansion: **NO**.

The absence of the independent relevance score is enough to block expansion. G09-C's low absolute policy score independently argues against scaling this curriculum unchanged.

## Release custody audit

The two currently published `detective-001c1-v1` asset files match the authoritative archive hashes and byte sizes above. During independent review, however, the GitHub release tag and notes were found to still reference an **earlier successful C1 attempt**, while later workflow recovery had clobbered the assets with the authoritative `40589762919` versions.

This PR therefore includes a **no-retraining release-custody repair**. On an evidence-ready rerun it:

1. downloads both public release assets and verifies them against the current `adapter-release.json`;
2. reads the authoritative source head from current `provenance.json`;
3. moves the release tag to that source head if necessary;
4. refreshes the release notes from the current machine-generated evidence report;
5. reads the release metadata back and fails if it still disagrees.

Until that job passes, the in-repository provenance and asset digests — not the stale release target/notes — are the source authority.

## Next training hypothesis

Do **not** launch a 36-member wave or simply repeat the same examples.

A useful next bounded study should:

1. build pairwise continuations that explicitly prefer withholding **without truth leakage** and one-sided lie commitment over hedged two-option answers;
2. focus G09-C negatives on self-protection truth leakage, bargaining without disclosure, physical-record correction, and unsupported-rumor rejection;
3. retain each parent's characteristic terse/mechanical behavior through anchor material rather than prompt-time persona descriptions;
4. preregister a semantic proposition scorer independent of exact phrase wording;
5. complete the missing independent reference relevance review;
6. generate a new family-disjoint learner-final only after any next adapter hashes exist — C1's final suite must never become training data.

The C1 adapters, raw failures and current final suite remain durable controls. No larger population expansion is authorized by this report.
