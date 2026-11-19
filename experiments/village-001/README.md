# VILLAGE-001 experiment control

VILLAGE-001 scales the BABBLE/PERSONA line into a 36-member Sprocket-derived SLM population.

Authority:

- campaign design: [`docs/VILLAGE_001_CYBERGOBLIN_FOUNDRY.md`](../../docs/VILLAGE_001_CYBERGOBLIN_FOUNDRY.md)
- epic: #9
- execution issues: #10, #11, #12, #13, #14
- machine-readable configuration: [`config.json`](config.json)
- deterministic member compiler: [`scripts/village_001_manifest.py`](../../scripts/village_001_manifest.py)

The initial bootstrap intentionally does **not** launch the full cloud-generation/training campaign. The preflight first proves that all 36 immutable member identities resolve to the durable BABBLE-001/BABBLE-002 ancestry.

Expected member IDs are `V-G01-A` through `V-G12-C`.

The final durable data archive will live at `datasets/village-001/`. Generated adapters remain outside normal Git history and are represented by hashes/retrieval metadata in the repository.

## VILLAGE-001A contract gate

Issue #10 adds two explicit row contracts:

- [`member-corpus-row.schema.json`](member-corpus-row.schema.json) — member identity, ancestral lineage, positive/negative target and provider provenance;
- [`sprocket-feedback.schema.json`](sprocket-feedback.schema.json) — post-SFT raw Sprocket feedback with immutable parent/member identity.

`scripts/village_001_validate.py` is the deterministic semantic validator. It checks constraints JSON Schema alone cannot express, including member-ID ↔ lineage/sibling consistency, duplicate normalized inputs, source-guide mismatch and the rule that parsed feedback is required when a feedback record is marked valid.

`scripts/village_001_provider_preflight.py` performs sequential protocol checks against the configured NVIDIA compiler, critic and adjudicator. Authorization is never written to disk; all archived request/response/error records are scrubbed against the live secret before serialization. The provider gate fails closed if the secret is absent, a configured model is unavailable, transport fails, or the provider response shape is incompatible.

Provider preflight evidence is written under `experiments/village-001/provider-preflight/run-<github-run-id>/` by the VILLAGE-001A workflow.

`scripts/village_001_gate.py` is the reusable fail-closed launch gate for #11 onward. It requires a durable successful provider preflight and verifies that its endpoint, roles and exact configured model IDs still match the current campaign configuration. Any later change to a provider model therefore invalidates the old gate until a new preflight is archived.


## VILLAGE-001B member-corpus foundry

Issue #11 expands each immutable member from its 100-row BABBLE-002 ancestral shard to a 160-row SFT corpus.

- A/B/C use different fixed surface quotas; this is curriculum diversity, not a hyperparameter sweep.
- MiniMax M3.1 Flash compiles the 60 new rows/member from the exact BABBLE-001 Sprocket conversations, durable packet, and sibling specialization.
- Nemotron 3.5 Bolt independently scores behavioral fidelity, genericness, and style dependence.
- Borderline cases alone are sent to Nemotron 3.5 Super for sparse adjudication.
- Failed or adjudicator-rejected rows are repaired by the compiler and re-reviewed; unresolved rows fail closed.
- All provider calls are retained before parsing and repair.
- The Actions matrix has 36 member jobs but cloud concurrency is one. A manual rerun may provide a prior workflow run ID to reuse complete member artifacts and resume partial ones.
- Publication is all-or-nothing: the assembler requires 36 canonical 160-row shards before replacing the durable `datasets/village-001/` stage.

The corresponding implementation is `scripts/village_001b_generate_member.py`, `scripts/village_001b_assemble.py`, and `.github/workflows/village-001b-member-corpora.yml`.


## VILLAGE-001C CPU SFT village

Issue #12 turns the 36 promoted VILLAGE-001B member corpora into independent SmolLM2-135M LoRA cartridges.

- Body/revision remain exactly the qualified PERSONA baseline.
- Training is 2 epochs, LoRA r8/alpha16/dropout0.05, assistant-only loss, 320 optimizer steps/member.
- Each member uses the immutable seed derived by `village_001_manifest.py`; there is no hyperparameter sweep.
- The common 24-scenario PERSONA-001 held-out suite is reused with the same neutral system prompt. No lineage/member description is supplied during evaluation.
- Matrix concurrency is 12 GitHub-hosted `ubuntu-24.04` CPU runners.
- Every 40 steps, the lane checkpoints adapter weights, optimizer state, Torch RNG state, next schedule step and cumulative telemetry. Failed lanes can resume from prior workflow artifacts.
- Completed lanes retain their full loss trace, source corpus hash/IDs, CPU/package identity, peak RSS, throughput, 24 raw held-out responses and adapter directory hash.
- The population assembler independently verifies all 36 lanes and packages deterministic adapter archives.
- Adapter bytes are published under release tag `village-001-sft-v1`; hashes/results/aggregate evidence are committed under `experiments/village-001/sft-evidence/`.

The workflow auto-starts only when the durable VILLAGE-001B dataset manifest/member shards are promoted, not merely when this training implementation is merged.


## VILLAGE-001D Sprocket-feedback maturation

Issue #13 feeds each provisional SFT villager back through Sprocket and then applies the proven PERSONA-002 preference continuation objective.

**Completed and independently read back.** The [accepted D report](preference-evidence/REPORT.md), [acceptance record](recovery-20261118/review/stage-d-acceptance.json) and [published readback](recovery-20261118/review/published-readback.json) retain all 36 terminal members and the accepted [adapter release manifest](preference-evidence/adapter-release.json). All 36 passed the frozen relative admission gate, which permits Stage E evaluation and does not establish a usable village. **0/3,456 Sprocket feedback records were schema-valid**; every raw reaction was preserved and the preference pairs were independently curated. Individual metric declines and failed provider attempts remain in the evidence.

Resource flow:

1. **36 local feedback lanes, max 12 parallel.** Each pinned Sprocket instance reacts three times to the member's 24 common held-out responses plus eight held-out cross-sibling member probes: 96 raw reactions/member, 3,456 total.
2. **36 NVIDIA pair-compilation lanes, cloud concurrency 1.** MiniMax compiles 60 chosen/rejected pairs/member from both the common T-suite feedback and all eight member-specific P probes (24/60 pairs are P-probe-derived); Bolt independently audits behavioral validity; only borderline pairs go to Nemotron Super; rejected pairs are repaired and re-audited.
3. **36 CPU preference-training lanes, max 12 parallel.** Each member runs 120 deterministic pairwise optimizer steps with the PERSONA-002 beta/margin/NLL-anchor recipe and 30-step restart checkpoints.
4. **36 blind before/after evaluations, cloud concurrency 1.** Parent and matured responses are shuffled under X/Y labels. Bolt scores task relevance, genericness, raw lineage fit, style dependence, behavior-only fit and A/B/C sibling prediction.
5. **Durable assembly/release.** All feedback, pair lineage, provider calls, before/after evidence and cartridge hashes are retained. Regressed members are not silently replaced.

The eight member probes are genuinely held out from the target cartridge: four deterministic cloud-expansion situations are selected from each of the two same-lineage sibling corpora after excluding every normalized input seen by the target member.

A VILLAGE-001D member passes the maturation gate only if blind task relevance does not regress by more than 0.5/4 and an apparent raw-lineage gain is not merely a style-only gain accompanied by worse behavior-only fit. Full 12-way lineage and 3-way sibling confusion matrices remain the responsibility of VILLAGE-001E (#14).

Durable outputs:

- feedback: `datasets/village-001/sprocket-feedback/`
- preference pairs: `datasets/village-001/preference/`
- raw NVIDIA pair evidence: `datasets/village-001/provider/preference/`
- curation records: `datasets/village-001/preference-quality/`
- population evidence: `experiments/village-001/preference-evidence/`
- matured adapter release: [`village-001d-recovery-40613393744-20261118`](https://github.com/techrote/neurounner/releases/tag/village-001d-recovery-40613393744-20261118)

## VILLAGE-001E final tournament and village ecology

Issue #14 is the terminal qualification stage. It consumes the immutable 36-member `village-001d-recovery-40613393744-20261118` population recorded in the accepted D adapter manifest and does not retrain or replace failed members.

**Completed on 19 November 2026; usable-village verdict FAIL.** The [published final report](final-evidence/REPORT.md) and [independent readback/interpretation](qualification-20261119/review/COMPLETION.md) retain 888 fresh responses and 180 ecology turns. Behavioral lineage passes at 222/864, but conditioned sibling recognition is 310/864 and only 14/36 members pass every individual criterion. All 36 pass raw relevance, raw genericness and D maturation; the remaining individual failures concern recognition. The [qualification release](https://github.com/techrote/neurounner/releases/tag/village-001e-final-20261119) retains all evidence and references the unchanged D adapters. [Stage F #256](https://github.com/techrote/neurounner/issues/256) is a distinct bounded diagnostic continuation.

- After preference maturation, all 36 released cartridges plus the untuned BASE answer a fresh `F01`–`F24` suite that was not used for SFT or preference training. Those responses are evaluated twice: raw output and compiler-neutralized behavior summaries.
- Level 1 is blind 12-way ancestral-lineage identification.
- Level 2 is a separate blind A/B/C sibling classification with the correct ancestral lineage supplied for every response; member IDs and true siblings remain hidden. Its denominator is all 864 member responses, not the subset with a correct Level 1 lineage prediction.
- The assembler emits full 12×12 lineage and 3×3 conditioned-sibling matrices. The 36×36 member matrices combine the blind lineage prediction with the separate true-lineage-conditioned sibling prediction; their accuracy measures joint correctness across those two tasks, not an additional unconstrained blind 36-way classification. Exactly five nearest member confusions are retained for every villager.
- Population gates retain the original literal thresholds: behavior-only lineage accuracy >= `0.1666666667` and conditioned behavior-only sibling accuracy >= `0.4166666667`. Across 864 member responses, these require at least **145/864** and **361/864** correct predictions respectively; substituting exact `1/6` or `5/12` would change the boundary by one hit. The remaining gates require mean raw genericness below the untuned BASE measured on the same fresh `F01`–`F24` suite and at least 30/36 individual quality passes.
- Ecology uses 12 deterministic same-lineage pairs and 12 cross-lineage three-agent groups for three rounds, with opaque handles and no prompt-time member/persona descriptions.
- Ecology records differentiation, generic convergence, sharing/hoarding, conflict persistence/repair, cross-lineage contamination, dominance/deference and outcome progress.
- Every failed or ambiguous villager remains in every population denominator, in the evidence and in the final model registry; failures are not dropped or silently replaced.

Authoritative outputs are promoted under `experiments/village-001/final-evidence/`, including raw provider transcripts, `confusion-matrices.json`, `per-member-verdicts.json`, population/ecology reports, `model-registry.json`, dataset provenance and hashes.


## VILLAGE-001F matched mask and feedback diagnostics

**Completed on 19 November 2026; working-effect criteria NOT MET.** The [complete diagnostic interpretation](diagnostic-20261119/review/COMPLETION.md) and [independent acceptance](diagnostic-20261119/review/final-acceptance.json) retain six matched member pairs, 1,440 optimizer steps, 300 new Q responses, 36 native Sprocket outputs and complete raw judging evidence. Corrected and reproduced D both average 1.75/4 relevance: exact gain zero, three positive member means, worst member −0.25. Both Sprocket prompts produce 0/18 structured critiques; the compact prompt has no structure-required grounded/actionable gain.

The [input release](https://github.com/techrote/neurounner/releases/tag/village-001f-pilot-inputs-20261119) publishes all twelve new adapters separately from historical D. The [final release](https://github.com/techrote/neurounner/releases/tag/village-001f-mask-sprocket-20261119) preserves all results, with 53 logical provider requests against the 114 ceiling. Native pairing, numerical reproduction, fixed-forward mechanisms, identical-text rating variability and contextual token-limit qualifications remain visible. This bounded diagnostic does not qualify a replacement village; epic #9 remains open.
