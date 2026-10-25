# VILLAGE-001 — cybernetic goblin village foundry

Status: **campaign design / orchestration bootstrap**  
Primary sources: issues #1, #2, #3, #5 and #6  
Population target: **36 final SLM villagers** from **12 Sprocket-derived lineages × 3 sibling specializations**

## Contents

1. **Objective** — scale the proven BABBLE/PERSONA pipeline into a measurable 36-member population.
2. **Evidence inherited from existing work** — identifies which prior results are authoritative and what they imply.
3. **Population design** — defines immutable villager IDs, sibling specializations and lineage rules.
4. **Training and feedback pipeline** — specifies SFT, Sprocket feedback, preference continuation and recovery.
5. **NVIDIA cloud roles** — separates high-volume compilation, independent criticism and sparse adjudication.
6. **Parallelism and capacity** — maps the campaign onto dozens of GitHub-hosted CPU training lanes without conflating runner differences.
7. **Evaluation** — defines hierarchical blind tests that scale beyond the four-cartridge PERSONA tournament.
8. **Evidence, storage and lineage** — preserves raw generations, provider responses, adapter hashes and failed lanes.
9. **Acceptance gates** — states what constitutes a complete campaign and a qualified villager.
10. **Execution decomposition** — gives a dependency-safe workstream split for the follow-on issues.

## 1. Objective

Build a **village**, not one monolithic assistant.

The first production-sized population is 36 small language models:

- 12 durable BABBLE-001 / BABBLE-002 source lineages: G01–G12;
- 3 independently trained sibling specializations per lineage;
- one immutable orchestrator-owned member ID for every final cartridge;
- independent SFT and preference-continuation evidence for every member;
- common held-out tests plus member-specific Sprocket feedback;
- population-level blind evaluation that distinguishes learned behavior from merely distinctive wording.

The default body remains the pinned `HuggingFaceTB/SmolLM2-135M-Instruct` revision already qualified by TRAIN-001, PERSONA-001, PERSONA-002 and TRAIN-004. LoRA remains the default because TRAIN-004 showed full tuning was only about 1.10× slower and 1.77× larger in peak RSS on a controlled CPU comparison, but produced a ~54× larger resumable checkpoint.

Full-parameter training is therefore reserved for later selected elders/survivors if a measured LoRA capacity limit appears. It is not the default village construction method.

## 2. Evidence inherited from existing work

### TRAIN-002b (#1)

Authoritative lesson: modest LR/rank/dropout changes did not beat normal seed variation; **training-material weighting did**. The failure-weighted lane was the best controlled perturbation.

VILLAGE-001 therefore spends diversity budget on curriculum and feedback differences before hyperparameter sweeps.

### BABBLE-002 (#2)

Authoritative corpus:

- 12 personas;
- 100 positive examples per lineage;
- 1,200 total positive SFT examples;
- one generic/sanitized negative paired with every positive;
- exact guide/packet/babble lineage;
- raw NVIDIA provider history retained.

This is the common ancestral corpus, not a disposable staging artifact.

### PERSONA-001 (#3)

Four same-body LoRA cartridges trained from different BABBLE lineages reached:

- 46.9% raw blind identification;
- 41.7% style-neutral behavior-only identification;
- 25% four-way chance.

This established that the Sprocket → compiled corpus → 135M LoRA path can produce measurable behavioral phenotypes.

### PERSONA-002 (#6)

Pairwise anti-sanitization continuation improved the fixed tournament:

- raw blind identification: 46.9% → 71.9%;
- behavior-only: 41.7% → 51.0%;
- G03↔G12 behavior cross-confusions: 18 → 5;
- abstract-unfamiliar behavior-only: 25% → 81.25%.

This makes preference continuation a normal maturation stage rather than an optional embellishment.

### TRAIN-004 (#5)

Full-parameter 135M CPU training is technically practical, but its resumable checkpoint is approximately 1.6 GB versus ~30 MB for LoRA. The village uses modular LoRA cartridges by default so dozens of independent inhabitants remain cheap to store, compare, swap and resume.

## 3. Population design

Canonical member IDs are:

```text
V-G01-A  V-G01-B  V-G01-C
...
V-G12-A  V-G12-B  V-G12-C
```

Human-readable names are decorative metadata and may be proposed later. **Names never become identity or lineage authority.**

Each sibling inherits its Gxx BABBLE lineage and then receives a different curriculum emphasis:

| Sibling | Working specialization | Overweighted surfaces | Purpose |
|---|---|---|---|
| A | failure-hardened | failure/recovery, conflicting motives, physical reaction | persistent, awkward recovery behavior |
| B | local tinkerer | local memory, mundane low stakes, physical reaction | habits, territorial/local detail, maintenance behavior |
| C | social scavenger | conversation/social, embodied decision, conflicting motives | bargaining, opportunism, social friction and choices |

These are training ecologies, not prompt-time personas. The finished SLM receives no hidden member description during held-out evaluation.

Every member gets a deterministic campaign seed derived from the campaign seed + lineage + sibling code. No two sibling lanes share both corpus order and seed.

## 4. Training and feedback pipeline

### Stage A — member-specific corpus growth

For every member:

1. Start from the 100 BABBLE-002 rows for its Gxx lineage.
2. Preserve the original positive and generic-negative pair.
3. Add approximately 60 member-specific examples with the sibling surface weighting above.
4. Compile new examples from the **raw BABBLE-001 Sprocket conversations plus the durable packet**, not from packet prose alone.
5. Keep source babble IDs, source packet ID, source guide ID, member ID, compiler call ID and sibling specialization on every new row.
6. Reject exact duplicates deterministically; archive malformed provider output rather than deleting it.

Target initial SFT material: approximately **160 examples/member**, or ~5,760 positive rows across the village.

### Stage B — 36-way SFT wave

Run one independent GitHub-hosted `ubuntu-24.04` job per member.

Default recipe inherits PERSONA-001:

- SmolLM2-135M-Instruct exact pinned revision;
- LoRA r8 / alpha16 / dropout0.05;
- 2 epochs;
- assistant-only loss;
- deterministic schedule;
- common optimizer recipe;
- member-specific seed;
- complete training telemetry and adapter hash.

The workflow may execute in waves (for example 12–16 jobs concurrently), but each member remains an independent training lane.

### Stage C — Sprocket feedback pass

After SFT, every provisional member answers the common held-out suite plus a small member-specific probe set.

For each provisional response:

- three independent Sprocket workers receive the situation and response;
- each Sprocket produces a **tiny bounded feedback record**, not unrestricted prose;
- raw Sprocket feedback is retained even when malformed;
- the orchestrator owns member/scenario/parent identities;
- Sprocket is never allowed to rewrite lineage or scores.

Suggested feedback record:

```json
{
  "keep": ["short behavioral element"],
  "mutate": ["short behavioral element"],
  "failure_guess": "one concrete failure mode",
  "genericness": "low|medium|high"
}
```

The purpose is not to make Sprocket an authoritative judge. It is a cheap, lineage-related mutation/feedback source whose disagreements become training material.

### Stage D — cloud compilation into preference pairs

For each member, combine:

- ancestral BABBLE rows;
- provisional SLM response;
- three raw Sprocket feedback records;
- deterministic scenario/member lineage.

Produce approximately **60 chosen/rejected preference pairs per member**.

The chosen target must improve the behavior while retaining odd lineage-specific details. The rejected target may be the provisional answer, a genericized repair, or a deterministic negative selected by the compiler. The compiler must explain which Sprocket feedback was accepted or rejected.

### Stage E — 36-way preference continuation

Continue each member independently from its SFT adapter using the PERSONA-002 objective:

- reference-free length-normalized pairwise logistic objective;
- chosen-response NLL anchor;
- identical optimization recipe across members;
- only the member corpus differs.

This creates a second wave of 36 training jobs. The normal campaign therefore contains **72 independent CPU training lanes** before bounded recovery/retry jobs.

### Stage F — selective recovery, not silent replacement

Failed or weak members remain evidence. Recovery may:

- resume a checkpoint;
- repair missing corpus shards;
- rerun an infrastructure failure;
- execute one predeclared curriculum rescue.

Do not silently regenerate a weak villager until it passes. Preserve failed phenotypes and compare them.

## 5. NVIDIA cloud roles

All cloud requests use the existing NVIDIA OpenAI-compatible endpoint and the repository `NVKEY` secret. API output remains a remote/unpinned backend and is archived under the dataset filing contract.

Current model roles for the campaign are deliberately asymmetric:

| Role | Model | Use |
|---|---|---|
| high-volume compiler | `minimaxai/minimax-m3.1-flash` | member-specific corpus expansion, repair, chosen/rejected pair synthesis |
| independent critic | `nvidia/nemotron-3.5-bolt-e4b` | behavior-vs-style criticism, genericization checks, disagreement review |
| sparse adjudicator | `nvidia/nemotron-3.5-super-vl-120b-a12b` | only disputed compiler/critic cases, elite selection and final population review |

Model availability is time-sensitive. Before a campaign launch, the workflow must execute a tiny provider preflight for every configured cloud model and fail closed if a model has disappeared or changed protocol.

Cloud concurrency is intentionally much lower than CPU-training concurrency. The compiler/critic stages should be serialized or narrowly bounded to avoid rate-limit storms and to preserve restartable per-call evidence.

No cloud model gets canonical lineage authority.

## 6. Parallelism and capacity

The village intentionally separates **data-generation parallelism** from **training parallelism**.

### Cloud stages

- bounded concurrency: 1–2 calls/jobs;
- retry with preserved failed responses;
- chunked member work;
- partial-success recovery;
- no all-or-nothing rerun if 35/36 members already have valid corpus shards.

### CPU training stages

- 36 independent SFT lanes;
- 36 independent preference lanes;
- default `max-parallel`: 12;
- each job has its own adapter/checkpoint/result artifact;
- aggregation happens only after all lane artifacts are independently validated.

Runner CPU identity is recorded. Throughput from unlike CPUs must not be interpreted as model-quality evidence.

At PERSONA-001/002 scale, 135M LoRA training used roughly 1.6–2.3 GiB peak RSS, so the standard four-vCPU runner class has substantial memory headroom. The primary scale risk is GitHub job concurrency / wall-clock scheduling, not per-lane memory.

## 7. Evaluation

A 36-way flat identity tournament alone is not informative enough. VILLAGE-001 uses hierarchical blind evaluation.

**Final-holdout custody:** the common T01–T24 suite may be used during SFT diagnosis and post-SFT preference maturation, so it is not authoritative final held-out evidence after Stage E. VILLAGE-001E therefore generates responses from every final released cartridge plus the untuned base on a separate fixed F01–F24 suite that is checked against the durable SFT and preference training inputs before inference. The raw and style-neutral final tournament uses only those fresh F-suite responses.

### Level 1 — lineage identification

Blindly predict G01–G12 from responses.

- chance: 8.33%;
- run on raw text and style-neutral behavior summaries;
- record full 12×12 confusion matrices.

### Level 2 — sibling identification

Given the correct Gxx lineage, identify A/B/C.

- chance: 33.33%;
- measures whether member-specific feedback/curriculum produced behavior beyond the common ancestor.

### Level 3 — nearest-neighbor confusion

For every member, record the five cartridges most often confused with it. This is more useful than a single population accuracy number when 36 classes are present.

### Level 4 — genericness and task relevance

Retain PERSONA-001/002 scoring concepts:

- persona/member fit;
- generic-assistant regression;
- task relevance;
- raw vs behavior-only gap;
- abstract-unfamiliar transfer;
- failure/recovery transfer.

### Level 5 — village ecology probes

Run fixed pair and small-group interactions with no hidden persona prompt.

Measure:

- dominance/deference patterns;
- willingness to share or hoard information;
- conflict persistence and repair;
- convergence into generic assistant voice;
- behavioral collapse when talking to another trained villager;
- cross-lineage contamination.

Ecology results do not replace single-agent qualification; they are an additional deployment-facing layer.

## 8. Evidence, storage and lineage

All durable VILLAGE-001 data follows `docs/DATASET_FILING_SYSTEM.md`.

Expected archive:

```text
datasets/village-001/
  README.md
  manifest.json
  ancestors/
  member-corpora/
  sprocket-feedback/
  provider/
    minimax/
    nemotron-bolt/
    nemotron-super/
  training/
  preference/
  evaluation/
  provenance/
    source-artifacts.json
    hashes.sha256
```

Per member, preserve:

- immutable member ID and ancestor Gxx;
- every source BABBLE ID used;
- SFT corpus hash;
- preference corpus hash;
- exact base revision;
- training seed and recipe;
- runner CPU/model telemetry;
- adapter archive hash;
- raw held-out outputs;
- raw Sprocket feedback;
- raw NVIDIA requests/responses/repairs;
- evaluator requests/responses;
- qualification verdict.

Adapters remain outside normal Git history and should be published as release assets or a model registry with an in-repo manifest and hashes.

## 9. Acceptance gates

### Campaign-complete gate

The campaign is complete only when:

- all 36 member IDs exist in the canonical manifest;
- 36 SFT lanes have terminal evidence;
- 36 preference lanes have terminal evidence or an explicitly recorded terminal failure;
- all successful cartridges have immutable hashes and retrievable assets;
- all cloud and Sprocket source records used for training are archived;
- blind evaluation covers every successful member;
- no Actions artifact is the sole surviving copy of authoritative text evidence.

### Initial population-quality gate

These are first-pass gates and may be tightened after the complete baseline exists:

- overall behavior-only Gxx lineage identification ≥ **16.7%** (2× twelve-way chance);
- overall behavior-only sibling identification ≥ **41.7%**;
- population mean genericness must improve versus the untuned base;
- no qualified member may regress task relevance by >0.5/4 against its SFT parent after preference continuation;
- preference continuation must not qualify solely because raw/style accuracy improved while behavior-only identification regressed;
- at least 30/36 members should pass individual quality checks before the population is called a usable village.

The remaining members are still archived as failures/oddities, not erased.

## 10. Execution decomposition

The campaign should be implemented as five narrow workstreams:

1. **#10 — VILLAGE-001A: population contract and corpus foundry**  
   Deterministic 36-member manifest, member-specific corpus schema, Sprocket feedback schema, NVIDIA provider abstraction and archive contract.

2. **#11 — VILLAGE-001B: source/feedback generation**  
   Generate member-specific Sprocket material and bounded cloud-compiled SFT/preference shards with restartable partial recovery.

3. **#12 — VILLAGE-001C: 36-lane SFT wave**  
   Generalize the proven PERSONA trainer to arbitrary member IDs, train all 36 independent SFT cartridges and publish hashes/artifacts.

4. **#13 — VILLAGE-001D: feedback preference maturation**  
   Collect Sprocket feedback against provisional cartridges, compile pairs, run 36 preference-continuation lanes and preserve before/after evidence.

5. **#14 — VILLAGE-001E: hierarchical tournament and village ecology**  
   Run clan/sibling blind tournaments, behavior-only normalization, confusion analysis, genericness/relevance gates and fixed multi-agent ecology probes.

TRAIN-003 / PR #8 is complementary and can proceed in parallel. Its semantic-mechanical compiler corpus should not become a hidden dependency of VILLAGE-001 unless a later issue explicitly imports it.

---

The governing principle is: **high-volume weirdness from Sprocket, deterministic identity from the orchestrator, bounded synthesis/criticism from multiple NVIDIA models, cheap independent CPU training at population scale, and blind behavioral evidence before a goblin is considered real.**


### Stage-B repromotion hygiene

The first VILLAGE-001B promotion creates the durable `datasets/village-001/` root. Later
stages are additive beneath that same root, so **a future B rerun must never replace the
whole dataset directory**.

After later-stage evidence exists, B repromotion is permitted only when:

- the candidate independently passes the complete 36-member B validation contract;
- downstream preference/final manifests carry the same immutable member-contract SHA;
- if SFT evidence exists, every candidate `member-corpora/V-*.jsonl` SHA-256 exactly
  matches the `training_dataset_sha256` recorded by the corresponding trained cartridge;
- only B-owned paths are replaced transactionally;
- D/E-owned paths, manifests, evaluation evidence and the newest-stage root README survive;
- B/D/E hash inventories are regenerated and independently verified before the staged tree
  replaces the live tree.

A same-ID or same-member-contract corpus is **not** sufficient compatibility once models
have been trained. Byte-different B training data beneath existing SFT/preference/final
evidence is an ancestry violation and must fail closed.

Implementation: `scripts/village_001b_promote.py` / issue #31.
