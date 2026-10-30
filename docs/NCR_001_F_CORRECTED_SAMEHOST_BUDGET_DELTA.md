# NCR-001F — original corrected Sprocket CPU wall-envelope / MAC-proxy source delta

Parent research issue #73, admission issue #72; draft source-only work stacked on original evidence custody PR #201 at 55026d914efd3ef4548da0644ccd27bb8de3074f. Original physical forward-only PR #199 was at 2e2d7c86b066e96207afd6355ddbfbe535d3718e; original corrected three-job physical Actions run is 40904839128. No model, protocol, source, dataset, optimizer, sealed fixture, original receipt or archived failed run is modified.

## What is newly tested

The frozen-host run used exactly three interleaved complete-model forward samples per path (frozen baseline, manually activated **untrained** NCA and untrained gated MLP), one warmup, synthetic source-matched 512- and 2048-unit inputs. On separate four-thread AMD EPYC 7763 runners, the original *within-model* complete-host NCA/MLP median ratios were Chat **1.01038** at 512 and **0.99904** at 2048; root Instruct **1.00594** and **1.00263** respectively. No cross-model pooled performance estimate is admissible.

The new auditor first runs PR #201's independently source- and blob-bound original custodian, requiring the three complete original Actions ZIPs, three complete decoded job logs, immutable SHA-256 / Git-blob / physical report identities, complete original job/run-source metadata, source tests, corrected actual four threads (startup threads were two), frozen-weight integrity and explicitly negative training gates. After those checks it reopens the *already committed original report.json ZIP members*, recalculates medians from each raw nanosecond sample, and derives:

- Original within-model whole-frozen-host NCA/MLP medians and both adapter/frozen ratios.
- An **observed-sample unpaired extreme ratio envelope**: min(three NCA times)/max(three MLP times) through max(three NCA times)/min(three MLP times.
- Whether that observed ratio envelope contains unity, without turning it into any inferential PASS.

That envelope bounds ratios of the **recorded nine path timings** under arbitrary NCA/MLP sample pairing; it is NOT a statistical confidence interval, not a measurement of independent repetitions, and not a guarantee for unobserved runs. Three interleaved samples per path do not justify narrow uncertainty claims. Neither NCA/frozen nor MLP/frozen full-host ratio isolates adapter wall time: the common frozen-backbone forward is included.

## Independent source MAC arithmetic

The pinned but separately stacked PR #98 helper at commit c8139cc36b1105f6bcafd585b766ff470e8c9e1d, script Git blob 6ab8ffb3e3327d7a6a92eb3cfb0075679f2425f1, defines the *source-only* Sprocket three-tap primary adapter geometry: 1280 hidden width, 8×8×4 field and 3 cellular steps per tap, alongside the 256-unit gated MLP.

Independent recalculation in this branch yields **2,371,584 NCA dense/Conv MAC per token** versus **1,966,080 MLP dense MAC per token**, a **1.20625** ratio and **405,504** additional NCA local Conv MACs; trainable parameters are 658,904 NCA and 658,432 MLP (472 difference, 0.0716% of NCA count, within 5% parameter match). Unregistered geometry and using the separately preregistered one-step ablation as the primary are refused. The PR #98 helper is referenced by exact source identity but is **not executed** by this source-only branch; this branch performs its own fixed formula computation.

A source MAC is not a measured FLOP, wall time, training cost, memory bandwidth, optimizer compute or cache traffic. The proxy excludes nonlinearities, normalization, frozen backbone and autograd. A 20.625% increase in counted adapter MAC alongside approximately 1× whole-model median timings is not contradictory, and cannot qualify equal actual compute/training budgets. The original NCA and MLP untrained readout interventions also used different coefficients (0.03/0.05), so this is no learned-quality comparison.

## Fail-closed scope and evidence

New standard-library regression tests verify existing permanent original archive custody, independently repeated per-variant 512/2048 medians, original effective four-thread metadata, two separate Chat and root-Instruct checkpoint hashes, the arithmetic proxy and an adversarial matrix of corrupt median/ratio/raw sample count, zero and boolean nanoseconds, NaNs, missing/extra paths, wrong rotation, changed contexts, changed warmups, unregistered geometry and forged measured-FLOP/compute-admission flags.

The new workflow uses only read access, exact PR literal-head checkout, explicit direct-parent SHA and a five-file allowlist. It refuses changed immutable protocol/source Git blobs or any alteration to committed original ZIP/log evidence. It retains the source-only decision, tests, runner identity and even failed logs as a transient supplemental Actions artifact; original six physical evidence streams already remain in permanent Git custody under #201. A CI green result is **not** new physical inference.

Distinct frozen lanes outside these two models (SmolLM2 and void.1) retain their separate #72 and #73 gates. The root-Instruct original six-token incomplete generated tool prefix remains negative, not valid tool/memory competence. void.1's actual EOS ID is **258**, with historical TRAIN-002b outcome impact **UNMEASURED**. Original earlier startup-threads metadata-defective run #40894595108 remains separately archived by draft #200. No draft stack merge, comparative training or held-out evaluation is authorized.

**Decision:** source-only description of original within-model untrained full-host timing samples and independent MAC arithmetic. **NOT** actual matched training compute, learned usefulness, final #72 source/runtime scientific admission, generated tool/memory semantics, four-model synthesis or training permission.

train_authorized=false; full_72_conformance_pass=false; actual_compute_budget_qualified=false.
