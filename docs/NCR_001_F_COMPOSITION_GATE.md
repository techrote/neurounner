# NCR-001F G9 — eight-source current-main composition

## Objective

Integration readiness for NCR-001 #67 / #73, without merging or reexecuting an
original model. Four task-owned drafts passed individual source CI, but had not
yet been tested as a single composition on current main. This extends existing
draft #223 with a disposable local Git composition harness, NOT a fifth PR.

## Dependency and source identity

The source-only order is strictly:

1. #199 → #201 → #209 → #212 → #227: original Sprocket forward-only timing,
   corrected original complete ZIP/log custody, source MAC interpretation,
   four-lane planned-phase registry and independent frozen checkpoint lock.
2. #224: original historical Actions event classifier.
3. #226: shared frozen-backbone leased MLP import and stable checkpoint save.
4. #223: original preregistration drift guards plus this composition preflight.

Read the pinned base and original head SHA of EVERY stage from
scripts/ncr_001f_composition_gate.py. No source head is substituted with a
mutable PR branch name. Each Git binary delta is independently fingerprinted
and applied to a DISPOSABLE worktree from freshly fetched current origin/main.
Original source ancestry, changed-file disjointness across independent groups,
original binary diffs and protocol/source-pins Git objects are checked before
a pass is possible. A missing commit, wrong ancestry, changed source files,
overlapping independent PRs, protected material or patch conflict is an
explicit REFUSAL, not a green check.

The resulting machine report identifies the actual current main SHA and the
eight original source heads/diff SHA-256s, with a categorical source-only
status. It deliberately leaves all_domain_tests_passed = null because the
separate CI steps must establish actual combined test results. The Git worktree
is never committed or pushed; it has no GitHub write credentials.

## Source-only CI acceptance

Execute positive and negative fake-Git tests under ordinary and optimized
Python, followed by actual composite-worktree offline test cases for event
attribution, synthetic four-lane MLP atomic load/save, shared decoder lease,
and three original corrected-Sprocket archived ZIP/log integrity. CPU Torch
only. The original model/test workflows are NOT dispatched. This step must not
run model inference, actual model generation, optimizers, held-out scoring or
comparative training. The pre-existing TRAIN-003 held-out-sensitive planner
suite is intentionally NOT re-executed here.

A passing report alone is NOT combined-CI acceptance. Preserve complete
original source/negative job logs and check exact source SHA, Action event,
runner head and distinct tests before declaring source qualification.
If current main or any participating PR head changes, recompose and requalify.

## Scientific and merge boundary

Four model identities and the exact original protocol/source-pins are
unchanged. Original source-MAC proxy, forward-only Sprocket wall timing, frozen
adapter toy reload, original raw custody and current-main source composition
do not measure matched backward/optimizer FLOPs, real three-seed supervised
tokens, learnable advantage, source semantics or independent #72 acceptance.
The historical root-Instruct incomplete tool-call and void.1 EOS-258 impact
remain negative/unmeasured.

train_authorized=false, full_72_conformance_pass=false,
actual_compute_budget_qualified=false. The four task-owned PRs remain
DRAFT/UNMERGED; actual merge-down belongs to the separate authorized worker.
This source-only composition cannot itself authorize training, admission,
PR readiness, release or main integration.

## Preserved first-source CI failure and correction

Initial literal source f513acbe2c275d0dc704df43f364abc76aa99061
failed BOTH no-model workflow runs 41187586980 (push) and 41187591034
(PR) after the original eight fake-Git tests passed 8/8 ordinary and 8/8
optimized. First composition incorrectly required draft-specific #223 anchor
e53ab3a to be on main, impossible while #223 remains draft, rather than
checking the original PR base. Corrected the main ancestry requirement to
12f7bf while retaining draft anchor verification on owned HEAD. Added a
ninth positive Git regression for this exact separate-branch condition.
Original failed jobs remain evidence; later source-composition PASS
requires newly verified exact-head CI.

## Preserved second-source failure and explicit historical index treatment

Source 30249110d71332a1340be121c6fa985e92b4fcf2 passed nine ordinary
and nine optimized fake-Git tests, but its exact-head push Actions
41187686016 REFUSED actual composition at #201 RAG.md. This exposed a
real historical indexing overlap, not a raw evidence or source-model defect:
old #201, #209 and #212 modify the shared README-style RAG index by
strictly additive signposts (2/1/1 lines). Current RAG has since evolved
independently and must NOT be reset to an old source version. G9 now
fingerprints the entire original binary source diff of each old PR and
requires exactly those historical additive RAG changes, while omitting ONLY
these three historical index hunks from the disposable current-main
application. Each omitted path is explicitly disclosed in the report,
with its effective applied diff hashed separately. No old source, binary
archive, historical review or current main RAG was modified. The new
fake-Git tests accept an exact additive historic signpost and refuse a
nonadditive mutation. Source-only PASS remains contingent on fresh CI.
