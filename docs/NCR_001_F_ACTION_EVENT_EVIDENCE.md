# NCR-001F — exact-head GitHub Actions event-aware evidence classification

Parent [#67](https://github.com/techrote/neurounner/issues/67), matched controls [#73](https://github.com/techrote/neurounner/issues/73). **Source-only / no model / no training / draft only.**

## Defect and narrow correction

An independent #73 PR-only workflow lookup on 30 October 2026 reported *no NCR-specific pull_request checks* on source-frozen [#99](https://github.com/techrote/neurounner/pull/99) and [#198](https://github.com/techrote/neurounner/pull/198). That finding was correctly scoped to PR-event lookup, but could be misread as the complete CI status. On 31 October, a fresh **all-event exact-head Actions search** established both PR heads also had a **successful source-specific push workflow**. The unrelated green `VILLAGE-001 population preflight` PR workflow is never accepted as NCR provenance.

| Frozen draft | Source head | Genuine source-scoped **push** check | Unrelated **PR** check |
|---|---|---|---|
| #99 void.1 matched gated MLP original receipt | `463b55410a1eb9f962960c605e7fa52844ffc2f5` | [40818768640](https://github.com/techrote/neurounner/actions/runs/40818768640), job `123266025303`: original source-byte receipt audit | [40818773779](https://github.com/techrote/neurounner/actions/runs/40818773779), VILLAGE |
| #198 corrected synthetic CPU original ZIP/log custody | `5ea19b3f9b04fa8542ff3d4b9e70882a811522a9` | [40875571415](https://github.com/techrote/neurounner/actions/runs/40875571415), job `123453388884`: read-only original custody | [40875664960](https://github.com/techrote/neurounner/actions/runs/40875664960), VILLAGE |

The independently checked original source workflows were `.github/workflows/ncr-001f-void-archive.yml` (#99) and `.github/workflows/ncr001f-corrected-cpu-archival-bootstrap.yml` (#198), not pull_request-triggered domain workflows. **No new physical model job was run by this audit.** #99's historical [actual frozen Qwen3 MLP prepass 40523446884](https://github.com/techrote/neurounner/actions/runs/40523446884) is distinct from the later archival push check; #198 is corrected synthetic adapter-only CPU custody, not a frozen-host result.

## Independent, fail-closed procedure

[Auditor](../scripts/ncr_001f_action_event_audit.py) uses read-only GitHub REST to fetch each still-open draft PR head, the **complete per-head all-event workflow inventory** (`head_sha`, 100 per page, explicitly rejects partial pagination and extra runs) and separate job lists for both historical runs. It verifies GitHub repository, PR branch, exact 40-hex source SHA, workflow file path, event type, status, conclusion, attempt and exact job ID/name/sha. PR-event VILLAGE success and real source-specific push success receive **different** machine-readable fields; neither yields scientific or training admission.

[Adversarial offline tests](../tests/test_ncr_001f_action_event_audit.py) reject head/branch swaps, false draft status, changed run IDs, wrong event/workflow, extra/missing/paginated runs, foreign repository, failure/skipped jobs, altered attempts and false attribution. [Read-only exact-head workflow](../.github/workflows/ncr-001f-action-event-audit.yml) runs the tests with normal and optimized Python, verifies the original protocol and source-pin Git blobs, then performs the live API readback. GitHub API errors or new runs at the original historical head are treated as **audit failure/reconciliation-needed**, not silent success. Transient Actions artifacts do not supersede original Git-custodied evidence.

## Limits

This is a historical **event-scope and provenance classifier** for two exact frozen heads only. A successful **push read-only archival check is not a PR-domain check**, a fresh model-host execution, a new original-model no-op/causal acceptance, training-compute measurement, or an independent human scientific review. Preserve original physical/ZIP/log custody and all negative evidence in existing historical branches; no original bytes are rewritten.

Explicit locked conclusions: `train_authorized=false`, `full_72_conformance_pass=false`, `actual_compute_budget_qualified=false`, `model_executed=false`, `optimizer_steps=0`, `heldouts_opened=false`. No protocol, source pins, fixture, matched model parameterization or main merge is changed.
