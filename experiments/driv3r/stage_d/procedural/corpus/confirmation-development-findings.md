# D-2 preregistered structural confirmation cohort (synthetic; PROVISIONAL)

**Owner scope:** CyberSand #570 D-2, Draft/HOLD PR #598; no native D-2 integration, model inference, GPU/API expense, training, Actions experiment-matrix dispatch, Ready or merge. The M1 common contract remains owner-accepted only at commit `8fa90de6be209ed0b7bd125039e6ec8b332fb1a6`, schema SHA-256 `ebe2fa211148602c614cba0f0a40f285cba797784b70efa035832d833877f80d`. The separate M3-A #601 single real-native BACKTRACK_CURSOR smoke **does not authorize D-2 native execution or native material Undo**.

## Preregistered without examining outcomes

The eight new synthetic structural inputs were frozen in Git commit **`8b2fe242666ba8ee4c6459a43e39de0ef0f582a6` before creating or running `confirmation_study.py`**. This is a chronological within-branch input freeze, **not** an independent third-party blind holdout. The immutable fixture source SHA-256 is `de9ca09eca3ec6a67d106c40e8fb2084b53c719a18259552518bfc4184bfe14c`; `corpus/confirmation-prereg-v1.json` SHA-256 is `fcdf3103ffba3da992d46ceee61ce21829d5a185a75a2fece673e431d079b9fb`. Both source and prereg bytes are checked at execution; outcomes do not edit them.

Eight geometries: diagonal fence, nested courtyard, paired vaults, staggered shelves, misleading scalar preference over a forbidden region, spiral islands, a single-cell upper choke and an impassable separator. Distinct full 32×32 hard permitted masks, original material, fixed seeds (1401–1408) and original positive u16le guidance have exact hashes in the prereg file. These **are not** translations of the previous seven DEVELOPMENT fixtures; nevertheless they were authored by the same researcher, so call them **preregistered synthetic evaluation**, not an unbiased external test.

**Frozen controls / metrics:** identical 44-new-cell painting cap, 120-hop synthetic brush-up bound, existing `pareto_study.py` five policies `weighted-0/2/8/32/guidance-strict` pinned at Git blob `7fe2f1c49f765730354077703c1aae76c40a41b9`. Each receives an identical hard mask, initial material, scalar field and geometry; each is compared with an adjacent-frontier guidance-greedy control using precisely its **actual** number of new material cells. Primary: four-neighbour final connected material components, lower better. Secondary: soft-threshold violations, edits, brush-up hops, failures and physical hard-mask reachability (which is distinct from feasibility within 44 edits). The hard separator must stay disconnected. Preserve all eight cases including failures.

## Provisional first-pass findings; no exact-policy qualification claim

The independently sourced local helper reproduced the previous seven-case DEVELOPMENT report **byte-for-byte** at SHA-256 `4b8d837e5380354314ead7adfa0e390491019e5cd9ec9c119fb7ccd6e6abe4bb`, an important behavioral cross-check. It is **not** byte-identical to the committed Git blob of `pareto_study.py`. Accordingly the new report explicitly sets `policy_source_verified=false` and labels itself `UNQUALIFIED_LOCAL_POLICY_REIMPLEMENTATION`, despite 11/11 new focused local tests PASS and two independent outcome generations with matching bytes. Do not call this exact-head, original-policy, native or third-party-confirmed evidence. Exact Git policy source will set `policy_source_verified=true` when run in a checkout.

The local shadow pass gives:

| Frozen case | Globally hard-mask reachable? | Weighted-0 final components / added cells | Weighted-32 final components / added cells | Strict-guidance components | Equal-edit guidance-greedy components (weight-0 budget) |
|---|---|---|---|---|---|
| diagonal-fence | yes | 1 / 26 | 1 / 28 | 2 | 2 |
| nested-courtyard | yes | 1 / 29 | 1 / 29 | 2 | 2 |
| paired-vaults | yes | 1 / 37 | 1 / 37 | 2 | 2 |
| staggered-shelves | yes | 2 / 0 | 2 / 0 | 2 | 2 |
| misleading-scalar | yes | 1 / 29 | 1 / 29 | 2 | 2 |
| spiral-islands | yes | 3 / 29 | 3 / 29 | 5 | 5 |
| one-cell-choke | yes | 1 / 41 | 1 / 41 | 2 | 2 |
| hard-separator | **no** | 2 / 0 | 2 / 0 | 2 | 2 |

Five of seven hard-mask-reachable cases fully connect at weight 0 under 44 new cells; *staggered-shelves* cannot find an affordable link, and *spiral-islands* reduces five components to three. Hard-mask reachability alone is insufficient to guarantee success within budget. Absolute treatment of the low guidance preference as a ban fails to connect every case. Equal-actual-edit local greedy controls do not connect any initially separated case; that is one baseline, **not** proof of algorithmic optimality or useful artwork. The separated hard-wall negative remains disconnected. Weighted-32 only differs from weight-0 on the diagonal-fence case here (28 vs 26 cells and 9 vs 11 sub-threshold cells), so this cohort does not demonstrate a robust continuous Pareto improvement.

Visual inspection suggests the original raster output remains **thin connector geometry**, not evidence of attractive or useful CyberSand game art. Original PNGs, guidance fields, every event journal and a per-policy matched-control raster were generated locally as **provisional shadow evidence**, not committed as bulk media. No independent human ratings or native snapshots/readbacks are claimed.

## Replay / next qualification

Use a source checkout of `research/570-d2-procedural-discovery`:

```sh
python -m unittest discover -s experiments/driv3r/stage_d/procedural -p 'test_*.py' -v
cd experiments/driv3r/stage_d/procedural
python confirmation_study.py --out /tmp/d2-confirmation
```

Verify the committed `pareto_study.py` actual Git blob `7fe2f1c49f765730354077703c1aae76c40a41b9`, the frozen fixture/source hashes and `policy_source_verified=true`, plus all 8/8 original PNGs and 40 per-policy material-event replays. Independent complete exact-branch test qualification and formal blinded human utility ratings remain to be done. Retain this first-pass provisional report and the failures without retuning the frozen inputs. This is not the unauthorized M4 model or A1/A2/A3 trial.

Reversal distinctions remain unchanged: synthetic brush-up BACKTRACK_CURSOR and RELOCATE_TO_BRANCH preserve material; UNDO_MATERIAL_EDIT is unavailable without a real owner-authorized native history and source-fenced receipt.
