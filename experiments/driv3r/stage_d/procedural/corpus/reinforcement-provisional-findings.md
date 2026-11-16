# D-2 post-freeze bridge reinforcement — 15 November 2026

**Status: exploratory DEVELOPMENT synthetic research only.** This is a new post-freeze algorithm; the original frozen eight-case fixtures and registered baseline policy remain unchanged. Input archive SHA-256: b7048f1e6dcb27956d6cb23b96aea57f84c5f7cf5f7e7c54bfae9deb10186b03. Its baseline has policy_source_verified=false / UNQUALIFIED_LOCAL_POLICY_REIMPLEMENTATION. Results MUST NOT be promoted to confirmation, native World execution, material receipts, learned model benefit or human artwork acceptance.

## Question and method

Can a small number of extra cells reinforce single-cell material bridges? The new reinforcement_study.py independently reads 32x32 original and weighted-0 baseline PNGs from the archived provisional evidence; verifies PNG and u16le SHA-256s, original material preservation and hard permissions; then computes *all* four-neighbour articulation points with iterative Tarjan. A bounded cardinal min-new-edit bypass is sought around each critical pivot, and is applied only when the complete bypass demonstrably lowers whole-graph articulation count. Radius 3, at most 8 added cells per proposed bypass, 24 total new cells per geometry, explicit terminal reason. This is only a synthetic path/edit plan, NOT a native brush-up route or host-owned Undo.

Each case has exact same actual new cell count for reinforcement and two controls: greedy high-u16 guidance-first frontier, and seeded random legal frontier. The script preserves all original material, stores ordered planned edits, cardinal routes, before/after material hashes, new material pixels, component count, articulation count and per-cell soft-threshold misses. Equal total edits do NOT imply equal route or native execution cost.

## Complete provisional eight-case comparison, 24-cell cap

| Case | Extra edits | Prior articulation cells | Reinforce | Guidance-first | Random legal frontier |
| --- | ---: | ---: | ---: | ---: | ---: |
| diagonal-fence | 23 | 28 | 5 | 33 | 30 |
| nested-courtyard | 24 | 31 | 9 | 40 | 38 |
| paired-vaults | 24 | 39 | 15 | 53 | 41 |
| staggered-shelves | 0 | 0 | 0 | 0 | 0 |
| misleading-scalar | 24 | 31 | 9 | 42 | 38 |
| spiral-islands | 24 | 32 | 7 | 31 | 37 |
| one-cell-choke | 24 | 43 | 19 | 55 | 44 |
| hard-separator | 0 | 0 | 0 | 0 | 0 |

The new policy lowers the articulation count on six of eight archive cases. Both zero-edit cases are explicit negative results. The source guidance-first control uses no below-threshold cells on five of the six nonempty cases, while reinforcement uses 10–24 such cells; misleading-scalar has all 24 below threshold in all policies. This confirms a topology-versus-soft-guidance trade-off under synthetic assumptions. These cases are *the same structures motivating the policy*, so no held-out generalization or aesthetic-quality conclusion is warranted. Fragility/articulation is a geometric proxy, not a test of simulated material load-bearing or art utility.

## Reproducibility and custody

Local isolated execution with exact new Git blob bytes: 11/11 tests PASS when the original ZIP is available, including dense 32x32 iterative Tarjan, 4-neighbour routing, permission refusal, budget-zero STOP, exact-edit paired controls, repeat deterministic report, and hostile guidance-byte tampering. Original local report SHA-256: e4e3faa2cb898eb645cd9844204b46d6db94e03c746d9d9ae836fab3ef69bfec. The output's qualification tag is POST_FREEZE_EXPLORATORY_PROVISIONAL_ONLY and includes the original archive SHA; archived evidence stays outside Git.

From the procedural directory with the original historical ZIP supplied separately:

    python -m unittest -v test_reinforcement_study.py
    python reinforcement_study.py --archive /path/to/provisional-evidence.zip --budget 24 --out /tmp/d2-reinforcement-report.json

Two archive-based tests will skip when provisional-evidence.zip is not adjacent to the local test module; the remaining nine tests run without the historical ZIP. Do not mistake an optional skip for complete corpus qualification. The earlier full exact-head D-2 suite and actual frozen-policy eight-case regeneration still remain distinct outstanding qualification gates.

## Gates and follow-up

M1 D-COMMON-001 owner-accepted source: 8fa90de6be209ed0b7bd125039e6ec8b332fb1a6 / schema SHA-256 ebe2fa211148602c614cba0f0a40f285cba797784b70efa035832d833877f80d. M2 source-pinned offline preparation is authorized; the separate M3/M4 native/Q8 work does NOT confer D-2 native or model-experiment authority. No native Undo, training, API spend, GPU work, experiment-matrix dispatch, merge or Ready promotion.

Next: qualify the complete exact-branch source and regenerate frozen eight-case results using the original committed policy, then prepare genuinely independent post-freeze novelty geometries and human material/artwork usefulness review.