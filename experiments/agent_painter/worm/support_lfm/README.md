# WORM-001 LFM support: local action decisions v1

Authority: [WORM-001 #612](https://github.com/techrote/cybersand/issues/612) and [owner 15-Nov pivot](https://github.com/techrote/cybersand/issues/568#issuecomment-6891678794). **Draft/HOLD; no model optimizer, native application, production provider or historical #574 edits.**

This is a runnable **W0 local-action supervision v2** dataset, paired legal baselines, and a seed/obstacle-matched **96×96 non-native PNG gallery**. It imports the shepherd-owned W0 host at `../common/worm.py` (initial qualified source `4bb9d6e1d0d0d78183f87d1e3c29648edafacf61`). The host—not this curriculum—owns obstacles, bounds, legal actions, raw-invalid fallback, Stop and rendering. Re-qualify if W0 changes. This code never accesses the old whole-scene TRAIN/DEVELOPMENT/FINAL.

## Action/observation contract

Same experimental W0 `cybersand.worm.observation-action/0-experimental`: `STRAIGHT LEFT RIGHT BRANCH RETRACE STOP`; heading 0 east/1 south/2 west/3 north; 5×5 local `#` obstacle/out-of-bounds, `+` previously occupied, `.` fresh, `@` current; `legal`, `last_actions`, `tick` and `family`. Teacher is a deterministic, *untrained* heuristic: seek fresh cells, sometimes branch at junctions, family-specific zigzag/straight preferences, retrace at blocked/revisited local fronts. Teacher labels are always legal; this is not a claim that all its trajectories are visually good. Some real decisions may be off-policy states drawn from W0 `wall`/`fidget` demonstrations.

- `train-v2.jsonl`: 12 distinct obstacle-world seeds 1101–1112, three families, two non-model source policies, first 32 observed decisions or until Stop; exact local prompts duplicated within TRAIN or present in DEVELOPMENT are removed.
- `development-v2.jsonl`: separate obstacle seeds 2101–2106 with identical procedure; duplicate prompt instances are deduplicated. **Never** optimize on these labels. Multiple trajectories or themes sharing a seed remain within the same split and are **not independent design structures**.
- **Adverse companion**: `adverse-train.jsonl` (6 distinct local motif groups × 3 families × 4 tick formulations = 72 records) and `adverse-development.jsonl` (4 disjoint motifs × 3 × 4 = 48 records) cover blocked fronts, blind alleys with/without retrace, repeated/visited arms, rare STOP and junctions. These are explicitly **synthetic LOCAL counterfactual observations**, NOT claims of W0-replayed world states or native Art receipts. Do not treat four tick formulations as four independent designs; do not optimize on adverse DEVELOPMENT. The historical W0-replayed **v1 artifact** remains immutable, but its old `train.jsonl` and `development.jsonl` are NOT admitted for model training because exact local prompts leak across independent obstacle seeds. v2 emits new filenames, a provenance-rich prompt-dedup audit, and tests literal cross-split prompt intersection equals zero.
- `world_group`, exact decision IDs, plain `prompt`, and `target_action` are present.
- **Correctness fix (15 November):** World seed segregation alone is insufficient for a 5×5-observation learner: the original v1 archive contains 125 identical literal prompt texts in both splits (training and held-out source worlds), even though seeds differ. Keep v1 as a negative provenance example; use **v2 only**, which preserves DEV prompts and drops any colliding TRAIN examples, deduplicating within each split. The new manifest records all filtered counts and affected seed groups. This is an action-identity leak correction, not proof of model generalization. The prompt is **plain UTF-8 text**, *not* an LFM chat template or proof of token fit. LFM primary alone wraps it using pinned official `Lfm2VlProcessor`, chooses EOS7 and owns the versioned WORM LoRA recipe.
- The generator rejects duplicate example identities, cross-split obstacle/motif groups, illegal labels, unknown action strings, and exact local-prompt overlap even across different obstacle seed groups. No private held-out complete-scene Art targets/images are used.

## Reproduce now

```sh
python3 -m unittest discover -s experiments/agent_painter/worm/support_lfm -p 'test_curriculum.py' -v
python3 experiments/agent_painter/worm/support_lfm/curriculum.py --output /tmp/worm-lfm-v1 --steps 240
```

Python 3.11+ standard library only; import resolves the current shared W0 baseline in the stacked branch. `manifest.json` contains full train/DEVELOPMENT source seed lists, counts, and canonical JSONL hashes. `gallery/summary.json` inventories each actual PNG and exact-trace JSON with SHA-256 and per-run metrics. `--steps` bounds gallery decisions to 1–512; training observations are capped independently at 32. GitHub-hosted workflow runs these commands and uploads the actual archives (no LFM model download, no GPU, no native Workbench claims).

## Fair paired evaluation

`save_gallery()` emits random-legal W0, deterministic wall W0 and a closed-loop heuristic oracle on the **same held-out seed, obstacle map, theme, grid and decision budget**. The oracle uses `rollout()`: each next observation arrives via the W0 authoritative callback after host legality checks, never teacher-forced. It also draws `TELEPORT` raw-invalid and constant-`LEFT` failure controllers, with original decisions and host refusals in adjacent JSON traces. These are **illustrative non-model failures**, not LFM first passes.

`statistics()` includes decision count, invalid raw count/rate, host unique coverage of obstacle-free cells, novelty per decision, W0 revisit contacts, branches, retraces, blocked-local decisions, recovery by retrace, longest repeated-action streak and applied-action entropy. Retraces are tracked separately rather than silently counted as new trail visits. No metric alone establishes aesthetic value; assess original PNGs alongside the numbers.

To compare an **actual model** without falsifying its output, LFM primary imports:

```python
from curriculum import evaluate_policy, statistics, worm
pairs = evaluate_policy("lfm-raw-stepNN", choose, seeds=(2101,2102), steps=48)
# choose(observation: dict) -> str, unmodified first-pass raw text
# Each (record, metadata) holds W0 events/raw_choice, host applied action, obstacles.
```

`choose` must invoke the genuine pinned model with the current observation. It MUST NOT see a DEVELOPMENT `target_action`, use answer repair, teacher fallback, hidden retries or altered host legality. W0 invokes the genuine policy once per decision through its immutable deep-copied observation and caps these evaluations at 512 actions; real-time native integration is a later separate W4 gate. If model outputs `"LEFT\n"`, it is **invalid** as a raw W0 action and remains invalid in the receipt.

The workflow does **not** report model-vs-baseline because no model endpoint/explicit optimiser lease has been provided. Current output is support-source evidence only. LFM primary can consume exact corpus hash, prompt-v1 format, held-out action interface and W0 PNG schema before launching its own CPU model training. The owner-authoritative Art2 complete-scene FINAL remains sealed.

## Raw-versus-applied action evaluation (15 November 2026)

A learned policy can repeatedly emit literal `STRAIGHT` while the W0 host
legally substitutes `RETRACE` (or `STOP`) when forward motion is blocked.
Therefore, **applied-action diversity and host recovery are not model ability**.
`statistics(record)` now additionally emits `raw_action_counts`,
`raw_action_entropy_bits`, `max_same_raw_streak`,
`raw_straight_fraction` and `host_fallback_count`. Always report raw and
applied metrics separately for LFM W1/W2; preserve every original token/action
receipt and never count fallback as an independently learned recovery.
The gallery summary marks the additive `worm-lfm-raw-applied-metrics-v1` schema;
W0 PNG/trace bytes and versioned v2 TRAIN/DEVELOPMENT teacher targets are unchanged.

`checked_splits(train, dev)` now fails closed on literal TRAIN/DEVELOPMENT
prompt overlap, including counterfactual companion rows, rather than requiring
callers to remember a separate deduplication helper. The existing v1 corpus
remains a retained negative example. This check does **not** prove semantic
novelty across near-identical but non-equal local observations.


## Primary W3 model-gallery independent source replayer (support audit v2)

Use `primary_gallery_audit.py --gallery <extracted-LFM-primary-W3-output>/gallery --seed 2101 --budget 32 --report audit.json` against **the original primary W3 artifact**, never hand-reconstructed first passes. The checker requires all 3 families × (model/random/wall/oracle), replays actual raw choices through the unchanged W0 host, compares complete observations, legal choices, positions, metrics and original PNG bytes, and validates EOS7/token-id/host first-pass receipt consistency. It reports paired **raw vs applied** metrics, action-streak/entropy, unique cells, branches, revisits, invalids, fallbacks and decision-count mismatches. Strict falsification cases are exercised in `test_primary_gallery_audit.py`.

This check is **source-faithful Python W0 provenance**, not a second model evaluation or native Godot evidence. Tokenizer decoding, checkpoint tensors, model license, and genuine optimizer steps must still be verified separately from primary receipts. Comparisons represent one shared seed, not independent generalization; early model STOP means a nominal equal decision budget does not imply equal executed decisions. Do not call a less-complicated but visually amusing line a statistically superior controller.

An additive independent audit-v2 metric detects the longest **exact raw-action motif of period 1–6 repeated at least three times**, returning its start, literal motif, full repetitions, span and fraction of choices. This catches the BRANCH/RETRACE/RIGHT loop seen in learned W3, which the previous longest-identical-action streak could miss. Model and random/wall/oracle controls are compared on identical W0 seeds and budgets. It is a *repetition diagnostic*, not an aesthetic or generalization score, and NEVER substitutes host-applied actions for raw model text. The original W3 artifact bytes and dataset are unchanged.

The support CI additionally produces **18 original W0 PNGs and 18 action traces** under `negative-controls/` for three themes against random/wall/oracle and three explicitly **untrained simulated failure selectors** (always STRAIGHT, always LEFT, illegal TELEPORT). These are visually useful reproducible bad behaviours, not secretly trained LFM outputs. Input W3 artifact content is NEVER downloaded or modified by this workflow. The support branch remains Draft/HOLD and does not dispatch any optimizer.


## W7 source-fenced anti-RIGHT challenge (15 November 2026)

The old 96-request stimulus suite contained only **one of 24** legal-mask edits that excluded the learned W7 action RIGHT. Moreover, the 24 original W0-observed local prompts collapse to only **15 distinct prompt byte strings** (wall states repeat across four independent obstacle seeds). Do not equate 24 labels or 96 prompt requests with 24/96 independent local decision states.

The new support-owned `w7_anti_right_audit.py` reads the **actual immutable** original support artifact #14165324490 (SHA256 `458db32c6239536f6cbd4f421fc76356f734be4f9b7f9efb850a208875f42794`) and original trained W7 checkpoint-inference artifact #14166972797 (SHA256 `51f6b2cb5214b818864dc739a958da79f433075d7dc7c0b71e352aa0480ea6ff`). It independently checks first-pass EOS/token identities, legal masks, prompt digests and six-code logits, preserves all 24 original W0 PNG+trace pairs, and produces **24 RIGHT-excluded synthetic prompt variations with 15 unique byte-level requests**, retaining all source IDs for traceability. These are **NOT W0-executed revised states or new learned model responses**.

To evaluate the 15 unique requests, LFM primary may consume `anti-right-15-unique.jsonl`, preserving `id`, `prompt_sha256`, `provenance_case_ids`, unmodified EOS7/token IDs and the host legal-mask rule; **first-pass validity (digit+true EOS) must be scored separately from raw host legality**. The primary's W7 prior summary labelled 95/96 as `raw_valid_eos` because it included the legal mask; actual original unmasked W7 first passes were **96/96 valid digit+EOS, 95/96 host-legal**, all 96 RIGHT. A replayed host-masked argmax computed from old original logits is strictly an **offline diagnostic**, not a model response to the new prompt or a new painting. Frozen dataset identity and complete original evidence are in the GitHub-hosted `agent-painter-worm-lfm-support-w7-anti-right.yml` artifact. No optimizer, Godot-native work, sealed FINAL access or production promotion occurs here.
