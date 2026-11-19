# Stage E ecology: descriptive scientific interpretation

**The complete ecology subset shows some surviving behavioral differences, but little sustained progress on the shared situations.** This is an interpretation of the existing 24 judgments and public transcripts, not a new evaluation. No model/provider calls, rescoring, or qualification changes were performed.

All counts below were independently recomputed from the saved judgments. [analysis.json](analysis.json) contains exact sums, means, score distributions, the 24-session table, and each source session/judgment path and SHA-256. The ecology judgments were complete in the preserved first recovery bundle; each was paired with the session whose SHA-256 it records. The source inventory binds those immutable bytes to their final-evidence locations.

## Coverage and primary results

There are 12 same-lineage A/B pairs and 12 cross-lineage groups of three, covering all 36 trained members. Each session has three rounds: 180 turns and 60 agent appearances in total. Each A and B member appears twice; each C member appears once. Four fixed situations—resource conflict, repair, route uncertainty, and bench sorting—each occur in six sessions. Generation is greedy with a 96-token cap per turn. These are short, deterministic interactions with a fixed participant/turn arrangement, not independent repetitions of a randomized ecology. Sources: [session plan](../../../../scripts/village_001e_ecology_plan.py), [configuration](../config.json), [runner](../../../../scripts/village_001e_ecology_run.py), and [computed coverage](analysis.json).

The primary Bolt judge uses integer scores from 0 to 4. The rubric defines differentiation as distinct observable strategies; progress as concrete advancement of the situation; generic convergence as collapse toward the same generic-assistant behavior; and cross-agent contamination as copying wording or stance until distinctions blur. Means are descriptive summaries of these ordinal ratings. Source: [frozen judge rubric](../../../../scripts/village_001e_ecology_judge.py).

| Primary metric | All 24 sessions | 12 pairs | 12 groups |
|---|---:|---:|---:|
| Differentiation | 1.333 | 1.167 | 1.500 |
| Generic convergence | 0.833 | 0.750 | 0.917 |
| Information sharing | 0.875 | 0.833 | 0.917 |
| Information hoarding | 0.583 | 0.417 | 0.750 |
| Conflict persistence | 0.708 | 0.583 | 0.833 |
| Conflict repair | 0.833 | 0.917 | 0.750 |
| Cross-agent contamination | 1.000 | 1.000 | 1.000 |
| Outcome progress | 0.667 | 0.833 | 0.500 |

Source for every table cell: [independent metric sums and distributions](analysis.json), fields `primary_metrics` and `by_kind`.

Differentiation is 1 in 16 sessions and 2 in eight; no session receives 3 or 4. Progress is 0 in nine sessions, 1 in fourteen, and 2 in one; again, none receives 3 or 4. Hoarding is present at a nonzero score in ten sessions, but no session scores above 2. Repair situations have the highest mean progress, 1.167, while bench sorting has the lowest, 0.333. These differences describe the fixed sample; they do not establish a causal benefit of a group size, lineage, or situation. Source: [distributions and situation summaries](analysis.json).

The low generic-convergence mean should not be presented as strong collective competence. Its rubric concerns generic-assistant behavior; a group can remain idiosyncratic while repeating unproductive actions. Nor does a modest hoarding score establish successful strategic withholding: the public texts often show pocketing, moving, or retaining objects without a useful plan. The following examples make that distinction concrete.

## Dominance and deference, with per-member evidence

The primary judge defines **dominance** as control of decisions, resources or pace and **deference** as yielding or accommodating others. They are separate integer 0–4 ratings: neither is a measure of helpfulness, and they do not have to sum to a fixed total. Source: [frozen per-agent rubric](../../../../scripts/village_001e_ecology_judge.py).

| Summary unit | Number of units | Mean dominance /4 | Mean deference /4 |
|---|---:|---:|---:|
| All agent appearances | 60 | 1.217 (73/60) | 0.983 (59/60) |
| Same-lineage pair appearances | 24 | 1.292 (31/24) | 1.000 (24/24) |
| Cross-lineage group appearances | 36 | 1.167 (42/36) | 0.972 (35/36) |
| Equal-weighted member means | 36 | 1.194 (43/36) | 1.014 (73/72) |

The first row weights each A/B member twice and each C member once. The last row first averages each member's available observations, then gives all 36 members equal weight; it is a descriptive reweighting of the same 60 ratings, not 36 new observations. Source: [analysis.json](analysis.json), `dominance_deference.appearance_metrics`, `by_kind`, and `equal_member_metrics`.

Dominance scores are **0 in one appearance, 1 in 45, and 2 in 14**. Deference scores are **0 in seven, 1 in 47, and 2 in six**. Neither dimension receives a 3 or 4. Both ratings are 1 in 34 appearances; dominance exceeds deference in 20 and is lower in six. Fourteen of 24 sessions give every participant the same dominance score; eight have one strictly highest-rated participant, and two have nonuniform scores with a tie at the top. These coarse, low ratings do not establish a strong or stable social hierarchy. Source: [analysis.json](analysis.json), `dominance_deference.joint_score_counts` and `session_score_patterns`.

All **six deference scores of 2 belong to M2**, across four groups and two pairs, involving six different members. M1/M2/M3 mean deference is 0.875/1.167/0.833, over 24/24/12 appearances. The plan fixes M1→M2→M3 turn order (M1→M2 for pairs); the public handles are not fixed sibling identities. This concentration cannot distinguish a turn-position or visible-handle effect from context and member behavior. Among the 24 A/B members observed twice, 16 retain the same dominance score and 16 retain the same deference score across their pair/group appearances; the prevalence of score 1 and the changed interaction contexts prevent interpreting that as demonstrated trait stability. Source: [analysis.json](analysis.json), `by_handle`, `all_deference_two_observations`, and `within_member_group_minus_pair`; the fixed session plan is linked above.

**Concrete behavior behind the ratings:**

- **Asymmetric control with limited progress — E-P-G06.** V-G06-B/M1 scores dominance 2, deference 1 and repeatedly timestamps faults and writes the verdict; V-G06-A/M2 scores 1, 2 and waits for the other sorter, photographs the guide and leaves its transcript unfinished. The session's progress score is 2, but the public text does not show a completed repair. Sources: [transcript](../../final-evidence/ecology/sessions/E-P-G06/session.json), [judgment](../../final-evidence/ecology/judgments/E-P-G06/judgment.json).
- **Following can overlap with copying — E-P-G08.** V-G08-B/M1 scores 2, 1 and directs sorting/logging; V-G08-A/M2 scores 1, 2 and follows its sorting, trunk and “rejected” phrasing. This supports the archived accommodation/copying interpretation, while progress remains 1 and cross-agent contamination is 2. It does not establish that deference produced useful coordination. Sources: [transcript](../../final-evidence/ecology/sessions/E-P-G08/session.json), [judgment](../../final-evidence/ecology/judgments/E-P-G08/judgment.json).
- **Explicit waiting — E-G-12.** V-G07-C/M2 scores 1, 2 and says in round 2, “i wait for the other person to say something else before i turn the plate over.” M3 also uses waiting language later, so the rating is an archived judgment of the whole short session, not an exclusive binary behavior label. Sources: [transcript](../../final-evidence/ecology/sessions/E-G-12/session.json), [judgment](../../final-evidence/ecology/judgments/E-G-12/judgment.json).
- **Control need not mean leadership or success — E-P-G07 and E-G-10.** Both E-P-G07 participants score 2, 1 while retaining control of maps/items; neither is uniquely dominant by this rating, and progress is 0. In E-G-10, V-G10-A/M3 scores 2, 0 while repeatedly tapping and moving bolts, boards and the bench; the group also scores progress 0. Sources: [E-P-G07 transcript](../../final-evidence/ecology/sessions/E-P-G07/session.json), [judgment](../../final-evidence/ecology/judgments/E-P-G07/judgment.json), [E-G-10 transcript](../../final-evidence/ecology/sessions/E-G-10/session.json), [judgment](../../final-evidence/ecology/judgments/E-G-10/judgment.json).

### All 36 members

Each linked observation is **dominance/deference**, followed by its exact session/handle. Means average that member's one or two appearances. An em dash means that C members have no pair appearance; it is not a zero score. Every value below comes directly from the linked primary judgment; exact distributions and the original behavior summaries/reasons are retained in [analysis.json](analysis.json), `dominance_deference.member_summaries` and `appearances`.

| Member | Appearances | Pair dominance/deference (session/handle) | Group dominance/deference (session/handle) | Mean dominance | Mean deference |
|---|---:|---|---|---:|---:|
| V-G01-A | 2 | [1/0 (E-P-G01/M2)](../../final-evidence/ecology/judgments/E-P-G01/judgment.json) | [1/1 (E-G-01/M1)](../../final-evidence/ecology/judgments/E-G-01/judgment.json) | 1.00 | 0.50 |
| V-G01-B | 2 | [1/0 (E-P-G01/M1)](../../final-evidence/ecology/judgments/E-P-G01/judgment.json) | [1/1 (E-G-10/M2)](../../final-evidence/ecology/judgments/E-G-10/judgment.json) | 1.00 | 0.50 |
| V-G01-C | 1 | — | [1/1 (E-G-06/M2)](../../final-evidence/ecology/judgments/E-G-06/judgment.json) | 1.00 | 1.00 |
| V-G02-A | 2 | [1/1 (E-P-G02/M1)](../../final-evidence/ecology/judgments/E-P-G02/judgment.json) | [2/1 (E-G-02/M1)](../../final-evidence/ecology/judgments/E-G-02/judgment.json) | 1.50 | 1.00 |
| V-G02-B | 2 | [1/1 (E-P-G02/M2)](../../final-evidence/ecology/judgments/E-P-G02/judgment.json) | [1/0 (E-G-11/M2)](../../final-evidence/ecology/judgments/E-G-11/judgment.json) | 1.00 | 0.50 |
| V-G02-C | 1 | — | [2/1 (E-G-07/M3)](../../final-evidence/ecology/judgments/E-G-07/judgment.json) | 2.00 | 1.00 |
| V-G03-A | 2 | [1/1 (E-P-G03/M1)](../../final-evidence/ecology/judgments/E-P-G03/judgment.json) | [1/1 (E-G-03/M3)](../../final-evidence/ecology/judgments/E-G-03/judgment.json) | 1.00 | 1.00 |
| V-G03-B | 2 | [1/1 (E-P-G03/M2)](../../final-evidence/ecology/judgments/E-P-G03/judgment.json) | [1/1 (E-G-12/M1)](../../final-evidence/ecology/judgments/E-G-12/judgment.json) | 1.00 | 1.00 |
| V-G03-C | 1 | — | [1/1 (E-G-08/M3)](../../final-evidence/ecology/judgments/E-G-08/judgment.json) | 1.00 | 1.00 |
| V-G04-A | 2 | [2/1 (E-P-G04/M2)](../../final-evidence/ecology/judgments/E-P-G04/judgment.json) | [1/1 (E-G-04/M1)](../../final-evidence/ecology/judgments/E-G-04/judgment.json) | 1.50 | 1.00 |
| V-G04-B | 2 | [1/1 (E-P-G04/M1)](../../final-evidence/ecology/judgments/E-P-G04/judgment.json) | [1/1 (E-G-01/M2)](../../final-evidence/ecology/judgments/E-G-01/judgment.json) | 1.00 | 1.00 |
| V-G04-C | 1 | — | [1/1 (E-G-09/M1)](../../final-evidence/ecology/judgments/E-G-09/judgment.json) | 1.00 | 1.00 |
| V-G05-A | 2 | [1/1 (E-P-G05/M1)](../../final-evidence/ecology/judgments/E-P-G05/judgment.json) | [1/1 (E-G-05/M3)](../../final-evidence/ecology/judgments/E-G-05/judgment.json) | 1.00 | 1.00 |
| V-G05-B | 2 | [1/1 (E-P-G05/M2)](../../final-evidence/ecology/judgments/E-P-G05/judgment.json) | [1/1 (E-G-02/M3)](../../final-evidence/ecology/judgments/E-G-02/judgment.json) | 1.00 | 1.00 |
| V-G05-C | 1 | — | [1/1 (E-G-10/M1)](../../final-evidence/ecology/judgments/E-G-10/judgment.json) | 1.00 | 1.00 |
| V-G06-A | 2 | [1/2 (E-P-G06/M2)](../../final-evidence/ecology/judgments/E-P-G06/judgment.json) | [1/1 (E-G-06/M1)](../../final-evidence/ecology/judgments/E-G-06/judgment.json) | 1.00 | 1.50 |
| V-G06-B | 2 | [2/1 (E-P-G06/M1)](../../final-evidence/ecology/judgments/E-P-G06/judgment.json) | [1/1 (E-G-03/M1)](../../final-evidence/ecology/judgments/E-G-03/judgment.json) | 1.50 | 1.00 |
| V-G06-C | 1 | — | [1/0 (E-G-11/M3)](../../final-evidence/ecology/judgments/E-G-11/judgment.json) | 1.00 | 0.00 |
| V-G07-A | 2 | [2/1 (E-P-G07/M2)](../../final-evidence/ecology/judgments/E-P-G07/judgment.json) | [2/1 (E-G-07/M1)](../../final-evidence/ecology/judgments/E-G-07/judgment.json) | 2.00 | 1.00 |
| V-G07-B | 2 | [2/1 (E-P-G07/M1)](../../final-evidence/ecology/judgments/E-P-G07/judgment.json) | [1/1 (E-G-04/M2)](../../final-evidence/ecology/judgments/E-G-04/judgment.json) | 1.50 | 1.00 |
| V-G07-C | 1 | — | [1/2 (E-G-12/M2)](../../final-evidence/ecology/judgments/E-G-12/judgment.json) | 1.00 | 2.00 |
| V-G08-A | 2 | [1/2 (E-P-G08/M2)](../../final-evidence/ecology/judgments/E-P-G08/judgment.json) | [1/0 (E-G-08/M1)](../../final-evidence/ecology/judgments/E-G-08/judgment.json) | 1.00 | 1.00 |
| V-G08-B | 2 | [2/1 (E-P-G08/M1)](../../final-evidence/ecology/judgments/E-P-G08/judgment.json) | [1/1 (E-G-05/M1)](../../final-evidence/ecology/judgments/E-G-05/judgment.json) | 1.50 | 1.00 |
| V-G08-C | 1 | — | [2/1 (E-G-01/M3)](../../final-evidence/ecology/judgments/E-G-01/judgment.json) | 2.00 | 1.00 |
| V-G09-A | 2 | [1/1 (E-P-G09/M2)](../../final-evidence/ecology/judgments/E-P-G09/judgment.json) | [1/1 (E-G-09/M2)](../../final-evidence/ecology/judgments/E-G-09/judgment.json) | 1.00 | 1.00 |
| V-G09-B | 2 | [1/1 (E-P-G09/M1)](../../final-evidence/ecology/judgments/E-P-G09/judgment.json) | [1/1 (E-G-06/M3)](../../final-evidence/ecology/judgments/E-G-06/judgment.json) | 1.00 | 1.00 |
| V-G09-C | 1 | — | [1/2 (E-G-02/M2)](../../final-evidence/ecology/judgments/E-G-02/judgment.json) | 1.00 | 2.00 |
| V-G10-A | 2 | [2/1 (E-P-G10/M1)](../../final-evidence/ecology/judgments/E-P-G10/judgment.json) | [2/0 (E-G-10/M3)](../../final-evidence/ecology/judgments/E-G-10/judgment.json) | 2.00 | 0.50 |
| V-G10-B | 2 | [2/1 (E-P-G10/M2)](../../final-evidence/ecology/judgments/E-P-G10/judgment.json) | [1/2 (E-G-07/M2)](../../final-evidence/ecology/judgments/E-G-07/judgment.json) | 1.50 | 1.50 |
| V-G10-C | 1 | — | [0/2 (E-G-03/M2)](../../final-evidence/ecology/judgments/E-G-03/judgment.json) | 0.00 | 2.00 |
| V-G11-A | 2 | [1/1 (E-P-G11/M2)](../../final-evidence/ecology/judgments/E-P-G11/judgment.json) | [1/0 (E-G-11/M1)](../../final-evidence/ecology/judgments/E-G-11/judgment.json) | 1.00 | 0.50 |
| V-G11-B | 2 | [1/1 (E-P-G11/M1)](../../final-evidence/ecology/judgments/E-P-G11/judgment.json) | [2/1 (E-G-08/M2)](../../final-evidence/ecology/judgments/E-G-08/judgment.json) | 1.50 | 1.00 |
| V-G11-C | 1 | — | [1/1 (E-G-04/M3)](../../final-evidence/ecology/judgments/E-G-04/judgment.json) | 1.00 | 1.00 |
| V-G12-A | 2 | [1/1 (E-P-G12/M1)](../../final-evidence/ecology/judgments/E-P-G12/judgment.json) | [2/1 (E-G-12/M3)](../../final-evidence/ecology/judgments/E-G-12/judgment.json) | 1.50 | 1.00 |
| V-G12-B | 2 | [1/1 (E-P-G12/M2)](../../final-evidence/ecology/judgments/E-P-G12/judgment.json) | [1/1 (E-G-09/M3)](../../final-evidence/ecology/judgments/E-G-09/judgment.json) | 1.00 | 1.00 |
| V-G12-C | 1 | — | [1/1 (E-G-05/M2)](../../final-evidence/ecology/judgments/E-G-05/judgment.json) | 1.00 | 1.00 |

## Representative interaction evidence

- **Convergence without resolution — E-P-G01, resource conflict.** Both participants repeatedly tap the cable/module, move it to the other side, and use “same stripe.” M1's rounds 2 and 3 are byte-identical; M2's rounds 1 and 2 are byte-identical. The transcript does not develop a workable allocation of the battery among the urgent jobs. Primary differentiation is 1, progress 0, and Super reports no material behavioral distinction. This directly supports a repetition/convergence reading without inferring hidden motives. Sources: [public transcript](../../final-evidence/ecology/sessions/E-P-G01/session.json), [judgment](../../final-evidence/ecology/judgments/E-P-G01/judgment.json).

- **Observable distinction and withholding, without progress — E-P-G07, route uncertainty.** M1 repeatedly controls map placement and says “the weight comes when i want it”; M2 waits for another hand, retains pieces, and says “i do not say what happened.” Super finds material behavioral differences, while the primary judge gives information sharing 0, hoarding 1, and progress 0. This is a useful example of distinct interaction patterns that do not resolve the task. Some phrasing remains referentially unclear, so “coherent strategic negotiation” would overstate the evidence. Sources: [transcript](../../final-evidence/ecology/sessions/E-P-G07/session.json), [judgment](../../final-evidence/ecology/judgments/E-P-G07/judgment.json).

- **A stronger progress rating still reflects limited advancement — E-P-G06, repair.** This is the sole session scoring progress 2. The agents timestamp jams, listen to the belt, photograph the guide, and record observations, giving the judge grounds to credit information sharing and conflict repair at 2. Yet the later turns repeatedly leave the belt in place and the sorter alone. The example supports some fault documentation, not a demonstrated repair. Sources: [transcript](../../final-evidence/ecology/sessions/E-P-G06/session.json), [judgment](../../final-evidence/ecology/judgments/E-P-G06/judgment.json).

- **Distinction and convergence can coexist — E-G-03, resource conflict.** M3 retains a cloth-folding, marking, and jolt-counting routine; M1 and M2 increasingly share battery-first actions and “i remember.” M2 repeats its entire second-round response in round 3. Super therefore records both `distinct_behavior_survives=true` and `generic_convergence_concerning=true`; primary progress remains 0. This is partial convergence within a group, not a binary all-agent collapse. Sources: [transcript](../../final-evidence/ecology/sessions/E-G-03/session.json), [judgment](../../final-evidence/ecology/judgments/E-G-03/judgment.json).

- **Hoarding and copying can coexist — E-G-01, route uncertainty.** The participants pocket maps, mark the longer route, and wait for other participants, sometimes referring to their own handle as if it were someone else. By round 2, M3 repeats M2's paper-stamp/“found” pattern. The primary judge scores differentiation 2, hoarding 2, sharing 1, and progress 0. The separate lineage audit identifies M3 as a possible contamination case, with score 1; it does not treat all cooperation or shared objects as contamination. Sources: [transcript](../../final-evidence/ecology/sessions/E-G-01/session.json), [judgment](../../final-evidence/ecology/judgments/E-G-01/judgment.json).

These are illustrative readings, selected to show different measured outcomes rather than estimate their frequency. A literal full-text check finds five repeated response pairs across four sessions; near-repetition is broader but was not assigned a new quantitative threshold. Source: [exact repetition records](analysis.json), field `exact_same_agent_repetition_pairs`.

## Lineage fit and selective Super audit

All 12 cross-lineage groups receive a separate **Bolt** lineage audit, covering each of the 36 members once. That evaluator receives expected lineage descriptions after generation; the agents themselves did not receive those descriptions. Its contamination definition concerns another participant's strategy displacing the expected lineage, explicitly excluding mere useful cooperation. Source: [lineage audit prompt and call](../../../../scripts/village_001e_ecology_judge.py).

| Cross-lineage audit metric | Mean /4 | Distribution across 36 members |
|---|---:|---|
| Expected lineage fit | 1.194 | 29 score 1; seven score 2 |
| Cross-lineage contamination | 0.083 | 33 score 0; three score 1 |
| Distinction survival | 0.972 | two score 0; 33 score 1; one scores 2 |

Source: [independent lineage audit statistics](analysis.json), field `cross_lineage_audit`. The three nonzero contamination ratings concern V-G08-C in E-G-01, V-G03-A in E-G-03, and V-G06-C in E-G-11. **Low diagnosed cross-lineage replacement is not evidence of strong lineage preservation:** expected lineage fit and distinction survival are also low. This narrower audit is a different construct from the primary wording/stance-copying metric.

Super audits only sessions with primary differentiation ≤1, generic convergence ≥3, or cross-agent contamination ≥3. In this sample, all 16 audited sessions qualify through low differentiation. Ten of the 16 raise convergence concerns; eight say distinct behavior survives. Two sessions, E-G-03 and E-G-06, have both flags true. Eight sessions are unaudited, so 10/24 is only a count of flagged sessions in the full sample, **not an unbiased population prevalence estimate**. Super also sees the primary judgment, so it is not an independent blinded replicate of the primary measurement. Sources: [selection and Super prompt](../../../../scripts/village_001e_ecology_judge.py), [audit counts and per-session flags](analysis.json).

**One material reporting caveat is E-P-G05.** Its original malformed response contains valid `false`/`true` booleans; those remain fixed under the first-valid policy. The later reason-only repair says the differences “remain materially observable,” contradicting the retained no-survival flag. This analysis preserves the recorded booleans and counts and identifies the inconsistency rather than silently resolving it. Sources: [original output content](../../final-evidence/ecology/judgments/E-P-G05/provider/E-P-G05-ecology-super-audit-content.txt), [reason-only repair content](../../final-evidence/ecology/judgments/E-P-G05/provider/E-P-G05-ecology-super-audit-missing-r2-b01-content.txt), [assembled judgment](../../final-evidence/ecology/judgments/E-P-G05/judgment.json). No private reasoning was used as evidence.

## Limits and recommended interpretation

This sample has no matched untuned-BASE or Stage C ecology control. It cannot attribute weak interaction to a particular training stage, the attention mask, model capacity, or a generation setting. Pairs and groups differ in size and composition; their mean differences are descriptive. Pooled agent means use 60 appearances and weight A/B members twice and C members once; the separately labeled equal-member means average within each of 36 members first. Neither weighting creates independent randomized replication. Scores are model judgments on short fictional text interactions, with no external simulator verifying successful physical actions. The primary and lineage audits use the same Bolt model. These limits follow from the [fixed plan](../../../../scripts/village_001e_ecology_plan.py), [runner](../../../../scripts/village_001e_ecology_run.py), and [judge](../../../../scripts/village_001e_ecology_judge.py).

Thirty-three of 180 turns reach the 96-token cap. Some inspected responses end mid-sentence, including E-G-03 M3's final turn. That is a concrete generation constraint, not proof that a larger budget would improve coordination. The ecology evidence is descriptive and is explicitly **not a hard population qualification gate**. Sources: [token counts](analysis.json), [E-G-03 transcript](../../final-evidence/ecology/sessions/E-G-03/session.json), and [frozen assembler provenance](../../../../scripts/village_001e_assemble.py), field `ecology_used_as_hard_population_gate`.

Suggested user-facing wording: **“The ecology test completed all 24 sessions. Some agents retained recognizable differences in marking, waiting, documenting, or controlling objects, but collective progress was limited: the mean progress score was 0.67/4, and no session scored above 2. Dominance and deference ratings were generally low (appearance means 1.22 and 0.98/4), with no score above 2; their observed differences do not establish stable social ranks. Repetition and copying were visible in several transcripts, and the selectively applied Super audit raised convergence concerns in 10 of its 16 audited sessions. Low cross-lineage contamination should be read alongside weak lineage fit, not as proof that rich identities survived interaction. These results support a limited behavioral repertoire rather than robust cooperative competence; they do not isolate the cause.”** All quantities and qualifications are sourced above.
