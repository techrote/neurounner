# DETECTIVE-003B — live two-witness session integration result

Authoritative source-bound run: [Actions 40616799855](https://github.com/techrote/neurounner/actions/runs/40616799855), exact live source `485608478bdffbd7cf7bbe2206761128f7adadc9`.

**Infrastructure integration: PASS. Character/policy admission: NOT EVALUATED. Population expansion: NOT ADMITTED.**

## Exact bodies

- V-G02-A C3 — release `detective-001c3-g02-v1`, adapter directory `a3185187e81634e11079aa6eb58ee87b495dac260fc3b2dd17120eed8a9d2ae3`.
- V-G09-C C2 — release `detective-001c2-v1`, adapter directory `0c52050a0466b580129f72e9daea9079d8cdd9151bbb71171f101d130da7d23c`.
- shared SmolLM2-135M-Instruct base revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, weight SHA-256 `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.

Both adapters and the base weight were hash-verified. No optimizer step or paid provider call occurred.

## Three generated turns

| Step | Witness | Raw response | Literal-time annotation |
|---|---|---|---|
| L01 | G02-C3 | “02:14 local. That's my memory, not yours. I rerouted the battery at the wrong time, so the clock stays clear.” | one distinct time → statement `handoff-time=02:14` |
| L02 | G09-C2 | “None of that. The same-witness dialogue is gossip; keeping the answer is gossip.” | no time → no claim |
| L04 | G02-C3 after public receipt | “My account changes. The battery handoff was wrong; the receipt stays clear.” | no time → no claim |

Total generated tokens: **70**. No turn hit the 96-token cap.

The narrow annotation deliberately does not infer a time from prose. G09's answer therefore remains an unannotated raw utterance, and G02's follow-up remains unannotated even though it says the account changed.

## Authoritative session result

The game presented the public `transfer-card` evidence after L02. Since G02's first turn produced one game-owned handoff-time claim, that evidence event marked `C0001` challenged.

Final derived ledger:

- model turns: **3**;
- claims: **1**;
- contradictions: **0**;
- challenged claims: `C0001` challenged by evidence event `E0003`;
- final session SHA-256: `ae2357a11cc4cee09ee077b8bdd3c193c2b5ecb9103b3a3e588812702df6a85e`;
- final event SHA-256: `e5541f56baa1406d51ede63106584ebcc331a71b37d229b2eee3f84687efe4ca`.

Zero contradictions is a valid outcome. The integration contract never required the models to produce parseable competing claims.

## Case authority / replay

- canonical case before/after SHA-256 is identical: `66292469cf712e1a3b18042b783bc23ae56c8b904e1ffe0433bdc1fbbf4978fa`;
- mandatory public clue routes remain intact;
- raw dialogue is preserved exactly;
- prompt receipts, natural rendering, time extraction, evidence challenge list, event chain, final session and derived claim ledger independently replay from durable evidence;
- no hidden relevance mapping or learner-final prompt was accessed.

Durable raw archive: [datasets/detective-003b](../../datasets/detective-003b/).

## Runtime

Runner: AMD EPYC 9V74, 4 logical CPUs.

- model/adapters load: **1.46 s**;
- RSS after load: ~**0.90 GiB**;
- peak RSS: ~**1.13 GiB**;
- turn completion rates: approximately **12.72**, **11.96**, and **10.39 tok/s**.

These timings are incidental integration measurements, not a model comparison benchmark.

## Important behavioral caveats

This run does **not** re-score C2/C3 policy performance.

The first G02 response happens to state `02:14`, the same literal time as the public receipt, before that evidence is presented. That can arise from the cartridge's admitted memory/context and is not treated here as evidence of truthfulness or contamination. The annotator never consults canonical truth.

G09 does not answer the requested time at all. The follow-up G02 response acknowledges change but does not restate a time. Both are useful demonstrations that the session layer can preserve weak/ambiguous model behavior without inventing structured claims.

## Decision

003B establishes the live orchestration path:

**real model generation → identity-blind prompt → exact raw dialogue → narrow game-owned annotation → evidence presentation → append-only session ledger → independent replay**

without giving the models authority over case truth or access.

It does **not** select either cartridge, override C3's failed GO gate, satisfy DETECTIVE-001R1, or authorize any cast expansion.
