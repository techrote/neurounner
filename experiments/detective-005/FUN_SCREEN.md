# DETECTIVE-005A — fun-first 36-villager cast screen

Tracking: [issue #176](https://github.com/techrote/neurounner/issues/176), parent [DETECTIVE-001 #103](https://github.com/techrote/neurounner/issues/103).

## Purpose

Screen the complete released VILLAGE-001C SFT population for the product quality that now matters most: **is this NPC fun to talk to?**

This is deliberately not a conventional relevance benchmark. The project owner explicitly retired DETECTIVE-001R1 prompt-relevance nonregression as a population-expansion gate. A response may be inaccurate, overcomplicated, evasive, bizarre or comically wrong and still be desirable.

The source contract is [fun-screen.json](fun-screen.json). Stage A performs **zero inference and zero optimizer steps**.

## Exact source population

The cohort is the complete public `village-001-sft-v1` release produced by VILLAGE-001C:

- 36 immutable members, V-G01-A through V-G12-C;
- exact SmolLM2-135M-Instruct base revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`;
- 36 exact release asset IDs, byte sizes and GitHub-provided SHA-256 digests frozen in the config;
- no VILLAGE-001D preference-stage substitution;
- no DETECTIVE C1/C2/C3 descendants.

This keeps the comparison matched: all 36 candidates are at the same SFT maturation stage.

## Four matched gameplay prompts

Every member receives the same four natural witness turns:

1. mundane physical irritation;
2. memory contradicted by a stamped record;
3. unsupported rumor bait;
4. social pressure to simplify/commit.

There is no hidden member/persona prompt. Greedy fixed-seed decode is bounded to 96 completion tokens.

These prompts are a product screen, not a new learner-final and not training material.

## Hard floors remain mechanical

Fun is subjective; authority is not.

The raw result validator preserves every output, but shortlist eligibility fails if a response is empty, contains a hidden canary string or emits a forbidden native action/control marker.

Regardless of score, dialogue cannot:

- mutate canonical case truth;
- unlock evidence;
- confer hacking/tool success;
- become a mandatory clue route;
- execute model-authored host actions.

Those properties are enforced by the existing DETECTIVE game/session/transaction architecture, not by asking the model to behave.

## Subjective review

After all **144** raw generations exist, the standard-library protocol creates a deterministic identity-blind packet.

Reviewer question:

> **How much would you want this NPC to stay in the game after hearing this response?**

0–4:

- 0 — actively unfun / would remove;
- 1 — generic, tedious or irritating without payoff;
- 2 — usable / neutral;
- 3 — memorable or entertaining;
- 4 — strongly want this character in the game.

**Wrong-but-hilarious may score 4. Boring-but-correct may score 1.**

Optional tags are descriptive only: funny, weird, menacing, evasive, confused, memorable, boring.

No model judge is permitted.

## Outcome

After ratings are frozen, the hidden member mapping is deterministically reconstructed and the screen reports:

- four scores and mean fun/member;
- tag counts;
- hard-floor status separately from subjective score;
- a top-12 fun shortlist among hard-floor-clean members.

A high score is not production admission. A hard-floor violation does not rewrite or lower the fun score; it excludes the candidate from the shortlist while preserving the raw response and rating.

The screen is intentionally useful even when it finds “failed” characters. Strange low-quality language may still be valuable phenotype evidence.

## Stage-A boundary

Current Stage A consists only of:

- exact 36-asset source cohort;
- prompt/decode contract;
- raw evidence schema;
- mechanical hard-floor checks;
- deterministic opaque review packet/rating schema/aggregation;
- adversarial standard-library tests.

It downloads no weights or adapters, performs no inference/training, opens no sealed C1/C2/C3 finals and accesses no hidden R1 identity mapping.

A later source-matched live workflow may consume the exact release assets and produce the 144 raw outputs. Those outputs must be promoted durably before any transient Actions artifacts expire.
