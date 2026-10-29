# DETECTIVE-003A — authoritative multi-NPC interview session ledger

Parent: [DETECTIVE-001 #103](https://github.com/techrote/neurounner/issues/103). Tracking: [#137](https://github.com/techrote/neurounner/issues/137).

This slice is **game/orchestration infrastructure only**. It performs no model inference, no training, no relevance scoring, no body selection, no hacking, and no population expansion.

## Authority boundary

The immutable DETECTIVE case remains the sole authority for:

- canonical facts;
- witness admitted knowledge;
- evidence contents and fact bindings;
- public/private evidence access;
- terminal grants and synthetic credentials;
- mandatory clue routes.

A model response is always an **untrusted raw utterance**.

A claim is a **game/reviewer annotation about an utterance**, never a model-authored fact. Claims can be compared with one another and challenged by presented evidence, but they cannot mutate case truth or unlock anything.

## Event-sourced session

`scripts/detective_003a_session.py` implements `detective-interview-session-v1`.

Private session header:

- deterministic `session_id`;
- case ID;
- private game-owned `session_key`;
- orchestrator participant IDs;
- append-only event log;
- session SHA-256.

The session ID is derived from case + participant list + session key and is revalidated during replay. Changing the ID and recomputing the session hash still refuses.

Every event contains:

- monotonic `E0001...` ID;
- previous-event SHA-256;
- event SHA-256;
- strict event fields only.

Two event kinds exist:

### `interview_turn`

Stores:

- monotonic interview turn;
- orchestrator witness ID;
- exact player question;
- exact raw response;
- raw-response SHA-256;
- zero or more **game/reviewer-owned** claim annotations.

Claims receive monotonic `C0001...` IDs. Allowed epistemic classes:

- `statement`;
- `denial`;
- `uncertainty`;
- `withholding`.

Withholding cannot carry a hidden value.

### `evidence_presented`

Stores:

- witness ID;
- evidence asset ID in private ledger metadata;
- public-access authority or a hash-valid DETECTIVE-001A evidence receipt;
- zero or more existing same-witness claim IDs marked as challenged.

Evidence presentation never changes the underlying case object and never rewrites an earlier claim.

Private/locked evidence requires an existing DETECTIVE-001A authority receipt whose visible asset set includes that asset. Public evidence can be presented directly by game authority.

## Prompt projection

`build_prompt(...)` returns a **private prompt receipt** binding orchestrator identity and a nested identity-blind `prompt` object for model use.

The model-facing projection contains only:

- case ID;
- selected witness's admitted memory **text** with canonical fact IDs removed;
- evidence **excerpts** explicitly presented to that witness, with asset/fact/terminal IDs removed;
- optional prior question/raw-response history from that **same witness only**;
- current player question.

It never contains:

- witness/member ID;
- other NPC identity/backstory/motivation/fault;
- claim IDs or contradiction metadata;
- canonical fact IDs;
- evidence asset IDs;
- terminal IDs or credentials;
- unpresented evidence;
- hidden relevance/persona labels;
- claim truth adjudication.

### History redaction

The raw ledger is immutable and preserves model text byte-for-byte, even if the model guesses strings such as a fact ID or synthetic credential.

When prior dialogue is projected back into a prompt, the projection redacts:

1. known authority tokens: fact IDs, asset IDs, terminal IDs, synthetic credentials;
2. exact canonical fact text that the selected witness has neither admitted knowledge of nor received through presented evidence.

This is projection-only. The event log always retains the exact original response.

A player question that directly injects a private authority token or exact hidden canonical fact is rejected before prompt construction.

## Contradiction derivation

Contradictions are recomputed deterministically from the append-only claim ledger.

For the same proposition key:

- two `statement` claims with different normalized values -> `value_mismatch`;
- `statement` versus `denial` of the same normalized value -> `assertion_denial`.

Each contradiction records:

- deterministic `X0001...` ID;
- proposition key;
- the two claim IDs;
- contradiction kind;
- `same_witness` or `cross_witness` scope.

Contradictions say only that **claims conflict**. They contain no canonical truth/winner/correctness field.

`uncertainty` and `withholding` claims do not create contradiction edges by themselves.

## Replay

`replay_session(case, session)` validates:

- strict session fields;
- deterministic session ID;
- case/participant identity;
- session SHA;
- monotonic event IDs;
- event hash chain;
- event SHA values;
- monotonic turn/claim IDs;
- raw-response digests;
- evidence authority;
- challenge references.

It then derives:

- turn count;
- per-witness presented evidence;
- complete historical claim list;
- challenge metadata;
- same/cross-witness contradictions;
- final event SHA.

No derived state is authoritative unless it reproduces from the event log.

## Bay Seventeen fixture

`experiments/detective-003/session-fixture.json` uses the existing two Bay Seventeen witnesses.

Sequence:

1. V-G02-A claims handoff time `02:05`;
2. V-G09-C claims `02:09`;
3. public transfer receipt is presented to V-G02-A and challenges the first claim;
4. V-G02-A revises to `02:14`.

Expected ledger:

- 3 turns;
- 3 claims;
- at least 3 pairwise contradiction edges;
- old `02:05` and revised `02:14` claims both remain;
- challenge is metadata only;
- public mandatory evidence route stays unchanged.

The model-facing follow-up receives the receipt excerpt, not `transfer-card`, `F-SWAP`, or any truth label.

## Security / adversarial coverage

Offline tests cover:

- same- and cross-witness contradiction derivation;
- statement/denial contradiction semantics;
- uncertainty/withholding non-contradiction;
- stale prompt receipt refusal;
- deterministic session-ID tamper;
- event/raw-response/hash-chain tamper;
- strict extra-field refusal;
- model text attempting to invent claim IDs/unlock evidence;
- private evidence without authority;
- cross-witness challenge misuse;
- same-witness-only history;
- hidden-fact and credential redaction;
- canonical case immutability and mandatory public route preservation.

## Non-authority

This ledger cannot:

- adjudicate whether a witness is truthful;
- mutate canonical facts;
- turn dialogue into evidence;
- make private evidence public;
- execute model-authored shell/code/network actions;
- access DETECTIVE-001R1 hidden mappings;
- train or select any model;
- admit population expansion.

It is the deterministic game-facing substrate on which later multi-turn model integration can be tested safely.
