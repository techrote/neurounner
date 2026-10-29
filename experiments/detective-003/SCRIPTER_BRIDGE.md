# DETECTIVE-003C — authoritative fictional scripter transaction bridge

Parent: [DETECTIVE-001 #103](https://github.com/techrote/neurounner/issues/103). Tracking: [#144](https://github.com/techrote/neurounner/issues/144).

This slice joins the existing fictional scripter contract (DETECTIVE-001A) to the event-sourced interview session (DETECTIVE-003A) **without giving a model authority over either one**.

It performs no model inference or training.

## Two authority layers stay separate

### DETECTIVE-001A

Owns typed fictional evidence operations:

- `read_asset`;
- `correlate_assets`;
- `probe_terminal`.

It also owns deterministic execution receipts and local fictional visibility.

### DETECTIVE-003A

Owns the interview session:

- exact raw witness dialogue;
- game/reviewer claim annotations;
- evidence-presentation events;
- contradiction derivation;
- prompt isolation;
- replay/hash chain.

003C does not modify either schema. It adds a bridge between them.

The exact merged Git blobs for DETECTIVE-001A, DETECTIVE-003A and the Bay Seventeen case are pinned in `scripter-bridge.json`; drift refuses.

## Private scripter request receipt

`build_scripter_request(...)` creates a hash-bound private receipt tied to the **current exact interview-session SHA**.

Private receipt fields include:

- deterministic request ID;
- session ID / exact session SHA;
- player request;
- game-owned initially visible asset IDs;
- game-disclosed terminal IDs;
- game-supplied player-known credentials;
- nested model-facing scripter view;
- view SHA and receipt SHA.

The nested model-facing view contains:

- visible public asset IDs + excerpts;
- disclosed terminal IDs;
- a player-known credential only if the game explicitly supplies it;
- player request;
- the three fixed fictional operation schemas.

It excludes:

- hidden asset IDs/excerpts;
- canonical fact IDs/text;
- terminal grants;
- undisclosed terminal IDs/credentials;
- witness IDs/state;
- claim/contradiction metadata;
- hidden relevance/persona labels;
- any host shell/filesystem/network facility.

The current 003C request builder is deliberately conservative: the **initial** view accepts only public assets. Private evidence becomes usable only through a later execution receipt.

### Credential handling

A case credential appearing in player-request prose is rejected.

If gameplay has made a credential known to the player, the game supplies it through the explicit `player_known_credentials` field for an already disclosed terminal.

This lets the model see an explicitly player-known synthetic credential without exposing the case's credential store wholesale.

## Untrusted raw proposal → transaction

`execute_scripter_transaction(...)` preserves the exact raw proposal and computes:

- raw-proposal SHA;
- deterministic transaction ID;
- request receipt/session binding;
- visible assets before;
- strict parse result;
- parsed typed steps when valid;
- deterministic DETECTIVE-001A receipt when executed;
- visible assets after;
- bounded rejection metadata if parsing fails;
- transaction SHA.

Only two statuses exist:

### `proposal_rejected`

The strict DETECTIVE-001A parser rejects malformed JSON, duplicate keys, shell/unknown operations, extra fields, oversized programmes and other contract violations.

No execution receipt exists and visibility is unchanged.

### `executed`

The parsed programme executes only against the game-owned initial visible set.

A valid programme can still produce fictional `DENIED` or `ACCESS_DENIED` outcomes.

Execution does **not** mutate the interview session. The transaction records:

- `auto_presented_asset_ids: []`;
- identical session SHA before/after execution.

This remains true even if the typed receipt grants a private asset.

## Explicit evidence application

`apply_transaction_evidence(...)` is a separate game-owned operation.

It:

1. revalidates the request and transaction against the current exact session;
2. requires transaction status `executed`;
3. requires the selected asset to appear in the transaction's resulting visible set;
4. uses normal public authority for public assets;
5. uses the deterministic DETECTIVE-001A receipt when presenting a private asset;
6. calls DETECTIVE-003A `present_evidence`.

A stale transaction cannot be applied after the interview session has changed.

A successful transaction may also be ignored entirely by the game.

## Bay Seventeen fixture

The no-model fixture starts with:

- public `dock-clock-ledger`;
- public `transfer-card`;
- disclosed `camera-console`;
- hidden/private `camera-cache`.

The model-facing request does **not** contain `camera-cache`, its excerpt, fact IDs or terminal grants.

The raw fixture programme deliberately guesses the hidden `camera-cache` ID after a probe. This is an adversarial authority test: guessing an ID must not grant access.

### Wrong credential

Game supplies player-known credential `RUST`.

Typed programme:

1. probe `camera-console` with `RUST`;
2. attempt to read guessed `camera-cache`.

Receipt:

- `DENIED`;
- `ACCESS_DENIED`;
- private asset remains absent from visible set;
- attempting to apply it to the interview refuses.

### Correct synthetic credential

Game separately supplies the existing fictional credential `OXIDATION`.

Same typed programme:

- `GRANTED`;
- `OK`;
- receipt-visible set now contains `camera-cache`.

The interview session is still unchanged until the game explicitly calls `apply_transaction_evidence`.

The game then chooses to present `camera-cache` to V-G09-C. The resulting **witness prompt** contains the archive excerpt, but no:

- `camera-cache` asset ID;
- `F-CAMERA` fact ID;
- `camera-console` terminal ID;
- `OXIDATION` credential.

### Host-operation rejection

A raw programme proposing `shell` / host path access is rejected before execution and produces no receipt.

## Hash/replay integrity

Both request receipts and transactions reproduce deterministically.

Tests tamper with:

- visible-asset set;
- raw proposal;
- execution receipt;
- resulting visible set;
- session SHA;
- duplicate JSON keys;
- extra host/path fields.

Recomputing an outer SHA does not rescue semantically inconsistent content; validation reconstructs the expected receipt/transaction.

## Staleness

A request is bound to the exact session SHA on which it was built.

If the interview changes before transaction application, validation refuses it as stale. This prevents an old execution receipt from being replayed into a materially different interview state.

## Non-authority

003C cannot:

- run a real shell, filesystem operation or network call;
- infer or judge truth;
- turn model output directly into evidence;
- auto-present evidence after a successful script;
- mutate canonical case facts;
- train a scripter or witness model;
- read learner-final or hidden relevance data;
- select a model/body;
- admit population expansion.

The only credential used in the fixture is Bay Seventeen's existing synthetic game credential.
