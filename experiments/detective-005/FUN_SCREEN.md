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

Only after all **144** raw generations exist and pass source-bound dataset readback may a trusted, separate custodian prepare the reviewer-only packet. The old publicly deterministic item ordering is disabled: a fresh 32-byte private key permutes item identities. The raw dataset contains **no packet**; the key and identity mapping must remain outside the repository until ratings have been sealed.

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
- private-key deterministic review packet/rating schema/aggregation, with post-rating unblinding;
- adversarial standard-library tests.

It downloads no weights or adapters, performs no inference/training, opens no sealed C1/C2/C3 finals and accesses no hidden R1 identity mapping.

A later source-matched live workflow may consume the exact release assets and produce the 144 raw outputs. Those outputs must be promoted durably before any transient Actions artifacts expire.

## Interrupted live-run custody

The source-bound live runner now journals **each validated raw response immediately**
to `out/<lineage>/results.partial.jsonl` and flushes it to disk before
starting the next generation. On a Python exception, it writes an
`interrupted.json` receipt naming the error class, exact source head, run ID,
partial record count and partial-file SHA-256. The unconditional lineage
artifact upload retains these records (while excluding temporary unpacked
adapter weights).

Before loading models, the runner now **exclusively creates** its lineage
output directory. Even an already-existing *empty* directory is treated as an
owned or ambiguous output and refused. An interleaved competing creator
cannot be bypassed between preflight and output creation: the loser raises a
read-only collision without writing another interruption receipt. This is
local output ownership, not a distributed runner lease or proof of artifact
promotion.

A torn/malformed final JSONL append cannot suppress the interruption receipt:
the runner counts only intact, newline-terminated JSON object rows, records
`partial_journal_problem` where needed and retains the **original partial
file bytes and full SHA-256**, including the damaged tail. Never replace those
bytes with a repaired version or mistake the intact-row count for a complete
lineage.

Complete-lineage assembly now rejects substituted runner-script hashes,
cross-lineage/mixed Actions run IDs and mismatched exact source heads. The
dataset carries the verified dispatch SHA, Actions run ID and a
`provenance/source-artifacts.json` name/hash inventory. Artifact IDs are
explicitly unavailable to the assembler; they must not be fabricated.

This is a failure-recovery measure, **not a claim of durable promotion**.
Partial files are transient Actions artifacts with a 30-day retention window;
they must be retrieved and archived under `datasets/` before expiration.
The 144-response assembler remains strict: no incomplete lineage is accepted,
no subjective packet is constructed from partial results, and no training,
model judging or population admission is authorized.

## Failed-workflow durable recovery (source merged; live unqualified)

As of 29 October 2026, source custody PRs #180, #183 and #189 have
been merged on main in dependency order after recorded independent
correction acceptance. Green offline tests and synthetic Actions ZIP
readbacks do not establish actual cancelled-workflow survival or
durable Git evidence-branch write/readback. The main-only synthetic
protocol dispatch is a separate no-model push/fetch/verification probe:
its output branch is explicitly named `evidence/detective-005a-synthetic-probe-<run-id>`.
It never constitutes real villager output or a live-screen authorization.
Both complete and failed-run live promotion steps now require a remote
branch fetch, exact pushed-commit identity and raw-archive verification
after git push. These are source safeguards until exercised on real runs.
All three promotion paths now use Git's absent-ref `--force-with-lease`
precondition, so a run-scoped branch created between the existence check
and push must be rejected even if a normal push would fast-forward.
An adversarial local bare-Git regression covers that race; this does not
qualify GitHub authorization, cancellation or remote push behavior.

An additional adversarial offline test now creates a fresh **local bare Git**
repository, commits an interrupted raw journal, pushes/fetches an isolated
evidence ref, extracts `git archive FETCH_HEAD`, verifies the original
bytes even after the working copy is changed, and refuses a tampered
extracted journal. This exercises actual local Git persistence/verification
rather than static workflow string checks. **Local Git is not GitHub Actions**:
remote authentication, permissions, cancellation and hosted branch durability
remain unqualified until the separately reviewed main-only synthetic probe.

**Artifact API pagination safeguard (source-only):** the live recovery job
requests `?per_page=100`. An HTTP-success first-page response does not prove
all original artifacts were listed. Recovery requires a non-boolean integer
`total_count` equal to the returned `artifacts` count; missing or inconsistent
counts are quarantined as invalid metadata. Surviving downloaded raw files
and original API response bytes are preserved, but `artifact_transport_complete`
remains false and the final custody gate RED. Unlisted originals require
independent retrieval before retry; this is not live cancellation/push proof.

**Recursive/oversized untrusted JSON (source-only):** a downloaded raw
JSONL journal, interruption receipt, final manifest or same-run artifact API
response can exceed Python's JSON parser nesting or integer-conversion limits.
These can raise `RecursionError` or `ValueError` rather than
`JSONDecodeError`. Recovery conservatively validates zero prefix rows for
an unparseable journal, quarantines recursive receipt/manifest/API metadata,
preserves all allowed original raw bytes and surviving independent lineages,
and fails transport certification for invalid metadata. Two adversarial
no-model tests cover deeply nested JSON across all four custody surfaces and
an oversized integer literal. This is source-only validation, not proof of
real failed-run archival or an authorized inference retry.

**Artifact identity uniqueness (source-only):** GitHub Actions artifact IDs must
be positive, non-boolean integers and unique across the *entire* returned API
page, not merely unique within one expected lineage name. Two different
lineage artifact names sharing an ID cannot establish two original sources;
that listing is quarantined as invalid and the transport gate remains RED.
This rule preserves all safely recovered raw siblings and the unmodified API
response for later review. The adversarial no-model test exercises both the
collision refusal and a valid distinct-ID control. It does not prove a real
failed/cancelled workflow or successful remote evidence-branch promotion.

The merged `recover-failed-evidence` job is strictly conditional: it runs
only after a confirmed-main `RUN-144-NO-TRAINING` plan succeeds and the
lineage matrix or full evidence assembler fails. It does **not** initiate any
generation or permit a retry. It retrieves already-uploaded, same-run
`detective005a-G*` artifacts, records their GitHub metadata and copies
the original bytes into `datasets/detective-005a-recovery-<run-id>/` on an
isolated new branch `evidence/detective-005a-partial-<run-id>`.

The standard-library `scripts/detective_005a_recover.py`:

- refuses unexpected files, symlinks, unpacked weight directories and
  duplicate per-lineage filenames;
- keeps raw `results.partial.jsonl`, `results.jsonl`,
  `manifest.json` and `interrupted.json` **byte-identical**,
  including any torn or unparseable tail;
- validates intact, source-matched rows against exact member/prompt order,
  adapter/base weight hashes, dispatch commit and Actions run;
- separately records missing artifacts, absent interruption receipts,
  malformed rows, identity mismatches and incomplete versus complete lineages;
- writes a run-specific manifest, source API provenance and full SHA-256
  inventory, then independently re-verifies the archived bytes before push;
- never builds a subjective fun packet or grants any admission.

`complete_lineage_recovered` requires all 12 source-valid raw results,
a valid original manifest, a present byte-identical **fsynced incremental
journal**, and a matching same-run GitHub Actions artifact API identity.
It applies to **one recovered lineage only**, not the 144-response screen.
A locally complete lineage without API metadata is `recovery_only`, not
certified custody. Invalid/foreign rows remain retained but quarantined.

Truncated, malformed or ambiguous GitHub artifact API metadata does not
abort preservation of already-downloaded raw files. The original broken
API bytes are retained and SHA-inventoried, with
`artifact_api_metadata_invalid` recorded. The retrieval gate remains
**RED**; this does not prove missing artifacts were recovered.

`artifact_transport_complete` measures only same-run artifact retrieval,
not a valid full screen: the API and download steps must report success;
listed original lineage artifacts must be present, and every downloaded
artifact needs a valid API-bound source-run identity. A failed download
or invalid API response can still produce a best-effort archive, but
**cannot leave the final recovery job green**. No rerun is permitted
until original surviving evidence is durably promoted or a verified
no-output determination is documented.

**Push-failure fallback (source-only, not live-qualified):** after the
recovery dataset passes local readback, the workflow attempts a separate
`detective005a-recovery-fallback-<run-id>` Actions upload **before** the
non-overwriting Git push. Its 30-day staging retention may preserve an
additional copy if the evidence-branch push fails; upload failure does not
prevent the durable push attempt. Neither this transient fallback nor a
passing offline test proves successful branch promotion, verifies the
original Actions ZIP bytes, or permits lineage reruns. Confirm a durable
archive (or independently establish no output) before any retry.

**Operational limits:** cancellation of the entire workflow may also cancel
the recovery job, and artifact retrieval or branch push can fail. Such red runs
need manual investigation; a recorded failure is never evidence of successful
archival. Reruns must await verified durable preservation or a documented
no-output determination. This design remains behind independent source-review
and merge gates for PR #180 and its dependent draft and has not processed a
live run.

## GitHub Actions synthetic transport qualification (NOT a live model run)

A separate pair of no-model jobs in the DETECTIVE-005A offline protocol
exercises the **real pinned** `actions/upload-artifact` and
`actions/download-artifact` actions using exactly the live workflow's positive
path, exclusion glob and `detective005a-G*` naming scheme. It constructs one
synthetic, source-contract-shaped response, appends a deliberately torn JSONL
tail and writes an interruption receipt. A second GitHub-hosted job independently
reads the same-run artifact API, downloads the ZIP, verifies byte equality of
both files and runs the recovery/archive SHA-inventory verifier. The synthetic
`work/` file must be excluded. No model files, inference or training are used,
and CI transport artifacts are **never** promoted as live research evidence.

The GitHub artifact API includes `workflow_run.id` and
`workflow_run.head_sha`. Both are checked against the source-run identity;
invalid/missing artifact ID, SHA-256 digest, size or expiry metadata remain
explicit warnings, and a would-be complete lineage is demoted to
`recovery_only` rather than silently admitted. Raw bytes are still preserved.
A `pull_request` Actions event uses a synthetic merge SHA for `github.sha`
while the artifact API reports the PR branch HEAD, so this **synthetic-only**
test pins checkout and receipt to `github.event.pull_request.head.sha`. The
actual DETECTIVE live workflow continues to require `workflow_dispatch` on
`main`, where the dispatch SHA is the verified source head. Do not relax that
rule.

Successful synthetic upload/download/readback establishes GitHub Actions
artifact *transport* compatibility, not the success of a real failed-run
recovery job, not guaranteed handling of entire-workflow cancellation, and not
the durable promotion of any of the 144 future model responses.

## Private blinded-review custody — proposed correction for issue #187

The Stage-A public fixed `REVIEW_SALT` was insufficient for genuine blinded
human review: the known 36×4 roster and code could reconstruct R001–R144
ordering. The proposed replacement requires a **fresh random 32-byte key** and
HMAC-SHA256 ordering. The key and complete mapping are not published in
`datasets/detective-005a`. Even though public code can now reproduce the
algorithm, it cannot recover the ordering from public information alone.

The complete 144-response evidence assembler stores **only the raw dataset,
source/lineage provenance, SHA-256 inventory and mechanical hard floors**.
Its verification refuses a public review/ folder or reviewer packet. It
continues to require all 12 source-matched lineages from one exact run.

After complete evidence is committed, an authorized trusted custodian uses
`scripts/detective_005a_blind.py` locally, with directories **outside
the checkout and separated from each other**, never in public GitHub Actions:

1. `prepare` verifies the raw dataset, generates a CSPRNG 256-bit key,
   constructs an opaque 144-item reviewer packet and blank 0–4 rating
   template, and retains the seed plus exact item-to-member mapping in a
   distinct permission-restricted private directory. The public packet holds
   a binding hash commitment, not the key or the mapping.
2. The independent human reviewer gets only the packet and template; they
   must not have access to the raw archive, full source output histories or
   private custodian directory while scoring. The custodian verifies all 144
   ratings against the packet hash and original item order.
3. `freeze` preserves the **original rating bytes** and a SHA-256 receipt,
   still without member IDs or seed. A missing score or changed packet refuses.
4. Only an explicitly confirmed `reveal` after freeze can authenticate the
   private commitment, verify unchanged source/packet/rating hashes, and
   reconstruct member scores/shortlist in a separate custodian-only output.
   Admission/training gates remain false.

The private custodian requires POSIX permission controls on a vetted
local filesystem. On Linux/WSL it checks the most-specific mount in
`/proc/self/mountinfo` and rejects Windows-backed DrvFs/9p/NTFS,
FUSE and network filesystems (including CIFS/NFS). **WSL on /mnt/c
must not hold the key**: `os.name == "posix"` and `chmod(0600)`
do not establish a private Windows ACL. Use a trusted local ext4
filesystem (e.g. WSL's native Linux filesystem); unknown Linux
mounts fail closed. Native Windows remains unsupported until an
independently reviewed ACL implementation exists.
The local `freeze` receipt is not a trusted timestamp or proof of rating
independence by itself. The hash of the unpublished map/packet must be
externally committed before ratings, and the signed/independently witnessed
rating-file SHA must be recorded before disclosure to establish an audit
timeline. Do not include the private key or mapping in either public receipt.

The mapped packet cannot be considered cryptographically anonymous to a
reviewer with access to the original, labelled raw records, whose response
text can be matched against the packet. This is an **operational access
separation requirement**. The repository does not provide such a controlled
reviewer distribution channel automatically: the custodian must share only
the sanitized packet through a separate authorized channel. A public GitHub
review comment or publicly committed packet alongside raw data cannot
establish blindness.

This proposal does not prepare any actual reviewer packet, generate model
responses or use unrelated DETECTIVE-001R1 mappings. It is a new, distinct
review gate stacked after PR #183. The existing 0/144 live evidence status and
the separate independent source reviews/explicit runtime authorization are
unchanged.

## Private-custodian CLI example (after full 144-row evidence and review)

Run locally on an access-controlled POSIX environment with a vetted local
filesystem (WSL native ext4, **not** /mnt/c DrvFs), with paths outside the
repository and reviewer distribution channel. Do not run these commands
in GitHub-hosted public CI or publish the custodian directory.

```sh
python3 scripts/detective_005a_blind.py prepare \
  --dataset datasets/detective-005a \
  --reviewer-dir /secure/reviewer-005a \
  --custody-dir /secure/owner-005a

# Give ONLY reviewer-005a/fun-packet.json and its matching blank ratings
# template to the independent human reviewer; do not share the raw dataset.

python3 scripts/detective_005a_blind.py freeze \
  --dataset datasets/detective-005a \
  --reviewer-dir /secure/reviewer-005a \
  --custody-dir /secure/owner-005a \
  --ratings /secure/received-human-ratings.json \
  --freeze-dir /secure/frozen-005a

# Witness and record the original ratings SHA and prior mapping/packet
# commitment through an independent archival channel before any reveal.

python3 scripts/detective_005a_blind.py reveal \
  --dataset datasets/detective-005a \
  --reviewer-dir /secure/reviewer-005a \
  --custody-dir /secure/owner-005a \
  --freeze-dir /secure/frozen-005a \
  --out-dir /secure/unblinded-005a \
  --confirm UNBLIND-AFTER-FROZEN-144-RATINGS
```

These are instructions, **not evidence of performed runs**. All 144 scores
must originate from a human reviewer, and the owner must document their
independence and their lack of raw/identity access before interpreting scores
as blinded. The output is a subjective candidate shortlist only.
