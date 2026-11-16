# D-1B four-stream offline handoff — 15 November 2026

**Research-only.** For #570 / Draft PR #596. This is an optional input/sample
generator for the Wormfleet Commander, **not a fleet scheduler, actual model
prompt adapter, native observation, native receipt, or M1-conformant event
journal**. No workflow, runner, model, PNG renderer, material editor, or Undo is
started. Source writes remain strictly in D-1B.

## Accepted source and boundaries

- Accepted M1 source: 8fa90de6be209ed0b7bd125039e6ec8b332fb1a6;
  schema SHA-256 ebe2fa211148602c614cba0f0a40f285cba797784b70efa035832d833877f80d.
  This module names the approved vocabulary; **it does not validate a complete
  M1 event journal**. Full five-record conformance is still a central-owner gate.
- Q8 native #611 evidence is **text-only** (--no-mmproj), one paired seed;
  the independent D-1B archive auditor and its original material receipts are
  neither altered nor assumed for these synthetic fixtures. No vision d1 claim.
- Goal: study modest, sometimes wayward line steering, not A* supremacy or an
  autonomous art supervisor. A native host alone owns the legal candidate
  catalog, materials, protection, STOP, revision fencing and rollback.

## Paired design, not a four-worm claim

For each declared synthetic seed/source, build_stream freezes the identical
snapshot, World/material SHA-256, legal action IDs/footprints and host geometry.
It produces four **independently mutable** sample arms:

| Arm | Pre-choice scalar | After-choice shadow critique |
|---|---|---|
| sham | u16 neutral zero; all optional preference channels zero | none |
| guidance | source-derived u16 positive scalar | none |
| critic | u16 neutral zero; all optional preference channels zero | hypothetical KEEP/UNDO/STOP |
| combined | source-derived u16 positive scalar | hypothetical KEEP/UNDO/STOP |

The pre-choice **sham == critic** model views and
**guidance == combined** model views have identical SHA-256 at every optional
information level. **Important correction:** the previous preview only zeroed
its scalar core. Its sham *preference/all* variants still exposed the active
signed-preference signal, confounding guidance-versus-sham experiments.
Current sham/critic views now neutralize every registered preference channel
(`u16le` attraction/avoidance and `i16le` signed preference) and hash the
actual neutral little-endian bytes. Factual `geometry_metric` distance/flow
channels remain *identical* in all arms. Thus preference/all views compare
active versus neutral **preferences**, whereas geometry views compare the
core scalar with identical geometry context. They are **not** geometry-blind
controls. No semantic field or M1 contract byte was changed. The critic is strictly post-choice; **it cannot leak a
suggested action or future metric into a controller's pre-choice prompt**.
A proposed shadow UNDO is never actual native rollback.

For each arm, the same source state supports core, preference, geometry,
and all numeric-channel ablations. Channel vocabularies/encodings follow
the accepted M1's positive u16, signed i16 and XY i16 registrations; each
numeric tile is capped at 4,096 cells and hashed from its explicit little-endian
bytes. Omitted semantic identities/hashes appear in audit-only data,
never in the model view. The exported record is a **D-1B research preview**, not
a replacement for M1's typed source schema or native safety checks.

build_four_streams refuses fewer/more than four sources, duplicate stream
IDs, seeds or snapshot IDs, **any repeated World digest, any repeated material
digest, or mixed host source revisions**. Previously distinct World/material
*tuples* were enough: two nominally separate sources could alias one World
while only varying a material hash (or vice versa). Such a bundle is now
rejected, even when all four seeds differ. The four sources must use the same
software revision but distinct declared World and material identities.
**These hashes are declarations, not authentication of original World
bytes.** Each stream must still be independently reverified by the
integrating host. Four synthetic streams imply **zero** model PIDs or resident
worms; only Commander's original leases/PIDs/job IDs establish concurrency.

## Reproduction (offline stdlib; no GitHub Actions dispatch)

From repo root:

~~~bash
python3 -m unittest experiments.driv3r.stage_d.guidance.test_fleet_samples -v
python3 - <<'PY'
import json
from experiments.driv3r.stage_d.guidance.fleet_samples import build_four_streams
from experiments.driv3r.stage_d.guidance.test_fleet_samples import fixture
bundle = build_four_streams([fixture(i) for i in range(4)])
print(json.dumps({"accepted_m1_source": bundle["accepted_m1_source"],
                  "stream_count": bundle["sample_stream_count"],
                  "resident_worm_count": bundle["resident_worm_count"],
                  "content_sha256": bundle["content_sha256_without_self_digest"]},
                 sort_keys=True))
PY
python3 -m experiments.driv3r.stage_d.guidance.qualification --timeout-seconds 180
~~~

Golden fixture: seeds **101, 203, 307, 409**; four source-isolated
synthetic World/material pairs; **4 arms × 4 optional-field views × 4
source seeds = 64 pre-choice input previews**; eight post-choice shadow sets
with three illustrative KEEP/UNDO/STOP cases each. Canonical bundle
digest **excluding its self-digest**:
dfaffcdceb26751e950b217daf64e55120097094fabead06af16c54f9946ba8e.
The earlier `e81d076d...` golden is superseded. The new digest was
independently recomputed only after an offline Python reconstruction exactly
reproduced the old golden, to check the serialization and 16-bit byte model.
This digest is for the synthetic fixture, not a native artifact.

**Historical qualification:** 16/16 focused stdlib tests passed for the
previous fixture source, with source-blob matching. The current repair adds
**five** adversarial tests (all three optional preference encodings, factual
geometry parity, neutral raw-byte hashes, and fixed golden custody), giving
**21** fleet-fixture methods and a fail-closed full-suite minimum of
**220 methods in 20 modules**. The independent fixture reconstruction matched
the original golden and computed the repaired golden above, but it is **not**
a run of the five new committed tests. The subsequent source-alias/revision isolation repair adds **three** more
adversarial methods, for **24 fleet tests and 223 declared test methods in 20
modules**, independently inventoried from exact GitHub blobs. The golden
fixture is unchanged because its sources already had distinct World/material
digests and one revision. The current exact-head clean-checkout **223-test
suite remains unexecuted**; do not cite Stage A/source-scope CI or a separate
reconstruction as proof of the committed tests.

## Commander integration / M2 handoff

Before actual model/native inference, Commander must bind each sample stream to
a real source-checked native World and distinct wrangler ID, lease ID,
model/checkpoint hash, run/job ID, loaded PID, original observation/receipt
hashes and bounded decision budget. Only host-produced catalogs are legal.
The Commander, not this producer, must verify raw original Q8 receipts, isolate
run-level seeds and native episodes, and measure actual one-vs-two model
co-residency. Real KEEP/UNDO/STOP needs native rollback/revision acceptance by
the central owner. No 128-turn, training, paid compute or graphics support is
implied by these offline samples.
