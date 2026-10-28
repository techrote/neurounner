# DETECTIVE-004A: synthetic scripter-specialization training corpus

**Status: Stage-A corpus archived; no model inference, no model training and no body/population admission.**

This archive is the complete deterministic **36-example** candidate curriculum from [issue #148](https://github.com/techrote/neurounner/issues/148) and [Stage-A protocol](../../experiments/detective-004/STAGE_A.md).

The six novel fictional training families are `amber-quay`, `cinder-span`, `ferrite-yard`, `opal-canal`, `signal-orchard` and `violet-works`. Each has four publisher-native tool-selection turns, one no-tool restraint and one host-owned result-continuation turn. Aggregate: **8 each** `read_asset`, `correlate_assets`, `probe_terminal`; **6** restraint; **3** successful-result use and **3** access denial.

- `training.jsonl` contains every exact synthetic message, canonical rendering, audit-only assistant-trained character spans, model-visible view and deterministic expected answer.
- `manifest.json` binds the source config and generator **Git blob SHA-1**, corpus SHA-256, all 36 IDs, source checkpoint/publisher identities, counts, and explicitly zero optimizer/provider/model calls.
- `provenance/source-artifacts.json` records the exact source-head GitHub Actions run that produced the promoted training bytes.
- `provenance/hashes.sha256` inventories durable archive files except itself.

The records are authored by deterministic source code, **not raw model generations**; consequently there are no provider raw responses, model failure records, retry/repair history or inference runner measurements. Historical B3/003D records remain in their original datasets and were not used as training examples. All credentials here are **synthetic game strings**; none is a host/system secret.

The generated archive was promoted from the source-matched offline CI run on **28 October 2026** after two independent deterministic generation passes compared byte-for-byte. Reproduce its core files with:

```sh
python3 scripts/detective_004a_protocol.py --verify-training datasets/detective-004a
```

The evaluation final is **not present**. A later, independently controlled post-trained-adapter-hash materialization must use the frozen three-family recipe and sealed-final custody rules. There is no direct path from this archive to population expansion or production-game scripter admission.
