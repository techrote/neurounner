# Ecology analysis provenance and recomputation

The existing [analysis.json](analysis.json) and [interpretation](ECOLOGY_INTERPRETATION.md) remain unchanged. The analysis identifies its original generator with `analysis_script_sha256`; that field does **not** identify the new reusable recomputation script.

## Original v2 generator

[extend_ecology_analysis_original.py](provenance/extend_ecology_analysis_original.py) is a byte-for-byte copy of the original `extend_ecology_analysis.py` source. Its SHA-256 is:

```text
ede0f0c777098965df0d4b822890ef4c4d2c676078498d1cfbd6b4243f1c775f
```

This exactly matches `analysis.json.analysis_script_sha256`. The original source used the previous scratch analysis/report, verified every recorded raw session/judgment hash, added the dominance/deference calculations, and rendered local and repository-relative report variants. It is preserved as a historical source record; its fixed scratch paths make it unsuitable as the portable entry point.

Original invocation context:

```bash
cd /workspace/scratch/17aa561a5219/e-ecology-interpretation
python extend_ecology_analysis.py
```

Its original v1 inputs remain preserved in that directory:

| Input | SHA-256 |
|---|---|
| `ecology-analysis.json` | `6480250de76d508c38b30580202901ab40ad99e84be8c2b2acfe318f9b79c5fb` |
| `ECOLOGY_INTERPRETATION.md` | `d8e43fa75cf1feed859a1964770c1a8f30467ee15868151a0674660e1c8c3b55` |

The final v2 source includes the editorial correction that **M3**, specifically, also uses waiting language in E-G-12. The corresponding report sentence and generator-identity metadata were corrected together before the candidate was frozen; no ratings or numerical results changed.

## Reusable read-only recomputation

[recompute_ecology.py](recompute_ecology.py) is a separate standard-library implementation. It reads the sibling analysis as the expected source-hash inventory, checks the exact archived session/judgment bytes and frozen config, independently recomputes all quantitative sections, and compares the results with the existing analysis. It neither replaces that analysis nor regenerates editorial prose or model judgments.

After final assembly and publication of this review directory, run:

```bash
python experiments/village-001/qualification-20261119/review/recompute_ecology.py \
  --repo /path/to/neurounner \
  --out /path/to/new-ecology-recomputation.json
```

`--out` must be a new file and must not be inside the raw final-evidence tree. The default reads the relative final-evidence locations in `analysis.json`; missing files or changed hashes fail closed. The receipt records its own script hash, the compared analysis hash, all raw source hashes, the verified sections, and whether the assembled paths were checked. It performs no network, model, tokenizer, optimizer, or provider operation.

## Verification before final assembly

The portable script was run against the actual frozen generation sessions and preserved first-recovery judgments using explicit source roots:

```bash
python repo-ready/review/recompute_ecology.py \
  --repo /workspace/scratch/17aa561a5219/neurounner-e-acceptance \
  --sessions /workspace/scratch/17aa561a5219/neurounner-e-acceptance/experiments/village-001/qualification-20261119/generation/ecology \
  --judgments /workspace/scratch/17aa561a5219/e-recovery/recovery-44443194861/bundle/lanes \
  --out /workspace/scratch/17aa561a5219/e-ecology-interpretation/recomputation-v2-current-sources.json
```

All **36 numerical/data sections matched**, including the primary and lineage metrics, selective-audit counts, exact repetition checks, token counts, all 60 per-appearance ratings, all 36 member summaries, and the weighting/handle/context breakdowns. The receipt explicitly reports `source_layout="explicit_preassembly"` and `final_assembled_paths_verified=false`; it does not claim final assembly acceptance.

| Verification artifact | SHA-256 |
|---|---|
| Portable recomputation script | `d95e42c004dd1d006b1e26d35a405c4bd79796b7ef182dec92854a723c02803f` |
| Preassembly recomputation receipt | `46b97979d5560f634cdf9e289a562b6cb2ae8e8d76d85ce315a3e37d81322971` |
| Unchanged candidate analysis | `25a426b0bf8cd0675b6616a748206c6ed990cf028299fd0454eaf28ff74d5b90` |
| Unchanged candidate interpretation | `5e3d02f2d29662da0cb5a5aa57e346eaeafa3f0675d7bc4ed8379facd21eee74` |

Before integrating the candidate, rerun the portable script without source overrides against the assembled tree and resolve the interpretation's relative links. The qualitative G05 boolean/reason inconsistency remains an explicitly documented interpretation of archived content; this numerical recomputation does not rejudge or resolve it.
