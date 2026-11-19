# VILLAGE-001E isolated final qualification

This execution follows accepted Stage D, merged in PR #254 at b3d98688814a720e6b4902887a911540c3be736e and independently read back from main before issue #13 closed. The complete 36-member D population is fixed; weak or failed members remain in every evaluation.

## Frozen evidence and recipe

The generated final suite is frozen at SHA-256 b404422c5a32ccdc53abf78f849a92a82938c0e88c2585d9d4ac90f3519631da. It has 24 scenarios, six families, and zero exact collisions against 3,383 training/diagnostic inputs. Its seed derives from the exact D summary, D adapter-release manifest and original final-suite recipe. Configuration differs from the original only in the frozen suite path, accepted D release tag and isolated E release tag. No scientific gate, model, inference recipe or training input changes.

## Two execution phases

1. Generate all 888 final responses (36 members plus BASE, each on 24 scenarios) and 180 ecology turns across 24 sessions. Preserve native rendered prompts, input/output token IDs, per-row checkpoints, identities and terminal records. Independently replay every result offline. Commit all generation evidence and its hash inventory to the owned branch before any cloud judging.
2. Run 24 tournament and 24 ecology judging lanes against those fixed bytes with a single global provider lane at a time. First valid judgments are immutable. Resume only matching source identities and exact request payloads; repair missing judgments in bounded 3/2/1 batches. Independently replay terminal judgments offline, then assemble all population and individual gates without dropping failures.

All adapter downloads select the accepted D manifest release and verify SHA-256/directory identity. The local inference runtime is fixed to torch 2.14.1+cpu, transformers 4.57.6 and peft 0.21.2; the SmolLM2-135M revision remains 12fd25f77366fa6b3b4b768ec3050bf629380bac. There is no new training.

## Publication and recovery

The workflow runs only after request.json changes on work/village-001e-qualification-20261119. Pull requests run offline checks only. Evidence commits use an unchanged-head lease and nonforce push solely to that owned branch. Final evidence publishes to the distinct prerelease village-001e-final-20261119; the adapter registry references the existing accepted D release. Full raw provider and generation evidence enters both Git and a complete release TAR. Actions artifacts are staging/recovery copies.

A later request revision can name a completed prior run in resume_run_id. Every recovered ZIP is bound to the owned workflow/branch/ancestor, exact artifact identity, byte size and digest, then safely extracted. Generation and provider identities independently reject source drift. Raw failures and prior accepted judgments are preserved.

Baseline judging is 468 logical requests, up to 564 with every existing optional audit. The total hard logical ceiling remains 1,104: 42 per tournament scenario, 3 per pair session and 5 per cross-lineage session, including schema repairs and persisted across resumes. Each logical request permits three archived transport launches, each with six HTTP attempts; the theoretical aggregate transport ceiling is 19,872 and job timeouts also bound work. No arbitrary all-population rerun or gate relaxation is permitted.

## Interpretation

Stage D admission is a relative regression screen, not a usable-village verdict. E measures fresh F-suite task relevance, genericness against contemporaneous BASE, raw and style-neutral lineage/sibling identification, transfer families, and multi-agent ecology. The sibling experiment supplies the correct lineage for every response; its denominator is all member responses, not only responses with a correct first-stage lineage guess. Frozen population gates require at least 30 individual passes, sufficient behavioral lineage and sibling accuracy, and lower genericness than BASE. Ecology is reported in full but does not enter the original usable-village conjunction.

Follow-on work will respond to E results while preserving this suite as final evaluation evidence. It must not turn these test failures into new training examples.
