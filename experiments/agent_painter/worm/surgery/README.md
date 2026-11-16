# SPROCKET-LOBOTOMY-001 — WORM-001 destructive weight surgery

Owner-authorised experimental sibling of [#612](https://github.com/techrote/cybersand/issues/612); not a new training lane, native implementation, or provider promotion. Historical [SPROCKET #576](https://github.com/techrote/cybersand/pull/576) and its W2 48-step all-RIGHT checkpoint are read-only. The experiment is housed on a separate Draft branch and uses two separate ordinary GitHub-hosted CPU VMs (targeted and chaos). One frozen original model + independently restored *in-memory* LoRA copies per VM; **no optimizer steps**.

The owner requests broad, deliberately destructive examples, including extreme and random weights. The **first campaign deliberately operates on the learned rank-8 LoRA adapter's 4,739,072 FP32 scalar parameters**, not on half of the 500M frozen base. Per-variant restoration and frozen-base hash equality are mandatory, to preserve causal interpretability.

Arms (the baseline is repeated on the separate VMs):
1. **attributed-zero10**: zero top 10% positive RIGHT-attributed adapter scalars;
2. **attributed-zero10-plus-crowbar50**: zero top 10% then overwrite the next 50% RIGHT-attributed adapter scalars with `torch.finfo(torch.float32).min`;
3. **attributed-extreme50**: set strongest half of RIGHT-attributed adapter scalars to minimum finite FP32;
4. **attributed-mid25**: top quarter of RIGHT-attributed values to original per-tensor range midpoint;
5. **global-adapter-extreme50 / mid25 / random25**: random chosen half set to minimum finite FP32, random quarter to midpoint, or random quarter to tensor-local uniformly sampled values; **global-adapter-zero50** is the matched nonspecific-damage control.

This is a **probe, not a recommended network repair**: minimum finite FP32 is about -3.4e38 and is likely to induce nonfinite activations. These catastrophic outputs count as negative observations, not a failed experiment or grounds to sanitize original model outputs. All random selectors have frozen seeds. The reference W2 adapter and full source/model SHA/provenance are pinned. W2 source: `7683d06d9ffb26417ffc9694a07e8cf25e165fc6`, Actions run `43697235301`, checkpoint `checkpoint-000048`, root Instruct model `kandivault/sprocket-500m@eb157d4ddddc4619e661d4a2426bb2dd000fc668`.

RIGHT-circuit **candidates** are ranked by positive gradient×LoRA-weight contributions to the RIGHT first-token logit relative to other legal action tokens on precommitted W0 observations. Separate attention/MLP residual path activation×gradient readings are recorded for inspection. Correlation/attribution does **not** alone prove circuitry: comparing targeted versus random weight ablations on identical prompts and worlds is the causal evidence. If first action tokens collide the run fails explicitly; it does not silently use an invalid proxy.

Per arm: record source/selected tensor indices and exact changed count/bytes digest, single-step six-action logits, raw greedy completions with EOS status and host legality, paired obstacle-seed W0 Python PNG+JSON traces, loops/revisits/coverage, numeric breakdowns and source provenance. All original invalid actions stay invalid and W0 owns host fallback/Stop. No sealed FINAL inspection, Art v2 curriculum relabel, native Godot material or unqualified model freeze. Preserve the strongest candidate for optional later stochastic/activation steering with no merge.

**Intended trigger:** an explicit committed `surgery-request-v1.json` on this dedicated branch starts a two-job push-triggered Actions experiment. Support/reviewer inspect outputs independently before any subsequent irreversible or global-base experiments.