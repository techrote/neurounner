# DREAMGRAFT-003 — Can inherited laws produce a novel interaction?

**Status:** frozen inference-only MEMPLANT experiment. No new model training, adapter updates, or merges. Stacked on [DG002 PR #233](https://github.com/techrote/neurounner/pull/233).

## The actual hypothesis

The receiver has previously been trained only to emit two factual properties supplied by a frozen Qwen2.5-0.5B donor: heating **hardens/softens** an alloy, and vibration makes its dust **float/sink**. The receiver never trained on the effects of a relay, furnace bridge or stabilizer.

We now introduce **three new public physical mechanisms** (their rules are in the task prompt, never in donor source or prefix training):

| Challenge | Public mechanism | Answer rule | Positive frequency across four source worlds |
| --- | --- | --- | --- |
| **XOR relay** | Energizes if exactly one of two sensors is on | hardens XOR floats | 2/4 |
| **AND furnace bridge** | Engages if both sensors are on | hardens AND floats | 1/4 |
| **Agreement stabilizer** | Activates if both sensors match (both on or both off) | hardens XNOR floats | 2/4 |

A receiver can only correctly predict the novel outcome after obtaining both inherited laws (or guessing). It is not asked to repeat the encoded words; the answer is **yes/no** to a new mechanistic consequence.

### Frozen components and source custody

- Source: exactly [DREAMGRAFT-002 run #43736088404](https://github.com/techrote/neurounner/actions/runs/43736088404), original `memplant-dg002-source-a1` artifact. SHA-256 `473cdb663922705cbddd99b57bfc5ea63048e113ec318e238574b2420511ae75` for the Qwen pooled attention-V feature array.
- Recipient: frozen SmolLM2-1.7B-Instruct; exact prototype checkpoint from `memplant-dg002-prefixes-a1`, SHA-256 `89fdb9c2edf8f2d3fffba6a124e6bcb62177229a4ee5d70522a17ed02eba8cc4`.
- Predeclared source readout: the **same lambda=1 two-output ridge recipe**, fitted only to DG002 training-family features. No intervention-task labels, prototype adjustments, fine-tuning, or new source examples are admitted.
- Test set: all **24 already-held-out DG002 cases (six world families × four counterfactual law combinations)**. The source cases have been viewed in previous research; the **interaction tasks are new**, not a fresh source-distribution confirmation.

### Strict comparisons

Each task and case is scored against the **actual presented donor** when a counterfactual swap occurs:

1. Actual V-derived continuous prefix, heat-only swap, vibration-only swap, and both-law swap.
2. B-native direct **oracle-prefix** selection from the evaluator's true pair of laws. This is an interface/reasoning diagnostic, **not** donor-state communication.
3. Hard-gated two-bit source classifier, a fixed equal blend, eight-zero prefix, norm-matched random prefix, and **true no-prefix**.
4. The two genuine laws explicitly supplied in text (**text-facts** baseline), and the complete original donor diary (**full-text** baseline).

A positive finding is not merely that responses change after a donor swap. For an XOR mechanism, swapping either one law *should* change the answer, while swapping both should *not*. For the AND mechanism, a one-law swap changes the answer only when the other sensor's state makes that rule relevant. All expected and actual flips are recorded individually, including **unexpected flips where the logical consequence should remain unchanged**.

The program scores two complete yes/no answer sequences in one batched frozen-model forward, preserving total conditional log-likelihood, selected result and source provenance. It records error counts, family sizes and inference cost; no invented availability or output claims.

### What could be demonstrated

**If successful:** a frozen recipient can derive a new mechanistic consequence from **two numerical attributes** transmitted from another model's real attention state, under a withheld task definition and causal source swaps.

**What success would not establish:** general world-model transfer, model-independent memory, direct cross-family KV-cache migration, a numerically editable latent allele, or any information inaccessible to a symbolic two-bit decoder. The mechanistic rule is supplied as text to the recipient, and the latent carries only two supervised properties.

**If unsuccessful:** native prefix and explicit-text oracles separate lack of receiver task ability from loss of donor information. Negative findings stay recorded.

### Creative output

One XOR run also retains six unedited samples in which the recipient invents a machine using the two inherited laws. This is exploratory art only; coherence and source fidelity are not silently inferred from the structured multiple-choice result.

## Run and review

The [isolated workflow](../../../.github/workflows/memplant-dg003.yml) runs three independent capped CPU evaluations and publishes a concise summary plus raw per-task evidence. No automatic next gate or merge. The true **Counterfactual Allele**—altering a latent memory *without re-encoding new source worlds*—is a later, separately authorised research question.
