# G1B Dream Postcard — a deliberately simple numerical rescue

The first end-to-end V/hidden/embedding MLP compilers failed to produce held-out donor-dependent decisions after 48 updates. That is a useful negative result. G1B explicitly isolates **source recognition** and **recipient continuous-prefix interpretation**.

**Mechanism:** frozen Qwen2.5-0.5B V cache → fixed 2,048-scalar fingerprint → small supervised ridge readout → either hard-select or continuously blend two directly learned SmolLM2-1.7B-native soft prefixes. The receiving model is frozen and never sees the donor text in latent arms. Source labels are used to fit the ridge and training prototypes, so this is **supervised feature classification + target-native pseudo-embedding compilation**, **not** native KV transplantation, a shared model-independent state, or novel causal memory inheritance.

### Comparison
A fixed-regularisation linear classifier is trained on 32 original paired memories and checked on eight original validation cases. **Four new fictional material families with eight opposite-rule observations** form the independent G1B test (not used or inspected in G1). At runtime, true and swapped source features are compared under the same B prompt, with an equal-norm random prefix, zero prefix, fixed prototypes, a continuous blend and separately labelled full-text B input.

All numerical conditions use the same two recipient-native learned prefixes; the **hard-gated channel** carries just a single classified bit, not the complete donor computational state. The **continuous blend** tests whether a confidence-bearing scalar can influence B without discrete selection. Both are needed to separate “one-bit route works” from “a smooth JIT pseudo-embedding works.”

Eight fresh cases cannot establish reliable generalisation. The science question is whether the low-bandwidth route produces any source-specific recipient decisions and survives a donor swap. Sixteen raw creative outputs are preserved separately. There is no editor, selective gallery or automatic progression to G2/G3.

### Execution
The isolated [G1B workflow](../../../.github/workflows/memplant-g1b-postcards.yml) prepares real Qwen V features and then runs frozen 1.7B native-prefix fitting/evaluation, each in a bounded hosted CPU job. Model weights are immutable and verified; small evidence and adapter tensors are retained outside Git. The parent G0, G1 and documentation PRs remain Draft/HOLD.
