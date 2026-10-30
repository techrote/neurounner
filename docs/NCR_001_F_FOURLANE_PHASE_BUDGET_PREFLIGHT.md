# NCR-001F — four-lane adapter MAC and actual training-phase budget preflight

**SOURCE-ONLY / NO TRAIN / NO SCIENTIFIC ADMISSION.** Research #67, conformance #72, matched-compute #73 and four independent learned comparisons #74/#75/#80/#81.

This child is stacked directly on exact original #209 head **28353d042fe479589e04fc7a85fbc56a90fb7707**; #209 is stacked on #201's permanent original three-ZIP/three-decoded-log custody from corrected physical Sprocket run [40904839128](https://github.com/techrote/neurounner/actions/runs/40904839128). This child does not alter original model source, weights, raw archives, logs, protocol, candidate fixtures or previous review decisions.

## Source-only four-lane accounting

The unchanged model-specific three post-block taps, 8×8×4 field and three cellular update steps give **adapter-only counted dense/Conv multiply-accumulates (MACs) per token**.

| Frozen lane | Hidden | Gated MLP | 3-step NCA primary | 1-step NCA | Primary NCA / MLP counted MAC |
|---|---:|---:|---:|---:|---:|
| SmolLM2-135M Instruct | 576 | 884,736 | 1,290,240 | 1,019,904 | 1.458333333× |
| Sprocket Chat (chat weights) | 1280 | 1,966,080 | 2,371,584 | 2,101,248 | 1.20625× |
| Sprocket distinct root Instruct (root weights) | 1280 | 1,966,080 | 2,371,584 | 2,101,248 | 1.20625× |
| dotlabs/void.1 Qwen3 / ByteTokenizer | 640 | 983,040 | 1,388,544 | 1,118,208 | 1.4125× |

The reset-at-depth and permuted-neighbour three-step source proxies equal the primary three-step proxy; these interventions may still differ in actual memory/state overhead. Conventional zero-init gated MLP has **514H+512** parameters, versus **514H+984** for NCA: 472 more NCA parameters, within the preregistered 5% count margin. Sprocket Chat and root Instruct share *geometry*, not weights or semantic baselines. void.1 UTF-8 byte units are not interchangeable with BPE units.

Each three-step NCA primary adds 405,504 counted local Conv MAC/token beyond the matched dense projection, at three taps. This does **not** measure hardware FLOPs, actual training compute or memory bandwidth. The proxy excludes frozen transformer cost, backward/autograd, normalization, nonlinearities, allocations, checkpoint IO and CPU scheduling. Never multiply an unobserved training-token count by this proxy and call it measured compute.

## Existing physical evidence is forward-only

Original #199/#201/#209 evidence contains three interleaved **untrained complete-model forward** samples per path for each Sprocket variant at 512 and 2048 source units, one warmup, effective four-thread AMD EPYC 7763 runners (separate hosts between model variants). The NCA/MLP ratios are descriptive: Chat **1.01038× (512), 0.99904× (2048)**; root Instruct **1.00594×, 1.00263×**. The previously computed extreme unpaired-ratio ranges are **not** confidence intervals or matched training-compute evidence.

No completed comparative NCA/MLP optimizer-run profile exists in these archives. No matched backward or optimizer time, actual per-step supervised token ledger, multi-seed training-time variance, per-phase peak RSS, or matched learned-quality result exists. The separate void.1 single public NCA-only diagnostic SGD step is not an MLP comparison; its startup threads=2 field was recorded before the loader set effective threads to four.

## Future actual-cost physical evidence required before admission

The new source-only machine preflight enumerates, without fabricating, the required genuine evidence:

- Exact frozen source/adapter source head and checkpoint hash for **each** model lane; original complete Actions ZIP and decoded job-log SHA/IDs.
- Verified same host/CPU identity **within** a paired frozen-model experiment; effective Torch thread count *after* model load; warmup; peak RSS and original runner/package inventory.
- Locked optimizer and three preregistered seeds (20261026, 20261027, 20261028), identical sample ordering, real input/supervised token counts with valid assistant-only masks. void.1 must separately report UTF-8 bytes, ByteTokenizer units and EOS.
- Actual, source-matched paired **forward, backward, optimizer** nanosecond observations, raw repetitions and uncertainty for both NCA and MLP. Never treat original forward-only samples as training phases.
- Complete raw, negative, partial and failure evidence under permanent Git custody, followed by distinct source-bound scientific review and explicit permission to train.

The predeclared **192 updates/seed × three seeds = 576 planned updates per arm** have not been performed here. The script refuses alteration of the four variant weights/hidden dimensions/taps, NCA three-step default, six preregistered comparison arms, frozen-base policy, training prohibition and the nine-check #72 gate. Its CLI re-audits #209's original Git-custodied three ZIPs/three decoded logs before deriving all four source MAC accounts. Tests mutate those contracts. A green source-only CI is not a new physical run.

**Status:** train_authorized=false; full_72_conformance_pass=false; actual_compute_budget_qualified=false. No held-out contents, model weights, original archives or preregistration were changed.
