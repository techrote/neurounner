# NCR-001E — original frozen Sprocket 2048-unit source/cache evidence and custody

**Research:** [NCR-001 #67](https://github.com/techrote/neurounner/issues/67); [independent model-specific gate #72](https://github.com/techrote/neurounner/issues/72); [stacked draft PR #100](https://github.com/techrote/neurounner/pull/100), on #97→#95→#88→#77. **This is an actual frozen-model prepass, not complete conformance or model training.**

## Real source-bound execution (27 October 2026)

**Original physical-model Actions run [40544384293](https://github.com/techrote/neurounner/actions/runs/40544384293) completed both jobs successfully.** Two original pinned Sprocket 500M checkpoints, 4-CPU Ubuntu 24.04; Python 3.12.3, torch 2.14.1+cpu, Transformers 4.57.6, genuine complete model safetensors bytes verified against exact revision. Original PR source head during the run `f2716ab5a3f25c90fbeca1ba7057cf324372b8a3`; synthetic Actions merge SHA `2fd70b173a56742239cfd31327d63ac2cefb981f`. NCA manually activated once, **without optimizer**.

| Original read-only test | Chat subfolder | Instruct root |
|---|---:|---:|
| 512 full vs 504-prefill +8 incremental max absolute logit error | 1.8358230590820312e-05 | 2.09808349609375e-05 |
| 1024 full vs 1016-prefill +8 incremental | 1.71661376953125e-05 | 1.9073486328125e-05 |
| **2048 full vs 2040-prefill +8 incremental** | **2.193450927734375e-05** | **2.0503997802734375e-05** |
| Original frozen base vs zero adapter logits at 512/2048 | exact / exact | exact / exact |
| Fixed required absolute / relative tolerance | 5e-4 / 5e-4 | 5e-4 / 5e-4 |
| Manually untrained readout effect at 2048 | 0.009396791458129883 | 0.019722938537597656 |
| Strict 512 future-suffix / A→B→A drift | **0 / 0** | **0 / 0** |
| Four literal *raw* token ID cached transport at 60+4 | 2.5033950805664062e-05 | 6.198883056640625e-05 |
| Peak process RSS (KiB) | 3,834,396 | 3,836,460 |
| Original runner | GitHub Actions 1000045238 | GitHub Actions 1000045239 |

- Source `kandivault/sprocket-500m@eb157d4ddddc4619e661d4a2426bb2dd000fc668`; exact original full 1,002,207,984-byte checkpoint SHA **Chat** `551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97`, **Instruct** `1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864`. Chat checkpoint is under `chat/model.safetensors`, Instruct is `model.safetensors`; they are **different frozen models**.
- Original immutable NCA integration source hook commits Chat `83a56835a7476f17a17980d28dd1a7c14bb6f2f6`, Instruct `b95c010f9b2ba3cbd9f81317981316bb9c8acd19`; exact source README SHA `09aec879dc5b387bdc031dc3c74d77a9ba44362a55163f9a8b456761b68142c9`. Synthetic ordinary-role UTF-free repeated engineering prompt hash of original actual tokenizer IDs `0f4094d5e98a558601c281297191bd1bd0bc00acfa5a14472811758f4ae6970c` (both variants).
- Four control tokens were required as **source-dedicated vocabulary IDs**, tool-call **6**, tool-result **7**, memory-read **10**, memory-write **11**. They were injected as *bare IDs* into a synthetic input to check raw transport through full and cached forwards. This **DOES NOT test a complete, source-supported tool-call/result/memory dialogue**, nor establish valid argument syntax, recipient/continuation, or semantic tool or memory competence. Chat upstream training excluded tool/memory examples; Instruct's purported exposure is not protocol admission.
- 2048 is the pinned model's reported context limit, but each long cached test independently decodes **only eight final positions** after a full prefill; the full 2048-prefix scan and all-context/long-multi-turn/tool-memory workloads remain unqualified.

## Durable read-only custody (no new model execution)

The original actual-host artifact ZIPs were downloaded in full and SHA-256 checked. Their two complete **original JSON receipts, preserved byte-for-byte**, are in [original-jsons.tar.gz.b64](../experiments/ncr-001/evidence-review/sprocket-extended-context/original-jsons.tar.gz.b64), with [machine-readable full source/runner/hashes/provenance](../experiments/ncr-001/evidence-review/sprocket-extended-context/provenance.json). The source gzip SHA-256 is `60446ea2f60e5f0b7cc22333dabe45f6fa6f53d4099fda86fa9480c3987bb4af` (2,203 bytes), containing precisely two original JSONs; no regenerated version has been mislabeled original.

| Original Actions artifact | Chat | Instruct |
|---|---|---|
| Artifact ID | [12753100555](https://github.com/techrote/neurounner/actions/runs/40544384293/artifacts/12753100555) | [12752646313](https://github.com/techrote/neurounner/actions/runs/40544384293/artifacts/12752646313) |
| Full artifact ZIP SHA-256 | `ef5fd03c53818136fde2ab34c8b935d311628354b7dcfd7ba872d79fdd602482` | `c9fdcf5a64d6463c2aa5b84b4ba7f2aad261cfdc0f52208ccb8d620568e87816` |
| Full original JSON bytes | 4,678 | 4,801 |
| Original JSON SHA-256 | `12423aee3dfb9cd493698388309c5b36749db051f074f3d48f23215906a10d24` | `4f47f690270ef58e051960e23a187573b04461ae2ed9e6ef6e68347c2a366a59` |

The original ZIPs remain in Actions artifact retention, while their exact receipt contents and ZIP SHA are recorded in Git. [Independent stdlib auditor](../scripts/ncr_001e_sprocket_extended_receipt_audit.py) and [offline source-tamper tests](../tests/test_ncr_001e_sprocket_extended_receipt_audit.py) must succeed before promoting this original custody as independently read back; **that auditing is not another frozen-model execution**.

**Negative evidence remains visible:** the first independent readback [Actions 40546305601](https://github.com/techrote/neurounner/actions/runs/40546305601) **FAILED** after six tests passed, because the positive case referenced `train_authorized` instead of the auditor report's `training_authorized`. The failure was preserved and corrected in source; not evidence of a model/causality failure or permission to weaken any gate.

## Limits and admission

Both Sprocket variants have now passed a **256-unit separate Python-interpreter frozen reload** via [original #97 run 40515083104](https://github.com/techrote/neurounner/actions/runs/40515083104), and these *new* 2048-context partial incremental/cache readouts on actual pinned models. That is stronger than original 48-unit short-source parity, but **not** a complete #72 independent per-model PASS. All model-specific full causal/padding/state/optimizer qualification, actual source-approved Instruct tool/memory grammar, final #68/#69 dependency review and final #72 acceptance remain open. SmolLM2 and void.1 are distinct lanes. No sealed held-outs, no NCA/MLP training, **zero optimizer steps**, no quality claims, no model weights changed, no draft merge. The preregistered four-lane protocol **train_authorized=false remains unchanged**.

Do not retroactively attribute the void.1 byte EOS defect #92 to these Sprocket checks.
