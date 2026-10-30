# NCR-001E — void.1 four-cohort scientific interpretation, without admission

**SOURCE-BOUND SCIENTIFIC REVIEW / NO TRAIN / NO NEW MODEL / NOT SCIENTIFIC ADMISSION.** Parent #67, #72, #79, #92 and matched-compute #73. This derived interpretation is strictly stacked on unchanged original custody [#211](https://github.com/techrote/neurounner/pull/211) head 0c1a47c4d562a24c7d5a81305affb4a2501fd80e, itself on original physical [#210](https://github.com/techrote/neurounner/pull/210) f78d302ab6062c3739315c3e04317a9a97b97191. The original merged #171 report and #172/#173/#175 preliminary diagnostic remain separate and unchanged.

## Frozen authority

Exact dotlabs/void.1 revision 9c2be5087772867d45961abd11f6b35110617563; Qwen3ForCausalLM, 24×640, post-block taps [7,15,23], 360622744-byte checkpoint SHA256 ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3; reviewed ByteTokenizer code SHA256 279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03. EOS is **integer 258**, never literal printable EOS bytes. Corrected serializer commit 9fbdee880697903aaf5b91cc29442c9e53479e50; unchanged source-specific frozen tensor fingerprint 7673957ba36e8eb5427e5ecc94dd52de71fbf9e6718bb0c2a94e4d7361b3f904. Four-lane protocol Git blob 14ce5fbaa38f8806c8c8738f19c01b91385eebaf and source pins 59fff1aed872b5563aa6898171983973e0cdb02c; original absolute/relative causal-KV tolerance **5e-4**. The four original implementation/source HEADs differ even though checkpoint and reviewed tokenizer identity agree.

## Four separate source-bound original physical cohorts

| Cohort | Original run; exact source HEAD | Physical observation | Limitation |
|---|---|---|---|
| A — post-lease pretraining/short/cold | [40747035876](https://github.com/techrote/neurounner/actions/runs/40747035876); 9aed5ca0574a6885c5d60c07d16561e0f73d5043 | Corrected causal-EOS forward, 32/128 cache, A→B→A request isolation, separate Python pretraining NCA and MLP cold replay | Zero optimizer steps; NOT post-update replay |
| B — pretraining 2048/max context | [40755894558](https://github.com/techrote/neurounner/actions/runs/40755894558); 1438f45efc0dab8ed669a814b4897fc7a839302f | Actual 512/1024/2048 incremental KV and future-suffix/GenerationMixin cache on/off under frozen/manual-untrained readout; NCA 2048 max absolute error **8.58306884765625e-06** | No post-step long-context test, learning or learned EOS behavior |
| C — original one-step gradient | [40765025332](https://github.com/techrote/neurounner/actions/runs/40765025332); 2810e800ef890f0c99f9ea44adca29f879b5d0ea | One **public synthetic** NCA-only SGD diagnostic step; 31 units, 21 causal completion/EOS targets, finite same-example loss **3.065654993057251 → 1.981663703918457**, nonzero readout/upstream gradients, 3 ordered field states, frozen backbone unchanged | No cold process after update, no comparative MLP/held-out/long-context or generalizable learning |
| D — independent post-step 31-unit cold/KV | [41058648266](https://github.com/techrote/neurounner/actions/runs/41058648266); f78d302ab6062c3739315c3e04317a9a97b97191 | A **separate** one-step public 31-unit NCA diagnostic and second Python process on same runner; bit-exact post-update logits, A→B→A reset and four genuine incremental final positions, max absolute error **9.5367431640625e-06**; 4 effective Torch threads after load | Not 2048 post-update, not all 31 positions, not a new host, not masked-batch/future-suffix/native generate post-update |

The **public prompt and shifted label/mask hashes are equal between C and D**, but the physical original runs, diagnostic updates and adapters are not the same execution. D's original on-disk intermediate receipt deliberately says NEW_DIAGNOSTIC_NOT_ORIGINAL_172. Do not combine C gradients and D KV as if measured on the same adapter instance. Do not equate 2048 pre-step source behavior in B with 2048 post-step behavior in D.

The raw Git custody is independent: A/B source and host original ZIP evidence under experiments/ncr-001/controls-evidence/void-postlease-20261028/ and void-postlease-maxctx-20261028/; C original complete ZIPs/logs and failed archival evidence under void-nca-learning-20261028/ (PR #173); D both original complete ZIPs (369/6770 bytes), full original source/host decoded job logs (22795/53480 bytes), and eight ZIP members under void-poststep-cold-20261030/ (PR #211). D manifest Git blob is 3b56c4b3e05da323e65cc3a187e5d2d320a7a007. Source/auditor CI failures in the original custody history are preserved, **not** relabelled as Qwen3 numerical failures.

## All nine preregistered #72 checks: bounded observation versus acceptance

| Required #72 check | Original merged #171 status | Later observations / limitation |
|---|---|---|
| 1. same_base_logits | PASS | Original no-op/frozen physical A/B; final independent model-specific admission not granted |
| 2. base_weight_unchanged_after_step | BLOCKED_BY_TRAINING_POLICY | C and D each observed unchanged frozen Qwen3 after **one public diagnostic** update; does not prove broadly representative learning or final admission |
| 3. causal_suffix_invariance | PASS | Pre-step A/B physical, including 2048; **no after-step 2048 suffix sentinel** |
| 4. full_prefix_vs_incremental_cache | PASS | Pre-step A/B up to 2048, D only **four post-step final positions** at 31; no joint after-step 2048 or every-position after-step cache |
| 5. per_token_isolation | PASS_SOURCE_RUNTIME | A/B source/runtime isolation and causal sentinels; D one public single stream does **not** establish post-update masked batched state isolation |
| 6. per_request_reset | PASS | A before step and D after step A→B→A; bounded synthetic sequence not multi-session stress |
| 7. nonzero_adapter_gradient_after_warmup | BLOCKED_BY_TRAINING_POLICY | C and D diagnostics show nonzero after readout warmup; **single public case**, no learned usefulness admission |
| 8. finite_state_and_loss | PARTIAL_BLOCKED_WARMUP_LOSS | A finite source loss; C/D finite after-warmup diagnostic loss and ordered depth fields; no long-context multi-example learned stability or generalization |
| 9. cold_reload_state_dict | PASS_PRETRAIN | A pre-step separate-process NCA/MLP, D **post-step NCA-only** separate Python with bit-exact active logits; same runner, not new-host reboot/long-context cold replay |

**Nine required checks have source-bound positive observations across distinct cohorts, not nine independently scientifically admitted results.** The merged #171 historical statuses remain literally unchanged; do not retrospectively rewrite its three originally BLOCKED/PARTIAL checks as PASS. Administrative owner-accepted same-account COMMENTED review is neither GitHub APPROVED, a second independent person nor scientific gate admission.

## Precisely unproven joint premises (not retroactively new mandatory gates)

1. A *single* actual frozen-Qwen3 run combining **post-update NCA** with **2048** full-sequence/incremental-KV and causality. B is pretraining; D is post-update 31-unit.
2. Full post-step per-position cache parity: D verified only four final positions, not all 31 or all 2048.
3. Post-step 2048 strict future-suffix and B>1 left/right masked batch per-token isolation, distinct from previously tested pre-step cases.
4. Post-step long-context HF GenerationMixin cache-on/off, not inherited from B.
5. Fresh-host post-step reload/replication: D is a genuinely new interpreter **on the same runner**, not a new CPU host or persisted learned release checkpoint.
6. Multi-sample, multi-seed learned usefulness or NCA-vs-MLP matched quality/compute (requires separately authorized experiments, not implicit #72 training).
7. Explicit exact-source independent **per-model scientific #72 acceptance** and downstream dependency/review/merge authorization; do not conflate source/custody PASS with this decision.

These joint tests are **scientifically useful stronger extrapolations**, not additional preregistered criteria silently inserted into the nine-check protocol. The original 12/12 four-step generations without real EOS 258 remain negative. The causal effect of historical TRAIN-002b literal five-byte EOS serialization remains **UNMEASURED**. Sprocket root-Instruct incomplete generated tool-call prefix belongs to a separate lane and remains negative there.

## Source-only evidence and refusal architecture

The new stdlib script scripts/ncr_001_void_fourcohort_scientific_delta.py first re-executes the prior original A/B ZIP auditors and committed #171 matrix, then #175's C original ZIP/full decoded-log audit and frozen receipts, and finally #211's D literal Git-tree/SHA/ZIP/full decoded-log auditor. It independently opens D's host ZIP to bind producer/consumer/intermediate receipt identity; checks the same public tensor-ID/label/mask hashes as C, while requiring different physical run IDs, source heads and processes. Adversarial tests reject a forged 2048 post-step claim, C/D run fusion, token-length/cache-count substitutions, changed source/pins/EOS, invented science or training, missing negative results and relaxed numerical tolerances.

A green source-only workflow is **not** a fresh model run. It must execute at the new exact PR head with immutable protocol/pins/source checks; the Git-custodied original raw streams outrank the derived report. No model inference, optimizer step, held-out content, protocol/fixture change, raw-evidence mutation or draft merge is authorized or performed. No new checkpoint is released.

**Locked:** train_authorized=false; void1_conformance_pass=false; full_72_conformance_pass=false; actual_compute_budget_qualified=false.
