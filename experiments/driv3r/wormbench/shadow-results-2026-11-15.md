# WORMBENCH — first original-state CPU shadow results (15 Nov 2026)

**Scope:** exactly 72 originally native-verified d1 Q8 wormbrain *decision observations*, 36 free and 36 soft, source `313b836b3c9ce2c60e4659e3763df140f84347a6`, Actions [43678354515](https://github.com/techrote/cybersand/actions/runs/43678354515), original artifacts `14147579972` and `14147233975`. Frozen 72-state record SHA-256 `178aff09de061a1fe235b1039ace74665192a93651f826942b669ec9e7a78c15`. No new PAINT, native World mutation, learned-art quality or free-running trajectories. All alternative decisions remain independent one-turn shadows **on original Q8 states**, not full autonomous episodes.

## Exact measurements and failures

| Controller / output policy | Median decision seconds | Valid actions | Agreement with Q8 | Native? | Source run |
|---|---:|---:|---:|---|---|
| Original d1 Q8 System One | **13.3064** | 72/72 | 72/72 | Original native receipts/replay | 43678354515 |
| d1 Q4_K_M System One | **3.4953** | **72/72** | **66/72 (91.7%)** | shadow | 43747297416 |
| LFM2.5-350M Q4_K_M raw chat | **1.4549** | **72/72** | 29/72 (40.3%) | shadow | 43747297416 |
| Sprocket-500M Q4_K_M raw chat | **2.1740** | **0/72** | N/A (rejected) | shadow | 43747297416 |
| Sprocket-500M Q4_K_M, separately GBNF-constrained | **0.9321** | **72/72** | 21/72 (29.2%) | shadow | 43748299312 |
| SmolLM2-135M-Instruct Q4_K_M raw chat | **0.5131** | **0/72** | N/A (rejected) | shadow | 43748299312 |
| SmolLM2-135M-Instruct Q4_K_M, separately GBNF-constrained | **0.6511** | **72/72** | **11/72 (15.3%)** | shadow | 43748744338 |

The initial Smol repository `tensorblock/SmolLM2-135M-Instruct-GGUF` advertised but no longer contained its Q4 file. That exact acquisition attempt FAILED before inference; failure artifact `14174233477` was retained. The corrected model is `QuantFactory/SmolLM2-135M-Instruct-GGUF`, revision `476854d00ede130660aba430d15f9347ad2e7d0e`, `SmolLM2-135M-Instruct.Q4_K_M.gguf`, SHA-256 `8030f04528538d47bda434f6f0bdf3952c40a58123e4d5e755332f23731a8684`. This corrected model completed all 72 inputs but echoed instructions instead of returning legal candidate IDs. Results were **rejected, not sanitized**.

The exact Smol grammar arm at source `1c16bbb8b88a87240bc5085d71ab216f518fb595` used the SAME corrected QuantFactory Q4 bytes and produced 72/72 legal candidate IDs at 0.6511 seconds median, 11/72 matching Q8. It chose `STOP` on **19/72** fixed states and `RELOCATE_START` on **11/72**. These potentially shorten real episodes considerably; there was no live World evolution in this test. [Original artifact 14175541244](https://github.com/techrote/cybersand/actions/runs/43748744338/artifacts/14175541244).

The initial Sprocket generative runs completed all 72 requests with chatty output that did not satisfy the host's strict literal candidate-ID contract; 0/72 passed, intentionally retained. A second, separately labelled GBNF-only legal-ID experiment on the **same exact model weights** yielded 72/72 syntactically valid choices. The constraint changes the *decoding method* and may affect chosen action distributions: it does not establish reasoning accuracy or art quality. Notably, constrained Sprocket selected `STOP` in 14/72 fixed original states versus zero STOPs for Q4 and LFM; no autonomous episode length may be inferred from this fact.

## Model SHA and runtime pins

- Original Q8: `LiquidAI/d1-2B-GGUF` Q8_0, bytes `1,846,240,576`, SHA `073313d92b53b0ac8193db6e906cb3c78a98183d244b15b7b8f5426115c6738c`.
- Q4: the same LiquidAI GGUF repository/revision's `d1-2B-Q4_K_M.gguf`, bytes `1,099,629,888`, SHA `adca5789ad30ae6027e8e936749bb9f19673097492ab26fa1c5b69643ca50b4b`. Q4/Q8 first-choice agreement 66/72, mean total-variation distance between their System One probability distributions **0.091838**. Their matched System One input format is the strongest quantization comparison.
- LFM: `LiquidAI/LFM2.5-350M-GGUF` revision `657e078c94084481950a2d555a941481f715536b`, `LFM2.5-350M-Q4_K_M.gguf`, 229,312,224 bytes, SHA `7e6f72643caafc9a68256686638c4d7916f2cec76d1df478d4c3ddcd95a6aed4`.
- Sprocket: `kandivault/sprocket-500m` revision `eb157d4ddddc4619e661d4a2426bb2dd000fc668`, `sprocket-500m-q4_k_m.gguf`, 310,279,296 bytes, SHA `fa9456e200200ba13889a6bf93dc55fc2173106b73c3f165833882ef91f091c0`.
- Runtime: `ggml-org/llama.cpp` b11960 revision `8642541b255bfb5dc39cd351a178be09013a55d6`; server binary SHA `005c789a278eff50500a864ee4736e7f4a7edf657acfba13e7a5cf19992927fa`.
- All cases source-link to the same 72-state original digest; original response archives and exact separate source commits remain distinguishable.

## Real-world qualification limits

- Q4 is **3.81× faster** by matched-state median decision latency, and its GGUF file 40% smaller. Surprisingly, observed Q4 resident process RSS was **~2,270 MiB median**, greater than original Q8 **~1,932–1,934 MiB** on their respective runner sessions; this benchmark **does not substantiate a peak RAM saving**. Re-test host RAM and two-resident concurrency before scaling.
- LFM at 1.45 seconds is **9.15× faster** in shadow median vs original Q8, but the chat input is a reduced **different prompt/interface**, unlike matched Q8 versus Q4 System One. Do not treat these speed ratios as architecture-normalised apples-to-apples benchmarks.
- Sprocket with grammar at 0.93 seconds is **14.28× faster** by median; formatting constraints force legal strings, not necessarily useful decisions. Smol raw chat at 0.51 seconds is fast yet completely invalid; Smol with grammar at 0.65 seconds is **20.44× faster** and 72/72 format-valid. Neither result establishes interesting native art or safe multi-turn behaviour; speed alone is not qualification.
- No stored-material native editing by Q4/SLMs, actual two-model co-residency per 4-vCPU runner, 128-turn trial, GPU/paid inference, training, default-provider changes, model acceptance, or human artwork rating was executed. No uncontrolled attempt to repair/referee a legal-but-poor model choice.

**Evidence/CI:** original run [43747297416](https://github.com/techrote/cybersand/actions/runs/43747297416), corrective run [43748299312](https://github.com/techrote/cybersand/actions/runs/43748299312), single-arm Smol grammar request run [43748744338](https://github.com/techrote/cybersand/actions/runs/43748744338) **completed successfully**. All use bounded local CPU runners and exact native Q8 archived states. Draft/HOLD PR #621 targets independent #611 development without touching #611 original execution, D-2 #598, #568 painter-training or frozen M1.
