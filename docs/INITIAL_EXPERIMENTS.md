# Initial hosted experiments — 24 October 2026

Source head: `0e0163856493334421b6eda748f63cf728df72b6`  
Actions run: `40092737208`  
Runner class: GitHub-hosted `ubuntu-24.04`, four logical CPUs, ~15.6 GiB RAM.

This run established the first hosted baseline for neurounner. It is a feasibility/throughput observation, not a stable hardware benchmark: GitHub placed different jobs on different AMD EPYC host generations.

## Cache and startup

The prime job started from a cache miss. It downloaded and SHA-256-verified the pinned llama.cpp prebuilt, Sprocket GGUF, and LFM GGUF, then saved cache key:

`ubuntu-neurounner-v1-b11852-edc2ac32-bb741ebb`

The pinned prebuilt removed the roughly four-minute llama.cpp source-build overhead observed in the earlier HEXADECARUNNER Sprocket smoke. On this run the runtime/model downloads and verification completed in roughly ten seconds, and every experiment job subsequently reported cache hits for the extracted runtime and both models.

All resident model servers became ready in roughly 0.50–0.54 seconds.

## Results

| Experiment | Layout on one runner | Completion tokens | Inference wall | Aggregate completion rate |
|---|---|---:|---:|---:|
| `sprocket3` | 3 × Sprocket, 1 thread each, concurrent | 464 | 6.542 s | 70.93 tok/s |
| `lfm1` | 1 × LFM2.5-1.2B, 4 threads, sequential prompts | 246 | 5.232 s | 47.02 tok/s |
| `pair` | Sprocket + LFM, 2 threads each, concurrent | 278 | 4.540 s | 61.24 tok/s |
| `ring` | 3 × Sprocket + LFM, 1 thread each, eight sequential turns | 649 | 21.622 s | 30.02 tok/s |

### sprocket3

Hosted CPU: AMD EPYC 7763.

Three independent Sprocket servers loaded the same 294.8 MiB Q4_K_M model. Each reported about 620 MiB RSS after load.

Per-process observed completion rates:

- Sprocket A: 144 tokens / 6.404 s = 22.49 tok/s
- Sprocket B: 160 tokens / 5.306 s = 30.15 tok/s
- Sprocket C: 160 tokens / 6.538 s = 24.47 tok/s

The three requests overlapped, yielding 464 completion tokens in 6.542 seconds overall. This confirms that three separate Sprocket inference processes are practical on one standard runner.

The raw concept quality remained erratic. One response wandered into a recursive explanation of physics clocks, another repeated readability constraints, and another became confused about building a sandbox character. This is consistent with using Sprocket as a mutation/noise source rather than an authoritative designer.

### lfm1

Hosted CPU: AMD EPYC 9V45.

The LFM2.5-1.2B QAD-Q4_0 server reported about 1.31 GiB RSS after load. Three sequential concept prompts produced:

- 104 tokens / 2.126 s = 48.91 tok/s
- 65 tokens / 1.443 s = 45.06 tok/s
- 77 tokens / 1.662 s = 46.32 tok/s

The outputs were substantially more coherent than Sprocket but often generic. The model followed the requested idea shape reasonably well, mentioning persistent state, hysteresis/delayed response, and simple deterministic state-machine mechanisms.

### pair

Hosted CPU: AMD EPYC 9V74.

Sprocket and LFM were resident together and generated simultaneously with two inference threads each:

- Sprocket: 162 tokens / 3.746 s = 43.25 tok/s
- LFM: 116 tokens / 4.537 s = 25.57 tok/s
- Combined: 278 tokens / 4.540 s = 61.24 tok/s

Reported post-load RSS was ~610 MiB for Sprocket and ~1.31 GiB for LFM.

This is an important result: mixed-model concurrent inference is not merely possible within the runner memory budget; it retains useful aggregate throughput.

### ring

Hosted CPU: AMD EPYC 9V74.

Four local servers remained resident:

- Sprocket A, B, C: one thread each, roughly 620 MiB RSS each
- LFM synthesizer: one thread, roughly 1.31 GiB RSS

The summed process RSS was about 3.1 GiB. This overstates unique physical memory where file-backed pages are shared, but even the pessimistic sum is far below the runner's available memory.

The ring ran:

`Sprocket A → Sprocket B → Sprocket C → LFM → Sprocket A → Sprocket B → Sprocket C → LFM`

It generated 649 completion tokens in 21.622 seconds. Sprocket's one-thread turns ranged from 35.3 to 49.2 tok/s. LFM's one-thread synthesizer turns were about 9.4–10.0 tok/s.

The conversation did show primitive concept propagation. The initial "last push direction" idea drifted through a push-back mechanism, then a target/trigger zone, then a spring-loaded "button" reflex, and finally an incremental-push behaviour. The result was not yet a useful kernel: Sprocket repeatedly fell into repetition/meta-language and LFM tended to sanitize the idea into generic prose.

All six Sprocket turns in the ring hit their 96-token length cap, while both LFM turns stopped naturally at 36–37 tokens. That strongly suggests future Sprocket mutation turns should be much shorter, probably 32–64 tokens, with a stricter machine-readable target.

## Immediate implications

The compute side is already sufficient for the intended investigation:

- runtime compilation is no longer a meaningful overhead;
- both models restore from the same verified cache;
- three Sprockets can generate concurrently on one runner;
- Sprocket and LFM can infer concurrently on one runner;
- three Sprockets plus one LFM can remain resident together with large RAM headroom;
- local loopback orchestration works without external services.

The next bottleneck is therefore **behavioural protocol design**, not runner capacity.

Promising next measurements:

1. Compare three independent Sprocket processes with one Sprocket server using three parallel slots.
2. Sweep mixed-model CPU allocation: Sprocket/LFM thread splits of 1/3, 2/2, and 3/1.
3. Restrict Sprocket mutations to 32, 48, and 64 tokens and measure usable-idea yield.
4. Have three Sprockets emit deliberately divergent mutations, then give all three simultaneously to LFM for selection/normalization rather than passing prose serially around the ring.
5. Replace free prose with a tiny Behaviour Kernel schema so deterministic validation can reject repetition and malformed mechanisms before LFM spends inference on them.

## Evidence

Run artifacts (seven-day Actions retention at creation time):

- `neurounner-sprocket3-a1` — artifact `12560060054`, SHA-256 `7329ebaf9396fb06e600e2025718c3c3be802f1246f3afb6b56ff280536f2a16`
- `neurounner-lfm1-a1` — artifact `12559299174`, SHA-256 `70a8d15219e5814ed3876e3b4d8c717b656853c0b813cdf5ca5ed78fa4f9dbf8`
- `neurounner-pair-a1` — artifact `12559392334`, SHA-256 `a463ab601c658280ddc7d870235ec80b256f5f857f91f34cdddfbb84e43d1410`
- `neurounner-ring-a1` — artifact `12558694306`, SHA-256 `ebd437fb0de568ebacfc0a86fb86593f61de4bfca640f6823671f2221c946cfe`
