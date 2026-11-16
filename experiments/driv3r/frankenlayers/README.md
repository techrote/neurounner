# D1-FRANKENLAYERS-YOLO — GREYHOUND / SCRAMBLED EGGS / BRANCHWITCH

Owner-directed #619 destructive layer surgery. This is deliberately **not** a careful architecture-optimization project.

Start from the immutable d1-2B Q8 donor and the exact pinned llama.cpp b11960 source. Patch **only the LFM3 System One decision graph** so the original 24-layer execution loop follows one of three deliberately destructive flows:

| Brain | Executed source layers | Intended result |
|---|---|---|
| GREYHOUND | 0,1,2,3,4,6,8,10,14,18,20,22,23 | Maximum speed: 13/24 layer executions |
| SCRAMBLED EGGS | 0,1,2,3,4,5,7,13,9,15,11,6,10,8,12,14,20,21,22,23 | Maximum weirdness: shuffled middle stack |
| BRANCHWITCH | 0,1,2,3,4,5,6,7,9,10,11,10,16,17,18,19,20,21,22,23 | Intended personality mutation: middle removal + duplicated global-attention layer 10 |

The decision endpoint reads the entire prompt in one batch with no recurrent cache, making this a relatively clean place to experiment with skip/reorder/duplicate execution. Normal completion/recurrent graphs remain stock. The Q8 GGUF bytes are never edited.

Each cloud arm builds the exact pinned llama.cpp commit with the small graph patch, then runs a same-runner stock flow and one mutant flow against the unchanged source-pinned W0 legal-action host, seed 570, three families, max 16 choices per family. Record original typed System One probabilities, legal actions, trajectory/PNG, warm p50/p95 latency, input-token throughput, executed layer count and failures.

This is **real d1 inference through a mangled graph**, but the resulting W0 images are deterministic Python host projections, not native Workbench PAINT. There is no training, no new native CyberSand code, no provider/default change and no merge/Ready. Failed or bizarre brains are valid research outputs.

Draft/HOLD. Do not silently replace a flow that crashes or emits invalid answers.
