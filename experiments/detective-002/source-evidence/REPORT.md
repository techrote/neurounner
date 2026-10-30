# DETECTIVE-002A — compact NPC body source qualification

This is source metadata evidence only. No model weight bytes were downloaded, no inference was run and no optimizer step was taken.

| Candidate | Class | Resolved revision | License | Parameters | Weight SHA | Stage-B status |
|---|---|---|---|---:|---|---|
| lfm25-350m | direct_candidate | 9e6c6ccf47cd318696e137d381a7ded8fe4df09f | lfm1.0 | 354,483,968 | 1c9c77a4471a7f590f85240f74ed1fc26df7fbde88c3006724e2f93ca993ea4e | stage_b_candidate |
| lfm25-vl-450m-text-reference | runtime_reference | 1abed04b6fe71314d8c446a1371c03d7c332266d | lfm1.0 | ~450,000,000 nominal | 1093f1331319199bbcacdbd7ecc9aa6e5678db6b55073948d0f508be90c8ab68 | stage_b_reference |
| qwen25-0.5b-instruct | direct_candidate | 7ae557604adf67be50417f59c2c2f167def9a775 | apache-2.0 | 494,032,768 | fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe | stage_b_candidate |
| qwen3-0.6b | near_band_exception | c1899de289a04d12100db370d81485cdf75e47ca | apache-2.0 | 751,632,384 | f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b | stage_b_candidate |
| smollm2-135m-instruct | existing_control | 12fd25f77366fa6b3b4b768ec3050bf629380bac | apache-2.0 | 134,515,008 | 5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c | stage_b_candidate |
| smollm2-360m-instruct | direct_candidate | a10cc1512eabd3dde888204e902eca88bddb4951 | apache-2.0 | 361,821,120 | e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e | stage_b_candidate |
| sprocket-500m-chat | existing_control | eb157d4ddddc4619e661d4a2426bb2dd000fc668 | apache-2.0 | 501,090,560 | 551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97 | stage_b_candidate |
| sprocket-500m-instruct | existing_control | eb157d4ddddc4619e661d4a2426bb2dd000fc668 | apache-2.0 | 501,090,560 | 1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864 | stage_b_candidate |
| zenyx-mcen-delta-mamba-565m | mutation_only | f2605f81891326e145895809e4753ec8a53faba7 | apache-2.0 | ~565,000,000 nominal | n/a | mutation_protocol_only |

## Guards

- source gates passed: **9/9**;
- model weight bytes downloaded: **0**;
- optimizer steps: **0**;
- paid API calls: **0**;
- DETECTIVE-001R1 hidden relevance mapping accessed: **no**;
- population expansion admitted: **no**.

A source-gate failure is retained as evidence and blocks that body from Stage B. Production/game license gates are separate from research source qualification and are never silently cleared here.
