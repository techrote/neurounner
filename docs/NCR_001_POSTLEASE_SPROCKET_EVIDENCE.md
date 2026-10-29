# NCR-001 post-lease Sprocket — original frozen-host evidence readback

Original physical GitHub Actions run: [#40681920472](https://github.com/techrote/neurounner/actions/runs/40681920472), run attempt 1, original implementation SHA `2d5c0ed72c3d1c44f78ba58493bfffd8fab26d80` in draft [PR #155](https://github.com/techrote/neurounner/pull/155). Composed decoder-root lease source is [PR #154](https://github.com/techrote/neurounner/pull/154) SHA `e961a454bc6125d41c0b9b3cb6589f6637e15b61`.

The independent *artifact readback* retrieved all three original Actions ZIPs and verified byte-for-byte SHA-256s:

| Original original-host artifact | Exact archived ZIP SHA-256 |
| --- | --- |
| [Chat #12812399997](https://github.com/techrote/neurounner/actions/runs/40681920472/artifacts/12812399997) | `870ab4aa2df927827659c816d8f2d9d31974e5cb8f57a5925aa72a4dd1ecbe7f` |
| [Root Instruct #12814075234](https://github.com/techrote/neurounner/actions/runs/40681920472/artifacts/12814075234) | `f8602daca392f89739637c4c5dbb04eb392c38c19f7ac9bb665b8507eca27daf` |
| [Source/negative tests #12812348933](https://github.com/techrote/neurounner/actions/runs/40681920472/artifacts/12812348933) | `11c9e4a5f009b074243bc462b32e4e830d548b2463891583f3aed4270096228e` |

The [original-receipts.tar.gz.b64](../experiments/ncr-001/controls-evidence/postlease-sprocket-20261028/original-receipts.tar.gz.b64) archive (compressed SHA-256 `edc47d6a3416a57bc7fc4fff2fb8ddc129e2370574258c04c3d57bd4f6c57b24`) retains both lanes' exact producer/consumer JSON and logs, adapter metadata and the original two source-test logs. The [manifest](../experiments/ncr-001/controls-evidence/postlease-sprocket-20261028/manifest.json) binds **all 18 original ZIP members** by SHA-256 and byte length. Four untrained binary `adapter.pt` checkpoint members are represented by verified hashes and preserved in the original Actions ZIPs; their bytes are **not committed** to this branch. GitHub Actions artifacts expire, so if long-term binary retention is required, promote those blobs separately under the model-storage policy. The downloaded original ZIPs and noncommitted job logs were inspected; the original job logs are still retained in Actions and runner identities and package versions appear in the raw producer/consumer receipts.

**Verified actual pinned Sprocket frozen hosts:** Chat `chat/model.safetensors` SHA `551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97`; Instruct root `model.safetensors` SHA `1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864`. Both original jobs and 17/17 offline source/negative contract tests passed.

| Check (maximum absolute difference, 2048-unit context) | NCA | Gated MLP |
| --- | ---: | ---: |
| Chat | `1.811981201171875e-05` | `1.811981201171875e-05` |
| Instruct | `1.7642974853515625e-05` | `1.621246337890625e-05` |

Original tolerance: `5e-4` absolute/relative; original-source zero-init no-op, masked batches, full/incremental cache at 512 and 2048 units, cross-adapter disabled-reference denial, request reset, causal suffix and exact independent-Python-process cold replay all passed. The original unchanged producer log warns that `HF.generate` cannot infer an attention mask when pad and EOS IDs coincide; generation comparisons passed as actually executed, but no broad application-level padded-generation qualification is claimed. Control IDs `6/7/10/11` only pass *raw token transport*. Neither tool result/argument protocol nor memory continuity is accepted.

For reproducible **offline readback** from this branch: `python3 scripts/ncr_001_postlease_evidence_audit.py`. It checks original bytes and eight independently injected negative controls for variant swaps, admission forgeries, invalid cache, cold reload, semantic overclaim, optimizer use and ZIP corruption. This is not an independent final scientific sign-off.

**Admission remains blocked** for #72: frozen backbone unchanged after an optimizer step, nonzero adapter gradients after warmup and supervised finite-loss checks (learning/update) have no qualifying evidence. No optimizer steps or NCA-vs-MLP comparative training occurred; held-out fixtures were not opened; no draft PR was merged. The separate void.1 BYTE-EOS-001 #92 and other frozen-model lanes are not altered. `train_authorized=false` remains mandatory until separate independent scientific review and explicit protocol authorization.
