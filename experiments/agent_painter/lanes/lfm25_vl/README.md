# LFM2.5-VL-450M painter lane

This lane loads the complete official multimodal
[`LiquidAI/LFM2.5-VL-450M`](https://huggingface.co/LiquidAI/LFM2.5-VL-450M/tree/fc6221ca597f3315e4f82fc2df606783267b34ba)
at `fc6221ca597f3315e4f82fc2df606783267b34ba`. The concrete class is
`Lfm2VlForConditionalGeneration`; its Siglip2 vision tower and multimodal
projector remain present, frozen and in evaluation mode. The embedded text
configuration's historical `_name_or_path` is not a download instruction.
There is no substituted text-only checkpoint or remote Python execution.

**Qualification authority:** [issue #568](https://github.com/techrote/cybersand/issues/568).
The implementation and local preparation checks below do not claim a hosted
optimizer result. Real optimizer updates, checkpoint tensor readback and
continuation on a separate runner are recorded in that issue at their actual
source SHA. The profile's status points to that ledger and is deliberately stable
across subsequent resumes.

## Model and processor identity

`profile.json` records all nine root snapshot files, their expected SHA-256s,
the exact dependency versions and a complete language-module inventory. The
weight SHA-256 comes from the official pinned revision's LFS metadata; the
hosted downloader and lane loader must also verify the downloaded weight bytes.
The other eight hashes were calculated from retrieved official file bytes.

The actual `Lfm2VlProcessor`, `TokenizersBackend` and `Lfm2VlImageProcessor`
are loaded from that snapshot. Text input uses the processor with `images=None`,
no padding, no added special tokens and no truncation. The tokenizer has 64,400
entries; the model has a padded 65,536-wide vocabulary. BOS is 1, PAD is 0,
assistant EOS is 7, and the image placeholder is 396. Image placeholders and
image-boundary tokens are refused by this text-training lane.

The original [chat template](https://huggingface.co/LiquidAI/LFM2.5-VL-450M/blob/fc6221ca597f3315e4f82fc2df606783267b34ba/chat_template.jinja)
renders system, user and assistant roles. The lane supplies no tools, images or
thinking fields. The assistant suffix is `<|im_end|>` followed by a newline.
The lane removes only that final template separator, after EOS, so the supervised
sequence ends at true EOS. Every completion byte remains intact. Full target
tokenization and prompt-prefix stability are checked; over-budget examples fail.
`encode_prompt` uses precisely the same prefix as training.

The processor wrapper's `save_pretrained` serializes the actual processor,
reloads it locally, and compares its complete tokenizer backend, special-token
map, chat template and image-processor configuration. It also preserves exact
upstream `LICENSE` and `UPSTREAM_README.md` bytes and writes a derivative notice.
The common checkpoint inventory binds these files with the adapter tensors and
optimizer/RNG state.

## Language-only LoRA and CPU implementation

PEFT receives an anchored full-path expression restricted to
`model.language_model.layers`. The separate 92-path inventory is checked before
injection and after PEFT construction/resume:

| Language layers | Allowed linear modules | Count |
| --- | --- | ---: |
| All 16 layers | `feed_forward.w1`, `w2`, `w3` | 48 |
| Attention layers 2, 5, 8, 10, 12, 14 | `self_attn.q_proj`, `k_proj`, `v_proj`, `out_proj` | 24 |
| Convolution layers 0, 1, 3, 4, 6, 7, 9, 11, 13, 15 | `conv.in_proj`, `out_proj` | 20 |

The expression avoids PEFT's automatic compaction of long target lists. A broad
`q_proj` suffix would also match vision layers and is not admitted. The public
trainability gate requires real CPU FP32 parameters, one default LoRA adapter,
rank 8, alpha 16, dropout 0.05 and bias `none`. It refuses alternative LoRA
variants, rank/alpha patterns, extra saved modules, trainable embeddings, an
adapter outside the inventory, a changed frozen component, or a vision/projector
component left in training mode. The shared trainer checks every frozen parameter
byte before and after optimization; the lane's structural check does not replace
that byte check.

The pinned Transformers implementation accepts the shared
`model(input_ids, attention_mask, labels, use_cache=False).loss` call and PEFT's
`CAUSAL_LM` wrapper. It bypasses image feature extraction when `pixel_values` is
absent. No extra forward-loss hook is necessary. Eager attention is required for
both submodels. Short convolution uses the shipped Torch `F.conv1d` path.
Optional `causal_conv1d` and `kernels` packages are refused, preventing a silent
switch away from the reviewed fallback. The common runner configures the CPU
numerical profile before importing Torch.

Required additions to the shared environment are `torchvision==0.29.2+cpu`
from the PyTorch CPU wheel index and `Pillow==13.0.0`. These versions were tested
with `torch==2.15.0+cpu`, `transformers==5.22.2` and `peft==0.21.2`.
Transformers 5.22.2's LFM image processor imports `torchvision.transforms.v2`
directly, even though this lane supplies no images. The CPU torchvision wheel's
metadata requires Torch 2.14 or newer. No GPU, timm or external kernel package
is required for this path.

| Inventory, from actual class construction on the meta device | Parameters |
| --- | ---: |
| Complete frozen multimodal base | 448,718,848 |
| Frozen language model, including tied vocabulary weights | 354,483,968 |
| Frozen Siglip2 vision model | 85,843,200 |
| Frozen multimodal projector | 8,391,680 |
| Trainable rank-8 LoRA | 2,998,272 |

The official root snapshot occupies 902,249,628 bytes, including the
897,484,568-byte safetensors file. FP32 base tensors alone require 1,794,875,392
bytes, about 1.672 GiB. Adapter tensors add 11,993,088 bytes; two FP32 Adam moment
sets add 23,986,176 bytes. These are storage calculations, not measured peak RSS:
activations, gradients, logits, loading temporaries and library overhead are
additional. Hosted receipts must report actual memory and elapsed time.

## Local preparation evidence and regressions

On 2026-11-13, starting from foundation source
`633920373a3d406a2ede61ce4137c324f76be23b`, all 13 lane tests passed using
official small files, the installed pinned stack plus isolated image dependencies,
and meta-only model construction. No model weight tensor download, model forward,
inference, loss, backward or optimizer step was performed locally.

The positively native-admitted pilot's 12 TRAIN rows passed custody validation
and full framing checks. Their complete sequences are 175–266 tokens, with
95–189 supervised tokens; every prefix matched exactly and every answer decoded
back to its admitted completion with one final EOS. FINAL targets were not read.
Tests additionally cover Unicode, over-budget refusal, image/EOS injection,
processor save/reload, text-model substitution, altered LoRA recipes, accidental
vision trainability, and refusal to treat meta parameters as a public training
pass. Meta-only verification uses a private inventory helper; the public gate
always requires CPU tensors.

The first test pass exposed PEFT target-list compaction and the official
processor's early refusal of image placeholders. The final implementation uses
an anchored expression and performs the lane's explicit text-only token check
before invoking the processor. Those failures were preparation defects, not
optimizer failures or model-quality measurements.

Run the model-free subset with ordinary Python:

```bash
python experiments/agent_painter/lanes/lfm25_vl/test_adapter.py
```

After the hosted runner installs the pinned environment and verifies its model
and native-corpus downloads, run every test:

```bash
LFM25_SNAPSHOT=/path/to/model PAINTER_CORPUS=/path/to/native/corpus \
  python experiments/agent_painter/lanes/lfm25_vl/test_adapter.py
```

Without those environment variables, processor/meta checks are explicitly
reported as skipped. Passing the subset alone is not full lane preparation.
Neither command counts as training. The shared hosted training workflow must
then execute genuine updates and independent-runner checkpoint continuation.

## Source review and licence

Reviewed primary sources are the exact upstream
[config](https://huggingface.co/LiquidAI/LFM2.5-VL-450M/blob/fc6221ca597f3315e4f82fc2df606783267b34ba/config.json),
[processor configuration](https://huggingface.co/LiquidAI/LFM2.5-VL-450M/blob/fc6221ca597f3315e4f82fc2df606783267b34ba/processor_config.json),
[model card](https://huggingface.co/LiquidAI/LFM2.5-VL-450M/blob/fc6221ca597f3315e4f82fc2df606783267b34ba/README.md),
[licence](https://huggingface.co/LiquidAI/LFM2.5-VL-450M/blob/fc6221ca597f3315e4f82fc2df606783267b34ba/LICENSE),
and the installed Transformers 5.22.2/PEFT 0.21.2 implementations. The model-card
metadata is `license=other`, `license_name=lfm1.0`, `license_link=LICENSE`;
the legal licence is **LFM Open License v1.0**. Its commercial-use section has
a US$10 million annual-revenue threshold, and redistribution requires retention
of the licence and attribution plus notice of modifications. This lane preserves
those materials with every adapter checkpoint. It does not relabel the model as
Apache-2.0 or infer unrestricted commercial rights from the public download.

| Reviewed installed Transformers source | SHA-256 |
| --- | --- |
| `models/lfm2_vl/modeling_lfm2_vl.py` | `13ed8a130221b238b6637db11762d2ae8858c88178de5345f0eb9c6676e3e39b` |
| `models/lfm2_vl/processing_lfm2_vl.py` | `8360729e518b0b144977aa255363f615fb00663ec6cff2434c6ac9c61f9b2b16` |
| `models/lfm2_vl/image_processing_lfm2_vl.py` | `c27e5e469b199665d4962b01f905b1e90a23ba1725feb8c701b574e9a23f2c40` |
| `models/lfm2/modeling_lfm2.py` | `ba9bc6952b8ab63805e3eaadc1440b543c0a1e00c31a46b1e647c55bcf397822` |

The shared downloader, custody checks, numerical recipe, optimizer, safe
checkpoint handling, Stop and fresh-runner comparison remain common code. This
lane owns only model loading, processor framing, exact language trainability and
processor preservation. Future image-conditioned training needs a separately
reviewed data/framing change; this text-only optimizer proof does not establish
visual grounding or painter quality.
