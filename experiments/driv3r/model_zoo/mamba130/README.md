# MZ-001: Mamba-130M frozen-backbone game-policy probe

**Research-only / Draft-HOLD.** The first new family candidate is [state-spaces/mamba-130m-hf](https://huggingface.co/state-spaces/mamba-130m-hf), a 130M-parameter selective state-space language model distinct from existing d1/LFM/Sprocket/Smol lanes. The CPU job resolves the Hugging Face main revision, requires SHA prefix `1e76775` (verified March 6, 2024), then downloads weights/tokenizer from the **exact full SHA**, not the moving main ref. It disables remote custom code.

The first bounded gate runs real pretrained model forwards on a compact 5×5 game percept, freezes the recurrent backbone, and **trains a real 7-way two-layer policy head** on 112 synthetic training examples (8 seeds × 14 steps). Forty-two held-out DEV examples measure generalization, and three unseen worlds are rolled out using the trained head, the identical untrained head, random legal (STOP allowed), random non-STOP (a fairer continuation control) and a heuristic teacher. The checkpoint contains only the trained head and the exact upstream revision. No giant model file is committed.

**This is a game-facing model training experiment, not a new game implementation.** The 11×11 world, obstacles, protected cells and legal action menu are an *external synthetic research abstraction*, not native Godot PAINT or authoritative CyberSand materials. Raw unmasked illegal predictions are reported separately from host-masked legal actions; host masking cannot prove that the model itself learned legality.

## Execute

```bash
python -m unittest -v experiments.driv3r.model_zoo.mamba130.test_world
# For pretrained model: CPU torch 2.5.1; transformers 4.51.3;
# huggingface_hub 0.34.4; pillow 11.3.0; safetensors 0.5.3;
# tokenizers 0.21.4; internet access to pinned Hugging Face revision.
python -m experiments.driv3r.model_zoo.mamba130.run --out mz-evidence
```

## Qualification and failure semantics

1. All five deterministic world-contract tests pass.
2. Actual pretrained 100M+ Mamba forward occurs on the pinned revision, not random hidden vectors; full upstream SHA recorded.
3. Actual optimizer updates to the policy head produce a nonempty checkpoint plus initial/trained DEV accuracy, train losses, CPU timing and RSS.
4. Original seed-labelled trained/untrained/random-nonstop/teacher traces and rendered PNG are retained with JSON, no invented native evidence.
5. If dependency/download/model-forward fails, label the candidate **not onboarded**. Passing source tests alone does not count as model inference or training.

Output from the bounded job is preserved as an immutable GitHub Actions artifact for 14 days, not checked into Git. Source HEAD, upstream SHA and checkpoint SHA-256 are recorded.

Next distinct candidate: 100M-class RWKV-7 [Hakureirm/rwkv7-0.1b-hf](https://huggingface.co/Hakureirm/rwkv7-0.1b-hf), subject to remote model-code review and independently pinned version. Wider transformer comparison: Qwen3-0.6B, only after CPU budget qualification.

**Hard scope:** never edit or import CyberSand gameplay code/assets/physics/rendering/Workbench. Game runtime may be read for research but is not modified, and default providers are unchanged. This branch is **Draft/HOLD**.
