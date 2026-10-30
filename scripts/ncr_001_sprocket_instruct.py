"""NCR-001J: source-bound frozen SprocketInstruct causal LM with NCA depth taps.

The NCA state is scoped to ONE decoder forward via local hook closures.
No hooks or field state remain installed between calls. No model edits.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import threading
from typing import Any, Iterator

import torch
from torch import Tensor, nn

from ncr_001_core import PersistentDepthNCA, DepthFieldState
from ncr_001_backbone_lease import shared_decoder_lock

MODEL_ID = "kandivault/sprocket-500m"
MODEL_REVISION = "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
HIDDEN_SIZE = 1280
LAYER_COUNT = 26
POST_BLOCK_TAPS = (7, 16, 25)
PINNED_ROOT_SAFETENSORS_SHA256 = "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864"


class FrozenSprocketInstructDepthNCA(nn.Module):
    """Attach NCA only after pinned Llama decoder blocks 7, 16 and 25.

    Each forward/generation call owns temporary hooks. The decoder-root prehook
    creates an independent *per-forward* state, so generation's incremental
    forwards cannot accidentally reuse the previous token's NCA state.
    Hooks are installed for the duration of one guarded operation and always
    removed, including on exceptions.
    """

    def __init__(self, backbone: nn.Module, *, verified_weight_sha256: str, taps: tuple[int, ...] = POST_BLOCK_TAPS):
        super().__init__()
        # Both Sprocket variants share model geometry, so config is not identity proof.
        if verified_weight_sha256 != PINNED_ROOT_SAFETENSORS_SHA256:
            raise ValueError("Sprocket root instruct checkpoint hash required, never substitute chat")
        cfg = getattr(backbone, "config", None)
        if cfg is None or getattr(cfg, "model_type", None) != "llama":
            raise ValueError("NCR-001J requires a pinned Llama-family SprocketInstruct base")
        if getattr(cfg, "hidden_size", None) != HIDDEN_SIZE:
            raise ValueError("wrong SprocketInstruct hidden width")
        if getattr(cfg, "num_hidden_layers", None) != LAYER_COUNT:
            raise ValueError("wrong SprocketInstruct decoder layer count")
        if tuple(taps) != POST_BLOCK_TAPS:
            raise ValueError("NCR-001J taps must match frozen protocol [7,16,25]")
        if getattr(backbone, "is_gradient_checkpointing", False):
            raise ValueError("backbone gradient checkpointing not supported by hook conformance")
        decoder = getattr(backbone, "model", None)
        layers = getattr(decoder, "layers", None)
        if layers is None or len(layers) != LAYER_COUNT:
            raise ValueError("SprocketInstruct model.model.layers[0:26] missing")
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        self.nca = PersistentDepthNCA(
            HIDDEN_SIZE, tap_count=len(POST_BLOCK_TAPS),
            grid_size=8, channels=4, steps_per_tap=3,
            observation_gain=0.2, update_gain=0.25,
        )
        self._instrument_lock = threading.Lock()
        self._backbone_lock = shared_decoder_lock(backbone)
        self._last_hook_trace: list[dict[str, Any]] = []

    def train(self, mode: bool = True):
        """Base stays in evaluation mode even during adapter-only training."""
        super().train(mode)
        self.backbone.eval()
        return self

    def frozen_parameters(self) -> int:
        return sum(p.numel() for p in self.backbone.parameters())

    def verify_frozen_base(self) -> None:
        if any(p.requires_grad for p in self.backbone.parameters()):
            raise RuntimeError("SprocketInstruct base parameters unexpectedly trainable")

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        self.verify_frozen_base()
        return iter(self.nca.parameters())

    @property
    def last_hook_trace(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._last_hook_trace]

    @contextmanager
    def _claim_backbone(self) -> Iterator[None]:
        """Refuse concurrent NCA/MLP use of the shared decoder root."""
        if not self._backbone_lock.acquire(blocking=False):
            raise RuntimeError("Shared frozen decoder already owned by another adapter")
        try:
            yield
        finally:
            self._backbone_lock.release()

    @contextmanager
    def _install_hooks(self) -> Iterator[None]:
        if not self._instrument_lock.acquire(blocking=False):
            raise RuntimeError("Concurrent instrumented forwards are prohibited")
        hooks = []
        state: DepthFieldState | None = None
        visited = 0
        traces: list[dict[str, Any]] = []
        is_open = False
        self._last_hook_trace = []

        def begin_forward(_module, _inputs):
            nonlocal state, visited, is_open
            if is_open:
                raise RuntimeError("Nested SprocketInstruct decoder forward prohibited")
            state = None
            visited = 0
            is_open = True

        def end_forward(_module, _inputs, output):
            nonlocal state, visited, is_open
            if visited != len(POST_BLOCK_TAPS):
                raise RuntimeError(
                    f"NCR only observed {visited}/{len(POST_BLOCK_TAPS)} depth taps"
                )
            if state is None or state.next_tap != len(POST_BLOCK_TAPS):
                raise RuntimeError("NCA depth state incomplete")
            traces.append({
                "tap_count": visited,
                "state_shape": list(state.field.shape),
                "finite": bool(torch.isfinite(state.field.detach()).all().item()),
            })
            state = None
            visited = 0
            is_open = False
            return output

        def make_tap_handler(tap_index: int):
            def tapped(_layer, _args, output):
                nonlocal state, visited
                if not is_open or visited != tap_index:
                    raise RuntimeError("NCA tap invoked outside ordered decoder forward")
                if isinstance(output, Tensor):
                    modified, state = self.nca.step(
                        output, state, tap_index=tap_index
                    )
                    result = modified
                elif isinstance(output, tuple) and output and isinstance(output[0], Tensor):
                    modified, state = self.nca.step(
                        output[0], state, tap_index=tap_index
                    )
                    result = (modified, *output[1:])
                else:
                    raise TypeError("Unexpected LlamaDecoderLayer output type")
                visited += 1
                return result
            return tapped

        try:
            root = self.backbone.model
            hooks.append(root.register_forward_pre_hook(begin_forward))
            # Posthook validates the complete ordered tap sequence before model.norm.
            hooks.append(root.register_forward_hook(end_forward))
            for index, block_index in enumerate(POST_BLOCK_TAPS):
                hooks.append(
                    root.layers[block_index].register_forward_hook(
                        make_tap_handler(index)
                    )
                )
            yield
        finally:
            for handle in reversed(hooks):
                handle.remove()
            self._last_hook_trace = traces
            self._instrument_lock.release()

    @contextmanager
    def _instrument(self) -> Iterator[None]:
        """Claim shared decoder ownership even for direct hook-window calls."""
        with self._claim_backbone():
            with self._install_hooks():
                yield

    def forward(self, *args, enabled: bool = True, **kwargs):
        self.verify_frozen_base()
        if not enabled:
            with self._claim_backbone():
                return self.backbone(*args, **kwargs)
        with self._instrument():
            return self.backbone(*args, **kwargs)

    def generate(self, *args, enabled: bool = True, **kwargs):
        """Delegates to the unmodified, pinned HF GenerationMixin machinery.

        The decoder-root prehook resets the NCA field for EACH prefill/decode
        forward, even though the hooks span the entire generate() invocation.
        """
        self.verify_frozen_base()
        if not enabled:
            with self._claim_backbone():
                return self.backbone.generate(*args, **kwargs)
        with self._instrument():
            return self.backbone.generate(*args, **kwargs)

    def save_adapter(self, destination: str | Path) -> dict[str, Any]:
        """Only ~659k adapter parameters, never the 500M-parameter backbone."""
        self.verify_frozen_base()
        directory = Path(destination)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 1,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_variant": "instruct",
            "checkpoint_sha256": PINNED_ROOT_SAFETENSORS_SHA256,
            "hidden_size": HIDDEN_SIZE,
            "post_block_taps": list(POST_BLOCK_TAPS),
            "adapter": self.nca.config(),
        }
        torch.save(self.nca.state_dict(), directory / "adapter.pt")
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return metadata

    def load_adapter(self, directory: str | Path) -> None:
        directory = Path(directory)
        observed = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_variant": "instruct",
            "checkpoint_sha256": PINNED_ROOT_SAFETENSORS_SHA256,
            "hidden_size": HIDDEN_SIZE,
            "post_block_taps": list(POST_BLOCK_TAPS),
            "adapter": self.nca.config(),
        }
        if observed != expected:
            raise RuntimeError("NCR-001J adapter provenance/config mismatch")
        state = torch.load(directory / "adapter.pt", map_location="cpu", weights_only=True)
        self.nca.load_state_dict(state, strict=True)
