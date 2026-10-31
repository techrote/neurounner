"""NCR-001F: real frozen-decoder post-block hooks for the matched gated MLP.

Four separate source identities; no NCA replacement or shared model checkpoints.
This is an adapter-only, no-training preflight, NOT independent #72 admission.
A caller-provided hash is an identity assertion, NOT proof that bytes were hashed:
the actual-host runner separately verifies the complete downloaded file.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import threading
from typing import Any, Iterator

import torch
from torch import Tensor, nn

from ncr_001f_controls import GatedMLPControl, matched_control_contract
from ncr_001_backbone_lease import shared_decoder_lock


# The SHA values below were independently checked against actual checkpoint
# bytes by the frozen-host #83/#84/#85/#86 and #89/#93/#95 source runs.
# Variant path matters: Sprocket chat and instruct are DIFFERENT checkpoints.
PINNED = {
    "smollm2": {
        "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        "weight_path": "model.safetensors",
        "weight_sha256": "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c",
        "model_type": "llama", "layers": 30, "width": 576, "taps": (9, 19, 29),
    },
    "sprocket_chat": {
        "model_id": "kandivault/sprocket-500m",
        "revision": "eb157d4ddddc4619e661d4a2426bb2dd000fc668",
        "weight_path": "chat/model.safetensors",
        "weight_sha256": "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97",
        "model_type": "llama", "layers": 26, "width": 1280, "taps": (7, 16, 25),
    },
    "sprocket_instruct": {
        "model_id": "kandivault/sprocket-500m",
        "revision": "eb157d4ddddc4619e661d4a2426bb2dd000fc668",
        "weight_path": "model.safetensors",
        "weight_sha256": "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864",
        "model_type": "llama", "layers": 26, "width": 1280, "taps": (7, 16, 25),
    },
    "void1": {
        "model_id": "dotlabs/void.1",
        "revision": "9c2be5087772867d45961abd11f6b35110617563",
        "weight_path": "model.safetensors",
        "weight_sha256": "ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3",
        "model_type": "qwen3", "layers": 24, "width": 640, "taps": (7, 15, 23),
    },
}


class FrozenDepthGatedMLP(nn.Module):
    """Attach the SAME ordinary, noncellular gate at the NCA's three depth taps.

    The control has no field, depth-state handoff or token-axis coupling.
    Hooks exist only during guarded calls, and always detach after failure.
    A model may be owned by only one active instrumented wrapper at a time.
    """

    @classmethod
    def _get_backbone_lock(cls, backbone: nn.Module) -> threading.Lock:
        """A single source-owned decoder-root lease for NCA and MLP."""
        return shared_decoder_lock(backbone)

    def __init__(
        self, backbone: nn.Module, *, lane: str,
        verified_weight_sha256: str, source_revision: str,
    ) -> None:
        super().__init__()
        if lane not in PINNED:
            raise ValueError("Unknown NCR-001 frozen-model lane")
        spec = PINNED[lane]
        if (source_revision != spec["revision"]
                or verified_weight_sha256 != spec["weight_sha256"]):
            raise ValueError("Unqualified model source/variant identity")
        cfg = getattr(backbone, "config", None)
        if (cfg is None
                or getattr(cfg, "model_type", None) != spec["model_type"]
                or getattr(cfg, "hidden_size", None) != spec["width"]
                or getattr(cfg, "num_hidden_layers", None) != spec["layers"]):
            raise ValueError("Backbone configuration does not match frozen model lane")
        if getattr(backbone, "is_gradient_checkpointing", False):
            raise ValueError("Gradient-checkpointed hook order is not qualified")
        root = getattr(backbone, "model", None)
        layers = getattr(root, "layers", None)
        if layers is None or len(layers) != spec["layers"]:
            raise ValueError("Required model.model.layers decoder interface absent")
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.lane = lane
        self.control = GatedMLPControl(spec["width"], tap_count=len(spec["taps"]))
        matched_control_contract(spec["width"], nca_parameters=514 * spec["width"] + 984)
        self._lock = threading.Lock()
        self._backbone_lock = self._get_backbone_lock(backbone)
        self._last_hook_trace: list[dict[str, Any]] = []

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def verify_frozen_base(self) -> None:
        if any(p.requires_grad for p in self.backbone.parameters()):
            raise RuntimeError("Frozen original backbone became trainable")

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        self.verify_frozen_base()
        return iter(self.control.parameters())

    @property
    def last_hook_trace(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._last_hook_trace]

    @contextmanager
    def _claim_backbone(self) -> Iterator[None]:
        """Fail closed across *different wrappers* sharing one frozen model.

        A per-wrapper lock alone permits two adapters to install overlapping
        hooks on the same backbone, contaminating even an 'enabled=False'
        reference call. No model weights, tokenizer or cellular state change.
        """
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent or nested instrumented model use prohibited")
        if not self._backbone_lock.acquire(blocking=False):
            self._lock.release()
            raise RuntimeError("Shared frozen backbone already owned by another adapter")
        try:
            yield
        finally:
            self._backbone_lock.release()
            self._lock.release()

    @contextmanager
    def _instrument(self) -> Iterator[None]:
        with self._claim_backbone():
            handles = []
            traces: list[dict[str, Any]] = []
            active = False
            visited = 0
            spec = PINNED[self.lane]
            self._last_hook_trace = []

            def begin(_root, _args):
                nonlocal active, visited
                if active:
                    raise RuntimeError("Nested decoder forward is prohibited")
                active = True
                visited = 0

            def finish(_root, _args, output):
                nonlocal active, visited
                if not active or visited != len(spec["taps"]):
                    raise RuntimeError("Incomplete/nonsequential gated MLP depth taps")
                traces.append({"tap_count": visited, "lane": self.lane})
                active = False
                visited = 0
                return output

            def make_tap(tap_index: int):
                def tap(_layer, _args, output):
                    nonlocal visited
                    if not active or visited != tap_index:
                        raise RuntimeError("Gated MLP tap invoked outside ordered forward")
                    if isinstance(output, Tensor):
                        new = self.control.step(output, tap_index=tap_index)
                    elif isinstance(output, tuple) and output and isinstance(output[0], Tensor):
                        new = (self.control.step(output[0], tap_index=tap_index), *output[1:])
                    else:
                        raise TypeError("Unsupported frozen decoder block return type")
                    visited += 1
                    return new
                return tap

            try:
                root = self.backbone.model
                handles.append(root.register_forward_pre_hook(begin))
                handles.append(root.register_forward_hook(finish))
                for step, layer_index in enumerate(spec["taps"]):
                    handles.append(root.layers[layer_index].register_forward_hook(make_tap(step)))
                yield
            finally:
                for handle in reversed(handles):
                    handle.remove()
                self._last_hook_trace = traces

    def forward(self, *args, enabled: bool = True, **kwargs):
        self.verify_frozen_base()
        if not enabled:
            with self._claim_backbone():
                return self.backbone(*args, **kwargs)
        with self._instrument():
            return self.backbone(*args, **kwargs)

    def generate(self, *args, enabled: bool = True, **kwargs):
        self.verify_frozen_base()
        if not enabled:
            with self._claim_backbone():
                return self.backbone.generate(*args, **kwargs)
        with self._instrument():
            return self.backbone.generate(*args, **kwargs)

    def _metadata(self) -> dict[str, Any]:
        spec = PINNED[self.lane]
        return {
            "schema_version": 1,
            "adapter_kind": "NCR-001F_conventional_gated_mlp",
            "lane": self.lane,
            "model_id": spec["model_id"],
            "model_revision": spec["revision"],
            "weight_path": spec["weight_path"],
            "verified_weight_sha256": spec["weight_sha256"],
            "model_type": spec["model_type"],
            "hidden_size": spec["width"],
            "layers": spec["layers"],
            "post_block_taps": list(spec["taps"]),
            "control": self.control.config(),
        }

    def save_adapter(self, directory: str | Path) -> dict[str, Any]:
        self.verify_frozen_base()
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        meta = self._metadata()
        torch.save(self.control.state_dict(), path / "adapter.pt")
        (path / "metadata.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return meta

    def load_adapter(self, directory: str | Path) -> None:
        """Validate the entire adapter before copying any live parameters.

        PyTorch strict load_state_dict can copy earlier valid tensors before
        rejecting a malformed later tensor. A rejected import must not change
        the matched control or interleave with another decoder-root user.
        This is an adapter-only operation; no original model is loaded.
        """
        with self._claim_backbone():
            self.verify_frozen_base()
            path = Path(directory)
            observed = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            if observed != self._metadata():
                raise RuntimeError("Gated MLP wrong model variant/source/config metadata")
            state = torch.load(path / "adapter.pt", map_location="cpu", weights_only=True)
            expected = self.control.state_dict()
            if not isinstance(state, dict) or set(state) != set(expected):
                raise RuntimeError("Gated MLP adapter state keys do not match")
            for name, reference in expected.items():
                incoming = state[name]
                if name == "_extra_state":
                    if type(incoming) is not dict or incoming != reference:
                        raise RuntimeError("Gated MLP adapter extra-state mismatch")
                    continue
                if (type(incoming) is not Tensor
                        or incoming.layout != torch.strided
                        or incoming.shape != reference.shape
                        or incoming.dtype != reference.dtype):
                    raise RuntimeError(f"Gated MLP adapter tensor metadata mismatch: {name}")
                if incoming.is_floating_point() and not bool(torch.isfinite(incoming).all()):
                    raise RuntimeError(f"Gated MLP adapter contains nonfinite values: {name}")

            # Preserve values for rollback even if a later copy raises.
            # The existing shared nonblocking lease covers validation, commit
            # and rollback, preventing forward/disabled-baseline interleaving.
            prior_parameters = {
                name: value.detach().clone()
                for name, value in self.control.named_parameters()
            }
            prior_buffers = {
                name: value.detach().clone()
                for name, value in self.control.named_buffers()
            }
            try:
                self.control.load_state_dict(state, strict=True)
            except BaseException:
                with torch.no_grad():
                    for name, value in self.control.named_parameters():
                        value.copy_(prior_parameters[name])
                    for name, value in self.control.named_buffers():
                        value.copy_(prior_buffers[name])
                raise
