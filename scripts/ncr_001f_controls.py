"""NCR-001F: ordinary parameter-matched, noncellular residual gated adapter.

Frozen LM weights and per-model tokenization are handled by later integrations.
This module never routes tokens, stores cross-depth state, or touches a model.
"""
from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F
import torch


class GatedMLPControl(nn.Module):
    """Share weights at identical post-block taps, but no spatial/recurrent state.

    Parameters:
        LayerNorm(H): 2H
        up Linear(H,256): 256H + 256
        feature gate: 256
        down Linear(256,H), bias=False: 256H
    Total: 514H + 512. NCR NCA total is 514H + 984, only 472 larger.

    Readout is initially zero, matching NCA's zero-readout training dynamics.
    Feature gate is initially 2*sigmoid(0)=1 and trains per feature.
    """

    schema_version = 1

    def __init__(self, hidden_size: int, *, bottleneck: int = 256,
                 tap_count: int = 3) -> None:
        super().__init__()
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if type(bottleneck) is not int or bottleneck <= 0:
            raise ValueError("bottleneck must be a positive integer")
        if type(tap_count) is not int or tap_count <= 0:
            raise ValueError("tap_count must be a positive integer")
        self.hidden_size = hidden_size
        self.bottleneck = bottleneck
        self.tap_count = tap_count
        self.norm = nn.LayerNorm(hidden_size)
        self.up = nn.Linear(hidden_size, bottleneck)
        self.feature_gate = nn.Parameter(torch.zeros(bottleneck))
        self.down = nn.Linear(bottleneck, hidden_size, bias=False)
        nn.init.zeros_(self.down.weight)

    def config(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "kind": "conventional_feature_gated_mlp_residual",
            "hidden_size": self.hidden_size,
            "bottleneck": self.bottleneck,
            "tap_count": self.tap_count,
            "cross_token_recurrence": False,
            "cross_depth_recurrence": False,
        }

    def get_extra_state(self) -> dict:
        return self.config()

    def set_extra_state(self, state: object) -> None:
        if state != self.config():
            raise RuntimeError("Conventional MLP architecture/variant mismatch")

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def step(self, hidden: Tensor, *, tap_index: int) -> Tensor:
        """A pure per-token map; no state moves between taps.

        A fresh call at any tap sees only that tap's hidden representation.
        """
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError("expected hidden [batch, sequence, hidden_size]")
        if type(tap_index) is not int or not (0 <= tap_index < self.tap_count):
            raise ValueError("invalid tap index")
        z = F.silu(self.up(self.norm(hidden)))
        z = z * (2.0 * torch.sigmoid(self.feature_gate))
        return hidden + self.down(z)

    def forward(self, hidden: Tensor) -> Tensor:
        # Forward is a single tap, not a simulated depth-persistent recurrence.
        return self.step(hidden, tap_index=0)


def matched_control_contract(hidden_size: int, *, nca_parameters: int,
                             limit: float = 0.05) -> dict:
    """Fail closed if conventional adapter no longer matches proposed NCA.

    This compares parameter *counts*, not FLOPs: the NCA's repeated local
    update has extra work and should be benchmarked separately.
    """
    control = GatedMLPControl(hidden_size)
    count = control.trainable_parameters()
    if type(nca_parameters) is not int or nca_parameters <= 0:
        raise ValueError("NCA parameter count must be positive")
    if not 0.0 <= limit <= 0.05:
        raise ValueError("only preregistered <=5% discrepancy is allowed")
    delta = abs(count - nca_parameters) / nca_parameters
    if delta > limit:
        raise RuntimeError(f"noncellular adapter not parameter matched: {delta:.6%}")
    return {
        "hidden_size": hidden_size,
        "nca_trainable_parameters": nca_parameters,
        "mlp_trainable_parameters": count,
        "difference": nca_parameters - count,
        "relative_absolute_difference": delta,
        "within_limit": True,
        "compares_FLOPs": False,
    }
