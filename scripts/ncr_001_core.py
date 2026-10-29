"""NCR-001B: explicit depth-persistent, per-token neural cellular field.

Pure PyTorch. No model-specific imports or token-to-token recurrence. The caller
must pass the returned state between transformer *depth* taps of ONE forward.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DepthFieldState:
    """Ephemeral state for one model forward, never cached on the module.

    next_tap makes tap ordering explicit. field has shape
    [batch, sequence, channels, height, width]; sequence positions do not
    communicate through the cellular update. Do not reuse this across calls.
    """

    field: Tensor
    next_tap: int


class PersistentDepthNCA(nn.Module):
    """A small shared local rule with hidden-width-specific projections.

    The zero-initialized readout gives an exact additive identity at creation.
    Since readout gradients are nonzero first, gradients reach the upstream
    cellular rule after the first readout update, not before it.
    """

    schema_version = 1

    def __init__(
        self,
        hidden_size: int,
        *,
        grid_size: int = 8,
        channels: int = 4,
        tap_count: int = 3,
        steps_per_tap: int = 3,
        observation_gain: float = 0.2,
        update_gain: float = 0.25,
    ) -> None:
        super().__init__()
        if any(type(v) is not int or v <= 0 for v in (hidden_size, grid_size, channels)):
            raise ValueError("hidden_size/grid_size/channels must be positive integers")
        if type(tap_count) is not int or tap_count <= 0:
            raise ValueError("tap_count must be a positive integer")
        if type(steps_per_tap) is not int or steps_per_tap <= 0:
            raise ValueError("steps_per_tap must be a positive integer")
        if not (0.0 <= observation_gain <= 1.0 and 0.0 < update_gain <= 1.0):
            raise ValueError("observation_gain/update_gain must be bounded")

        self.hidden_size = hidden_size
        self.grid_size = grid_size
        self.channels = channels
        self.tap_count = tap_count
        self.steps_per_tap = steps_per_tap
        self.observation_gain = float(observation_gain)
        self.update_gain = float(update_gain)
        self.field_size = channels * grid_size * grid_size

        self.input_norm = nn.LayerNorm(hidden_size)
        self.observe = nn.Linear(hidden_size, self.field_size)
        # Every cell sees only a zero-padded 3x3 square; same weights all cells/taps.
        self.perception = nn.Conv2d(channels, channels * 4, 3, padding=1)
        self.rule = nn.Conv2d(channels * 4, channels * 2, 1)
        self.readout = nn.Linear(self.field_size, hidden_size, bias=False)
        nn.init.zeros_(self.readout.weight)

    def config(self) -> dict[str, int | float]:
        return {
            "schema_version": self.schema_version,
            "hidden_size": self.hidden_size,
            "grid_size": self.grid_size,
            "channels": self.channels,
            "tap_count": self.tap_count,
            "steps_per_tap": self.steps_per_tap,
            "observation_gain": self.observation_gain,
            "update_gain": self.update_gain,
        }

    def get_extra_state(self) -> dict[str, int | float]:
        """Persist architecture identity with the state_dict."""
        return self.config()

    def set_extra_state(self, state: object) -> None:
        if state != self.config():
            raise RuntimeError("NCR-001B architecture metadata mismatch")

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _check_field(self, hidden: Tensor, state: DepthFieldState, tap_index: int) -> None:
        if not isinstance(state, DepthFieldState) or state.next_tap != tap_index:
            raise ValueError("wrong depth-tap order or non-NCA state")
        expected = (hidden.shape[0], hidden.shape[1], self.channels,
                    self.grid_size, self.grid_size)
        if state.field.shape != expected:
            raise ValueError(f"field shape {tuple(state.field.shape)} != {expected}")
        if state.field.device != hidden.device or state.field.dtype != hidden.dtype:
            raise ValueError("field device/dtype must match hidden tensor")

    def evolve(self, field: Tensor, *, steps: int | None = None) -> Tensor:
        """Apply the same 3x3 local rule at every cell and iteration.

        Public for locality and boundary-condition tests; does not change module
        state. A 1-step impulse cannot propagate farther than one cell.
        """
        if field.ndim != 5 or tuple(field.shape[2:]) != (
                self.channels, self.grid_size, self.grid_size):
            raise ValueError("expected [B,T,channels,grid_size,grid_size] field")
        iterations = self.steps_per_tap if steps is None else steps
        if type(iterations) is not int or iterations <= 0:
            raise ValueError("steps must be a positive integer")
        b, t = field.shape[:2]
        z = field.reshape(b * t, self.channels, self.grid_size, self.grid_size)
        for _ in range(iterations):
            perceived = F.silu(self.perception(z))
            delta, gate = self.rule(perceived).chunk(2, dim=1)
            z = torch.tanh(z + self.update_gain * torch.sigmoid(gate) * torch.tanh(delta))
        return z.reshape(b, t, self.channels, self.grid_size, self.grid_size)

    def step(
        self,
        hidden: Tensor,
        state: DepthFieldState | None,
        *,
        tap_index: int,
        iterations: int | None = None,
        reset_each_tap: bool = False,
    ) -> tuple[Tensor, DepthFieldState]:
        """Return updated hidden and state; caller threads state to next tap.

        Call first with state=None, tap_index=0, then pass the exact returned
        state for taps 1, 2, ... . Discard state at end of this model forward.
        No field survives across generated tokens or independent requests.
        """
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError("expected hidden [batch,seq,hidden_size]")
        if type(tap_index) is not int or not 0 <= tap_index < self.tap_count:
            raise ValueError("tap index outside configured depth-tap count")
        if state is None:
            if tap_index != 0:
                raise ValueError("depth field must start at first tap")
        else:
            self._check_field(hidden, state, tap_index)
        seed = torch.tanh(self.observe(self.input_norm(hidden)))
        seed = seed.reshape(hidden.shape[0], hidden.shape[1],
                            self.channels, self.grid_size, self.grid_size)
        if state is None or reset_each_tap:
            field = seed
        else:
            field = torch.tanh(state.field + self.observation_gain * seed)
        field = self.evolve(field, steps=iterations)
        delta = self.readout(field.reshape(hidden.shape[0], hidden.shape[1], -1))
        return hidden + delta, DepthFieldState(field=field, next_tap=tap_index + 1)
