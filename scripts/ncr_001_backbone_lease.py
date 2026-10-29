"""Shared NCR-001 frozen decoder ownership, no-training/actual-host-neutral.

All NCA and conventional gated-MLP wrappers import this one registry. The lock
is keyed on the *decoder root* (backbone.model), not the outer LM wrapper: two
outer modules aliasing a decoder must not overlap hooks or reference forwards.
Only the owning wrappers participate; direct unwrapped calls and separate
Python processes are NOT protected. This stores no weights, tokens or fields.
"""
from __future__ import annotations

import threading
import weakref
from torch import nn

_registry_guard = threading.Lock()
_decoder_locks: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def shared_decoder_lock(backbone: nn.Module) -> threading.Lock:
    """Return one non-reentrant, nonblocking-use lease per decoder object."""
    decoder = getattr(backbone, "model", None)
    if not isinstance(decoder, nn.Module):
        raise TypeError("NCR frozen backbone needs a decoder nn.Module at .model")
    with _registry_guard:
        lock = _decoder_locks.get(decoder)
        if lock is None:
            lock = threading.Lock()
            _decoder_locks[decoder] = lock
        return lock
