"""Helpers shared by the entry-point scripts.

Device resolution lives here rather than in one script that the others import
from: "auto" must be turned into a concrete device string before it reaches
``torch.load(map_location=...)`` or ``Tensor.to()``, which reject it, and every
script needs that.
"""

import torch


def resolve_device(spec):
    """Resolve a user-supplied device string ("auto"/None -> cuda|cpu)."""
    if spec in ("auto", None):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return spec


def device_type(device):
    """The device family ("cuda" from "cuda:1"), as autocast/GradScaler want it."""
    return str(device).split(":", 1)[0]


__all__ = ["resolve_device", "device_type"]
