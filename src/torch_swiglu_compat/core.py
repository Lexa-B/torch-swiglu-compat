from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["install", "_swiglu_impl", "_SwiGLUModule"]

def _swiglu_impl(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Negative-dim handling to mirror PyTorch behavior.
    if dim < 0:
        dim += input.dim()
    # Dtype check (the PR requires a floating dtype).
    if not input.is_floating_point():
        raise RuntimeError(f"swiglu expected floating dtype for input, got {input.dtype}")
    # Size check: last dimension must be even (since we split into two equal chunks).
    size_on_dim = input.size(dim)
    if size_on_dim % 2 != 0:
        raise RuntimeError(
            f"swiglu input size on dim {dim} should be an even number, got {size_on_dim}"
        )
    # Split into two halves, apply SiLU to the first, then multiply by the second.
    a, b = input.chunk(2, dim=dim)
    return F.silu(a) * b

class _SwiGLUModule(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Call through the functional for parity with upstream registration paths.
        return torch.nn.functional.swiglu(input, dim=self.dim)

def install() -> None:
    """
    Install PR-compatible symbols into torch if missing.

    Adds (only if not already present):
      - torch.nn.functional.swiglu(input, dim=-1)
      - torch.nn.SwiGLU(dim=-1)

    Safe to call multiple times; no-ops if upstream already defines them.
    """
    # Functional
    if not hasattr(torch.nn.functional, "swiglu"):
        setattr(torch.nn.functional, "swiglu", _swiglu_impl)
    # Module
    if not hasattr(torch.nn, "SwiGLU"):
        setattr(torch.nn, "SwiGLU", _SwiGLUModule)
