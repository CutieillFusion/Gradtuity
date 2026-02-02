"""
Optimizer utilities for Gradtuity.

- clip_grad_norm_: Clip gradient norm by scaling grads in place.
"""

from __future__ import annotations

import math
from typing import Iterable

from ..tensor import Tensor


def clip_grad_norm_(
    params: Iterable[Tensor],
    max_norm: float,
    eps: float = 1e-6,
) -> float:
    """
    Clip the gradient norm of the parameters.

    Computes total_norm = sqrt(sum(|g|^2 over all param grads)), then if
    total_norm > max_norm, scales all grads in place by max_norm / (total_norm + eps).
    Parameters with no grad are skipped.

    In DDP: call after sync_grads() so grads are consistent; v1 does not
    allreduce the norm.

    Args:
        params: Iterable of parameters (typically model.parameters()).
        max_norm: Maximum norm; grads are scaled if total_norm exceeds this.
        eps: Small value added to norm for numerical stability.

    Returns:
        Pre-clip total norm (float).
    """
    total_sq = 0.0
    for p in params:
        if p.grad is None:
            continue
        g2 = p.grad.mul(p.grad)
        s = g2.sum()
        total_sq += s.item()
    total_norm = math.sqrt(total_sq)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for p in params:
            if p.grad is None:
                continue
            p.grad.mul_scalar_inplace_(scale)
    return total_norm
