"""
Activation checkpointing: trade compute for memory.

`checkpoint(fn, *inputs)` runs `fn` without saving intermediate activations. During
backward, it re-runs `fn` with autograd to recompute the subgraph, then propagates
gradients into the original input tensors.

Memory win: peak activation memory inside `fn` is paid only when `fn` is recomputed
(during backward), instead of being held from forward through backward.
Compute cost: ~1 extra forward pass through `fn`.
"""

from __future__ import annotations

from typing import Callable

from .tensor import Tensor, accum_grad, ensure_grad


def checkpoint(fn: Callable[..., Tensor], *inputs: Tensor) -> Tensor:
    """
    Run `fn(*inputs)` without saving intermediate activations.

    The returned tensor's backward re-runs `fn` on detached copies of `inputs`
    (with grad-tracking) to rebuild the subgraph, propagates gradients via the
    rebuilt graph, and accumulates them into the original input tensors.

    Constraints:
        - `fn` must produce a single Tensor output.
        - `fn` must be deterministic across the two calls (no randomness, or
          a seeded RNG that produces the same draws). Dropout in `fn` should be
          OFF or use a fixed seed; otherwise grads are wrong.
        - `inputs` are Tensors. Other args (ints, configs) should be captured
          via closure on `fn`.
    """
    # Forward: detach so fn builds NO graph (detached parents have requires_grad=False
    # so _set_graph short-circuits requires_grad propagation).
    detached_fwd = [x.detach() for x in inputs]
    out_no_grad = fn(*detached_fwd)
    # out_no_grad has requires_grad=False (no parent required grad).

    saved_inputs = inputs
    saved_fn = fn

    def _backward(out_grad: Tensor) -> None:
        # Recompute with grad-tracking on fresh detached copies that propagate requires_grad
        recompute_inputs = []
        for x in saved_inputs:
            ri = x.detach()
            ri.requires_grad = x.requires_grad
            recompute_inputs.append(ri)
        recompute_out = saved_fn(*recompute_inputs)

        # Local topo + reverse traversal seeded with out_grad.
        # Mirrors Tensor.backward()'s topo+free pattern.
        visited: set[int] = set()
        topo: list[Tensor] = []

        def build(v: Tensor) -> None:
            if id(v) in visited or not v.requires_grad:
                return
            visited.add(id(v))
            for p in v._parents:
                build(p)
            topo.append(v)

        build(recompute_out)
        recompute_out.grad = out_grad

        for i in range(len(topo) - 1, -1, -1):
            node = topo[i]
            if node._backward is not None and node.grad is not None:
                node._backward(node.grad)
            if node._parents:
                node._parents = ()
                node._backward = None
                node._ctx = None
            topo[i] = None

        # Propagate recomputed-input grads into the real saved_inputs
        for ri, si in zip(recompute_inputs, saved_inputs):
            if ri.grad is not None and si.requires_grad:
                ensure_grad(si)
                accum_grad(si, ri.grad)

    out_no_grad._set_graph(parents=tuple(saved_inputs), backward_fn=_backward)
    return out_no_grad
