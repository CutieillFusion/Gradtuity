"""
Deterministic RNG state for dropout (Philox-style, seed + counter).

Used so dropout can regenerate the same mask in backward without storing it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


_default_rng: "DropoutRNG | None" = None


@dataclass
class DropoutRNG:
    """
    Deterministic RNG state for dropout.

    Each dropout call consumes numel random values: offset = rng.counter,
    then rng.counter += numel. Forward/backward kernels use (seed, offset + i)
    so the same mask is regenerated in backward.
    """

    seed: int
    counter: int = 0

    def advance(self, n: int) -> None:
        self.counter += n


def default_rng() -> DropoutRNG:
    """Return the global default DropoutRNG (lazily created)."""
    global _default_rng
    if _default_rng is None:
        seed = int(os.environ.get("GRADTUITY_DROPOUT_SEED", 0))
        _default_rng = DropoutRNG(seed=seed, counter=0)
    return _default_rng


def dropout_rng_state_dict() -> dict[str, int]:
    """Return current dropout RNG state for checkpointing (seed, counter)."""
    rng = default_rng()
    return {"seed": rng.seed, "counter": rng.counter}


def load_dropout_rng_state(state: dict[str, int]) -> None:
    """Restore dropout RNG state from checkpoint."""
    global _default_rng
    _default_rng = DropoutRNG(seed=state["seed"], counter=state["counter"])
