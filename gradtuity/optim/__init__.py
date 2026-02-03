"""
Optimizers for Gradtuity (SGD, AdamW).

Common API: params, step(), zero_grad(), state_dict(), load_state_dict(), save(), load().
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import triton

from ..cuda_mem import cuda_memset
from ..kernels.optim_kernels import adamw_step_kernel, sgd_update_kernel
from ..tensor import Tensor
from ..tensor_io import load_safetensors, save_safetensors
from .utils import clip_grad_norm_

if TYPE_CHECKING:
    from typing import Any

ADAMW_STATE_VERSION = 1
SGD_STATE_VERSION = 1
BLOCK = 256


class Optimizer(ABC):
    """
    Base class for optimizers.

    Subclasses implement step(), state_dict(), load_state_dict(), and optionally
    override save() for state tensors. load() is a class method that dispatches by type.
    """

    def __init__(self, params: list[Tensor]) -> None:
        self.params = list(params)

    @abstractmethod
    def step(self) -> None:
        """Perform one update step."""
        ...

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero or clear gradients for all parameters."""
        for p in self.params:
            if p.grad is None:
                continue
            if set_to_none:
                p.grad = None
            else:
                cuda_memset(p.grad.ptr, 0, p.grad.nbytes)

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable state dict (type, version, hyperparams, param_count, param_shapes)."""
        ...

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Validate and restore optimizer state from state."""
        ...

    def save(self, path_prefix: str) -> None:
        """Write state_dict() to path_prefix + '.json'. Subclasses override to add .safetensors."""
        state = self.state_dict()
        json_path = path_prefix + ".json"
        with open(json_path, "w") as f:
            json.dump(state, f, separators=(",", ":"))

    @classmethod
    def load(cls, path_prefix: str, params: list[Tensor]) -> Optimizer:
        """Load optimizer by reading JSON type and dispatching to the correct subclass."""
        json_path = path_prefix + ".json"
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"optimizer state JSON not found: {json_path}")
        with open(json_path) as f:
            state = json.load(f)
        opt_type = state.get("type")
        if opt_type not in _OPTIMIZER_REGISTRY:
            raise ValueError(
                f"unknown optimizer type {opt_type!r}; known: {list(_OPTIMIZER_REGISTRY)}"
            )
        return _OPTIMIZER_REGISTRY[opt_type].load(path_prefix, params)


class AdamW(Optimizer):
    """
    AdamW optimizer: decoupled weight decay, fused per-parameter Triton update.

    State: exp_avg (m) and exp_avg_sq (v) on GPU; global step on CPU.
    Identity is by parameter list order (no parameter groups).
    """

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas[0], betas[1]
        self.eps = eps
        self.weight_decay = weight_decay
        n = len(self.params)
        self.m: list[Tensor | None] = [None] * n
        self.v: list[Tensor | None] = [None] * n
        self.step_count = 0

    def _validate_hyperparams(self) -> None:
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if not (0 <= self.beta1 < 1):
            raise ValueError(f"beta1 must be in [0, 1), got {self.beta1}")
        if not (0 <= self.beta2 < 1):
            raise ValueError(f"beta2 must be in [0, 1), got {self.beta2}")
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")

    def step(self) -> None:
        self._validate_hyperparams()
        self.step_count += 1
        t = self.step_count
        bc1 = 1.0 / (1.0 - self.beta1**t)
        bc2 = 1.0 / (1.0 - self.beta2**t)

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if p.grad.shape != p.shape:
                raise ValueError(
                    f"param[{i}] grad shape {p.grad.shape} does not match param shape {p.shape}"
                )
            if self.m[i] is None:
                self.m[i] = Tensor._zeros_like(p, requires_grad=False)
                self.v[i] = Tensor._zeros_like(p, requires_grad=False)
            grid = (triton.cdiv(p.numel, BLOCK),)
            adamw_step_kernel[grid](
                p.ptr,
                p.grad.ptr,
                self.m[i].ptr,
                self.v[i].ptr,
                n_elements=p.numel,
                lr=self.lr,
                beta1=self.beta1,
                beta2=self.beta2,
                eps=self.eps,
                weight_decay=self.weight_decay,
                bc1=bc1,
                bc2=bc2,
                BLOCK=BLOCK,
            )

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "AdamW",
            "version": ADAMW_STATE_VERSION,
            "hyperparams": {
                "lr": self.lr,
                "betas": [self.beta1, self.beta2],
                "eps": self.eps,
                "weight_decay": self.weight_decay,
            },
            "step": self.step_count,
            "param_count": len(self.params),
            "param_shapes": [list(p.shape) for p in self.params],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("type") != "AdamW":
            raise ValueError(f"state type must be 'AdamW', got {state.get('type')!r}")
        if state.get("version") != ADAMW_STATE_VERSION:
            raise ValueError(
                f"state version must be {ADAMW_STATE_VERSION}, got {state.get('version')}"
            )
        n = state.get("param_count")
        if n is None or n != len(self.params):
            raise ValueError(
                f"state param_count {n} does not match len(params) {len(self.params)}"
            )
        shapes = state.get("param_shapes")
        if shapes is None or len(shapes) != n:
            raise ValueError("state param_shapes missing or length mismatch")
        for i, p in enumerate(self.params):
            if list(p.shape) != shapes[i]:
                raise ValueError(
                    f"param[{i}] shape {list(p.shape)} does not match saved {shapes[i]}"
                )
        self.step_count = state.get("step", 0)
        hp = state.get("hyperparams", {})
        if hp:
            self.lr = hp.get("lr", self.lr)
            betas = hp.get("betas")
            if betas is not None and len(betas) >= 2:
                self.beta1, self.beta2 = betas[0], betas[1]
            self.eps = hp.get("eps", self.eps)
            self.weight_decay = hp.get("weight_decay", self.weight_decay)
        if "m" in state and "v" in state:
            m_list = state["m"]
            v_list = state["v"]
            self.m = [m_list[i] if i < len(m_list) else None for i in range(n)]
            self.v = [v_list[i] if i < len(v_list) else None for i in range(n)]

    def save(self, path_prefix: str) -> None:
        state = self.state_dict()
        json_path = path_prefix + ".json"
        with open(json_path, "w") as f:
            json.dump(state, f, separators=(",", ":"))

        tensor_dict: dict[str, Tensor] = {}
        for i in range(len(self.params)):
            if self.m[i] is not None:
                tensor_dict[f"m/{i}"] = self.m[i]
                tensor_dict[f"v/{i}"] = self.v[i]
        if tensor_dict:
            save_safetensors(
                path_prefix + ".safetensors",
                tensor_dict,
                metadata={"optimizer": "AdamW", "version": str(ADAMW_STATE_VERSION)},
            )

    @staticmethod
    def load(path_prefix: str, params: list[Tensor]) -> AdamW:
        json_path = path_prefix + ".json"
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"optimizer state JSON not found: {json_path}")
        with open(json_path) as f:
            state = json.load(f)
        if state.get("type") != "AdamW":
            raise ValueError(f"state type must be 'AdamW', got {state.get('type')!r}")
        if state.get("version") != ADAMW_STATE_VERSION:
            raise ValueError(
                f"state version must be {ADAMW_STATE_VERSION}, got {state.get('version')}"
            )
        n = state.get("param_count")
        if n is None or n != len(params):
            raise ValueError(
                f"state param_count {n} does not match len(params) {len(params)}"
            )
        shapes = state.get("param_shapes")
        if shapes is None or len(shapes) != n:
            raise ValueError("state param_shapes missing or length mismatch")
        for i, p in enumerate(params):
            if list(p.shape) != shapes[i]:
                raise ValueError(
                    f"param[{i}] shape {list(p.shape)} does not match saved {shapes[i]}"
                )
        hp = state.get("hyperparams", {})
        lr = hp.get("lr", 1e-3)
        betas_list = hp.get("betas", [0.9, 0.999])
        betas = (betas_list[0], betas_list[1]) if len(betas_list) >= 2 else (0.9, 0.999)
        eps = hp.get("eps", 1e-8)
        weight_decay = hp.get("weight_decay", 0.01)
        opt = AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        opt.step_count = state.get("step", 0)

        st_path = path_prefix + ".safetensors"
        if os.path.isfile(st_path):
            loaded = load_safetensors(st_path, requires_grad=False)
            for i in range(n):
                key_m, key_v = f"m/{i}", f"v/{i}"
                opt.m[i] = loaded.get(key_m)
                opt.v[i] = loaded.get(key_v)
        return opt


class SGD(Optimizer):
    """
    SGD optimizer: param -= lr * grad.

    Stateless (no m/v); save/load persist only hyperparams and param metadata (JSON only).
    """

    def __init__(self, params: list[Tensor], lr: float = 1e-3) -> None:
        super().__init__(params)
        self.lr = lr

    def step(self) -> None:
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if p.grad.shape != p.shape:
                raise ValueError(
                    f"param[{i}] grad shape {p.grad.shape} does not match param shape {p.shape}"
                )
            grid = (triton.cdiv(p.numel, BLOCK),)
            sgd_update_kernel[grid](p.ptr, p.grad.ptr, self.lr, p.numel, BLOCK=BLOCK)

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "SGD",
            "version": SGD_STATE_VERSION,
            "hyperparams": {"lr": self.lr},
            "param_count": len(self.params),
            "param_shapes": [list(p.shape) for p in self.params],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("type") != "SGD":
            raise ValueError(f"state type must be 'SGD', got {state.get('type')!r}")
        if state.get("version") != SGD_STATE_VERSION:
            raise ValueError(
                f"state version must be {SGD_STATE_VERSION}, got {state.get('version')}"
            )
        n = state.get("param_count")
        if n is None or n != len(self.params):
            raise ValueError(
                f"state param_count {n} does not match len(params) {len(self.params)}"
            )
        shapes = state.get("param_shapes")
        if shapes is None or len(shapes) != n:
            raise ValueError("state param_shapes missing or length mismatch")
        for i, p in enumerate(self.params):
            if list(p.shape) != shapes[i]:
                raise ValueError(
                    f"param[{i}] shape {list(p.shape)} does not match saved {shapes[i]}"
                )
        hp = state.get("hyperparams", {})
        if hp:
            self.lr = hp.get("lr", self.lr)

    @staticmethod
    def load(path_prefix: str, params: list[Tensor]) -> SGD:
        json_path = path_prefix + ".json"
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"optimizer state JSON not found: {json_path}")
        with open(json_path) as f:
            state = json.load(f)
        if state.get("type") != "SGD":
            raise ValueError(f"state type must be 'SGD', got {state.get('type')!r}")
        if state.get("version") != SGD_STATE_VERSION:
            raise ValueError(
                f"state version must be {SGD_STATE_VERSION}, got {state.get('version')}"
            )
        n = state.get("param_count")
        if n is None or n != len(params):
            raise ValueError(
                f"state param_count {n} does not match len(params) {len(params)}"
            )
        shapes = state.get("param_shapes")
        if shapes is None or len(shapes) != n:
            raise ValueError("state param_shapes missing or length mismatch")
        for i, p in enumerate(params):
            if list(p.shape) != shapes[i]:
                raise ValueError(
                    f"param[{i}] shape {list(p.shape)} does not match saved {shapes[i]}"
                )
        lr = state.get("hyperparams", {}).get("lr", 1e-3)
        return SGD(params, lr=lr)


_OPTIMIZER_REGISTRY: dict[str, type[Optimizer]] = {"AdamW": AdamW, "SGD": SGD}

__all__ = ["AdamW", "Optimizer", "SGD", "clip_grad_norm_"]
