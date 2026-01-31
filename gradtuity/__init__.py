"""Gradtuity: From-scratch tensor autodiff engine with Triton kernels."""

from .tensor import Tensor
from .functional import (
    zeros,
    zeros_like,
    ones,
    ones_like,
    randn,
    full,
    full_like,
    zero_grad,
    sgd_step,
)
from .nn import CNN, Conv2d, Flatten, Linear, MaxPool2d, MLP, Module

__all__ = [
    "Tensor",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "randn",
    "full",
    "full_like",
    "zero_grad",
    "sgd_step",
    "Module",
    "Linear",
    "Flatten",
    "Conv2d",
    "MaxPool2d",
    "CNN",
    "MLP",
]
