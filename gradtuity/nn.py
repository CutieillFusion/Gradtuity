"""
Neural network modules for Gradtuity.

Provides high-level abstractions for building neural networks:
- Module: Base class for all neural network modules
- Linear: Fully connected layer (Y = X @ W + b)
- Flatten: Reshape to 2D (batch, -1)
- Conv2d: 2D convolution
- MLP: Multi-layer perceptron
"""

from __future__ import annotations

import math

from .functional import randn, zero_grad, zeros
from .tensor import Tensor


class Module:
    """
    Base class for all neural network modules.

    Subclasses should implement __call__ for the forward pass.
    Parameters are automatically collected from Linear layers.
    """

    def parameters(self) -> list[Tensor]:
        """
        Return a list of all trainable parameters in this module.

        Returns:
            List of Tensor objects with requires_grad=True.
        """
        params = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Tensor) and attr.requires_grad:
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self) -> None:
        """
        Zero out gradients for all parameters.

        Should be called before loss.backward() in training loops.
        """
        zero_grad(self.parameters())

    def __call__(self, x: Tensor) -> Tensor:
        """
        Forward pass. Subclasses must implement this.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        raise NotImplementedError("Subclasses must implement __call__")


class Linear(Module):
    """
    Fully connected linear layer: Y = X @ W + b

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.

    Attributes:
        weight: Learnable weight matrix of shape (in_features, out_features).
        bias: Learnable bias vector of shape (out_features,).
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """
        Initialize a Linear layer.

        Weights are initialized using Xavier/Glorot initialization:
        std = sqrt(2 / (in_features + out_features))
        Bias is initialized to zeros.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
        """
        self.in_features = in_features
        self.out_features = out_features

        # Xavier/Glorot initialization for better gradient flow
        std = (2.0 / (in_features + out_features)) ** 0.5
        self.weight = randn((in_features, out_features), requires_grad=True, std=std)

        self.bias = zeros((out_features,), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Forward pass: Y = X @ W + b (fused operation)

        Args:
            x: Input tensor of shape (batch, in_features).

        Returns:
            Output tensor of shape (batch, out_features).
        """
        return x.linear(self.weight, self.bias)

    def parameters(self) -> list[Tensor]:
        """Return weight and bias tensors."""
        return [self.weight, self.bias]

    def __repr__(self) -> str:
        return (
            f"Linear(in_features={self.in_features}, out_features={self.out_features})"
        )


class Flatten(Module):
    """
    Flatten dimensions from start_dim to the end into a single dimension.

    For input (N, C, H, W), output is (N, C*H*W). Used to connect conv layers
    to linear layers.

    Args:
        start_dim: First dimension to flatten (default 1; keep batch dim).
    """

    def __init__(self, start_dim: int = 1) -> None:
        self.start_dim = start_dim

    def __call__(self, x: Tensor) -> Tensor:
        """
        Forward: reshape to (dim_0, ..., dim_start_dim-1, -1).

        Args:
            x: Input tensor of any rank >= 2.

        Returns:
            Output tensor of shape (x.shape[0], ..., prod(rest)).
        """
        s = x.shape
        if len(s) < 2:
            raise ValueError(f"Flatten requires at least 2D input, got shape {s}")
        if self.start_dim < 0:
            start_dim = len(s) + self.start_dim
        else:
            start_dim = self.start_dim
        if start_dim <= 0 or start_dim >= len(s):
            raise ValueError(
                f"Flatten start_dim must be in [1, {len(s) - 1}], got {self.start_dim}"
            )
        kept = s[:start_dim]
        rest = s[start_dim:]
        flat_size = math.prod(rest)
        out_shape = kept + (flat_size,)
        return x.view(out_shape)

    def parameters(self) -> list[Tensor]:
        """Flatten has no parameters."""
        return []

    def __repr__(self) -> str:
        return f"Flatten(start_dim={self.start_dim})"


class Conv2d(Module):
    """
    2D convolution: Y = conv2d(X, weight) + bias.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size (int or tuple; int means square).
        stride: Stride (default 1).
        padding: Padding (default 0).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        if isinstance(kernel_size, int):
            kH = kW = kernel_size
        else:
            kH, kW = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kH, kW)
        self.stride = stride
        self.padding = padding
        # Weight: (out_channels, in_channels, kH, kW). Kaiming-like init
        std = (2.0 / (in_channels * kH * kW)) ** 0.5
        self.weight = randn(
            (out_channels, in_channels, kH, kW),
            requires_grad=True,
            std=std,
        )
        self.bias = zeros((out_channels,), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Forward: conv2d(x, weight, bias, stride, padding).
        """
        return x.conv2d(
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]

    def __repr__(self) -> str:
        kH, kW = self.kernel_size
        return (
            f"Conv2d({self.in_channels}, {self.out_channels}, "
            f"kernel_size=({kH}, {kW}), stride={self.stride}, padding={self.padding})"
        )


class MaxPool2d(Module):
    """
    2D max pooling: (N, C, H, W) -> (N, C, H_out, W_out).

    Args:
        kernel_size: Window size (int or tuple; default 2).
        stride: Stride (defaults to kernel_size if not set).
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int] = 2,
        stride: int | tuple[int, int] | None = None,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, x: Tensor) -> Tensor:
        return x.maxpool2d(
            kernel_size=self.kernel_size,
            stride=self.stride,
        )

    def parameters(self) -> list[Tensor]:
        return []

    def __repr__(self) -> str:
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride})"


class CNN(Module):
    """
    Small CNN for MNIST: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> Linear -> ReLU -> Linear.

    Input (N, 1, 28, 28), output (N, 10).
    """

    def __init__(self) -> None:
        self.conv1 = Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)
        self.flatten = Flatten(start_dim=1)
        self.fc1 = Linear(64 * 7 * 7, 128)
        self.fc2 = Linear(128, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = x.relu()
        x = self.pool1(x)
        x = self.conv2(x)
        x = x.relu()
        x = self.pool2(x)
        x = self.flatten(x)
        x = x.linear_relu(self.fc1.weight, self.fc1.bias)
        x = x.linear(self.fc2.weight, self.fc2.bias)
        return x

    def parameters(self) -> list[Tensor]:
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        return params

    def __repr__(self) -> str:
        return "CNN(conv1->relu->pool1->conv2->relu->pool2->flatten->fc1->relu->fc2)"


class MLP(Module):
    """
    Multi-layer perceptron (fully connected neural network).

    Creates a sequence of Linear layers with ReLU activations.
    The final layer has no activation (linear output).

    Example:
        >>> model = MLP(2, [16, 16, 1])  # 2 inputs -> 16 -> 16 -> 1 output
        >>> print(model)
        MLP of [Linear(2, 16), Linear(16, 16), Linear(16, 1)]
    """

    def __init__(self, nin: int, nouts: list[int]) -> None:
        """
        Initialize an MLP.

        Args:
            nin: Number of input features.
            nouts: List of output sizes for each layer.
                   Example: [16, 16, 1] creates 3 layers.
        """
        self.nin = nin
        self.nouts = nouts

        # Build layers: nin -> nouts[0] -> nouts[1] -> ... -> nouts[-1]
        sizes = [nin] + nouts
        self.layers = [Linear(sizes[i], sizes[i + 1]) for i in range(len(nouts))]

    def __call__(self, x: Tensor) -> Tensor:
        """
        Forward pass through all layers.

        Uses fused linear+relu for hidden layers (more efficient).
        The final layer has no activation (linear output).

        Args:
            x: Input tensor of shape (batch, nin).

        Returns:
            Output tensor of shape (batch, nouts[-1]).
        """
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                # Hidden layer: fused linear + relu (1 kernel instead of 2)
                x = x.linear_relu(layer.weight, layer.bias)
            else:
                # Output layer: linear only (no activation)
                x = x.linear(layer.weight, layer.bias)
        return x

    def parameters(self) -> list[Tensor]:
        """Return all parameters from all layers."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __repr__(self) -> str:
        layer_strs = [
            f"Linear({self.layers[i].in_features}, {self.layers[i].out_features})"
            for i in range(len(self.layers))
        ]
        return f"MLP of [{', '.join(layer_strs)}]"
