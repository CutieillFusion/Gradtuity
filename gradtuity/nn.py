"""
Neural network modules for Gradtuity.

Provides high-level abstractions for building neural networks:
- Module: Base class for all neural network modules
- Linear: Fully connected layer (Y = X @ W + b)
- MLP: Multi-layer perceptron
"""

from __future__ import annotations

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
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})"


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
        self.layers = [
            Linear(sizes[i], sizes[i + 1]) for i in range(len(nouts))
        ]

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
