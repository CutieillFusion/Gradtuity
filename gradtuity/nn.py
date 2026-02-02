"""
Neural network modules for Gradtuity.

Provides high-level abstractions for building neural networks:
- Module: Base class for all neural network modules
- Linear: Fully connected layer (Y = X @ W + b)
- Flatten: Reshape to 2D (batch, -1)
- Embedding: Lookup table (row-gather) over learnable weight
- CausalSelfAttention: Multi-head causal self-attention
- Conv2d: 2D convolution
- LayerNorm: Layer normalization over last dimension (gamma, beta)
- MLP: Multi-layer perceptron
"""

from __future__ import annotations

import math

from .functional import ones, randn, zero_grad, zeros
from .random import DropoutRNG, default_rng
from .tensor import Tensor


class Module:
    """
    Base class for all neural network modules.

    Subclasses should implement __call__ for the forward pass.
    Parameters are automatically collected from Linear layers.
    """

    def __init__(self) -> None:
        self.training = True

    def train(self, mode: bool = True) -> "Module":
        """
        Set training mode. When True, dropout etc. are active; when False, eval behavior.

        Recurses into child Modules. Returns self for chaining.
        """
        self.training = mode

        def visit(obj: object) -> None:
            for attr_name in dir(obj):
                if attr_name.startswith("_"):
                    continue
                try:
                    attr = getattr(obj, attr_name)
                except AttributeError:
                    continue
                if isinstance(attr, Module):
                    attr.train(mode)
                elif isinstance(attr, list):
                    for item in attr:
                        if isinstance(item, Module):
                            item.train(mode)

        visit(self)
        return self

    def eval(self) -> "Module":
        """Set evaluation mode (training=False). Returns self for chaining."""
        return self.train(False)

    def parameters(self) -> list[Tensor]:
        """
        Return a list of all trainable parameters in this module.

        Order is deterministic (sorted state_dict key order) so that
        distributed training (e.g. sync_grads) sees the same parameter
        list on all ranks.

        Returns:
            List of Tensor objects with requires_grad=True.
        """
        state = self.state_dict()
        return [state[k] for k in sorted(state)]

    def zero_grad(self) -> None:
        """
        Zero out gradients for all parameters.

        Should be called before loss.backward() in training loops.
        """
        zero_grad(self.parameters())

    def state_dict(self) -> dict[str, Tensor]:
        """
        Return a dict mapping parameter names to tensors.

        Keys are dot-separated paths, e.g. "layers.0.weight", "fc1.bias".
        Numeric segments indicate list indices. Same traversal as parameters().
        """
        result: dict[str, Tensor] = {}

        def visit(obj: object, prefix: str) -> None:
            for attr_name in dir(obj):
                if attr_name.startswith("_"):
                    continue
                try:
                    attr = getattr(obj, attr_name)
                except AttributeError:
                    continue
                if isinstance(attr, Tensor) and attr.requires_grad:
                    result[prefix + attr_name] = attr
                elif isinstance(attr, Module):
                    visit(attr, prefix + attr_name + ".")
                elif isinstance(attr, list):
                    for i, item in enumerate(attr):
                        if isinstance(item, Module):
                            visit(item, prefix + attr_name + "." + str(i) + ".")

        visit(self, "")
        return result

    def load_state_dict(self, state: dict[str, Tensor], strict: bool = True) -> None:
        """
        Load parameters from a state dict (e.g. from state_dict() or load_safetensors).

        Replaces each parameter tensor with the corresponding value from state.
        Keys must be dot-separated paths matching state_dict() naming.

        Args:
            state: Dict mapping parameter names to tensors.
            strict: If True, raise if keys in state don't match model or model keys
                    are missing from state. If False, only load keys that exist.
        """
        own_keys = set(self.state_dict().keys())
        state_keys = set(state.keys())
        if strict:
            if state_keys - own_keys:
                raise ValueError(
                    f"load_state_dict: unexpected key(s) in state: {state_keys - own_keys}"
                )
            if own_keys - state_keys:
                raise ValueError(
                    f"load_state_dict: missing key(s) in state: {own_keys - state_keys}"
                )
        for key, tensor in state.items():
            if key not in own_keys:
                continue
            parts = key.split(".")
            current: object = self
            for segment in parts[:-1]:
                try:
                    if segment.isdigit():
                        current = current[int(segment)]  # type: ignore[index]
                    else:
                        current = getattr(current, segment)
                except (AttributeError, IndexError, KeyError) as e:
                    raise ValueError(
                        f"load_state_dict: cannot resolve key {key!r}"
                    ) from e
            last = parts[-1]
            try:
                setattr(current, last, tensor)
            except AttributeError as e:
                raise ValueError(f"load_state_dict: cannot set key {key!r}") from e

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


class Dropout(Module):
    """
    Inverted dropout: y = x * mask / (1-p) in train, y = x in eval.

    Obeys module.train() / module.eval(). Uses deterministic RNG (seed + counter)
    so the same mask is regenerated in backward (no large mask storage).

    Args:
        p: Drop probability (keep with prob 1-p).
        rng: DropoutRNG for deterministic mask; uses default_rng() if None.
    """

    def __init__(self, p: float = 0.1, rng: DropoutRNG | None = None) -> None:
        super().__init__()
        self.p = p
        self.rng = rng if rng is not None else default_rng()

    def __call__(self, x: Tensor) -> Tensor:
        return x.dropout(p=self.p, training=self.training, rng=self.rng)

    def parameters(self) -> list[Tensor]:
        return []

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


class Embedding(Module):
    """
    Lookup table (embedding) layer: out = weight[indices].

    Owns a single parameter weight of shape (num_embeddings, embedding_dim).
    Forward is row-gather: out[n, d] = weight[int(indices[n]), d].

    Args:
        num_embeddings: Size of the dictionary (vocabulary size).
        embedding_dim: Size of each embedding vector.

    Attributes:
        weight: Learnable weight matrix of shape (num_embeddings, embedding_dim).
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        """
        Initialize an Embedding layer.

        Weight is initialized with randn and std=1/sqrt(embedding_dim).

        Args:
            num_embeddings: Number of embeddings (vocabulary size).
            embedding_dim: Dimension of each embedding.
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        std = 1.0 / (embedding_dim**0.5)
        self.weight = randn(
            (num_embeddings, embedding_dim), requires_grad=True, std=std
        )

    def __call__(self, indices: list | tuple | Tensor) -> Tensor:
        """
        Forward pass: row-gather from weight by indices.

        Args:
            indices: Token IDs, 1D (N,) or 2D (B, S). List, tuple, or Tensor.

        Returns:
            Output tensor of shape (N, embedding_dim) or (B, S, embedding_dim).
        """
        return self.weight.embedding(indices)

    def parameters(self) -> list[Tensor]:
        """Return the single weight tensor."""
        return [self.weight]

    def __repr__(self) -> str:
        return f"Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})"


class PositionalEmbedding(Module):
    """
    Positional embedding (GPT-2 style): fixed max positions, learned vectors.

    Wraps an internal Embedding(max_positions, embed_dim). Forward returns
    (batch_size, seq_len, embed_dim) with the same position vector repeated
    across batch rows.

    Args:
        max_positions: Maximum sequence length (number of position indices).
        embed_dim: Dimension of each position vector.
    """

    def __init__(self, max_positions: int, embed_dim: int) -> None:
        self.max_positions = max_positions
        self.embed_dim = embed_dim
        self.embed = Embedding(max_positions, embed_dim)
        self._position_cache: dict[int, list[int]] = {}

    def _positions_1d(self, seq_len: int, start_pos: int) -> list[int]:
        if seq_len not in self._position_cache:
            self._position_cache[seq_len] = list(range(seq_len))
        base = self._position_cache[seq_len]
        return [start_pos + i for i in base]

    def __call__(
        self,
        seq_len: int,
        batch_size: int,
        start_pos: int = 0,
    ) -> Tensor:
        """
        Forward: positional embeddings for (batch_size, seq_len, embed_dim).

        Args:
            seq_len: Sequence length S.
            batch_size: Batch size B.
            start_pos: First position index (for KV-cache / sliding window).

        Returns:
            Tensor of shape (B, S, embed_dim).
        """
        if start_pos + seq_len > self.max_positions:
            raise ValueError(
                f"start_pos ({start_pos}) + seq_len ({seq_len}) > max_positions ({self.max_positions})"
            )
        positions_1d = self._positions_1d(seq_len, start_pos)
        indices = positions_1d * batch_size
        out_flat = self.embed(indices)
        return out_flat.view((batch_size, seq_len, self.embed_dim))

    def parameters(self) -> list[Tensor]:
        return self.embed.parameters()

    def __repr__(self) -> str:
        return f"PositionalEmbedding(max_positions={self.max_positions}, embed_dim={self.embed_dim})"


class TiedLMHead(Module):
    """
    LM head with weight tying: logits = h @ wte.weight.T.

    No separate weight; uses the embedding module's weight. Gradients flow
    into wte.weight so the optimizer sees it only once. state_dict stores
    the weight once (under the embedding).

    Args:
        embedding_module: nn.Embedding whose weight is used (e.g. wte).
    """

    def __init__(self, embedding_module: Embedding) -> None:
        self.embedding_module = embedding_module

    def __call__(self, h: Tensor) -> Tensor:
        """
        Forward: logits = h @ wte.weight.T.

        Args:
            h: Hidden states (B, S, E).

        Returns:
            Logits (B, S, V).
        """
        B, S, E = h.shape
        wte = self.embedding_module
        if E != wte.embedding_dim:
            raise ValueError(
                f"input last dim {E} must equal embedding_dim {wte.embedding_dim}"
            )
        h_flat = h.view((B * S, E))
        logits_flat = h_flat.linear_tied(wte.weight)
        return logits_flat.view((B, S, wte.num_embeddings))

    def parameters(self) -> list[Tensor]:
        return []

    @property
    def weight_ref(self) -> Tensor:
        """Reference to the shared weight (for tests)."""
        return self.embedding_module.weight

    def __repr__(self) -> str:
        return "TiedLMHead(tied to Embedding)"


class CausalSelfAttention(Module):
    """
    Multi-head causal self-attention (decoder-only, GPT-style).

    Input (B, S, E), output (B, S, E). Position i cannot attend to j > i.
    Uses separate Q, K, V projections and a single output projection.
    Optional attention and residual dropout (GPT-2 style).

    Args:
        embed_dim: Model dimension E (must equal input/output last dim).
        num_heads: Number of attention heads (embed_dim must be divisible).
        attn_pdrop: Dropout on attention weights after softmax (default 0.0).
        resid_pdrop: Dropout on output before residual add (default 0.0).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
    ) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError("embed_dim and num_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / (self.head_dim**0.5)
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop

        # Xavier for Q, K, V, and output projections
        std = (2.0 / (embed_dim + embed_dim)) ** 0.5
        self.Wq = randn((embed_dim, embed_dim), requires_grad=True, std=std)
        self.Wk = randn((embed_dim, embed_dim), requires_grad=True, std=std)
        self.Wv = randn((embed_dim, embed_dim), requires_grad=True, std=std)
        self.Wo = randn((embed_dim, embed_dim), requires_grad=True, std=std)
        self.bq = zeros((embed_dim,), requires_grad=True)
        self.bk = zeros((embed_dim,), requires_grad=True)
        self.bv = zeros((embed_dim,), requires_grad=True)
        self.bo = zeros((embed_dim,), requires_grad=True)

        self.attn_dropout = Dropout(p=attn_pdrop) if attn_pdrop > 0 else None
        self.resid_dropout = Dropout(p=resid_pdrop) if resid_pdrop > 0 else None

    def __call__(self, x: Tensor) -> Tensor:
        """
        Forward: causal multi-head self-attention.

        Args:
            x: Input tensor of shape (B, S, E).

        Returns:
            Output tensor of shape (B, S, E).
        """
        B, S, E = x.shape
        H = self.num_heads
        D = self.head_dim
        if E != self.embed_dim:
            raise ValueError(
                f"input last dim {E} must equal embed_dim {self.embed_dim}"
            )

        # Flatten to (B*S, E) for 2D linear
        x_flat = x.view((B * S, E))
        q_flat = x_flat.linear(self.Wq, self.bq)
        k_flat = x_flat.linear(self.Wk, self.bk)
        v_flat = x_flat.linear(self.Wv, self.bv)
        q = q_flat.view((B, S, E)).view((B, S, H, D)).transpose4d_12()
        k = k_flat.view((B, S, E)).view((B, S, H, D)).transpose4d_12()
        v = v_flat.view((B, S, E)).view((B, S, H, D)).transpose4d_12()
        # q, k, v: (B, H, S, D)

        scores = q.bmm(k.transpose4d_last2())
        scores = scores.scale(self.scale)
        scores = scores.apply_causal_mask()
        attn = scores.softmax(dim=-1)
        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn)
        ctx = attn.bmm(v)
        # ctx: (B, H, S, D)
        ctx = ctx.transpose4d_12().view((B, S, E))
        ctx_flat = ctx.view((B * S, E))
        out_flat = ctx_flat.linear(self.Wo, self.bo)
        if self.resid_dropout is not None:
            out_flat = self.resid_dropout(out_flat)
        return out_flat.view((B, S, E))

    def parameters(self) -> list[Tensor]:
        return [
            self.Wq, self.Wk, self.Wv, self.Wo,
            self.bq, self.bk, self.bv, self.bo,
        ]

    def __repr__(self) -> str:
        return (
            f"CausalSelfAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads})"
        )


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


class LayerNorm(Module):
    """
    Layer normalization over the last dimension: y = (x - mean) * rstd * gamma + beta.

    Owns scale (gamma) and shift (beta), both 1D of length normalized_shape.
    Supports 2D, 3D, and 4D inputs; 3D/4D are normalized over the last dimension.

    Args:
        normalized_shape: Size of the last dimension (e.g. d_model).
        eps: Epsilon for variance stability (default 1e-5).

    Attributes:
        gamma: Scale parameter, shape (normalized_shape,), init ones.
        beta: Shift parameter, shape (normalized_shape,), init zeros.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        """
        Initialize LayerNorm.

        Gamma is initialized to ones, beta to zeros.

        Args:
            normalized_shape: Size of the last dimension to normalize.
            eps: Epsilon added to variance (default 1e-5).
        """
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = ones((normalized_shape,), requires_grad=True)
        self.beta = zeros((normalized_shape,), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Forward pass: layer normalize over last dimension.

        Args:
            x: Input tensor of shape (N, H) or (N, ..., H) with H = normalized_shape.

        Returns:
            Output tensor of same shape as x.
        """
        return x.layer_norm(self.gamma, self.beta, eps=self.eps)

    def parameters(self) -> list[Tensor]:
        """Return gamma and beta."""
        return [self.gamma, self.beta]

    def __repr__(self) -> str:
        return f"LayerNorm(normalized_shape={self.normalized_shape}, eps={self.eps})"


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
