"""
GPT-2 small model for Gradtuity: decoder-only transformer with pre-LN blocks,
tied LM head, and optional dropout (embd, attn, resid).

input_ids (B, S) must be Tensor with float32 storage and exact integer values
(e.g. from uint16 token IDs converted to float in Python before Tensor(...)).
"""

from __future__ import annotations

from gradtuity import (
    CausalSelfAttention,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    Module,
    PositionalEmbedding,
    Tensor,
    TiedLMHead,
    checkpoint,
)

# GPT-2 small config (matches HuggingFace openai-community/gpt2)
GPT2_SMALL = {
    "vocab_size": 50257,
    "n_positions": 1024,
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "n_inner": 3072,  # 4 * n_embd
    "embd_pdrop": 0.1,
    "attn_pdrop": 0.1,
    "resid_pdrop": 0.1,
}


class GPT2Block(Module):
    """
    Single GPT-2 decoder block: pre-LN attention + residual, pre-LN MLP + residual.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_inner: int,
        attn_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        self.attn = CausalSelfAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )
        self.ln2 = LayerNorm(n_embd)
        self.mlp_fc1 = Linear(n_embd, n_inner)
        self.mlp_fc2 = Linear(n_inner, n_embd)

    def __call__(self, x: Tensor) -> Tensor:
        # pre-LN attention + residual
        x = x.add(self.attn(self.ln1(x)))
        # pre-LN MLP + residual: fc1 -> gelu -> fc2
        h = self.ln2(x)
        B, S, E = h.shape
        h_flat = h.view((B * S, E))
        h_flat = h_flat.linear(self.mlp_fc1.weight, self.mlp_fc1.bias).gelu()
        h_flat = h_flat.linear(self.mlp_fc2.weight, self.mlp_fc2.bias)
        x = x.add(h_flat.view((B, S, E)))
        return x

    def parameters(self) -> list[Tensor]:
        params = []
        params.extend(self.ln1.parameters())
        params.extend(self.attn.parameters())
        params.extend(self.ln2.parameters())
        params.extend(self.mlp_fc1.parameters())
        params.extend(self.mlp_fc2.parameters())
        return params


class GPT2Small(Module):
    """
    GPT-2 small: wte + wpe + drop + N x block + ln_f + lm_head (tied to wte).

    Forward: input_ids (B, S) float32 integer-valued -> logits (B, S, vocab_size).
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_inner: int | None = None,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if n_inner is None:
            n_inner = 4 * n_embd
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.use_checkpoint = use_checkpoint

        self.wte = Embedding(vocab_size, n_embd)
        self.wpe = PositionalEmbedding(n_positions, n_embd)
        self.drop = Dropout(p=embd_pdrop)
        self.blocks = [
            GPT2Block(n_embd, n_head, n_inner, attn_pdrop, resid_pdrop)
            for _ in range(n_layer)
        ]
        self.ln_f = LayerNorm(n_embd)
        self.lm_head = TiedLMHead(self.wte)

    def __call__(self, input_ids: Tensor) -> Tensor:
        """
        Forward: input_ids (B, S) -> logits (B, S, vocab_size).

        input_ids must be float32 with exact integer values (token IDs).
        """
        B, S = input_ids.shape
        # (B, S, n_embd)
        x = self.wte(input_ids).add(self.wpe(seq_len=S, batch_size=B, start_pos=0))
        x = self.drop(x)
        for block in self.blocks:
            x = checkpoint(block, x) if self.use_checkpoint else block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def parameters(self) -> list[Tensor]:
        params = []
        params.extend(self.wte.parameters())
        params.extend(self.wpe.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.ln_f.parameters())
        return params
