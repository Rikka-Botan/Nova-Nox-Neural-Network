# N4: Nova Nox Neural Network
# Official implementation
# coding = utf-8
# Copyright 2025 Rikka Botan. All rights reserved
# Licensed under "MIT License"

from torch import nn
import torch
import torch.nn.functional as F
from typing import Optional, Any, cast
import math


class BotanDyT(nn.Module):
    def __init__(
        self,
        hidden_size: int
    ):
        """
        ## DyT Normalization for Nova Nox

        hidden_size: int
        """
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size)
        )

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = F.tanh(hidden_states)
        hidden_states = self.weight * hidden_states
        return hidden_states


def ASGG(
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    ## adaptive swish-gelu gating implementation

    x: input tensor

    y: reference tensor

    z: gate tensor
    """
    if z is not None:
        return F.gelu(y) * (F.sigmoid(z) * x)
    else:
        gate, ref, x = x.chunk(3, dim=-1)
        return F.gelu(ref) * (F.sigmoid(gate) * x)


class BotanMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
        mlp_bias: bool = False,
        device: Any | None = None,
        dtype: Any | None = None
    ):
        """
        ## ASGG base MLP
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.up_proj = nn.Linear(
            in_features=hidden_size,
            out_features=intermediate_size*3,
            bias=mlp_bias,
            device=device,
            dtype=dtype
        )
        self.down_proj = nn.Linear(
            in_features=intermediate_size,
            out_features=hidden_size,
            bias=mlp_bias,
            device=device,
            dtype=dtype
        )
        nn.init.kaiming_normal_(self.up_proj.weight, a=math.sqrt(0.5))
        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(0.5))

    def forward(self, x: torch.Tensor):
        x = ASGG(self.up_proj(x))
        x = self.down_proj(x)
        return x


def Cummean(
    x: torch.Tensor,
    dim: int
) -> torch.Tensor:
    """
    ## Cummean function

    x: Tensor(batch, seq_len, embs)
    """
    denom = torch.cumsum(torch.ones_like(x), dim=dim)
    return torch.cumsum(denom * x, dim=dim) / denom**2


# Simple Nova Nox
class N4(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads: int,
        bias: bool = False,
        device: str = None,
        dtype: str = None
    ):
        """
        ## Nova Nox Neural Network implementation
        """
        super().__init__()
        self.qkvh_linear = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size*3+heads,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.out_linear = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.norm = BotanDyT(hidden_size // heads)
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.scale = hidden_size**0.5

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        bsz, seql, embs = hidden_states.size()
        QKV, H = torch.split(
            self.qkvh_linear(hidden_states),
            [
                embs*3,
                self.heads
            ],
            dim=-1
        )
        H = F.tanh(H).transpose(1, 2).unsqueeze(-1)
        QKV = QKV.reshape(bsz, seql, self.heads, -1).transpose(1, 2)
        Q, K, V = torch.chunk(QKV, chunks=3, dim=-1)
        K = Cummean(H * K, dim=2)
        matrix = torch.matmul(Q, K.transpose(2, 3)) / self.scale
        mask = torch.full((seql, seql), float('-inf'), device=hidden_states.device)
        tri_mask = torch.triu(torch.ones((seql, seql), device=hidden_states.device), diagonal=1).bool()
        mask = torch.where(tri_mask, mask, torch.zeros_like(mask))
        matrix = F.softmax(matrix + mask, dim=-1)
        outputs = torch.matmul(matrix, V)
        outputs = self.out_linear(outputs.transpose(1, 2).reshape(bsz, seql, -1))
        return outputs


class N4D(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        inter_size: int,
        heads: int,
        bias: bool = False,
        device: str = None,
        dtype: str = None
    ):
        """
        ## Nova Nox Neural Network Decoder
        """
        super().__init__()
        self.n4 = N4(
            hidden_size=hidden_size,
            heads=heads,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.mlp = BotanMLP(
            hidden_size=hidden_size,
            intermediate_size=inter_size,
            mlp_bias=bias,
            device=device,
            dtype=dtype
        )
        self.pre_mlp_norm = BotanDyT(hidden_size)
        self.post_mlp_norm = BotanDyT(hidden_size)
        self.pre_n4_norm = BotanDyT(hidden_size)
        self.post_n4_norm = BotanDyT(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        res = hidden_states
        hidden_states = self.pre_n4_norm(hidden_states)
        hidden_states = self.n4(hidden_states)
        hidden_states = self.post_n4_norm(hidden_states) + res
        res = hidden_states
        hidden_states = self.pre_mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_norm(hidden_states) + res
        return hidden_states


class N4B(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        inter_size: int,
        heads: int,
        layers: int,
        bias: bool = False,
        device: str = None,
        dtype: str = None
    ):
        """
        ## Base Language Model constructed by N4
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.n4d = nn.ModuleList([
            N4D(
                hidden_size=hidden_size,
                inter_size=inter_size,
                heads= heads,
                bias=bias,
                device=device,
                dtype=dtype
            )
            for _ in range(layers)
        ])
    def forward(
        self,
        inputs: torch.LongTensor
    ) -> torch.Tensor:
        hidden_states = self.embedding(inputs)
        for proj in self.n4d:
            hidden_states = proj(hidden_states)

        return hidden_states


class N4C(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        inter_size: int,
        heads: int,
        layers: int,
        bias: bool = False,
        device: str = None,
        dtype: str = None
    ):
        """
        ## Causal Language Model constructed by N4
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.n4b = N4B(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            inter_size=inter_size,
            heads=heads,
            layers=layers,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.lm_head = nn.Linear(
            in_features=hidden_size,
            out_features=vocab_size,
            bias=bias,
            device=device,
            dtype=dtype
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        hidden_states = self.n4b(input_ids)
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            return logits, loss

        return logits

    def generate_n4c(
        self,
        input_ids: torch.LongTensor = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 10,
        top_p: float = 2.0,
        eos_token_id: int = 2
    ):
        tokens = input_ids

        # Generate
        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = self(tokens)
            logits = out[0, -1]
            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, k=top_k)[0][-1]
                logits[indices_to_remove] = -torch.inf
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -torch.inf
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if next_token.item() == eos_token_id:
                return
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=-1)
            yield cast(int, next_token.item())
