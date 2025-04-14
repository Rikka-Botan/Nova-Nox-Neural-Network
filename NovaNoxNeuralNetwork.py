# N4: Nova Nox Neural Network
# Official implementation
# coding = utf-8
# Copyright 2025 Rikka Botan. All rights reserved
# Licensed under "MIT License"
from torch import nn
import torch
import torch.nn.functional as F


class PDS(nn.Module):
    def __init__(
        self,
        in_dim: int
    ):
        """
        Parametric Delay SiLU
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim))
        self.act = nn.SiLU()
        self.eps = nn.Parameter(torch.randn(1))

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = hidden_states * self.act(self.weight * hidden_states)
        delay_states = torch.roll(hidden_states, shifts=1, dims=1)
        delay_states[:, -1] = 0
        hidden_states = hidden_states + F.sigmoid(self.eps) * delay_states
        return hidden_states


class N4(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
        device: str = None,
        dtype: str = None
    ):
        """
        N4(NNNN): Nova Nox Neural Network
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.act = PDS(in_dim=in_dim)
        if bias:
            self.bias_weight = nn.Parameter(torch.randn(out_dim))
        self.propagation = nn.Parameter(torch.randn(out_dim))
        self.in_dim = in_dim
        self.device = device
        self.dtype = dtype
        self.bias = bias

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        in_device = hidden_states.device
        in_dtype = hidden_states.dtype
        if self.device is not None:
            hidden_states = hidden_states.to(self.device)
        if self.dtype is not None:
            hidden_states = hidden_states.to(self.dtype)
        hidden_states = self.act(hidden_states)
        if self.bias:
            hidden_states = F.linear(
                hidden_states, self.weight, self.bias_weight
            ) / self.in_dim**0.5
        else:
            hidden_states = F.linear(
                hidden_states, self.weight) / self.in_dim**0.5
        hidden_states = (F.tanh(self.propagation)/2+0.5) * hidden_states

        return hidden_states.to(in_device).to(in_dtype)


class N4H(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        scale: int,
        bias: bool = False,
        device: str = None,
        dtype: str = None
    ):
        """
        N4 based Hippocumpus module
        """
        super().__init__()
        self.in_n4 = N4(
            in_dim=in_dim,
            out_dim=scale,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.inter_n4 = N4(
            in_dim=in_dim,
            out_dim=in_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.out_n4 = N4(
            in_dim=in_dim,
            out_dim=out_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.in_dim = in_dim
        self.scale = scale

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        scaler = self.in_n4(hidden_states)
        scaled_states = torch.einsum(
            'bsl, bse -> ble', scaler, hidden_states
        ) / (self.in_dim * self.scale)**0.5
        scaled_states = self.inter_n4(scaled_states)
        hidden_states = torch.einsum(
            'ble, bse -> bse', F.tanh(scaled_states), hidden_states
        ) / (self.in_dim * self.scale)**0.5 + hidden_states
        hidden_states = self.out_n4(hidden_states)
        return hidden_states


# ----- Test part -----
model = nn.Sequential(
    N4(500, 1000),
    N4H(1000, 1000, 10),
    N4(1000, 500),
)
inputs = torch.randn(1, 1000, 500)
outputs = model(inputs)
print(inputs)
print(outputs)
