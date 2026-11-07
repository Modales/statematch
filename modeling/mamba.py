"""Small attention-free selective state-space model, implemented in portable PyTorch.

This follows the selective SSM recurrence introduced by Gu & Dao (Mamba, 2023).
It intentionally avoids CUDA-only extensions so the same checkpoint runs on a Vultr CPU VM.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    activity_vocab: int
    category_vocab: int = 8
    d_model: int = 48
    d_state: int = 8
    expand: int = 2
    n_layers: int = 2
    n_numeric: int = 3

    @classmethod
    def from_dict(cls, value):
        return cls(**{k: value[k] for k in cls.__dataclass_fields__ if k in value})


class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int, expand: int):
        super().__init__()
        inner = d_model * expand
        self.inner = inner
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, inner * 2, bias=False)
        self.conv = nn.Conv1d(inner, inner, 3, padding=2, groups=inner)
        self.x_proj = nn.Linear(inner, 1 + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(1, inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(inner, 1))
        self.D = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, d_model, bias=False)

    def forward(self, x):
        batch, length, _ = x.shape
        x_res, gate = self.in_proj(x).chunk(2, dim=-1)
        # Causal depthwise convolution: trim future positions from the padded kernel.
        x_conv = self.conv(x_res.transpose(1, 2))[..., :length].transpose(1, 2)
        x_conv = F.silu(x_conv)
        params = self.x_proj(x_conv)
        dt_raw = params[..., :1]
        B = params[..., 1:1 + self.d_state]
        C = params[..., 1 + self.d_state:]
        dt = F.softplus(self.dt_proj(dt_raw)).squeeze(-1)
        A = -torch.exp(self.A_log)
        state = x.new_zeros(batch, self.inner, self.d_state)
        outputs = []
        for step in range(length):
            delta = dt[:, step]
            transition = torch.exp(delta[..., None] * A)
            drive = delta[..., None] * B[:, step, None, :] * x_conv[:, step, :, None]
            state = transition * state + drive
            y = (state * C[:, step, None, :]).sum(dim=-1) + self.D * x_conv[:, step]
            outputs.append(y)
        y = torch.stack(outputs, dim=1)
        return self.out_proj(y * F.silu(gate))


class MambaBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.mixer = SelectiveSSM(config.d_model, config.d_state, config.expand)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class StateSpaceProcessModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.activity = nn.Embedding(config.activity_vocab, config.d_model, padding_idx=0)
        self.category = nn.Embedding(config.category_vocab, config.d_model)
        self.numeric = nn.Linear(config.n_numeric, config.d_model, bias=False)
        self.blocks = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.activity_vocab)

    def forward(self, activities, categories, numeric, mask=None):
        x = self.activity(activities) + self.category(categories).unsqueeze(1) + self.numeric(numeric)
        for block in self.blocks:
            x = block(x)
        hidden = self.norm(x)
        logits = self.head(hidden)
        return {"logits": logits, "hidden": hidden}

    def loss(self, activities, categories, numeric, targets, mask):
        result = self.forward(activities, categories, numeric, mask)
        logits = result["logits"].reshape(-1, self.config.activity_vocab)
        loss = F.cross_entropy(logits, targets.reshape(-1), reduction="none", ignore_index=0)
        loss = loss.reshape(targets.shape)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0), result


class TinyTransformer(nn.Module):
    """Parameter-matched attention baseline used only for evaluation."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.activity = nn.Embedding(config.activity_vocab, config.d_model, padding_idx=0)
        self.category = nn.Embedding(config.category_vocab, config.d_model)
        self.numeric = nn.Linear(config.n_numeric, config.d_model, bias=False)
        self.position = nn.Parameter(torch.randn(1, 257, config.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            config.d_model, 4, config.d_model * config.expand,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.activity_vocab)

    def forward(self, activities, categories, numeric, mask=None):
        length = activities.shape[1]
        x = self.activity(activities) + self.category(categories).unsqueeze(1) + self.numeric(numeric)
        x = x + self.position[:, :length]
        causal = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
        padding = None if mask is None else ~mask.bool()
        hidden = self.norm(self.encoder(x, mask=causal, src_key_padding_mask=padding))
        return {"logits": self.head(hidden), "hidden": hidden}

    def loss(self, activities, categories, numeric, targets, mask):
        result = self.forward(activities, categories, numeric, mask)
        logits = result["logits"].reshape(-1, self.config.activity_vocab)
        loss = F.cross_entropy(logits, targets.reshape(-1), reduction="none", ignore_index=0)
        loss = loss.reshape(targets.shape)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0), result


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
