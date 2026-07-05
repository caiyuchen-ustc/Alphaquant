"""AlphaGPT formula generator — a small autoregressive Transformer over the token vocab.

★ bug#2 fix: the original returned a critic `value` and MTPHead `task_probs` that never
entered the loss (dead code). Here we DROP the MTPHead entirely (a plain Linear head), and
keep a critic head whose value IS used as the REINFORCE baseline in engine.py. Nothing is
generated that the loss ignores.

Kept lightweight for CPU: d_model=64, 2 layers, 4 heads, causal self-attention.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Cfg
from .vocab import FORMULA_VOCAB


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class Block(nn.Module):
    def __init__(self, d, nhead, dff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True, dropout=dropout)
        self.n1 = RMSNorm(d)
        self.n2 = RMSNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Linear(dff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.n2(x)))
        return x


class AlphaGPT(nn.Module):
    def __init__(self):
        super().__init__()
        d = Cfg.D_MODEL
        self.vocab_size = FORMULA_VOCAB.size
        self.token_emb = nn.Embedding(self.vocab_size, d)
        self.pos_emb = nn.Parameter(torch.zeros(1, Cfg.MAX_FORMULA_LEN + 1, d))
        self.blocks = nn.ModuleList([Block(d, 4, 2 * d) for _ in range(2)])
        self.ln_f = RMSNorm(d)
        self.head = nn.Linear(d, self.vocab_size)       # policy logits
        self.critic = nn.Linear(d, 1)                   # ★ value baseline (used in loss)

    def forward(self, idx):
        B, T = idx.size()
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x)
        last = x[:, -1, :]
        return self.head(last), self.critic(last).squeeze(-1)   # logits [B,V], value [B]
