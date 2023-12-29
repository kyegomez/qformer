import torch
from torch import nn, Tensor
from zeta import nn as znn
from zeta.utils import enforce_types


class QFormer(nn.Module):
    @enforce_types
    def __init__(
        self, dim, depth, heads, mlp_dim, dropout=0.1, emb_dropout=0.1
    ):
        pass

    @enforce_types
    def forward(self, x: Tensor) -> Tensor:
        pass
