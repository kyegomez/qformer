import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, einsum, nn
from zeta import LayerNorm, default, exists, l2norm
from zeta.nn import (
    MultiQueryAttention,
    SimpleFeedForward,
)
from zeta.utils import enforce_types


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        dropout=0.0,
        norm_context=False,
        cosine_sim=False,
        cosine_sim_scale=16,
    ):
        super().__init__()
        self.cosine_sim = cosine_sim
        self.scale = (
            cosine_sim_scale if cosine_sim else (dim_head**-0.5)
        )
        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.norm_context = (
            LayerNorm(context_dim) if norm_context else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False), LayerNorm(dim)
        )

    def forward(self, x, context, mask=None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (
            self.to_q(x),
            *self.to_kv(context).chunk(2, dim=-1),
        )

        q, k, v = map(
            lambda t: rearrange(
                t, "b n (h d) -> b h n d", h=self.heads
            ),
            (q, k, v),
        )

        # add null key / value for classifier free guidance in prior net

        nk, nv = map(
            lambda t: repeat(t, "d -> b h 1 d", h=self.heads, b=b),
            self.null_kv.unbind(dim=-2),
        )

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        if self.cosine_sim:
            q, k = map(l2norm, (q, k))

        q, k = map(lambda t: t * math.sqrt(self.scale), (q, k))

        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.type(sim.dtype)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class ImgBlock(nn.Module):
    """
    ImgBlock is a module that performs multi-query attention, cross-attention, and feedforward operations on input tensors.

    Args:
        dim (int): The dimension of the input tensors.
        depth (int): The number of times the operations are applied.
        heads (int): The number of attention heads.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
        emb_dropout (float, optional): The embedding dropout probability. Defaults to 0.1.

    Attributes:
        dim (int): The dimension of the input tensors.
        depth (int): The number of times the operations are applied.
        heads (int): The number of attention heads.
        dropout (float): The dropout probability.
        emb_dropout (float): The embedding dropout probability.
        attn (MultiQueryAttention): The multi-query attention module.
        cross_attn (CrossAttention): The cross-attention module.
        feedforward (SimpleFeedForward): The feedforward module.

    Methods:
        forward(x: Tensor, img: Tensor) -> Tensor:
            Performs the forward pass of the ImgBlock module.

    """

    @enforce_types
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        super(ImgBlock, self).__init__(*args, **kwargs)
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dropout = dropout
        self.attn = MultiQueryAttention(dim, heads)
        self.cross_attn = CrossAttention(
            dim=dim,
            heads=heads,
            dropout=dropout,
        )
        self.feedforward = SimpleFeedForward(dim, dim * 4, dropout)

        self.self_attn_layers = nn.ModuleList([])
        self.cross_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])

        # Add the attn, cross attention, simple feedforward layers to the list
        for _ in range(depth):
            # Add the multi query attention layer
            self.self_attn_layers.append(
                MultiQueryAttention(dim, heads)
            )

            # Add the cross attention layer
            self.cross_attn_layers.append(
                CrossAttention(dim=dim, heads=heads, dropout=dropout)
            )

            # Add the simple feedforward layer
            self.ffn_layers.append(
                SimpleFeedForward(dim, dim * 4, dropout)
            )

    @enforce_types
    def forward(self, x: Tensor, img: Tensor) -> Tensor:
        """
        Performs the forward pass of the ImgBlock module.

        Args:
            x (Tensor): The input tensor.
            img (Tensor): The image tensor.

        Returns:
            Tensor: The output tensor after applying multi-query attention, cross-attention, and feedforward operations.

        """
        for self_attn, cross_attn, ffn in zip(
            self.self_attn_layers,
            self.cross_attn_layers,
            self.ffn_layers,
        ):
            x, _, _ = self_attn(x)
            x = cross_attn(x, img)
            x = ffn(x)

        return x


class TextBlock(nn.Module):
    """
    TextBlock module that performs self-attention and feedforward operations.

    Args:
        dim (int): The dimension of the input and output tensors.
        heads (int): The number of attention heads.
        depth (int): The number of layers in the module.
        dropout (float, optional): The dropout probability. Defaults to 0.1.

    Attributes:
        dim (int): The dimension of the input and output tensors.
        heads (int): The number of attention heads.
        depth (int): The number of layers in the module.
        dropout (float): The dropout probability.
        attn (MultiQueryAttention): The self-attention module.
        feedforward (SimpleFeedForward): The feedforward module.
        layers (nn.ModuleList): The list of layers in the module.

    Methods:
        forward(x: Tensor) -> Tensor:
            Performs the forward pass of the TextBlock module.

    """

    def __init__(
        self,
        dim: int,
        heads: int,
        depth: int,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.depth = depth
        self.dropout = dropout

        self.attn = MultiQueryAttention(dim, heads)
        self.feedforward = SimpleFeedForward(dim, dim * 4, dropout)
        self.layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(MultiQueryAttention(dim, heads))

            self.ffn_layers.append(
                SimpleFeedForward(dim, dim * 4, dropout)
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass of the TextBlock module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after self-attention and feedforward operations.

        """
        for attn, ffn in zip(self.layers, self.ffn_layers):
            x, _, _ = attn(x)
            x = ffn(x)
        return x


class QFormer(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        depth: int,
        dropout: float = 0.1,
        text_block_depth: int = None,
        img_text_block_depth: int = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.depth = depth
        self.dropout = dropout
        self.img_block = ImgBlock(dim, depth, heads, dropout)
        self.text_block = TextBlock(dim, heads, depth, dropout)
        self.img_layers = nn.ModuleList([])
        self.text_layers = nn.ModuleList([])

        # Add the img and text layers to the list
        for _ in range(depth):
            self.img_layers.append(
                ImgBlock(dim, img_text_block_depth, heads, dropout)
            )
            self.text_layers.append(
                TextBlock(dim, heads, text_block_depth, dropout)
            )

    def forward(self, x: Tensor, img: Tensor) -> Tensor:
        for text_block, img_block in zip(
            self.text_layers, self.img_layers
        ):
            x = text_block(x)
            out = img_block(x, img)
            out = out + x
        return out


