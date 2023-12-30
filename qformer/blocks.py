from torch import Tensor, nn
from zeta.nn import (
    MultiQueryAttention,
    SimpleFeedForward,
)
from zeta.nn.attention.cross_attention import CrossAttention


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

        # Create a list of layers
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
