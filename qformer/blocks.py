from einops import rearrange, reduce
from torch import Tensor, nn
from zeta.nn import (
    MultiQueryAttention,
    SimpleFeedForward,
)
from zeta.nn.attention.cross_attention import CrossAttention


def img_to_text(x: Tensor, seqlen: int, dim: int, norm: bool = True):
    """
    Convert an image tensor to a text tensor.

    Args:
        x (Tensor): Input image tensor of shape (batch_size, channels, height, width).
        seqlen (int): Length of the output text sequence.
        dim (int): Dimension of the intermediate representation.
        norm (bool, optional): Whether to apply layer normalization. Defaults to True.

    Returns:
        Tensor: Output text tensor of shape (batch_size, seqlen, dim).

    Example::
        >>> x = torch.randn(2, 3, 32, 32)
        >>> x = img_to_text(x, 100, 512)
        >>> x.shape
        torch.Size([2, 100, 512])
    """
    b, c, h, w = x.shape

    img = reduce(x, "b c h w -> b c (h w)", "mean")
    img = nn.Linear(h * w, dim)(img)
    img = rearrange(img, "b c d -> b d c")
    img = nn.Linear(c, seqlen)(img)
    img = rearrange(img, "b d c -> b c d")

    if norm:
        img = nn.LayerNorm(dim)(img)

    return img


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
            dim=dim, heads=heads, dropout=dropout, *args, **kwargs
        )

        # Create a list of layers
        self.self_attn_layers = nn.ModuleList([])
        self.cross_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])

        # Add the attn, cross attention, simple feedforward layers to the list
        for _ in range(depth):
            # Add the multi query attention layer
            self.self_attn_layers.append(
                MultiQueryAttention(dim, heads, *args, **kwargs)
            )
            # Add the cross attention layer
            self.cross_attn_layers.append(
                CrossAttention(
                    dim=dim,
                    heads=heads,
                    dropout=dropout,
                    *args,
                    **kwargs,
                )
            )
            # Add the simple feedforward layer
            self.ffn_layers.append(
                SimpleFeedForward(
                    dim, dim * 4, dropout, *args, **kwargs
                )
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
        b_t, s, d = x.shape
        b, c, h, w = img.shape
        img = img_to_text(img, s, d)

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
        self.layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                MultiQueryAttention(dim, heads, *args, **kwargs)
            )

            self.ffn_layers.append(
                SimpleFeedForward(
                    dim, dim * 4, dropout, *args, **kwargs
                )
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
