from torch import Tensor, nn
from zeta.nn import (
    MultiQueryAttention,
    SimpleFeedForward,
)
from zeta.nn.attention.cross_attention import CrossAttention
from qformer.masking import mask_top_right_quadrant


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


class QFormer(nn.Module):
    """
    QFormer is a transformer-based model for processing text and image inputs.

    Args:
        dim (int): The dimension of the model.
        heads (int): The number of attention heads.
        depth (int): The depth of the model.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        text_block_depth (int, optional): The depth of the text block. Defaults to None.
        img_text_block_depth (int, optional): The depth of the image text block. Defaults to None.

    Attributes:
        dim (int): The dimension of the model.
        heads (int): The number of attention heads.
        depth (int): The depth of the model.
        dropout (float): The dropout rate.
        img_block (ImgBlock): The image block of the model.
        text_block (TextBlock): The text block of the model.
        img_layers (nn.ModuleList): The list of image layers.
        text_layers (nn.ModuleList): The list of text layers.

    """

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
        """
        Forward pass of the QFormer model.

        Args:
            x (Tensor): The input tensor.
            img (Tensor): The image tensor.

        Returns:
            Tensor: The output tensor.

        """
        for text_block, img_block in zip(
            self.text_layers, self.img_layers
        ):
            x = text_block(x)
            x = mask_top_right_quadrant(x)
            out = img_block(x, img)
            out = out + x
        return out
