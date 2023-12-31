from torch import Tensor, nn

from qformer.blocks import ImgBlock, TextBlock
from qformer.masking import mask_top_right_quadrant


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

    Examples:
        >>> model = QFormer(dim=512, heads=8, depth=6, dropout=0.1, text_block_depth=2, img_text_block_depth=2)
        >>> x = torch.randn(1, 10, 512)
        >>> img = torch.randn(1, 3, 224, 224)
        >>> out = model(x, img)
        >>> out.shape
        torch.Size([1, 10, 512])
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
            x = text_block(x) + x
            x = mask_top_right_quadrant(x)
            out = img_block(x, img) + x
        return out
