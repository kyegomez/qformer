import torch
from qformer.model import QFormer


def test_qformer_init():
    model = QFormer(
        dim=512,
        heads=8,
        depth=6,
        dropout=0.1,
        text_block_depth=2,
        img_text_block_depth=2,
    )
    assert model.dim == 512
    assert model.heads == 8
    assert model.depth == 6
    assert model.dropout == 0.1
    assert model.img_block is not None
    assert model.text_block is not None
    assert len(model.img_layers) == 6
    assert len(model.text_layers) == 6


def test_qformer_forward():
    model = QFormer(
        dim=512,
        heads=8,
        depth=6,
        dropout=0.1,
        text_block_depth=2,
        img_text_block_depth=2,
    )
    x = torch.randn(1, 10, 512)
    img = torch.randn(1, 3, 224, 224)
    out = model(x, img)
    assert out.shape == torch.Size([1, 10, 512])
