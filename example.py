import torch
from qformer import QFormer

x = torch.randn(1, 32, 512)
img = torch.randn(1, 32, 512)

qformer = QFormer(512, 8, 8, 0.1, 2, 2)
y = qformer(x, img)
print(y.shape)