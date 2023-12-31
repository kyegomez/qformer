import torch
from qformer import QFormer

x = torch.randn(
    1, 32, 512
)  # Create a random tensor of shape (1, 32, 512)

img = torch.randn(
    1, 32, 512
)  # Create another random tensor of shape (1, 32, 512)

qformer = QFormer(
    512, 8, 8, 0.1, 2, 2
)  # Create an instance of the QFormer model

y = qformer(
    x, img
)  # Apply the QFormer model to the input tensors x and img

print(y.shape)  # Print the shape of the output tensor y
