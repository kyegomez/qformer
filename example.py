import torch
from qformer import QFormer

# Create a random tensor of shape (1, 32, 512)
x = torch.randn(1, 32, 512)

# Create a random image tensor of shape (1, 3, 224, 224)
img = torch.randn(1, 3, 224, 224)

# Create an instance of the QFormer model with the following parameters:
# - input_size: 512
# - num_heads: 8
# - num_layers: 8
# - dropout: 0.1
# - num_classes: 2
# - num_patches: 2
qformer = QFormer(512, 8, 8, 0.1, 2, 2)

# Apply the QFormer model to the input tensors x and img
y = qformer(x, img)

# Print the shape of the output tensor y
print(y.shape)
