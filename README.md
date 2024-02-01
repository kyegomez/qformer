[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)


# Qformer
Implementation of Qformer from BLIP2 in Zeta Lego blocks. The implementation is here straight from Figure 2. In particular the image block and text block.

## Install
`pip3 install qformer`


## Usage
```python
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


```


# License
MIT



# Citation
```bibtext
@misc{li2023blip2,
    title={BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models}, 
    author={Junnan Li and Dongxu Li and Silvio Savarese and Steven Hoi},
    year={2023},
    eprint={2301.12597},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```