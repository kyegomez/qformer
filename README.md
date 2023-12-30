[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)


# Qformer
Implementation of Qformer from BLIP2 in Zeta Lego blocks. The implementation is here straight from Figure 2. In particular the image block and text block.

## Install
`pip3 install qformer`


## Usage
```python
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