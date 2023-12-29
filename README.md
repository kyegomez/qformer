[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)


# Qformer


## Usage

```python
import torch
from qformer import ImgBlock


# 3d tensor, B x SEQLEN x DIM
x = torch.randn(1, 32, 1024)
image = torch.randn(1, 32, 1024)

attn = ImgBlock(1024, 8, 1024)
out = attn(x, image)
print(out.shape)
```


# License
MIT



