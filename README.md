[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)


# Qformer
Implementation of Qformer from BLIP2 in Zeta Lego blocks.

## Install
`pip3 install qformer`


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


### `TextBlock`

```python
import torch
from qformer import TextBlock

x = torch.randn(1, 32, 512)

model = TextBlock(512, 8, 8)
y = model(x)
print(y.shape)

```



### Qformer
```python



```


# License
MIT



