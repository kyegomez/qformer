import torch
from torch import nn

def bi_directional_self_attn_mask(img_tokens, text_tokens):
    """
    Creates a bi-directional self-attention mask for image-text matching tasks.
    All image and text tokens can attend to each other.

    Args:
        img_tokens (torch.Tensor): The tensor representing image tokens with shape [B, C, H, W].
        text_tokens (torch.Tensor): The tensor representing text tokens with shape [B, SEQLEN, Dim].

    Returns:
        torch.Tensor: A mask tensor where all elements are zero (allowing full attention).
    """
    batch_size, seq_len, _ = text_tokens.size()
    num_image_tokens = img_tokens.size(2) * img_tokens.size(3)
    total_seq_len = seq_len + num_image_tokens
    mask = torch.zeros((batch_size, total_seq_len, total_seq_len), dtype=text_tokens.dtype, device=text_tokens.device)
    return mask


def mmc_self_attn_mask(img, text, *args):
    total_tokens = img + text
    mask = torch.full(
        (total_tokens, total_tokens, *args), float("-inf")
    )
    mask[:img, :img] = 0
    mask[:img:, :img] = 0
    mask[:img:, img:] = torch.tril(
        torch.zeros((text, text, *args))
    )
    return mask


def uni_modal_self_attn_mask(img, text):
    total = img + text
    mask = torch.full(total, total), float("-inf")
    mask[:img, :img] = 0
    mask[img:, img:] = 0
    return 

x = torch.randn(1, 3, 224, 224)
y = torch.randn(1, 10, 768)

print(bi_directional_self_attn_mask(x, y).shape)
