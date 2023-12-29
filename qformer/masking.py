import torch


def multi_modal_causal_self_attention_mask(x):
    """
    Applies a multi-modal causal self-attention mask. This mask allows query tokens to attend to all
    other query tokens and text tokens to attend only to preceding text tokens and all query tokens.

    Args:
    - x (torch.Tensor): the input tensor of shape [batch_size, seqlen, dim]

    Returns:
    - torch.Tensor: the mask tensor of shape [batch_size, seqlen, seqlen] with 0s where attention is allowed
                    and float('-inf') where it is not, suitable for adding to the raw attention scores.
    """
    batch_size, seqlen, _ = x.shape
    # Initialize mask to all ones
    mask = torch.ones((seqlen, seqlen), dtype=torch.float32)
    # Create a causal mask for the text tokens
    causal_mask = torch.tril(
        torch.ones((seqlen // 2, seqlen // 2), dtype=torch.float32)
    )
    mask[-(seqlen // 2) :, -(seqlen // 2) :] = causal_mask
    # Invert the mask so that 0s are where attention is allowed and float('-inf') where it is not
    mask = torch.log(mask)

    # Expand the mask for the batch size
    mask = mask.repeat(batch_size, 1, 1)

    return mask


batch_size = 2
seqlen = 8
dim = 512


# Example to test the function with dummy data
x_dummy = torch.rand(batch_size, seqlen, dim)  # Dummy data
multi_modal_causal_mask = multi_modal_causal_self_attention_mask(
    x_dummy
)
print(multi_modal_causal_mask.shape)
