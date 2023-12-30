import torch


def mask_top_right_quadrant(tensor):
    """
    Masks the top right quadrant of a tensor.

    Args:
        tensor (Tensor): The input tensor.

    Returns:
        Tensor: The masked tensor.
    """
    rows, cols = tensor.shape[-2:]
    mask = torch.ones(rows, cols)
    mask[: rows // 2, cols // 2 :] = 0
    return tensor * mask
