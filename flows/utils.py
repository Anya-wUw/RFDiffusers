import torch

def unsqueeze_like(tensor: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    # https://pytorch-forecasting.readthedocs.io/en/latest/api/pytorch_forecasting.utils.unsqueeze_like.html
    n_unsqueezes = like.ndim - tensor.ndim
    if n_unsqueezes < 0:
        raise ValueError(f"tensor.ndim={tensor.ndim} > like.ndim={like.ndim}")
    elif n_unsqueezes == 0:
        return tensor
    else:
        return tensor[(...,) + (None,) * n_unsqueezes]
