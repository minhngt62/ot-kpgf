import numpy as np
import torch


def squared_euclidean(
    x: torch.Tensor, 
    y: torch.Tensor, 
    eps: int = 1e-10
) -> torch.Tensor:
    Cxy = x.pow(2).sum(dim=1).unsqueeze(1) + y.pow(2).sum(dim=1).unsqueeze(0) \
            - 2 * torch.matmul(x, y.t())
    return Cxy / (Cxy.max() + eps)

def softmax(
    x: torch.Tensor
) -> torch.Tensor:
    y = torch.exp(x - torch.max(x))
    fx = y / torch.sum(torch.exp(x - torch.max(x)), axis=-1, keepdims=True)
    return fx

def kl_div(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: int = 1e-10
) -> torch.Tensor:
    return torch.sum(x * torch.log(y + eps) - x * torch.log(y + eps), axis=-1)

def js_div(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: int = 1e-10
) -> torch.Tensor:
    x = torch.unsqueeze(x, dim=1)
    y = torch.unsqueeze(y, dim=0)
    return 0.5 * (kl_div(x, (x + y) / 2, eps) + kl_div(y, (x + y) / 2, eps))