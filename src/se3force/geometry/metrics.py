import torch


def relative_error(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (a - b).norm() / b.norm().clamp_min(eps)


def parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
