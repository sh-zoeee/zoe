import torch

def softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.ones(x.size()[0], dtype=torch.float32).to(x.device)+torch.exp(x))
    