import torch
from torch import nn

from oml.utils.misc import one_hot


class IdealOneHotModel(nn.Module):
    def __init__(self, emb_dim: int, shift: int = 0):
        super(IdealOneHotModel, self).__init__()
        self.emb_dim = emb_dim
        self.shift = shift

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        embeddings = torch.stack([one_hot(label + self.shift, self.emb_dim) for label in labels])
        return embeddings


class IdealClusterEncoder(nn.Module):
    def forward(self, labels: torch.Tensor, need_noise: bool = True) -> torch.Tensor:
        embeddings = labels + need_noise * 0.01 * torch.randn_like(labels, dtype=torch.float)
        embeddings = embeddings.view((len(labels), 1))
        return embeddings
