import torch
from torch import Tensor

from oml.interfaces.models import IPairwiseDistanceModel
from oml.utils.misc_torch import elementwise_dist


class SimpleSiamese(IPairwiseDistanceModel):
    def __init__(self, feat_dim: int, identity_init: bool):
        super(SimpleSiamese, self).__init__()
        self.feat_dim = feat_dim

        self.proj1 = torch.nn.Linear(in_features=feat_dim, out_features=feat_dim, bias=False)
        self.proj2 = torch.nn.Linear(in_features=feat_dim, out_features=feat_dim, bias=False)

        if identity_init:
            self.proj1.load_state_dict({"weight": torch.eye(feat_dim)})
            self.proj2.load_state_dict({"weight": torch.eye(feat_dim)})

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.proj1(x1)
        x2 = self.proj2(x2)
        y = elementwise_dist(x1, x2, p=2)
        return y
