import torch

from oml.interfaces.models import IPairwiseModel
from oml.utils.misc_torch import elementwise_dist


class SiameseL2(IPairwiseModel):

    def __init__(self, feat_dim: int, init_with_identity: bool):
        super(SiameseL2, self).__init__()
        self.feat_dim = feat_dim

        self.proj1 = torch.nn.Linear(in_features=feat_dim, out_features=feat_dim, bias=False)
        self.proj2 = torch.nn.Linear(in_features=feat_dim, out_features=feat_dim, bias=False)

        if init_with_identity:
            self.proj1.load_state_dict({"weight": torch.eye(feat_dim)})
            self.proj2.load_state_dict({"weight": torch.eye(feat_dim)})

    def forward(self, x1, x2):
        x1 = self.proj1(x1)
        x2 = self.proj2(x2)

        d12 = elementwise_dist(x1, x2, p=2)

        return d12
