from typing import Dict

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

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

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.proj1(x1)
        x2 = self.proj2(x2)

        d12 = elementwise_dist(x1, x2, p=2)

        return d12


class VectorsPairsDataset(Dataset):
    def __init__(self, x1: Tensor, x2: Tensor):
        assert len(x1) == len(x2)
        self.x1 = x1
        self.x2 = x2

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {"x1": self.x1[idx], "x2": self.x2[idx]}

    def __len__(self) -> int:
        return len(self.x1)


def extract_pairwise(model: IPairwiseModel, x1: Tensor, x2: Tensor, num_workers: int = 0) -> Tensor:
    device = "gpu" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(VectorsPairsDataset(x1, x2), batch_size=512, shuffle=False, num_workers=num_workers)
    model.to(device)
    outputs = []
    for batch in loader:
        output = model(batch["x1"].to(device), batch["x2"].to(device))
        outputs.append(output)
    return torch.stack(outputs)
