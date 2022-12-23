from typing import Dict

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from oml.const import PAIR_1ST_KEY, PAIR_2ND_KEY
from oml.interfaces.models import IPairwiseDistanceModel


class VectorsPairsDataset(Dataset):
    """
    Dataset to iterate over pairs of embeddings.

    """

    def __init__(self, x1: Tensor, x2: Tensor, pair_1st_key: str = PAIR_1ST_KEY, pair_2nd_key: str = PAIR_2ND_KEY):
        assert x1.shape == x2.shape

        self.pair_1st_key = pair_1st_key
        self.pair_2nd_key = pair_2nd_key

        self.x1 = x1
        self.x2 = x2

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {self.pair_1st_key: self.x1[idx], self.pair_2nd_key: self.x2[idx]}

    def __len__(self) -> int:
        return len(self.x1)


# fmt: off
def pairwise_inference(
        model: IPairwiseDistanceModel,
        x1: Tensor,
        x2: Tensor,
        num_workers: int = 0,
        batch_size: int = 512,
) -> Tensor:
    dataset = VectorsPairsDataset(x1=x1, x2=x2)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    outputs = []
    with torch.no_grad():
        for batch in loader:
            x1 = batch[dataset.pair_1st_key].to(model.device)
            x2 = batch[dataset.pair_2nd_key].to(model.device)
            outputs.append(model(x1=x1, x2=x2))

    return torch.stack(outputs).detach().cpu()


__all__ = ["VectorsPairsDataset", "pairwise_inference"]
