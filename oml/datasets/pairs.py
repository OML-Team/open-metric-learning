from typing import Dict, List, Tuple

from torch import Tensor

from oml.const import INDEX_KEY, PAIR_1ST_KEY, PAIR_2ND_KEY
from oml.interfaces.datasets import IBaseDataset, IPairDataset


class PairDataset(IPairDataset):
    """
    Dataset to iterate over pairs of items.

    """

    def __init__(
        self,
        base_dataset: IBaseDataset,
        pair_ids: List[Tuple[int, int]],
        pair_1st_key: str = PAIR_1ST_KEY,
        pair_2nd_key: str = PAIR_2ND_KEY,
        index_key: str = INDEX_KEY,
    ):
        self.base_dataset = base_dataset
        self.pair_ids = pair_ids

        self.pair_1st_key = pair_1st_key
        self.pair_2nd_key = pair_2nd_key
        self.index_key: str = index_key

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        i1, i2 = self.pair_ids[idx]
        key = self.base_dataset.input_tensors_key
        return {
            self.pair_1st_key: self.base_dataset[i1][key],
            self.pair_2nd_key: self.base_dataset[i2][key],
            self.index_key: idx,
        }

    def __len__(self) -> int:
        return len(self.pair_ids)


__all__ = ["PairDataset"]
