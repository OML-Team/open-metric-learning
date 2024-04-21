from typing import Dict, List, Tuple, Union

from torch import Tensor

from oml.const import INDEX_KEY, INPUT_TENSORS_KEY_1, INPUT_TENSORS_KEY_2
from oml.interfaces.datasets import IBaseDataset, IPairDataset


class PairDataset(IPairDataset):
    """
    Dataset to iterate over pairs of items of any modality.

    """

    def __init__(
        self,
        base_dataset: IBaseDataset,
        pair_ids: List[Tuple[int, int]],
        input_tensors_key_1: str = INPUT_TENSORS_KEY_1,
        input_tensors_key_2: str = INPUT_TENSORS_KEY_2,
        index_key: str = INDEX_KEY,
    ):
        self.base_dataset = base_dataset
        self.pair_ids = pair_ids

        self.input_tensors_key_1 = input_tensors_key_1
        self.input_tensors_key_2 = input_tensors_key_2
        self.index_key: str = index_key

    def __getitem__(self, item: int) -> Dict[str, Union[Tensor, int]]:
        i1, i2 = self.pair_ids[item]
        key = self.base_dataset.input_tensors_key
        return {
            self.input_tensors_key_1: self.base_dataset[i1][key],
            self.input_tensors_key_2: self.base_dataset[i2][key],
            self.index_key: item,
        }

    def __len__(self) -> int:
        return len(self.pair_ids)


__all__ = ["PairDataset"]
