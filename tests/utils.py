from typing import Sequence

import torch
from torch import Tensor


def check_if_sequence_of_tensors_are_equal(list1: Sequence[Tensor], list2: Sequence[Tensor]) -> bool:
    for l1, l2 in zip(list1, list2):
        if not torch.allclose(l1, l2):
            return False
    return True
