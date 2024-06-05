from typing import Sequence

import torch
from torch import FloatTensor, LongTensor, Tensor


def check_if_sequence_of_tensors_are_equal(list1: Sequence[Tensor], list2: Sequence[Tensor]) -> bool:
    if len(list1) != len(list2):
        return False

    for l1, l2 in zip(list1, list2):

        if isinstance(l1, FloatTensor) and isinstance(l2, FloatTensor):
            if len(l1) != len(l2) or (not torch.allclose(l1, l2)):
                return False

        elif isinstance(l1, LongTensor) and isinstance(l2, LongTensor):
            if len(l1) != len(l2) or (not (l1 == l2).all()):
                return False

        else:
            raise TypeError(f"Unsupported types: {type(l1)}, {type(l2)}.")

    return True
