from abc import ABC
from collections.abc import MutableMapping
from typing import Any, Dict, Hashable, Iterator, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch import Tensor, cdist

TSingleValues = Union[int, float, np.float_, np.int_, torch.Tensor]
TSequenceValues = Union[List[float], Tuple[float, ...], np.ndarray, torch.Tensor]
TOnlineValues = Union[TSingleValues, TSequenceValues]


def elementwise_dist(x1: Tensor, x2: Tensor, p: int = 2) -> Tensor:
    """
    Args:
        x1: tensor with the shape of [N, D]
        x2: tensor with the shape of [N, D]
        p: degree

    Returns: elementwise distances with the shape of [N]

    """
    assert len(x1.shape) == len(x2.shape) == 2
    assert x1.shape == x2.shape

    # we need an extra dim here to avoid pairwise behaviour of torch.cdist
    if len(x1.shape) == 2:
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

    dist = cdist(x1=x1, x2=x2, p=p).squeeze()

    return dist


def pairwise_dist(x1: Tensor, x2: Tensor, p: int = 2) -> Tensor:
    """
    Args:
        x1: tensor with the shape of [N, D]
        x2: tensor with the shape of [M, D]
        p: degree

    Returns: pairwise distances with the shape of [N, M]

    """
    assert len(x1.shape) == len(x2.shape) == 2
    assert x1.shape[-1] == x2.shape[-1]

    return cdist(x1=x1, x2=x2, p=p)


def _check_is_sequence(val: Any) -> bool:
    try:
        len(val)
        return True
    except Exception:
        return False


class OnlineCalc(ABC):
    """
    The base class to calculate some statistics online (on the steam of values).

    """

    def __init__(self, val: Optional[TOnlineValues] = None):
        self.result: float = 0.0
        if val is not None:
            self.update(val)

    def update(self, val: TOnlineValues) -> None:
        if _check_is_sequence(val):
            self.calc_with_batch(val)
        else:
            self.calc_with_single_value(val)

    def calc_with_single_value(self, val: TSingleValues) -> None:
        """
        Calculation with non iterable types: float, int / numpy and torch elements (array and elements of
        array have different types and methods)

        """
        self.calc_with_batch([val])

    def calc_with_batch(self, val: TSequenceValues) -> None:
        """
        Vectorized calculation of online value

        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self.__dict__)})"


class AvgOnline(OnlineCalc):
    def __init__(self, *args: Any, **kwargs: Any):
        self.n = 0
        super().__init__(*args, **kwargs)

    def calc_with_batch(self, val: TSequenceValues) -> None:
        len_val = len(val)
        self.n += len_val
        self.result = sum(val) / self.n + (self.n - len_val) / self.n * self.result  # type: ignore


class SumOnline(OnlineCalc):
    def calc_with_batch(self, val: TSequenceValues) -> None:
        self.result += float(sum(val))


class OnlineDict(MutableMapping):  # type: ignore
    """
    We don't inherite from built-in 'dict' due to internal C optimization. We mimic to dict with MutableMapping
    https://treyhunner.com/2019/04/why-you-shouldnt-inherit-from-list-and-dict-in-python/

    """

    online_calculator: Type[OnlineCalc]

    def __init__(self, input_dict: Optional[Dict[Hashable, TOnlineValues]] = None):
        self.dict: Dict[Hashable, OnlineCalc] = {}

        if input_dict:
            self.update(input_dict)

    def __setitem__(self, key: Hashable, value: TOnlineValues) -> None:
        self.dict[key] = self.online_calculator(value)

    def __getitem__(self, key: Hashable) -> float:
        return self.dict[key].result

    def __len__(self) -> int:
        return len(self.dict)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.dict)

    def __delitem__(self, key: Hashable) -> None:
        return self.dict.__delitem__(key)

    def update(self, input_dict: Dict[Hashable, TOnlineValues]) -> None:  # type: ignore
        for k, v in input_dict.items():
            if k in self.dict:
                self.dict[k].update(v)
            else:
                self[k] = v

    def __repr__(self) -> str:
        output = {k: v.result for k, v in self.dict.items()}
        return f"{self.__class__.__name__}({output})"

    def get_dict_with_results(self) -> Dict[str, float]:
        return {k: v for k, v in self.items()}


class OnlineAvgDict(OnlineDict):
    online_calculator = AvgOnline


class OnlineSumDict(OnlineDict):
    online_calculator = SumOnline


__all__ = [
    "elementwise_dist",
    "pairwise_dist",
    "OnlineCalc",
    "AvgOnline",
    "SumOnline",
    "OnlineDict",
    "OnlineAvgDict",
    "OnlineSumDict",
]
