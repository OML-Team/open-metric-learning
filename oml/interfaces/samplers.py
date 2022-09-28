from abc import ABC, abstractmethod
from typing import Iterator, List


class IBatchSampler(ABC):
    """
    We introduce our interface instead of using the default BatchSampler from Torch,
    because the last one is just a wrapper for the sequential sampler, which is not
    convenient for our purposes.

    """

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns:
             The number of batches in an epoch

        """
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self) -> Iterator[List[int]]:
        """
        Returns:
            Iterator contains indices for the batches

        """
        raise NotImplementedError()


__all__ = ["IBatchSampler"]
