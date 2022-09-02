from abc import ABC, abstractmethod
from typing import Iterator, List


class IBatchSampler(ABC):
    """
    We introduce our own IBatchSampler interface instead of using the default torch BatchSampler, because the last one
    is just a wrapper for sequential sampler and in some libraries, it should follow the default constructor. To
    avoid possible errors in these cases, we use our interface with extra methods.
    """

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of batches in an epoch
        """
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self) -> Iterator[List[int]]:
        """
        Each step of the iterator has to return indices for batch
        """
        raise NotImplementedError()
