from typing import Iterator, List

from torch.utils.data import Sampler


class IBatchSampler(Sampler[List[int]]):
    """
    We introduce our own IBatchSampler interface instead of using default torch BatchSampler, because the last on is
    just wrapper for sequential sampler and in some libraries it should follow default constructor. In order to
    avoid possible errors for this cases, we use out own interface with extra methods.
    """

    def __len__(self) -> int:
        """
        Return number of batches in epoch
        """
        raise NotImplementedError()

    def __iter__(self) -> Iterator[List[int]]:
        """
        Each step of iterator have to return indices for batch
        """
        raise NotImplementedError()

    def batch_size(self) -> int:
        """
        Return number of sampler in each batch. We assume that batches (including last) have the same size
        """
        raise NotImplementedError()
