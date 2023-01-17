from typing import Any

from torch import Tensor


class IDistancesPostprocessor:
    """
    This is a parent class for the classes which apply some postprocessing
    after the embeddings have been extracted and distance matrix has been calculated.
    For example, we may want to apply one of query-reranking techniques.

    """

    def process(self, distances: Tensor, queries: Any, galleries: Any) -> Tensor:
        """
        This method takes all the needed variables and returns
        the modified matrix of distances, where some distances are
        replaced with new ones.

        """
        raise NotImplementedError()


__all__ = ["IDistancesPostprocessor"]
