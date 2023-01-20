from typing import Any

from torch import Tensor


class IDistancesPostprocessor:
    """
    This is a parent class for the classes which apply some postprocessing
    after query-to-gallery distance matrix has been calculated.
    For example, we may want to apply one of re-ranking techniques.

    """

    def process(self, distances: Tensor, queries: Any, galleries: Any) -> Tensor:
        """
        This method takes all the needed variables and returns
        the modified matrix of distances, where some distances are
        replaced with new ones.

        Args:
            distances: Matrix with the shape of ``[Q, G]``
            queries: Queries in the amount of ``Q``
            galleries: Galleries in the amount of ``G``

        Returns:
            An updated distances matrix with the shape of ``[Q, G]``

        """
        raise NotImplementedError()


__all__ = ["IDistancesPostprocessor"]
