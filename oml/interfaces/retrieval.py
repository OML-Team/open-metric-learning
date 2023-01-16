from typing import Any, Optional

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


class IRetrievalRunner:
    """
    The goal of the class is to translate representations of queries and galleries into distance matrix.
    It's also responsible for applying post-processing (re-ranking) techniques.

    """

    post_processor: Optional[IDistancesPostprocessor]  # Post-processor can be passed to constructor

    def setup_gallery(self, *args, **kwargs) -> Any:  # type: ignore
        """
        The method setups a gallery for further usage (searching index).

        """
        raise NotImplementedError()

    def retrieve(self, *args, **kwargs) -> Any:  # type: ignore
        """
        The method returns distance matrix with the size of ``[query, gallery]``.

        """
        raise NotImplementedError()


__all__ = ["IRetrievalRunner", "IDistancesPostprocessor"]
