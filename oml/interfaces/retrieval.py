from typing import Optional

from torch import Tensor


class IPostprocessor:
    """
    This is a parent class for the classes which apply some postprocessing
    after the embeddings have been extracted and distance matrix was calculated.
    For example, we may want to apply one of query-reranking techniques.

    """

    def process(self, *args, **kwargs) -> Tensor:  # type: ignore
        """
        This method takes all the needed variables and returns
        the modified matrix of distances, where some distances are
        replaced with new ones.

        """
        raise NotImplementedError()


class IRetrievalRunner:
    """
    The goal of the class is to translate representations of queries and galleries into distance matrices.
    It is also responsible for applying post-processing (re-ranking) techniques.

    """

    def __init__(self, post_processor: Optional[IPostprocessor]):  # type: ignore
        """
        The method can take a post-processor as an input.

        """
        raise NotImplementedError()

    def setup_gallery(self, *args, **kwargs) -> Any:
        """
        The method setups a gallery for further usage (searching index).

        """
        raise NotImplementedError()

    def retrieve(self, *args, **kwargs) -> Any:
        """
        The method returns distance matrix with the size of ``[Query, Gallery]``.

        """
        raise NotImplementedError()


__all__ = ["IRetrievalRunner", "IPostprocessor"]
