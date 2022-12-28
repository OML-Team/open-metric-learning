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


__all__ = ["IPostprocessor"]
