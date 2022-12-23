from torch import Tensor


class IPostprocessor:
    """
    This is a parent class for the classes which apply some postprocessing.rst
    after the embeddings have been extracted and distance matrix was calculated.
    For example, we may want to apply some of the dimension reduction methods or
    one of query-reranking techniques.

    """

    def process(self, *args, **kwargs) -> Tensor:  # type: ignore
        """
        This method takes all the needed variables and returns
        the modified matrix of distances, where some of the distances were
        replaced with new ones.

        # todo: force not to change the shape of distance matrix
        """
        raise NotImplementedError()


__all__ = ["IPostprocessor"]
