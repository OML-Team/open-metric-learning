class IPostprocessor:
    """
    This is a parent class for the classes which apply some postprocessing
    after the embeddings have been extracted.
    For example, we may want to apply some method of dimension reduction or
    query-reranking.

    """

    def process(self, *args, **kwargs):  # type: ignore
        raise NotImplementedError()


__all__ = ["IPostprocessor"]
