class IRetrievalPostprocessor:
    # todo 522: update signatures and think one more time about classes hierarchy

    """
    This is a base interface for the classes which somehow postprocess retrieval results.
    """

    def process(self, distances, retrieved_ids, dataset):  # type: ignore
        raise NotImplementedError()


__all__ = ["IRetrievalPostprocessor"]
