from typing import Any


class IRetrievalPostprocessor:
    # todo 522: update signatures after we have RetrievalPrediction class

    """
    This is a parent class for the classes which somehow postprocess retrieval results.
    """

    def process(self, distances, retrieved_ids, dataset) -> Any:  # type: ignore
        raise NotImplementedError()


__all__ = ["IRetrievalPostprocessor"]
