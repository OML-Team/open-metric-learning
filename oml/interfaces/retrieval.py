from oml.interfaces.datasets import IDatasetQueryGallery
from oml.retrieval.prediction import RetrievalPrediction


class IRetrievalPostprocessor:
    """
    This is a base interface for the classes which somehow postprocess retrieval results.
    """

    def process(self, prediction: RetrievalPrediction, dataset: IDatasetQueryGallery) -> RetrievalPrediction:
        raise NotImplementedError()


__all__ = ["IRetrievalPostprocessor"]
