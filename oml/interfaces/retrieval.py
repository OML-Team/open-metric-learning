from oml.interfaces.datasets import IQueryGalleryDataset
from oml.retrieval.retrieval_results import RetrievalResults


class IRetrievalPostprocessor:
    """
    This is a base interface for the classes which somehow postprocess retrieval results.

    """

    def process(self, rr: RetrievalResults, dataset: IQueryGalleryDataset) -> RetrievalResults:
        raise NotImplementedError()


__all__ = ["IRetrievalPostprocessor"]
