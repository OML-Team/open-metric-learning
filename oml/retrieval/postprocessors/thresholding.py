from oml.interfaces.retrieval import IRetrievalPostprocessor
from oml.retrieval import RetrievalResults


class AdaptiveThresholding(IRetrievalPostprocessor):
    def __init__(self, th: float):
        self.th = th

    def process(self, rr: RetrievalResults) -> RetrievalResults:  # type: ignore
        return rr


__all__ = ["AdaptiveThresholding"]
