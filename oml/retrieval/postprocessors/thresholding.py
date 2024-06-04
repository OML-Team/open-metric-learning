from copy import deepcopy

from oml.interfaces.retrieval import IRetrievalPostprocessor
from oml.retrieval import RetrievalResults


class ConstantThresholding(IRetrievalPostprocessor):
    def __init__(self, th: float):
        self.th = th

    def process(self, rr: RetrievalResults) -> RetrievalResults:  # type: ignore
        """
        Args:
            rr: An instance of `RetrievalResults`.

        Returns:
            An updated instance of `RetrievalResults` where retrieved items having
            distance greater than `self.th` have been filtered out.

        """
        distances_upd = []
        retrieved_ids_upd = []

        for dists, rids in zip(rr.distances, rr.retrieved_ids):
            mask = dists < self.th
            dists = dists[mask]
            rids = rids[mask]

            distances_upd.append(dists)
            retrieved_ids_upd.append(rids)

        rr = RetrievalResults(distances=distances_upd, retrieved_ids=retrieved_ids_upd, gt_ids=deepcopy(rr.gt_ids))

        return rr


class AdaptiveThresholding(IRetrievalPostprocessor):
    def __init__(self, th: float):
        self.th = th

    def process(self, rr: RetrievalResults) -> RetrievalResults:  # type: ignore
        return rr


__all__ = ["AdaptiveThresholding", "ConstantThresholding"]
