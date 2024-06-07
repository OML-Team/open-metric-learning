from copy import deepcopy

from oml.interfaces.retrieval import IRetrievalPostprocessor
from oml.retrieval.retrieval_results import RetrievalResults
from oml.utils.misc_torch import AvgOnline


class ConstantThresholding(IRetrievalPostprocessor):
    def __init__(self, th: float):
        """
        Args:
            th: Distance threshold to limit the `RetrievalResults`.
        """
        self.th = th

    def process(self, rr: RetrievalResults) -> RetrievalResults:  # type: ignore
        if rr.is_empty():
            return rr.deepcopy()

        distances_upd = []
        rids_upd = []

        for dists, rids in zip(rr.distances, rr.retrieved_ids):
            mask = dists < self.th
            dists = dists[mask]
            rids = rids[mask]

            distances_upd.append(dists)
            rids_upd.append(rids)

        rr_upd = RetrievalResults(distances=distances_upd, retrieved_ids=rids_upd, gt_ids=deepcopy(rr.gt_ids))

        return rr_upd


class AdaptiveThresholding(IRetrievalPostprocessor):
    def __init__(self, n_std: float):
        """
        This postprocessor cuts `RetrievalResults` if a big gap in consecutive distances has been found.
        The big gap is defined as a gap greater than `n_std * avg_gap`.

        Args:
            n_std: the smaller value, the less `RetrievalResults` will be remained.

        """
        self.n_std = n_std

    def process(self, rr: RetrievalResults) -> RetrievalResults:  # type: ignore
        if rr.is_empty():
            return rr.deepcopy()

        avg_diff = AvgOnline()
        for dists in rr.distances:
            avg_diff.update(dists[1:] - dists[:-1])

        distances_upd = []
        rids_upd = []

        for dists, rids in zip(rr.distances, rr.retrieved_ids):
            diffs = dists[1:] - dists[:-1]
            mask_gaps = diffs > self.n_std * avg_diff.result

            if mask_gaps.sum() == 0:
                distances_upd.append(dists)
                rids_upd.append(rids)
            else:
                i_th = mask_gaps.nonzero()[0]
                distances_upd.append(dists[: i_th + 1])
                rids_upd.append(rids[: i_th + 1])

        rr_upd = RetrievalResults(distances=distances_upd, retrieved_ids=rids_upd, gt_ids=deepcopy(rr.gt_ids))

        return rr_upd


__all__ = ["AdaptiveThresholding", "ConstantThresholding"]
