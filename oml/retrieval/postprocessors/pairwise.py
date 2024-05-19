from typing import Sequence, Tuple

from torch import FloatTensor, LongTensor, concat

from oml.inference.abstract import pairwise_inference
from oml.interfaces.datasets import IQueryGalleryDataset
from oml.interfaces.models import IPairwiseModel
from oml.interfaces.retrieval import IRetrievalPostprocessor
from oml.retrieval.retrieval_results import RetrievalResults
from oml.utils.misc_torch import cat_two_sorted_tensors_and_keep_it_sorted


class PairwiseReranker(IRetrievalPostprocessor):
    def __init__(
        self,
        top_n: int,
        pairwise_model: IPairwiseModel,
        num_workers: int,
        batch_size: int,
        verbose: bool = False,
        use_fp16: bool = False,
    ):
        """

        Args:
            top_n: Model will be applied to the ``num_queries * top_n`` pairs formed by each query
                and ``top_n`` most relevant galleries.
            pairwise_model: Model which is able to take two items as inputs
                and estimate the *distance* (not in a strictly mathematical sense) between them.
            num_workers: Number of workers in DataLoader
            batch_size: Batch size that will be used in DataLoader
            verbose: Set ``True`` if you want to see progress bar for an inference
            use_fp16: Set ``True`` if you want to use half precision

        """
        assert top_n > 1, "The number of the retrieved results for each query to process has to be greater than 1."

        self._top_n = top_n
        self.model = pairwise_model

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_fp16 = use_fp16

    @property
    def top_n(self) -> int:
        return self._top_n

    def process(self, rr: RetrievalResults, dataset: IQueryGalleryDataset) -> RetrievalResults:  # type: ignore
        """

        Args:
            rr: RetrievalResults object.
            dataset: Dataset having query/gallery split.

        Returns:
            After model is applied to the ``top_n`` retrieved items, the updated RetrievalResults are returned.
            In other words, we permute the first ``top_n`` items, but the rest remains untouched.

        **Example 1** (for one query):

        .. code-block:: python

            rr.retrieved_ids = [[3,   2,   1,   0,   4  ]]
            rr.distances     = [[0.1, 0.2, 0.5, 0.6, 0.7]]

            # Let's say a postprocessor has been applied to the
            # first 3 elements and the new distances are: [0.4, 0.2, 0.3]

            # In this case, the updated values will be:
            rr.retrieved_ids = [[2,   1,   3,   0,   4  ]]
            rr.distances:    = [[0.2, 0.3, 0.4, 0.6, 0.7]]

        **Example 2** (for one query):

        .. code-block:: python

            # Note, the new distances to the top_n items produced by the pairwise model
            # may be rescaled to keep the distances order. Here is an example:
            rr.distances = [[0.1, 0.2, 0.3, 0.5, 0.6]]
            top_n = 3

            # Imagine, the postprocessor didn't change the order of the first 3 items
            # (it's just a convenient example, the general logic remains the same),
            # however the new values have a bigger scale:
            distances_new = [[1, 2, 5, 0.5, 0.6]]

            # Thus, we need to downscale the first 3 distances, so they are lower than 0.5:
            scale = 5 / 0.5 = 0.1
            # Finally, let's apply the found scale to the top 3 distances:
            rr_upd.distances = [[0.1, 0.2, 0.5, 0.5, 0.6]]

            # Note, if new and old distances are already sorted, we don't apply any scaling.

        """
        gt_ids = rr.gt_ids
        distances_upd, retrieved_ids_upd = self._process_raw(
            retrieved_ids=rr.retrieved_ids, distances=rr.distances, dataset=dataset
        )
        rr_upd = RetrievalResults(distances=distances_upd, retrieved_ids=retrieved_ids_upd, gt_ids=gt_ids)
        return rr_upd

    def _process_raw(
        self, retrieved_ids: Sequence[LongTensor], distances: Sequence[FloatTensor], dataset: IQueryGalleryDataset
    ) -> Tuple[Sequence[FloatTensor], Sequence[LongTensor]]:
        top_n = self.top_n

        # let's make list of pairs of queries and top_n gallery items for which we need to recompute distances
        # since queries may have different number of retrieved items, we also need bounds variable
        pairs = []
        bounds = [0]
        for iq, ids_gallery in enumerate(retrieved_ids):
            ids_gallery_global = dataset.get_gallery_ids()[ids_gallery][:top_n].tolist()
            ids_query_global = [dataset.get_query_ids()[iq].item()] * len(ids_gallery_global)

            pairs.extend(list(zip(ids_query_global, ids_gallery_global)))
            bounds.append(bounds[-1] + len(ids_gallery_global))

        distances_recomputed = pairwise_inference(
            model=self.model,
            base_dataset=dataset,
            pair_ids=pairs,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            verbose=self.verbose,
            use_fp16=self.use_fp16,
        )

        # now let's reshape flatten distances into the original structure of lists may be having different sizes
        distances_upd, retrieved_ids_upd = [], []
        for query_start, query_end, dist_orig, ri_orig in zip(bounds[:-1], bounds[1:], distances, retrieved_ids):
            dist_recomputed_q, ii_rerank = distances_recomputed[query_start:query_end].sort()

            distances_upd += [
                cat_two_sorted_tensors_and_keep_it_sorted(
                    dist_recomputed_q.view(1, -1), dist_orig[top_n:].view(1, -1)
                ).view(-1)
            ]
            retrieved_ids_upd += [concat([ri_orig[ii_rerank], ri_orig[top_n:]])]

        assert len(retrieved_ids_upd) == len(retrieved_ids) == len(distances) == len(distances_upd)

        for iq in range(len(retrieved_ids)):
            assert len(retrieved_ids[iq]) == len(retrieved_ids_upd[iq])
            assert len(distances[iq]) == len(distances_upd[iq])

        return distances_upd, retrieved_ids_upd


__all__ = ["PairwiseReranker"]
