from typing import Tuple

import torch
from torch import FloatTensor, LongTensor

from oml.datasets.base import DatasetQueryGallery
from oml.inference.pairs import pairwise_inference
from oml.interfaces.models import IPairwiseModel
from oml.interfaces.retrieval import IRetrievalPostprocessor
from oml.utils.misc_torch import cat_two_sorted_tensors_and_keep_it_sorted, take_2d


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
            pairwise_model: Model which is able to take two embeddings as inputs
                and estimate the *distance* (not in a strictly mathematical sense) between them.
            num_workers: Number of workers in DataLoader
            batch_size: Batch size that will be used in DataLoader
            verbose: Set ``True`` if you want to see progress bar for an inference
            use_fp16: Set ``True`` if you want to use half precision

        """
        assert top_n > 1, "Number of galleries for each query to process has to be greater than 1."

        self.top_n = top_n
        self.model = pairwise_model

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_fp16 = use_fp16

    def process(
        self,
        distances: FloatTensor,
        retrieved_ids: LongTensor,
        dataset: DatasetQueryGallery,
    ) -> Tuple[FloatTensor, LongTensor]:
        """

        Note, the new distances to the ``top_n`` items produced by the pairwise model may be adjusted
        to remain distances sorted. Here is an example:
        ``original_distances = [0.1, 0.2, 0.3, 0.5, 0.6], top_n = 3``
        Imagine, the postprocessor didn't change the order of the first 3 items (it's just a convenient example,
        the logic remains the same), however the new values have a bigger scale:
        ``distances_upd = [1, 2, 5, 0.5, 0.6]``.
        Thus, we need to rescale the first three distances, so they don't go above ``0.5``.
        The scaling factor is ``s = min(0.5, 0.6) / max(1, 2, 5) = 0.1``. Finally:
        ``distances_upd_scaled = [0.1, 0.2, 0.5, 0.5, 0.6]``.
        If concatenation of two distances is already sorted, we keep it untouched.

        """
        assert len(retrieved_ids) == len(distances)

        top_n = min(self.top_n, retrieved_ids.shape[1])
        n_queries = len(retrieved_ids)

        retrieved_ids = retrieved_ids.clone()
        distances = distances.clone()

        # let's list pairs of (query_i, gallery_j) we need to process
        ids_q = dataset.get_query_mask().nonzero().repeat_interleave(top_n)
        ii_g = dataset.get_gallery_mask().nonzero()
        ids_g = ii_g[retrieved_ids[:, :top_n]].flatten()
        assert len(ids_q) == len(ids_g)
        pairs = list(zip(ids_q.tolist(), ids_g.tolist()))

        distances_top = pairwise_inference(
            model=self.model,
            base_dataset=dataset,
            pair_ids=pairs,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            verbose=self.verbose,
            use_fp16=self.use_fp16,
        )
        distances_top = distances_top.view(n_queries, top_n)

        distances_top_ranked, ii_rerank = distances_top.sort()

        # Updating indices:
        unprocessed_ids = retrieved_ids[:, top_n:]
        retrieved_ids_upd = torch.concat(
            [take_2d(retrieved_ids, ii_rerank), unprocessed_ids],  # re-ranked top + old values
            dim=1,
        ).long()

        # Updating distances:
        unprocessed_distances = distances[:, top_n:]
        if unprocessed_distances.shape[1] > 0:
            distances_upd = cat_two_sorted_tensors_and_keep_it_sorted(distances_top_ranked, unprocessed_distances)
        else:
            distances_upd = distances_top_ranked

        assert distances_upd.shape == distances.shape
        assert retrieved_ids_upd.shape == retrieved_ids.shape

        return distances_upd, retrieved_ids_upd


__all__ = ["PairwiseReranker"]
