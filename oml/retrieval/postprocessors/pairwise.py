from typing import Tuple

import torch
from torch import FloatTensor, LongTensor

from oml.inference.pairs import pairwise_inference
from oml.interfaces.datasets import IDatasetQueryGallery
from oml.interfaces.models import IPairwiseModel
from oml.interfaces.retrieval import IRetrievalPostprocessor
from oml.utils.misc_torch import take_2d


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
        todo

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
        dataset: IDatasetQueryGallery,
    ) -> Tuple[FloatTensor, LongTensor]:
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
        retrieved_ids_upd = torch.concat(
            [take_2d(retrieved_ids, ii_rerank), retrieved_ids[:, top_n:]],  # re-ranked top  # old untouched values
            dim=1,
        ).long()

        # Updating distances:
        unprocessed_distances = distances[:, top_n:]

        # The idea of offset is that we want to keep distances consist after re-ranking. Here is an example:
        # distances     = [0.1, 0.2, 0.3, 0.5, 0.6], top_n = 3
        # imagine, the postprocessor didn't change the order of distances, but assigned values withing a different scale
        # distances_upd = [0.6, 0.8, 0.9, 0.5, 0.6], top_n = 3
        # to keep distances aligned we introduce offset = max(0.6, 0.8, 0.9) - min(0.5, 0.6) = 0.5
        # so, adjusted distances are
        # distances_upd = [0.6, 0.8, 0.9, 0.5 + 0.5, 0.6 + 0.5]  = [0.6, 0.8, 0.9, 1.0, 1.1]

        # todo
        # if unprocessed_distances.numel() > 0:
        #     offset = distances_top_ranked.max(dim=1) - unprocessed_distances.min(dim=1) + 1e-5
        #     unprocessed_distances += offset

        distances_upd = torch.concat(
            [distances_top_ranked, unprocessed_distances],  # re-ranked top  # unprocessed distances shifted by offset
            dim=1,
        ).float()

        assert distances_upd.shape == distances.shape
        assert retrieved_ids_upd.shape == retrieved_ids.shape

        return distances_upd, retrieved_ids_upd


__all__ = ["PairwiseReranker"]
