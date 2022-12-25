import torch
from torch import Tensor

from oml.datasets.vectors_pairs import pairwise_inference
from oml.interfaces.models import IPairwiseDistanceModel
from oml.interfaces.postprocessor import IPostprocessor
from oml.utils.misc_torch import assign_2d


class PairwiseEmbeddingsPostprocessor(IPostprocessor):
    """
    This postprocessor allows us to re-estimate the distances between queries and galleries
    closest to them. It creates pairs of embeddings which represent such queries and galleries
    and feeds them to a pairwise model.

    """

    def __init__(self, pairwise_model: IPairwiseDistanceModel, top_n: int):
        """
        Args:
            pairwise_model: Model which is able to take two embeddings as inputs
                and estimate the *distance* (not in a strictly mathematical sense) between them.
            top_n: Model will be applied to the ``num_queries * top_n`` pairs formed by each query
                and ``top_n`` most relevant galleries.

        """
        assert top_n > 0, "Number of galleries for each query to process has to be greater than 0."

        self.model = pairwise_model
        self.top_n = top_n

    def process(self, distances: Tensor, emb_query: Tensor, emb_gallery: Tensor) -> Tensor:  # type: ignore
        """
        Args:
            distances: Matrix with the shape of ``[Q, G]``
            emb_query: Embeddings with the shape of ``[Q, features_dim]``
            emb_gallery: Embeddings with the shape of ``[G, features_dim]``

        Returns:
            Matrix with the shape of ``[Q, G]`` containing updated *distances*.

        """
        n_queries = len(emb_query)
        n_galleries = len(emb_gallery)

        assert list(distances.shape) == [n_queries, n_galleries]

        # 1. Pick top n galleries for each query with the smallest distances
        top_n = min(self.top_n, n_galleries)
        ii_top = torch.topk(distances, k=top_n, largest=False)[1].view(-1)

        # 2. Create (n_queries * top_n) pairs of each query and related galleries and re-estimate distances for them
        emb_query = emb_query.repeat_interleave(top_n, dim=0)
        emb_gallery = emb_gallery[ii_top]
        distances_upd = pairwise_inference(self.model, emb_query, emb_gallery)
        distances_upd = distances_upd.view(n_queries, top_n)

        # 3. Update distances for top-n galleries
        # In order of keeping the relative order with the rest of the galleries,
        # we make all of them greater than the maximum distance among re-estimated distances
        offset = distances_upd.max(dim=1)[0] + torch.finfo(torch.float32).eps
        distances += offset.unsqueeze(-1)
        distances = assign_2d(x=distances, indices=ii_top.view(n_queries, top_n), new_values=distances_upd)

        assert list(distances.shape) == [n_queries, n_galleries]

        return distances


__all__ = ["PairwiseEmbeddingsPostprocessor"]
