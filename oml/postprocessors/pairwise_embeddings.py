import torch
from torch import Tensor

from oml.datasets.vectors_pairs import pairwise_inference
from oml.interfaces.models import IPairwiseDistanceModel
from oml.interfaces.postprocessor import IPostprocessor
from oml.utils.misc_torch import assign_2d


class PairwiseEmbeddingsPostprocessor(IPostprocessor):
    def __init__(self, pairwise_model: IPairwiseDistanceModel, top_n: int):
        assert top_n > 0

        self.model = pairwise_model
        self.top_n = top_n

    def process(self, distances: Tensor, emb_query: Tensor, emb_gallery: Tensor) -> Tensor:  # type: ignore
        n_queries = len(emb_query)
        n_galleries = len(emb_gallery)

        assert list(distances.shape) == [n_queries, n_galleries]

        # pick top n galleries for each query with the smallest distances
        top_n = min(self.top_n, n_galleries)
        ii_top = torch.topk(distances, k=top_n, largest=False)[1].view(-1)

        # create (n_queries * top_n) pairs of each query and related galleries and re-estimate distances for them
        emb_query = emb_query.repeat_interleave(top_n, dim=0)
        emb_gallery = emb_gallery[ii_top]
        distances_upd = pairwise_inference(self.model, emb_query, emb_gallery)
        distances_upd = distances_upd.view(n_queries, top_n)

        # update distances for top-n galleries, keeping the order of rest of the galleries (we use offset for it)
        offset = distances_upd.min(dim=1)[0] - distances.min(dim=1)[0] + torch.finfo(torch.float32).eps
        distances += offset.unsqueeze(-1)
        distances = assign_2d(x=distances, indeces=ii_top.view(n_queries, top_n), new_values=distances_upd)

        assert list(distances.shape) == [n_queries, n_galleries]

        return distances
