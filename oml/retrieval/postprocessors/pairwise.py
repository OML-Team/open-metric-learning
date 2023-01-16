import itertools
from abc import ABC
from pathlib import Path
from typing import List
from typing import Union

import torch
from oml.datasets.pairs import images_pairwise_inference
from oml.inference.pairwise import pairwise_inference_on_images, pairwise_inference_on_embeddings
from oml.interfaces.models import IPairwiseDistanceModel
from oml.interfaces.postprocessor import IPostprocessor
from oml.transforms.images.utils import TTransforms
from oml.utils.misc_torch import assign_2d
from torch import Tensor, nn
from oml.models.siamese import ImagesSiamese, SimpleSiamese


class PairwiseProcessor(IPostprocessor, ABC):
    """
    This postprocessor allows us to re-estimate the *distances* between queries and *top-n* galleries
    closest to them. It creates pairs of queries and galleries and feeds them to a pairwise model.

    """

    top_n: int

    def process(self,
                distances: Tensor,
                queries: Union[List[Path], Tensor],
                galleries: Union[List[Path], Tensor]
                ):
        """
        Args:
            distances: Matrix with the shape of ``[Q, G]``
            queries: *Q* queries, they may be paths or representations.
            galleries: *G* galleries, they may be paths or representations.

        Returns:
            Matrix with the shape of ``[Q, G]`` containing updated *distances*.

        """
        n_queries = len(queries)
        n_galleries = len(galleries)

        assert list(distances.shape) == [n_queries, n_galleries]

        # 1. Adjust top_n with respect to the actual gallery size
        top_n = min(self.top_n, n_galleries)

        # 2. Create (n_queries * top_n) pairs of each query and related galleries and re-estimate distances for them
        distances_upd = self.inference_on_pairs(queries=queries, galleries=galleries, distances=distances, top_n=top_n)
        distances_upd = distances_upd.view(n_queries, top_n)

        # 3. Update distances for top-n galleries
        # The idea is that we somehow permute top-n galleries, but rest of the galleries
        # we keep in the end of the list as before permutation.
        # To do so, we add an offset to these galleries (which did not participate in permutation)
        if top_n < n_galleries:
            # Here we use the fact that distances not participating in permutation start with top_n + 1 position
            min_in_old_distances = torch.topk(distances, k=top_n + 1, largest=False)[0][:, -1]
            max_in_new_distances = distances_upd.max(dim=1)[0]
            offset = max_in_new_distances - min_in_old_distances + 1e-5  # we also need some eps if max == min
            distances += offset.unsqueeze(-1)
        else:
            # Pairwise postprocessor has been applied to all possible pairs, so, there are no rest distances.
            # Thus, we don't need to care about order and offset at all.
            pass

        distances = assign_2d(x=distances, indices=ii_top.view(n_queries, top_n), new_values=distances_upd)

        assert list(distances.shape) == [n_queries, n_galleries]

        return distances

    def inference_on_pairs(self,
                           queries: Union[List[Path], Tensor],
                           galleries: Union[List[Path], Tensor],
                           distances: Tensor,
                           top_n: int
                           ) -> Tensor:
        """
        Depends on the exact types of queries/galleries this method may be implemented differently.

        """
        raise NotImplementedError()


class PairwiseEmbeddingsPostprocessor(PairwiseProcessor):

    def __init__(self, pairwise_model: IPairwiseDistanceModel, top_n: int):
        """
        Args:
            pairwise_model: Model which is able to take two embeddings as inputs
                and estimate the *distance* (not in a strictly mathematical sense) between them.
            top_n: Model will be applied to the ``num_queries * top_n`` pairs formed by each query
                and ``top_n`` most relevant galleries.

        """
        assert top_n > 1, "Number of galleries for each query to process has to be greater than 1."

        self.model = pairwise_model
        self.top_n = top_n

    def inference_on_pairs(self, queries: Tensor, galleries: Tensor, distances: Tensor, top_n: int) -> Tensor:
        ii_top = torch.topk(distances, k=top_n, largest=False)[1].view(-1)
        queries = emb_query.repeat_interleave(top_n, dim=0)
        galleries = galleries[ii_top]
        distances_upd = pairwise_inference_on_images(self.model, queries, galleries)
        distances_upd = distances_upd.view(n_queries, top_n)
        return distances_upd


class PairwiseImagesPostprocessor(IPostprocessor):
    def __init__(self, pairwise_model: nn.Module, top_n: int, image_transforms: TTransforms):
        self.model = pairwise_model
        self.top_n = top_n
        self.image_transforms = image_transforms

    def inference_on_pairs(self, queries: List[Path], galleries: List[Path], distances: Tensor, top_n: int) -> Tensor:
        ii_top = torch.topk(distances, k=top_n, largest=False)[1].view(-1)
        queries = list(itertools.chain.from_iterable(itertools.repeat(x, top_n) for x in queries))
        galleries = [galleries[i] for i in ii_top]
        distances_upd = images_pairwise_inference(self.model, queries, galleries, self.image_transforms)
        return distances_upd


__all__ = ["PairwiseProcessor", "PairwiseEmbeddingsPostprocessor", "PairwiseImagesPostprocessor"]
