import itertools
from abc import ABC
from pathlib import Path
from typing import List, Union

import torch
from torch import Tensor

from oml.inference.pairwise import (
    pairwise_inference_on_embeddings,
    pairwise_inference_on_images,
)
from oml.interfaces.models import IPairwiseDistanceModel
from oml.interfaces.retrieval import IDistancesPostprocessor
from oml.transforms.images.utils import TTransforms
from oml.utils.misc_torch import assign_2d


class PairwisePostprocessor(IDistancesPostprocessor, ABC):
    """
    This postprocessor allows us to re-estimate the *distances* between queries and *top-n* galleries
    closest to them. It creates pairs of queries and galleries and feeds them to a pairwise model.

    """

    top_n: int

    def process(
        self, distances: Tensor, queries: Union[List[Path], Tensor], galleries: Union[List[Path], Tensor]
    ) -> Tensor:
        """
        Args:
            distances: Matrix with the shape of ``[Q, G]``
            queries: *Q* queries, they may be paths or representations.
            galleries: *G* galleries, they may be paths or representations.

        Returns:
            Distance matrix with the shape of ``[Q, G]``,
            where ``top_n`` minimal values in each row have been updated by the pairwise model,
            other distances are shifted to keep the relative order.

        """
        n_queries = len(queries)
        n_galleries = len(galleries)

        assert list(distances.shape) == [n_queries, n_galleries]

        # 1. Adjust top_n with respect to the actual gallery size and find top-n pairs
        top_n = min(self.top_n, n_galleries)
        ii_top = torch.topk(distances, k=top_n, largest=False)[1].view(-1)

        # 2. Create (n_queries * top_n) pairs of each query and related galleries and re-estimate distances for them
        distances_upd = self.inference(queries=queries, galleries=galleries, ii_top=ii_top, top_n=top_n)
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

    def inference(
        self, queries: Union[List[Path], Tensor], galleries: Union[List[Path], Tensor], ii_top: Tensor, top_n: int
    ) -> Tensor:
        """
        Depends on the exact types of queries/galleries this method may be implemented differently.

        Args:
            queries: Queries with the length of ``Q``
            galleries: Galleries with the length of ``G``
            ii_top: Indices of the closest galleries with the shape of ``[Q, top_n]``
            top_n: Defines the number of the closest galleries to re-rank

        Returns:
            Updated distance matrix with the shape of ``[Q, G]``.

        """
        raise NotImplementedError()


class PairwiseEmbeddingsPostprocessor(PairwisePostprocessor):
    def __init__(
        self,
        top_n: int,
        pairwise_model: IPairwiseDistanceModel,
        num_workers: int,
        batch_size: int,
        verbose: bool = False,
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

        """
        assert top_n > 1, "Number of galleries for each query to process has to be greater than 1."

        self.top_n = top_n
        self.model = pairwise_model
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.verbose = verbose

    def inference(self, queries: Tensor, galleries: Tensor, ii_top: Tensor, top_n: int) -> Tensor:
        """
        Args:
            queries: Queries representations with the length of ``Q``
            galleries: Galleries representations with the length of ``G``
            ii_top: Indices of the closest galleries with the shape of ``[Q, top_n]``
            top_n: Defines the number of the closest galleries to re-rank

        Returns:
            Updated distance matrix with the shape of ``[Q, G]``.

        """
        queries = queries.repeat_interleave(top_n, dim=0)
        galleries = galleries[ii_top]
        distances_upd = pairwise_inference_on_embeddings(
            model=self.model,
            embeddings1=queries,
            embeddings2=galleries,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        return distances_upd


class PairwiseImagesPostprocessor(PairwisePostprocessor):
    def __init__(
        self,
        top_n: int,
        pairwise_model: IPairwiseDistanceModel,
        transforms: TTransforms,
        num_workers: int,
        batch_size: int,
        verbose: bool = True,
    ):
        """
        Args:
            top_n: Model will be applied to the ``num_queries * top_n`` pairs formed by each query
                and ``top_n`` most relevant galleries.
            pairwise_model: Model which is able to take two images as inputs
                and estimate the *distance* (not in a strictly mathematical sense) between them.
            transforms: Transforms that will be applied to an image
            num_workers: Number of workers in DataLoader
            batch_size: Batch size that will be used in DataLoader
            verbose: Set ``True`` if you want to see progress bar for an inference

        """
        assert top_n > 1, "Number of galleries for each query to process has to be greater than 1."

        self.top_n = top_n
        self.model = pairwise_model
        self.image_transforms = transforms
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.verbose = verbose

    def inference(self, queries: List[Path], galleries: List[Path], ii_top: Tensor, top_n: int) -> Tensor:
        """
        Args:
            queries: Paths to queries with the length of ``Q``
            galleries: Paths to galleries with the length of ``G``
            ii_top: Indices of the closest galleries with the shape of ``[Q, top_n]``
            top_n: Defines the number of the closest galleries to re-rank

        Returns:
            Updated distance matrix with the shape of ``[Q, G]``.

        """
        queries = list(itertools.chain.from_iterable(itertools.repeat(x, top_n) for x in queries))
        galleries = [galleries[i] for i in ii_top]
        distances_upd = pairwise_inference_on_images(
            model=self.model,
            paths1=queries,
            paths2=galleries,
            transform=self.image_transforms,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        return distances_upd


__all__ = ["PairwisePostprocessor", "PairwiseEmbeddingsPostprocessor", "PairwiseImagesPostprocessor"]
