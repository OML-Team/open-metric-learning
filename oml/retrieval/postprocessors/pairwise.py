import itertools
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch import Tensor

from oml.const import EMBEDDINGS_KEY, IS_GALLERY_KEY, IS_QUERY_KEY, PATHS_KEY
from oml.inference.pairs import (
    pairwise_inference_on_embeddings,
    pairwise_inference_on_images,
)
from oml.interfaces.models import IPairwiseModel
from oml.interfaces.retrieval import IDistancesPostprocessor
from oml.transforms.images.utils import TTransforms
from oml.utils.misc_torch import assign_2d


class PairwisePostprocessor(IDistancesPostprocessor, ABC):
    """
    This postprocessor allows us to re-estimate the distances between queries and ``top-n`` galleries
    closest to them. It creates pairs of queries and galleries and feeds them to a pairwise model.

    """

    top_n: int
    verbose: bool = False

    def process(self, distances: Tensor, queries: Any, galleries: Any) -> Tensor:
        """
        Args:
            distances: Matrix with the shape of ``[Q, G]``
            queries: Queries in the amount of ``Q``
            galleries: Galleries in the amount of ``G``

        Returns:
            Distance matrix with the shape of ``[Q, G]``,
                where ``top_n`` minimal values in each row have been updated by the pairwise model,
                other distances are shifted by a margin to keep the relative order.

        """
        n_queries = len(queries)
        n_galleries = len(galleries)

        assert list(distances.shape) == [n_queries, n_galleries]

        # 1. Adjust top_n with respect to the actual gallery size and find top-n pairs
        top_n = min(self.top_n, n_galleries)
        ii_top = torch.topk(distances, k=top_n, largest=False)[1]

        # 2. Create (n_queries * top_n) pairs of each query and related galleries and re-estimate distances for them
        if self.verbose:
            print("\nPostprocessor's inference has been started...")
        distances_upd = self.inference(queries=queries, galleries=galleries, ii_top=ii_top, top_n=top_n)
        distances_upd = distances_upd.to(distances.device).to(distances.dtype)

        # 3. Update distances for top-n galleries
        # The idea is that we somehow permute top-n galleries, but rest of the galleries
        # we keep in the end of the list as before permutation.
        # To do so, we add an offset to these galleries' distances (which haven't participated in the permutation)
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

        distances = assign_2d(x=distances, indices=ii_top, new_values=distances_upd)

        assert list(distances.shape) == [n_queries, n_galleries]

        return distances

    def inference(self, queries: Any, galleries: Any, ii_top: Tensor, top_n: int) -> Tensor:
        """
        Depends on the exact types of queries/galleries this method may be implemented differently.

        Args:
            queries: Queries in the amount of ``Q``
            galleries: Galleries in the amount of ``G``
            ii_top: Indices of the closest galleries with the shape of ``[Q, top_n]``
            top_n: Number of the closest galleries to re-rank

        Returns:
            An updated distance matrix with the shape of ``[Q, G]``

        """
        raise NotImplementedError()


class PairwiseEmbeddingsPostprocessor(PairwisePostprocessor):
    def __init__(
        self,
        top_n: int,
        pairwise_model: IPairwiseModel,
        num_workers: int,
        batch_size: int,
        verbose: bool = False,
        use_fp16: bool = False,
        is_query_key: str = IS_QUERY_KEY,
        is_gallery_key: str = IS_GALLERY_KEY,
        embeddings_key: str = EMBEDDINGS_KEY,
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
            is_query_key: Key to access a binary mask indicates queries in case of using ``process_by_dict``
            is_gallery_key: Key to access a binary mask indicates galleries in case of using ``process_by_dict``
            embeddings_key: Key to access embeddings in case of using ``process_by_dict``

        """
        assert top_n > 1, "Number of galleries for each query to process has to be greater than 1."

        self.top_n = top_n
        self.model = pairwise_model
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_fp16 = use_fp16

        self.is_query_key = is_query_key
        self.is_gallery_key = is_gallery_key
        self.embeddings_key = embeddings_key

    def inference(self, queries: Tensor, galleries: Tensor, ii_top: Tensor, top_n: int) -> Tensor:
        """
        Args:
            queries: Queries representations with the shape of ``[Q, *]``
            galleries: Galleries representations with the shape of ``[G, *]``
            ii_top: Indices of the closest galleries with the shape of ``[Q, top_n]``
            top_n: Number of the closest galleries to re-rank

        Returns:
            Updated distance matrix with the shape of ``[Q, G]``

        """
        n_queries = len(queries)
        queries = queries.repeat_interleave(top_n, dim=0)
        galleries = galleries[ii_top.view(-1)]
        distances_upd = pairwise_inference_on_embeddings(
            model=self.model,
            embeddings1=queries,
            embeddings2=galleries,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            verbose=self.verbose,
            use_fp16=self.use_fp16,
        )
        distances_upd = distances_upd.view(n_queries, top_n)
        return distances_upd

    def process_by_dict(self, distances: Tensor, data: Dict[str, Any]) -> Tensor:
        queries = data[self.embeddings_key][data[self.is_query_key]]
        galleries = data[self.embeddings_key][data[self.is_gallery_key]]
        return self.process(distances=distances, queries=queries, galleries=galleries)

    @property
    def needed_keys(self) -> List[str]:
        return [self.is_query_key, self.is_gallery_key, self.embeddings_key]


class PairwiseImagesPostprocessor(PairwisePostprocessor):
    def __init__(
        self,
        top_n: int,
        pairwise_model: IPairwiseModel,
        transforms: TTransforms,
        num_workers: int = 0,
        batch_size: int = 128,
        verbose: bool = True,
        use_fp16: bool = False,
        is_query_key: str = IS_QUERY_KEY,
        is_gallery_key: str = IS_GALLERY_KEY,
        paths_key: str = PATHS_KEY,
    ):
        """
        Args:
            top_n: Model will be applied to the ``num_queries * top_n`` pairs formed by each query
                and its ``top_n`` most relevant galleries.
            pairwise_model: Model which is able to take two images as inputs
                and estimate the *distance* (not in a strictly mathematical sense) between them.
            transforms: Transforms that will be applied to an image
            num_workers: Number of workers in DataLoader
            batch_size: Batch size that will be used in DataLoader
            verbose: Set ``True`` if you want to see progress bar for an inference
            use_fp16: Set ``True`` if you want to use half precision
            is_query_key: Key to access a binary mask indicates queries in case of using ``process_by_dict``
            is_gallery_key: Key to access a binary mask indicates galleries in case of using ``process_by_dict``
            paths_key: Key to access paths to images in case of using ``process_by_dict``

        """
        assert top_n > 1, "Number of galleries for each query to process has to be greater than 1."

        self.top_n = top_n
        self.model = pairwise_model
        self.image_transforms = transforms
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_fp16 = use_fp16

        self.is_query_key = is_query_key
        self.is_gallery_key = is_gallery_key
        self.paths_key = paths_key

    def inference(self, queries: List[Path], galleries: List[Path], ii_top: Tensor, top_n: int) -> Tensor:
        """
        Args:
            queries: Paths to queries with the length of ``Q``
            galleries: Paths to galleries with the length of ``G``
            ii_top: Indices of the closest galleries with the shape of ``[Q, top_n]``
            top_n: Number of the closest galleries to re-rank

        Returns:
            Updated distance matrix with the shape of ``[Q, G]``

        """
        n_queries = len(queries)
        queries = list(itertools.chain.from_iterable(itertools.repeat(x, top_n) for x in queries))
        galleries = [galleries[i] for i in ii_top.view(-1)]
        distances_upd = pairwise_inference_on_images(
            model=self.model,
            paths1=queries,
            paths2=galleries,
            transform=self.image_transforms,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            verbose=self.verbose,
            use_fp16=self.use_fp16,
        )
        distances_upd = distances_upd.view(n_queries, top_n)
        return distances_upd

    def process_by_dict(self, distances: Tensor, data: Dict[str, Any]) -> Tensor:
        queries = np.array(data[self.paths_key])[data[self.is_query_key]]
        galleries = np.array(data[self.paths_key])[data[self.is_gallery_key]]
        return self.process(distances=distances, queries=queries, galleries=galleries)

    @property
    def needed_keys(self) -> List[str]:
        return [self.is_query_key, self.is_gallery_key, self.paths_key]


__all__ = ["PairwisePostprocessor", "PairwiseEmbeddingsPostprocessor", "PairwiseImagesPostprocessor"]
