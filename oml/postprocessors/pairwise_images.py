import itertools
from pathlib import Path
from typing import List

import torch
from torch import Tensor, nn

from oml.datasets.pairs import images_pairwise_inference
from oml.interfaces.postprocessor import IPostprocessor
from oml.transforms.images.utils import TTransforms
from oml.utils.misc_torch import assign_2d


class PairwiseImagesPostprocessor(IPostprocessor):
    # todo: we need types here
    def __init__(self, pairwise_model: nn.Module, top_n: int, image_transforms: TTransforms):
        self.model = pairwise_model
        self.top_n = top_n
        self.image_transforms = image_transforms

    def process(self, distances: Tensor, paths_query: List[Path], paths_gallery: List[Path]) -> Tensor:  # type: ignore
        n_queries = len(paths_query)
        n_galleries = len(paths_gallery)

        assert list(distances.shape) == [n_queries, n_galleries]

        # 1. Pick top n galleries for each query with the smallest distances
        top_n = min(self.top_n, n_galleries)
        ii_top = torch.topk(distances, k=top_n, largest=False)[1].view(-1)

        # 2. Create (n_queries * top_n) pairs of each query and related galleries and re-estimate distances for them
        paths_query = list(itertools.chain.from_iterable(itertools.repeat(x, self.top_n) for x in paths_query))
        paths_gallery = [paths_gallery[i] for i in ii_top]

        distances_upd = images_pairwise_inference(self.model, paths_query, paths_gallery, self.image_transforms)
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


__all__ = ["PairwiseImagesPostprocessor"]
