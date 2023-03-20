from typing import Tuple

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from oml.const import MOCK_DATASET_PATH
from oml.inference.flat import inference_on_images
from oml.models.meta.siamese import TrivialDistanceSiamese
from oml.models.resnet import ResnetExtractor
from oml.retrieval.postprocessors.pairwise import PairwiseImagesPostprocessor
from oml.transforms.images.torchvision.transforms import get_normalisation_resize_torch
from oml.transforms.images.utils import TTransforms
from oml.utils.download_mock_dataset import download_mock_dataset
from oml.utils.misc_torch import pairwise_dist


def get_validation_results(model: nn.Module, transforms: TTransforms) -> Tuple[Tensor, Tensor, Tensor]:
    _, df_val = download_mock_dataset(MOCK_DATASET_PATH)
    is_query = np.array(df_val["is_query"]).astype(bool)
    is_gallery = np.array(df_val["is_gallery"]).astype(bool)

    paths = np.array(df_val["path"].apply(lambda x: MOCK_DATASET_PATH / x))
    queries = paths[is_query]
    galleries = paths[is_gallery]

    embeddings = inference_on_images(
        model=model,
        paths=paths.tolist(),
        transform=transforms,
        num_workers=0,
        batch_size=4,
        verbose=False,
        use_fp16=True,
    )

    distances = pairwise_dist(x1=embeddings[is_query], x2=embeddings[is_gallery], p=2)

    return distances, queries, galleries


@pytest.mark.parametrize("top_n", [2, 5, 100])
def test_trivial_processing_does_not_change_distances_order(top_n: int) -> None:
    extractor = ResnetExtractor(weights=None, arch="resnet18", normalise_features=True, gem_p=None, remove_fc=True)

    pairwise_model = TrivialDistanceSiamese(extractor)

    transforms = get_normalisation_resize_torch(im_size=32)

    distances, queries, galleries = get_validation_results(model=extractor, transforms=transforms)

    postprocessor = PairwiseImagesPostprocessor(
        top_n=top_n,
        pairwise_model=pairwise_model,
        transforms=transforms,
        num_workers=0,
        batch_size=4,
        verbose=False,
        use_fp16=True,
    )
    distances_processed = postprocessor.process(distances=distances.clone(), queries=queries, galleries=galleries)

    order = distances.argsort()
    order_processed = distances_processed.argsort()

    assert (order == order_processed).all(), (order, order_processed)

    if top_n <= len(galleries):
        min_orig_distances = torch.topk(distances, k=top_n, largest=False).values
        min_processed_distances = torch.topk(distances_processed, k=top_n, largest=False).values
        assert torch.allclose(min_orig_distances, min_processed_distances)
