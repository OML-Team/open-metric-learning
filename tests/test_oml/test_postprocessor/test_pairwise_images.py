from typing import Tuple

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from oml.const import MOCK_DATASET_PATH
from oml.inference.base import inference_on_images
from oml.models.siamese import ResNetSiamese
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
        model=model, paths=paths.tolist(), transform=transforms, num_workers=0, batch_size=4, verbose=False
    )

    distances = pairwise_dist(x1=embeddings[is_query], x2=embeddings[is_gallery], p=2)

    return distances, queries, galleries


@pytest.mark.parametrize("top_n", [2, 5])
def test_trivial_processing_does_not_change_distances(top_n: int) -> None:
    pairwise_model = ResNetSiamese(pretrained=False)

    embedder = pairwise_model.backbone
    transforms = get_normalisation_resize_torch(im_size=32)

    distances, queries, galleries = get_validation_results(model=embedder, transforms=transforms)

    postprocessor = PairwiseImagesPostprocessor(
        top_n=top_n, pairwise_model=pairwise_model, transforms=transforms, num_workers=0, batch_size=4, verbose=False
    )
    distances_processed = postprocessor.process(distances=distances, queries=queries, galleries=galleries)

    assert torch.allclose(distances, distances_processed)
