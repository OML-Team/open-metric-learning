from typing import Tuple

import pytest
import torch
from torch import Tensor, nn

from oml.const import MOCK_DATASET_PATH
from oml.datasets.images import ImageQueryGalleryLabeledDataset
from oml.inference import inference
from oml.interfaces.datasets import IQueryGalleryDataset
from oml.models.meta.siamese import TrivialDistanceSiamese
from oml.models.resnet.extractor import ResnetExtractor
from oml.retrieval.postprocessors.pairwise import PairwiseReranker
from oml.transforms.images.torchvision import get_normalisation_resize_torch
from oml.transforms.images.utils import TTransforms
from oml.utils.download_mock_dataset import download_mock_dataset
from oml.utils.misc_torch import pairwise_dist


def get_validation_results(model: nn.Module, transforms: TTransforms) -> Tuple[Tensor, IQueryGalleryDataset]:
    _, df_val = download_mock_dataset(MOCK_DATASET_PATH)

    dataset = ImageQueryGalleryLabeledDataset(df=df_val, transform=transforms, dataset_root=MOCK_DATASET_PATH)

    embeddings = inference(model, dataset, batch_size=4)

    distances = pairwise_dist(x1=embeddings[dataset.get_query_ids()], x2=embeddings[dataset.get_gallery_ids()], p=2)

    return distances, dataset


@pytest.mark.long
@pytest.mark.parametrize("top_n", [2, 5, 100])
@pytest.mark.parametrize("pairwise_distances_bias", [0, -100, +100])
def test_trivial_processing_does_not_change_distances_order(top_n: int, pairwise_distances_bias: float) -> None:
    extractor = ResnetExtractor(weights=None, arch="resnet18", normalise_features=True, gem_p=None, remove_fc=True)
    pairwise_model = TrivialDistanceSiamese(extractor, output_bias=pairwise_distances_bias)

    transforms = get_normalisation_resize_torch(im_size=32)
    distances, dataset = get_validation_results(model=extractor, transforms=transforms)

    postprocessor = PairwiseReranker(
        top_n=top_n,
        pairwise_model=pairwise_model,
        num_workers=0,
        batch_size=4,
        verbose=False,
        use_fp16=True,
    )
    distances_processed = postprocessor.process(distances=distances.clone(), dataset=dataset)

    assert (distances_processed.argsort() == distances.argsort()).all()

    if pairwise_distances_bias == 0:
        assert torch.allclose(distances_processed, distances)
    else:
        assert not torch.allclose(distances_processed, distances)
