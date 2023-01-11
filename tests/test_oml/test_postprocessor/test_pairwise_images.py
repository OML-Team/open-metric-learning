from random import sample

import torch

from oml.const import MOCK_DATASET_PATH
from oml.models.siamese import ImagesSiamese
from oml.postprocessors.pairwise_images import PairwiseImagesPostprocessor
from oml.transforms.images.torchvision.transforms import get_normalisation_torch
from oml.utils.download_mock_dataset import download_mock_dataset


def test_pairwise_images_processor_runs() -> None:
    download_mock_dataset(MOCK_DATASET_PATH)

    paths = list(MOCK_DATASET_PATH.glob("**/*.jpg"))
    paths_query = sample(paths, len(paths) // 2)
    paths_gallery = list(set(paths) - set(paths_query))

    distances = torch.randint(high=100, size=(len(paths_query), len(paths_gallery))).float()

    model = ImagesSiamese()
    processor = PairwiseImagesPostprocessor(model, top_n=5, image_transforms=get_normalisation_torch())

    distances_upd = processor.process(distances, paths_query, paths_gallery)
    assert distances.shape == distances_upd.shape
