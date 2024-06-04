from typing import Tuple

import pytest
from torch import nn

from oml.const import MOCK_DATASET_PATH
from oml.datasets.images import ImageQueryGalleryLabeledDataset
from oml.inference import inference
from oml.interfaces.datasets import IQueryGalleryDataset
from oml.models.meta.siamese import TrivialDistanceSiamese
from oml.models.resnet.extractor import ResnetExtractor
from oml.retrieval.postprocessors.pairwise import PairwiseReranker
from oml.retrieval.retrieval_results import RetrievalResults
from oml.transforms.images.torchvision import get_normalisation_resize_torch
from oml.transforms.images.utils import TTransforms
from oml.utils.download_mock_dataset import download_mock_dataset
from tests.utils import check_if_sequence_of_tensors_are_equal


def get_validation_results(model: nn.Module, transforms: TTransforms) -> Tuple[RetrievalResults, IQueryGalleryDataset]:
    _, df_val = download_mock_dataset(MOCK_DATASET_PATH)

    dataset = ImageQueryGalleryLabeledDataset(df=df_val, transform=transforms, dataset_root=MOCK_DATASET_PATH)

    embeddings = inference(model, dataset, batch_size=4)

    rr = RetrievalResults.from_embeddings(embeddings.float(), dataset=dataset, n_items=100)

    return rr, dataset


@pytest.mark.long
@pytest.mark.parametrize("top_n", [2, 5, 100])
@pytest.mark.parametrize("pairwise_distances_bias", [0, -100, +100])
def test_trivial_processing_does_not_change_distances_order(top_n: int, pairwise_distances_bias: float) -> None:
    extractor = ResnetExtractor(weights=None, arch="resnet18", normalise_features=True, gem_p=None, remove_fc=True)
    pairwise_model = TrivialDistanceSiamese(extractor, output_bias=pairwise_distances_bias)

    transforms = get_normalisation_resize_torch(im_size=32)
    rr, dataset = get_validation_results(model=extractor, transforms=transforms)

    postprocessor = PairwiseReranker(
        top_n=top_n,
        pairwise_model=pairwise_model,
        num_workers=0,
        batch_size=4,
        verbose=False,
        use_fp16=True,
    )
    rr_upd = postprocessor.process(rr, dataset=dataset)

    assert check_if_sequence_of_tensors_are_equal(rr.retrieved_ids, rr_upd.retrieved_ids)

    if pairwise_distances_bias == 0:
        assert check_if_sequence_of_tensors_are_equal(rr_upd.distances, rr.distances)
    else:
        assert not check_if_sequence_of_tensors_are_equal(rr_upd.distances, rr.distances)
