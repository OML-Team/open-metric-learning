from typing import Any, Dict, Tuple

import pytest
import torch
from torch import Tensor, nn

from oml.const import MOCK_DATASET_PATH
from oml.datasets.base import DatasetQueryGallery
from oml.inference.abstract import _inference
from oml.interfaces.models import IExtractor
from oml.models.meta.siamese import TrivialDistanceSiamese
from oml.models.resnet.extractor import ResnetExtractor
from oml.retrieval.postprocessors.pairwise import PairwiseReranker
from oml.transforms.images.torchvision import get_normalisation_resize_torch
from oml.utils.download_mock_dataset import download_mock_dataset
from oml.utils.misc_torch import pairwise_dist


@pytest.fixture
def validation_results() -> Tuple[Tensor, DatasetQueryGallery, IExtractor]:
    model = ResnetExtractor(weights=None, arch="resnet18", normalise_features=True, gem_p=None, remove_fc=True)

    _, df_val = download_mock_dataset(MOCK_DATASET_PATH)

    dataset = DatasetQueryGallery(
        df=df_val, dataset_root=MOCK_DATASET_PATH, transform=get_normalisation_resize_torch(im_size=32)
    )

    def _apply(model_: nn.Module, batch_: Dict[str, Any]) -> Tensor:
        return model_(batch_[dataset.input_tensors_key])

    embeddings = _inference(
        model=model,
        apply_model=_apply,
        dataset=dataset,
        num_workers=0,
        batch_size=4,
        verbose=False,
        use_fp16=True,
    )

    distances = pairwise_dist(x1=embeddings[dataset.get_query_mask()], x2=embeddings[dataset.get_gallery_mask()], p=2)

    return distances, dataset, model


@pytest.mark.long
@pytest.mark.parametrize("top_n,k", [(2, 2), (3, 4), (4, 3), (100, 3)])
def test_trivial_processing_does_not_change_distances_order(top_n: int, k, validation_results) -> None:  # type: ignore
    distances, dataset, extractor = validation_results

    pairwise_model = TrivialDistanceSiamese(extractor)

    # todo 522: use RetrievalPrediction here
    distances, retrieved_ids = torch.topk(distances, k=k, largest=False)

    postprocessor = PairwiseReranker(
        top_n=top_n,
        pairwise_model=pairwise_model,
        num_workers=0,
        batch_size=4,
        verbose=False,
        use_fp16=True,
    )
    distances_processed, retrieved_ids_upd = postprocessor.process(
        distances=distances, dataset=dataset, retrieved_ids=retrieved_ids
    )

    assert (retrieved_ids == retrieved_ids_upd).all()
    assert torch.isclose(distances, distances_processed).all()
