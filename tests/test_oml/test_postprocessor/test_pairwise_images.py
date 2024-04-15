from typing import Any, Dict, Tuple

import pytest
import torch
from torch import FloatTensor, Tensor, nn

from oml.const import MOCK_DATASET_PATH
from oml.datasets import ImagesDatasetQueryGallery
from oml.inference.abstract import _inference
from oml.interfaces.datasets import IDatasetQueryGallery
from oml.interfaces.models import IExtractor
from oml.models.meta.siamese import TrivialDistanceSiamese
from oml.models.resnet.extractor import ResnetExtractor
from oml.retrieval.postprocessors.pairwise import PairwiseReranker
from oml.retrieval.prediction import RetrievalPrediction
from oml.transforms.images.torchvision import get_normalisation_resize_torch
from oml.utils.download_mock_dataset import download_mock_dataset


@pytest.fixture
def validation_results() -> Tuple[FloatTensor, IDatasetQueryGallery, IExtractor]:
    model = ResnetExtractor(weights=None, arch="resnet18", normalise_features=True, gem_p=None, remove_fc=True)

    _, df_val = download_mock_dataset(MOCK_DATASET_PATH)

    dataset = ImagesDatasetQueryGallery(
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
    ).float()

    return embeddings, dataset, model


@pytest.mark.long
@pytest.mark.parametrize("top_n,k", [(2, 2), (3, 4), (4, 3), (100, 3)])
def test_trivial_processing_does_not_change_distances_order(top_n: int, k, validation_results) -> None:  # type: ignore
    embeddings, dataset, extractor = validation_results

    prediction = RetrievalPrediction.compute_from_embeddings(
        embeddings=embeddings, dataset=dataset, n_ids_to_retrieve=k
    )

    pairwise_model = TrivialDistanceSiamese(extractor)

    postprocessor = PairwiseReranker(
        top_n=top_n,
        pairwise_model=pairwise_model,
        num_workers=0,
        batch_size=4,
        verbose=False,
        use_fp16=True,
    )
    prediction_upd = postprocessor.process(prediction=prediction, dataset=dataset)

    assert (prediction.retrieved_ids == prediction_upd.retrieved_ids).all()
    assert torch.isclose(prediction.distances, prediction_upd.distances).all()
