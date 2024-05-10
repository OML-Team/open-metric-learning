from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
from torch import LongTensor, nn

from oml.const import LABELS_COLUMN, MOCK_DATASET_PATH, PATHS_COLUMN
from oml.datasets.images import (
    ImageQueryGalleryDataset,
    ImageQueryGalleryLabeledDataset,
)
from oml.inference import inference
from oml.interfaces.datasets import IVisualizableDataset
from oml.models import ResnetExtractor
from oml.retrieval.retrieval_results import RetrievalResults
from oml.utils.download_mock_dataset import download_mock_dataset
from tests.test_integrations.utils import (
    EmbeddingsQueryGalleryDataset,
    EmbeddingsQueryGalleryLabeledDataset,
)


def get_model_and_datasets_images(with_gt_labels):  # type: ignore
    datasets = []

    for df_name in ["df.csv", "df_with_bboxes.csv", "df_with_sequence.csv"]:
        _, df_val = download_mock_dataset(global_paths=True, df_name=df_name)
        df_val[PATHS_COLUMN] = df_val[PATHS_COLUMN].apply(lambda x: Path(MOCK_DATASET_PATH) / x)

        if with_gt_labels:
            dataset = ImageQueryGalleryLabeledDataset(df_val)
        else:
            del df_val[LABELS_COLUMN]
            dataset = ImageQueryGalleryDataset(df_val)

        datasets.append(dataset)

    model = ResnetExtractor(weights=None, arch="resnet18", gem_p=None, remove_fc=True, normalise_features=False)

    return datasets, model


def get_model_and_datasets_embeddings(with_gt_labels):  # type: ignore
    embeddings = torch.randn((6, 4)).float()
    is_query = torch.tensor([1, 1, 1, 0, 0, 0]).bool()
    is_gallery = torch.tensor([0, 0, 0, 1, 1, 1]).bool()

    if with_gt_labels:
        labels = torch.tensor([0, 1, 0, 1, 0, 1]).long()
        dataset = EmbeddingsQueryGalleryLabeledDataset(
            embeddings=embeddings, labels=labels, is_query=is_query, is_gallery=is_gallery
        )
    else:
        dataset = EmbeddingsQueryGalleryDataset(embeddings=embeddings, is_query=is_query, is_gallery=is_gallery)

    model = nn.Linear(4, 1)

    return [dataset], model


@pytest.mark.parametrize("with_gt_labels", [False, True])
@pytest.mark.parametrize("data_getter", [get_model_and_datasets_embeddings, get_model_and_datasets_images])
def test_retrieval_results_om_images(with_gt_labels, data_getter) -> None:  # type: ignore
    datasets, model = data_getter(with_gt_labels=with_gt_labels)

    for dataset in datasets:

        n_query = len(dataset.get_query_ids())

        embeddings = inference(model=model, dataset=dataset, num_workers=0, batch_size=4).float()

        top_n = 2
        rr = RetrievalResults.compute_from_embeddings(embeddings=embeddings, dataset=dataset, n_items_to_retrieve=top_n)

        assert rr.distances.shape == (n_query, top_n)
        assert rr.retrieved_ids.shape == (n_query, top_n)
        assert torch.allclose(rr.distances.clone().sort()[0], rr.distances)

        if with_gt_labels:
            assert rr.gt_ids is not None

        error_expected = not isinstance(dataset, IVisualizableDataset)
        if error_expected:
            with pytest.raises(TypeError):
                fig = rr.visualize(query_ids=[0, 3], dataset=dataset, n_galleries_to_show=3)
                fig.show()
                plt.close(fig=fig)
        else:
            fig = rr.visualize(query_ids=[0, 3], dataset=dataset, n_galleries_to_show=3)
            fig.show()
            plt.close(fig=fig)

    assert True


def test_retrieval_results_creation() -> None:
    with pytest.raises(RuntimeError):
        RetrievalResults(
            distances=torch.randn((2, 3)).float(),
            retrieved_ids=LongTensor([[1, 0, 2], [4, 0, 1]]),
            gt_ids=[LongTensor([0, 1, 3]), []],
        )
