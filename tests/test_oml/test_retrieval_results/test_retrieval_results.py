from pathlib import Path

import pytest

from oml.const import (
    IS_GALLERY_COLUMN,
    IS_QUERY_COLUMN,
    LABELS_COLUMN,
    MOCK_DATASET_PATH,
    PATHS_COLUMN,
)
from oml.datasets.images import (
    ImageQueryGalleryDataset,
    ImageQueryGalleryLabeledDataset,
)
from oml.inference.flat import inference_on_images
from oml.models import ResnetExtractor
from oml.retrieval.retrieval_results import RetrievalResults
from oml.transforms.images.torchvision import get_normalisation_torch
from oml.utils.download_mock_dataset import download_mock_dataset


@pytest.mark.parametrize("with_gt_labels", [True, False])
@pytest.mark.parametrize("df_name", ["df.csv"])
def test_retrieval_results_om_images(with_gt_labels: bool, df_name: str) -> None:
    _, df_val = download_mock_dataset(dataset_root=MOCK_DATASET_PATH, df_name=df_name)
    df_val[PATHS_COLUMN] = df_val[PATHS_COLUMN].apply(lambda x: Path(MOCK_DATASET_PATH) / x)

    n_query, n_gallery = df_val[IS_QUERY_COLUMN].sum(), df_val[IS_GALLERY_COLUMN].sum()

    if with_gt_labels:
        dataset = ImageQueryGalleryLabeledDataset(df_val)
    else:
        del df_val[LABELS_COLUMN]
        dataset = ImageQueryGalleryDataset(df_val)

    model = ResnetExtractor(weights=None, arch="resnet18", gem_p=None, remove_fc=True, normalise_features=False)
    embeddings = inference_on_images(
        model=model,
        paths=df_val[PATHS_COLUMN].tolist(),
        transform=get_normalisation_torch(),
        num_workers=0,
        batch_size=4,
    ).float()

    top_n = 2
    retrieval_results = RetrievalResults.compute_from_embeddings(
        embeddings=embeddings, dataset=dataset, n_items_to_retrieve=top_n
    )

    assert retrieval_results.distances.shape == (n_query, top_n)
    assert retrieval_results.retrieved_ids.shape == (n_query, top_n)

    if with_gt_labels:
        assert retrieval_results.gt_ids is not None

    fig = retrieval_results.visualize(query_ids=[0, 3], dataset=dataset, n_galleries_to_show=3)
    print(dir(fig))
    print(n_gallery)
