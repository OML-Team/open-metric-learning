import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from oml.const import MOCK_DATASET_PATH, SEQUENCE_COLUMN
from oml.datasets.base import DatasetQueryGallery
from oml.metrics.embeddings import EmbeddingMetrics, TMetricsDict_ByLabels
from oml.utils.download_mock_dataset import download_mock_dataset
from oml.utils.misc import compare_dicts_recursively, set_global_seed


def validation(df: pd.DataFrame) -> TMetricsDict_ByLabels:
    set_global_seed(42)
    extractor = nn.Flatten()

    val_dataset = DatasetQueryGallery(df, dataset_root=MOCK_DATASET_PATH)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, num_workers=0)
    calculator = EmbeddingMetrics(extra_keys=("paths",), sequence_key=val_dataset.sequence_key, cmc_top_k=(1,))
    calculator.setup(num_samples=len(val_dataset))

    with torch.no_grad():
        for batch in tqdm(val_loader):
            batch["embeddings"] = extractor(batch["input_tensors"])
            calculator.update_data(batch)

    metrics = calculator.compute_metrics()

    return metrics


def test_invariants_in_validation_with_sequences_1() -> None:
    # We check that metrics don't change if we assign unique sequence id
    # to every sample in validation set (so, ignoring logic is not applicable)

    _, df = download_mock_dataset(MOCK_DATASET_PATH)

    df_with_seq = df.copy()
    df_with_seq[SEQUENCE_COLUMN] = list(range(len(df_with_seq)))

    metrics = validation(df)
    metrics_with_sequence = validation(df_with_seq)

    assert compare_dicts_recursively(metrics_with_sequence, metrics)


def test_invariants_in_validation_with_sequences_2() -> None:
    # We check that metrics don't change in the case, when we put
    # a copy of every sample to gallery under the same sequence id

    _, df = download_mock_dataset(MOCK_DATASET_PATH)

    df_a = df.copy()
    df_a[SEQUENCE_COLUMN] = list(range(len(df_a)))
    df_a["is_query"] = True
    df_a["is_gallery"] = True

    df_b_1 = df_a.copy()
    df_b_1["is_query"] = True
    df_b_1["is_gallery"] = False
    df_b_2 = df_a.copy()
    df_b_2["is_query"] = False
    df_b_2["is_gallery"] = True
    df_b = pd.concat([df_b_1, df_b_2])
    df_b = df_b.reset_index(drop=True)

    metrics_a = validation(df_a)
    metrics_b = validation(df_b)

    assert compare_dicts_recursively(metrics_a, metrics_b)
