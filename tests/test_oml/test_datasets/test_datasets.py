import matplotlib
import pandas as pd
import torch
from torch import LongTensor

from oml.const import (
    BLACK,
    CATEGORIES_COLUMN,
    IS_GALLERY_COLUMN,
    IS_QUERY_COLUMN,
    LABELS_COLUMN,
    PATHS_COLUMN,
    TEXTS_COLUMN,
)
from oml.datasets import (
    ImageBaseDataset,
    ImageLabeledDataset,
    ImageQueryGalleryDataset,
    ImageQueryGalleryLabeledDataset,
    TextBaseDataset,
    TextLabeledDataset,
    TextQueryGalleryDataset,
    TextQueryGalleryLabeledDataset,
)
from oml.interfaces.datasets import (
    IBaseDataset,
    ILabeledDataset,
    IQueryGalleryDataset,
    IVisualizableDataset,
)
from oml.utils import get_mock_images_dataset, get_mock_texts_dataset


class ASCITokenizer:
    @staticmethod
    def tokenize(text: str, *args, **kwargs):  # type: ignore
        ids = LongTensor(list(map(ord, text)))
        data = {"input_ids": ids, "attention_mask": torch.ones(len(ids)).long()}
        return data

    def __call__(self, text: str, *args, **kwargs):  # type: ignore
        return self.tokenize(text=text, *args, **kwargs)  # type: ignore


def check_base(dataset_b: IBaseDataset) -> None:
    item = dataset_b[0]

    assert dataset_b.index_key in item
    assert dataset_b.input_tensors_key in item


def check_labeled(dataset_l: ILabeledDataset, df: pd.DataFrame) -> None:
    item = dataset_l[0]
    assert dataset_l.labels_key in item

    # test label2category()
    labels = list(dataset_l.get_label2category().keys())
    categories = list(dataset_l.get_label2category().values())
    assert set(labels) == set(df[LABELS_COLUMN].tolist())
    assert set(categories) == set(df[CATEGORIES_COLUMN].tolist())

    # test get_labels()
    assert set(dataset_l.get_labels().tolist()) == set(df[LABELS_COLUMN].tolist())


def check_query_gallery(dataset_qg: IQueryGalleryDataset, df: pd.DataFrame) -> None:
    q_ids = dataset_qg.get_query_ids()
    g_ids = dataset_qg.get_gallery_ids()

    assert 0 <= q_ids.max() <= len(dataset_qg) - 1
    assert 0 <= g_ids.max() <= len(dataset_qg) - 1

    assert len(q_ids) <= len(dataset_qg)
    assert len(g_ids) <= len(dataset_qg)

    for iq in q_ids:
        assert df[IS_QUERY_COLUMN].iloc[int(iq)]

    for ig in g_ids:
        assert df[IS_GALLERY_COLUMN].iloc[int(ig)]


def check_visaulization(dataset_v: IVisualizableDataset) -> None:
    current_backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    _ = dataset_v.visualize(item=0, color=BLACK)
    matplotlib.use(current_backend)
    assert True


def test_text_datasets() -> None:
    df_train, df_val = get_mock_texts_dataset()
    tokenizer = ASCITokenizer()

    df = pd.concat([df_train, df_val])

    # Base
    dataset_b = TextBaseDataset(
        df[TEXTS_COLUMN].tolist(), tokenizer, extra_data={CATEGORIES_COLUMN: df[CATEGORIES_COLUMN]}
    )
    check_base(dataset_b)

    # Labeled
    dataset_l = TextLabeledDataset(df_train, tokenizer=tokenizer)
    check_base(dataset_l)
    check_visaulization(dataset_l)
    check_labeled(dataset_l, df_train)

    # Query Gallery
    dataset_qg = TextQueryGalleryDataset(df_val, tokenizer=tokenizer)
    check_base(dataset_qg)
    check_visaulization(dataset_qg)
    check_query_gallery(dataset_qg, df_val)

    # Query Gallery Labeled
    dataset_qgl = TextQueryGalleryLabeledDataset(df_val, tokenizer=tokenizer)
    check_base(dataset_qgl)
    check_visaulization(dataset_qgl)
    check_query_gallery(dataset_qgl, df_val)
    check_labeled(dataset_qgl, df_val)


def test_image_datasets() -> None:
    df_train, df_val = get_mock_images_dataset(df_name="df_with_category.csv", global_paths=True)
    df = pd.concat([df_train, df_val])

    # Base
    dataset_b = ImageBaseDataset(paths=df[PATHS_COLUMN].tolist(), extra_data={CATEGORIES_COLUMN: df[CATEGORIES_COLUMN]})
    check_base(dataset_b)

    # Labeled
    dataset_l = ImageLabeledDataset(df_train)
    check_base(dataset_l)
    check_labeled(dataset_l, df_train)

    # Query Gallery
    dataset_qg = ImageQueryGalleryDataset(df_val)
    check_base(dataset_qg)
    check_query_gallery(dataset_qg, df_val)

    # Query Gallery Labeled
    dataset_qgl = ImageQueryGalleryLabeledDataset(df_val)
    check_base(dataset_qgl)
    check_query_gallery(dataset_qgl, df_val)
    check_labeled(dataset_qgl, df_val)
