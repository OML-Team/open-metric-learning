from pathlib import Path
from typing import Union

import pandas as pd

REQUIRED_FIELDS = ["label", "path", "split", "is_query", "is_gallery"]
BBOXES_FIELDS = ["x_1", "x_2", "y_1", "y_2"]


def check_retrieval_dataframe_format(df: Union[Path, str, pd.DataFrame], dataset_root: Path) -> None:
    if isinstance(df, (Path, str)):
        df = pd.read_csv(df, index_col=None)

    assert all(x in df.columns for x in REQUIRED_FIELDS), df.columns

    assert set(df["split"]) == {"train", "validation"}, set(df["split"])

    mask_train = df["split"] == "train"
    assert pd.isna(df["is_query"][mask_train].unique()[0]), df["is_query"][mask_train].unique()
    assert pd.isna(df["is_gallery"][mask_train].unique()[0]), df["is_gallery"][mask_train].unique()

    val_mask = ~mask_train

    for split_field in ["is_query", "is_gallery"]:
        unq_values = set(df[split_field][val_mask])
        assert unq_values in [{False}, {True}, {False, True}]

    assert df["label"].dtypes == int

    assert all(((df["is_query"][val_mask].astype(bool)) | df["is_gallery"][val_mask].astype(bool)).to_list())

    # we explicitly put ==True here because of Nones
    labels_query = set(df["label"][df["is_query"] == True])  # noqa
    labels_gallery = set(df["label"][df["is_gallery"] == True])  # noqa
    assert labels_query.intersection(labels_gallery) == labels_query

    assert all(df["path"].apply(lambda x: (dataset_root / x).exists()).to_list())

    # check bboxes if exist
    if set(BBOXES_FIELDS).intersection(set(list(df.columns))):
        assert all(x in df.columns for x in BBOXES_FIELDS), df.columns

        assert all((df["x_1"] < df["x_2"]).to_list())
        assert all((df["y_1"] < df["y_2"]).to_list())
        for coord in BBOXES_FIELDS:
            assert all((df[coord] >= 0).to_list()), coord
