from cProfile import label
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from ..functional.metrics import calc_gt_mask, calc_mask_to_ignore, validate_dataset

REQUIRED_FIELDS = ["label", "path", "split", "is_query", "is_gallery"]
BBOXES_FIELDS = ["x_1", "x_2", "y_1", "y_2"]


def check_retrieval_dataframe_format(
    df: Union[Path, str, pd.DataFrame], dataset_root: Optional[Path] = None, sep: str = ","
) -> None:
    """
    Function checks if the data is in the correct format.

    Args:
        df: Path to .csv file or pandas DataFrame
        dataset_root: Path to the dataset root
        sep: Separator used in .csv

    """
    if isinstance(df, (Path, str)):
        df = pd.read_csv(df, sep=sep, index_col=None)

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

    isq = df["is_query"][val_mask].astype(bool).values
    isg = df["is_gallery"][val_mask].astype(bool).values
    labels = df["label"][val_mask].values
    mask_gt = calc_gt_mask(labels=labels, is_query=isq, is_gallery=isg)
    mask_to_ignore = calc_mask_to_ignore(is_query=isq, is_gallery=isg)
    validate_dataset(mask_gt=mask_gt, mask_to_ignore=mask_to_ignore)

    if dataset_root is None:
        dataset_root = Path("")

    assert all(df["path"].apply(lambda x: (dataset_root / x).exists()).to_list())

    # check bboxes if exist
    if set(BBOXES_FIELDS).intersection(set(list(df.columns))):
        assert all(x in df.columns for x in BBOXES_FIELDS), df.columns

        bboxes_columns = df[BBOXES_FIELDS]

        # here we check that for one example bounding box consists of four None (no bounding box) or have 4
        # integers as corners (checking that we don't use float indexes for the array)
        assert np.all(
            np.logical_or(
                np.isnan(bboxes_columns.values).sum(axis=1) == 4,
                (np.mod(bboxes_columns.values, 1) == 0).sum(axis=1) == 4,
            )
        )

        bboxes_df = df[~(df["x_1"].isna())]
        assert all((bboxes_df["x_1"] < bboxes_df["x_2"]).to_list())
        assert all((bboxes_df["y_1"] < bboxes_df["y_2"]).to_list())
        for coord in BBOXES_FIELDS:
            assert all((bboxes_df[coord] >= 0).to_list()), coord

    # check categories format
    if ("category" in df.columns) and ("category_name" in df.columns):
        assert len(df["category"].unique()) == len(
            df["category_name"].unique()
        ), "Amount of unique categories and their names are not equal"

        assert df["category"].dtypes == int, "Category have to be int dtype"


__all__ = ["check_retrieval_dataframe_format"]
