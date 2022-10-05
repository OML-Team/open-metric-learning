import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from oml.const import (
    BBOXES_COLUMNS,
    CATEGORIES_COLUMN,
    IS_GALLERY_COLUMN,
    IS_QUERY_COLUMN,
    LABELS_COLUMN,
    OBLIGATORY_COLUMNS,
    PATHS_COLUMN,
    SPLIT_COLUMN,
    X1_COLUMN,
    X2_COLUMN,
    Y1_COLUMN,
    Y2_COLUMN,
)


def check_retrieval_dataframe_format(
    df: Union[Path, str, pd.DataFrame], dataset_root: Optional[Path] = None, sep: str = ",", verbose: bool = True
) -> None:
    """
    Function checks if the dataset is in the correct format.

    Args:
        df: Path to ``.csv`` file or pandas DataFrame
        dataset_root: Path to the dataset root, set ``None`` if you used absolute paths in your dataframe
        sep: Separator used in ``.csv``
        verbose: Set ``True`` if you want to see warnings

    """
    if isinstance(df, (Path, str)):
        df = pd.read_csv(df, sep=sep, index_col=None)

    assert all(x in df.columns for x in OBLIGATORY_COLUMNS), df.columns

    assert set(df[SPLIT_COLUMN]) == {"train", "validation"}, set(df[SPLIT_COLUMN])

    mask_train = df[SPLIT_COLUMN] == "train"
    assert pd.isna(df[IS_QUERY_COLUMN][mask_train].unique()[0]), df[IS_QUERY_COLUMN][mask_train].unique()
    assert pd.isna(df[IS_GALLERY_COLUMN][mask_train].unique()[0]), df[IS_GALLERY_COLUMN][mask_train].unique()

    val_mask = ~mask_train

    for split_field in [IS_QUERY_COLUMN, IS_GALLERY_COLUMN]:
        unq_values = set(df[split_field][val_mask])
        assert unq_values in [{False}, {True}, {False, True}]

    assert df[LABELS_COLUMN].dtypes == int

    assert all(((df[IS_QUERY_COLUMN][val_mask].astype(bool)) | df[IS_GALLERY_COLUMN][val_mask].astype(bool)).to_list())

    # we explicitly put ==True here because of Nones
    labels_query = set(df[LABELS_COLUMN][df[IS_QUERY_COLUMN] == True])  # noqa
    labels_gallery = set(df[LABELS_COLUMN][df[IS_GALLERY_COLUMN] == True])  # noqa
    assert labels_query.intersection(labels_gallery) == labels_query

    if dataset_root is None:
        dataset_root = Path("")

    assert all(df[PATHS_COLUMN].apply(lambda x: (dataset_root / x).exists()).to_list())

    # check bboxes if exist
    if set(BBOXES_COLUMNS).intersection(set(list(df.columns))):
        assert all(x in df.columns for x in BBOXES_COLUMNS), df.columns

        bboxes_columns = df[BBOXES_COLUMNS]

        # here we check that for one example bounding box consists of four None (no bounding box) or have 4
        # integers as corners (checking that we don't use float indexes for the array)
        assert np.all(
            np.logical_or(
                np.isnan(bboxes_columns.values).sum(axis=1) == 4,
                (np.mod(bboxes_columns.values, 1) == 0).sum(axis=1) == 4,
            )
        )

        bboxes_df = df[~(df[X1_COLUMN].isna())]
        mask_good_x1_x2 = bboxes_df[X1_COLUMN] < bboxes_df[X2_COLUMN]
        mask_good_y1_y2 = bboxes_df[Y1_COLUMN] < bboxes_df[Y2_COLUMN]
        n_bad_x1_x2 = (~mask_good_x1_x2).sum()
        n_bad_y1_y2 = (~mask_good_y1_y2).sum()
        assert not n_bad_x1_x2, f"Number of bad x1/x2 pairs {n_bad_x1_x2}"
        assert not n_bad_y1_y2, f"Number of bad y1/y2 pairs {n_bad_y1_y2}"
        for coord in BBOXES_COLUMNS:
            assert all((bboxes_df[coord] >= 0).to_list()), coord

    if CATEGORIES_COLUMN in df.columns:
        label_to_category = defaultdict(set)
        for _, row in df.iterrows():
            label_to_category[row[LABELS_COLUMN]].add(row[CATEGORIES_COLUMN])

        bad_categories = {k: v for k, v in label_to_category.items() if len(v) > 1}

        if bad_categories and verbose:
            warnings.warn(
                f"Note! You mapping between categories and labels is not bijection!"
                f"During the training and validation we will force it to be bijection by picking"
                f"one random category for each label."
                f"\n"
                f"{bad_categories}"
            )


__all__ = ["check_retrieval_dataframe_format"]
