from pathlib import Path
from typing import Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from oml.const import (
    BBOXES_COLUMNS,
    IS_GALLERY_COLUMN,
    IS_QUERY_COLUMN,
    LABELS_COLUMN,
    MOCK_DATASET_PATH,
    OBLIGATORY_COLUMNS,
    SPLIT_COLUMN,
    X1_COLUMN,
    X2_COLUMN,
    Y1_COLUMN,
    Y2_COLUMN,
)
from oml.utils.dataframe_format import check_retrieval_dataframe_format
from oml.utils.download_mock_dataset import download_mock_dataset

TDFrames = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


@pytest.fixture()
def mock_dfs() -> TDFrames:
    df_train, df_val = download_mock_dataset(MOCK_DATASET_PATH)
    df = pd.concat([df_train, df_val], ignore_index=True).reset_index(drop=True)
    return df_train, df_val, df


@pytest.fixture()
def mock_with_bboxes() -> pd.DataFrame:
    download_mock_dataset(MOCK_DATASET_PATH)
    return pd.read_csv(MOCK_DATASET_PATH / "df_with_bboxes.csv")


def test_mock_df_is_valid(mock_dfs: TDFrames) -> None:
    _, _, df = mock_dfs
    check_retrieval_dataframe_format(df, dataset_root=MOCK_DATASET_PATH)


def test_independent_splits(mock_dfs: TDFrames) -> None:
    df_train, df_val, _ = mock_dfs
    check_retrieval_dataframe_format(df_train, dataset_root=MOCK_DATASET_PATH)
    check_retrieval_dataframe_format(df_val, dataset_root=MOCK_DATASET_PATH)


def test_no_obligatory_cols(mock_dfs: TDFrames) -> None:
    _, _, df = mock_dfs
    for col in OBLIGATORY_COLUMNS:
        df_copy = df.copy()
        df_copy.drop(columns=[col], inplace=True)
        with pytest.raises(AssertionError):
            check_retrieval_dataframe_format(df_copy, dataset_root=MOCK_DATASET_PATH)


def test_wrong_splits(mock_dfs: TDFrames) -> None:
    _, _, df = mock_dfs
    df = df.copy()
    df.loc[0, SPLIT_COLUMN] = str(uuid4())
    with pytest.raises(AssertionError):
        check_retrieval_dataframe_format(df, dataset_root=MOCK_DATASET_PATH)


def test_train_qg_not_none(mock_dfs: TDFrames) -> None:
    df_train, _, _ = mock_dfs

    df = df_train.copy()
    df.loc[0, IS_QUERY_COLUMN] = 1
    with pytest.raises(AssertionError):
        check_retrieval_dataframe_format(df, dataset_root=MOCK_DATASET_PATH)

    df = df_train.copy()
    df.loc[0, IS_GALLERY_COLUMN] = 2
    with pytest.raises(AssertionError):
        check_retrieval_dataframe_format(df, dataset_root=MOCK_DATASET_PATH)


def test_val_qg_only_bool(mock_dfs: TDFrames) -> None:
    _, df_val, _ = mock_dfs

    df = df_val.copy()
    df.loc[0, IS_QUERY_COLUMN] = "asd"
    with pytest.raises(AssertionError):
        check_retrieval_dataframe_format(df, dataset_root=MOCK_DATASET_PATH)

    df = df_val.copy()
    df.loc[0, IS_GALLERY_COLUMN] = 123
    with pytest.raises(AssertionError):
        check_retrieval_dataframe_format(df, dataset_root=MOCK_DATASET_PATH)

    df = df_val.copy()
    df.loc[0, IS_GALLERY_COLUMN] = None
    with pytest.raises(AssertionError):
        check_retrieval_dataframe_format(df, dataset_root=MOCK_DATASET_PATH)


def test_q_without_g(mock_dfs: TDFrames) -> None:
    _, df_val, _ = mock_dfs

    df = df_val.copy()
    first_q_row = np.arange(len(df))[df[IS_QUERY_COLUMN] == True][0]  # noqa
    df.loc[first_q_row, LABELS_COLUMN] = max(df[LABELS_COLUMN]) + 1
    with pytest.raises(AssertionError):
        check_retrieval_dataframe_format(df, dataset_root=MOCK_DATASET_PATH)


def test_images_not_found(mock_dfs: TDFrames) -> None:
    _, _, df = mock_dfs
    with pytest.raises(AssertionError):
        check_retrieval_dataframe_format(df, dataset_root=Path(str(uuid4())))


def test_not_enough_bbox_columns(mock_with_bboxes: pd.DataFrame) -> None:
    for col in BBOXES_COLUMNS:
        df = mock_with_bboxes.copy()
        df.drop(columns=[col], inplace=True)
        with pytest.raises(AssertionError):
            check_retrieval_dataframe_format(df, dataset_root=MOCK_DATASET_PATH)


def test_all_coords_none(mock_with_bboxes: pd.DataFrame) -> None:
    df = mock_with_bboxes.copy()
    for col in BBOXES_COLUMNS:
        df[col] = np.nan
    check_retrieval_dataframe_format(df, dataset_root=MOCK_DATASET_PATH)


def test_all_coords_not_none(mock_with_bboxes: pd.DataFrame) -> None:
    for col in BBOXES_COLUMNS:
        df = mock_with_bboxes.copy()
        df[col] = np.nan
        with pytest.raises(AssertionError):
            check_retrieval_dataframe_format(df, dataset_root=MOCK_DATASET_PATH)


def test_not_empty_bbox(mock_with_bboxes: pd.DataFrame) -> None:
    df = mock_with_bboxes.copy()
    df.loc[0, X1_COLUMN] = df.loc[0, X2_COLUMN]
    with pytest.raises(AssertionError):
        check_retrieval_dataframe_format(df, dataset_root=MOCK_DATASET_PATH)

    df = mock_with_bboxes.copy()
    df.loc[1, Y1_COLUMN] = df.loc[1, Y2_COLUMN] + 1
    with pytest.raises(AssertionError):
        check_retrieval_dataframe_format(df, dataset_root=MOCK_DATASET_PATH)


def test_not_negative_coords(mock_with_bboxes: pd.DataFrame) -> None:
    for idx, col in enumerate(BBOXES_COLUMNS):
        df = mock_with_bboxes.copy()
        df.loc[idx, col] = -1
        with pytest.raises(AssertionError):
            check_retrieval_dataframe_format(df, dataset_root=MOCK_DATASET_PATH)
