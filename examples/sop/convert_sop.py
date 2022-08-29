from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from oml.const import (
    CATEGORY_COLUMN,
    IS_GALLERY_COLUMN,
    IS_QUERY_COLUMN,
    PATH_COLUMN,
    SPLIT_COLUMN,
)
from oml.utils.dataframe_format import check_retrieval_dataframe_format


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=Path)
    return parser


def build_sop_df(dataset_root: Path) -> pd.DataFrame:
    dataset_root = Path(dataset_root)

    ebay_train = dataset_root / "Ebay_train.txt"
    ebay_test = dataset_root / "Ebay_test.txt"

    for file in [ebay_train, ebay_test]:
        assert file.is_file(), f"File {file} does not exist."

    train_data = pd.read_csv(ebay_train, sep=" ")
    test_data = pd.read_csv(ebay_test, sep=" ")

    col_map = {"class_id": "label"}
    train_data = train_data.rename(columns=col_map)
    test_data = test_data.rename(columns=col_map)

    train_data[SPLIT_COLUMN] = "train"
    test_data[SPLIT_COLUMN] = "validation"
    train_data[IS_QUERY_COLUMN] = None
    train_data[IS_GALLERY_COLUMN] = None

    test_data[IS_QUERY_COLUMN] = True
    test_data[IS_GALLERY_COLUMN] = True

    df = pd.concat((train_data, test_data))
    df[PATH_COLUMN] = df[PATH_COLUMN].apply(lambda x: dataset_root / x)
    df[CATEGORY_COLUMN] = df[PATH_COLUMN].apply(lambda x: x.parent.name)

    check_retrieval_dataframe_format(df, dataset_root=dataset_root)
    return df


def main() -> None:
    print("SOP dataset preparation started...")
    args = get_argparser().parse_args()
    df = build_sop_df(args.dataset_root)
    df.to_csv(args.dataset_root / "df.csv", index=None)
    print("SOP dataset preparation completed.")
    print(f"DataFrame saved in {args.dataset_root}\n")


if __name__ == "__main__":
    main()
