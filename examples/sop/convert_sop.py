from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from oml.utils.dataframe_format import check_retrieval_dataframe_format


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=Path)
    return parser


def build_sop(dataset_root: Path) -> pd.DataFrame:
    ebay_train = dataset_root / "Ebay_train.txt"
    ebay_test = dataset_root / "Ebay_test.txt"

    for file in [ebay_train, ebay_test]:
        assert file.is_file(), f"File {file} does not exist."

    train_data = pd.read_csv(ebay_train, sep=" ")
    test_data = pd.read_csv(ebay_test, sep=" ")

    col_map = {"class_id": "label"}
    train_data = train_data.rename(columns=col_map)
    test_data = test_data.rename(columns=col_map)

    train_data["split"] = "train"
    test_data["split"] = "validation"
    train_data["is_query"] = None
    train_data["is_gallery"] = None

    test_data["is_query"] = True
    test_data["is_gallery"] = True

    df = pd.concat((train_data, test_data))
    df = df[["label", "path", "split", "is_query", "is_gallery"]]

    check_retrieval_dataframe_format(df, dataset_root=dataset_root)
    return df


def main() -> None:
    print("SOP dataset preparation started...")
    args = get_argparser().parse_args()
    df = build_sop(args.dataset_root)
    df.to_csv(args.dataset_root / "df.csv", index=None)
    print("SOP dataset preparation completed.")
    print(f"DataFrame saved in {args.dataset_root}\n")


if __name__ == "__main__":
    main()
