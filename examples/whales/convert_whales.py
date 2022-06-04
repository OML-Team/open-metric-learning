from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from random import sample

import pandas as pd

from oml.utils.dataframe_format import check_retrieval_dataframe_format
from oml.utils.misc import set_global_seed


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=Path)
    return parser


def build_whales(dataset_root: Path) -> pd.DataFrame:
    set_global_seed(42)

    assert (dataset_root / "meta.csv").is_file()

    df = pd.read_csv(dataset_root / "meta.csv")
    df["label"] = df["label"].apply(lambda x: int(x.replace("whale", "")))

    val_labels = sample(list(set(df.label)), 100)
    val_mask = df["label"].isin(val_labels)

    df["split"] = "train"
    df["split"][val_mask] = "validation"

    df["is_query"] = None
    df["is_gallery"] = None

    df["is_query"][val_mask] = True
    df["is_gallery"][val_mask] = True

    labels_good = [label for label, count in Counter(df["label"]).items() if count > 1]  # type: ignore
    df = df[df["label"].isin(labels_good)]

    check_retrieval_dataframe_format(df, dataset_root=dataset_root)

    return df


def main() -> None:
    print("Whales dataset preparation started...")
    args = get_argparser().parse_args()
    df = build_whales(args.dataset_root)
    df.to_csv(args.dataset_root / "df.csv", index=None)
    print("Whales dataset preparation completed.")
    print(f"DataFrame saved in {args.dataset_root}\n")


if __name__ == "__main__":
    main()
