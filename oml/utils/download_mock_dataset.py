from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple, Union

import gdown
import pandas as pd

from oml.const import MOCK_DATASET_PATH, MOCK_DATASET_URL


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=MOCK_DATASET_PATH)
    return parser


def check_mock_dataset_exists(dataset_root: Union[str, Path]) -> bool:
    dataset_root = Path(dataset_root)
    files_exist = [(dataset_root / "df.csv").exists()]
    for im in ["rectangle", "circle", "triangle", "cross"]:
        for i in range(1, 4):
            files_exist.append((dataset_root / "images" / f"{im}_{i}.jpg").exists())
    return all(files_exist)


def download_mock_dataset(dataset_root: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset_root = Path(dataset_root)

    if not check_mock_dataset_exists(dataset_root):
        gdown.download_folder(url=MOCK_DATASET_URL, output=str(dataset_root))
    else:
        print("Mock dataset has been downloaded already.")

    df = pd.read_csv(Path(dataset_root) / "df.csv")

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "validation"].reset_index(drop=True)

    return df_train, df_val


def main() -> None:
    args = get_argparser().parse_args()
    download_mock_dataset(dataset_root=args.dataset_root)


__all__ = ["download_mock_dataset", "check_mock_dataset_exists"]

if __name__ == "__main__":
    main()
