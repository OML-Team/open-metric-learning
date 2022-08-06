from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple, Union

import gdown
import pandas as pd

from oml.const import MOCK_DATASET_URL, MOCK_DATASET_PATH


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=MOCK_DATASET_PATH)
    return parser


def download_mock_dataset(dataset_root: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset_root = Path(dataset_root)

    files_exist = [(dataset_root / "df.csv").exists()]
    for im in ["rectangle", "circle", "triangle", "cross"]:
        for i in range(1, 4):
            files_exist.append((dataset_root / "images" / f"{im}_{i}.jpg").exists())

    if not all(files_exist):
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


__all__ = ["download_mock_dataset"]

if __name__ == "__main__":
    main()
