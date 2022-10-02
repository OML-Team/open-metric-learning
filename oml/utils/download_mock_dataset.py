from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple, Union

import gdown
import pandas as pd

from oml.const import MOCK_DATASET_MD5, MOCK_DATASET_PATH, MOCK_DATASET_URL_GDRIVE
from oml.utils.io import calc_folder_hash, download_folder_from_remote_storage


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=MOCK_DATASET_PATH)
    return parser


def check_mock_dataset_exists(dataset_root: Union[str, Path]) -> bool:
    return Path(dataset_root).exists() and calc_folder_hash(dataset_root) == MOCK_DATASET_MD5


def download_mock_dataset(dataset_root: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to download mock dataset which is already prepared in the required format.

    Args:
        dataset_root: Path to save the dataset

    Returns: Dataframes for the training and validation stages

    """
    dataset_root = Path(dataset_root)

    if not check_mock_dataset_exists(dataset_root):
        try:
            download_folder_from_remote_storage(remote_folder=MOCK_DATASET_PATH.name, local_folder=str(dataset_root))
        except:
            gdown.download_folder(url=MOCK_DATASET_URL_GDRIVE, output=str(dataset_root))
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
