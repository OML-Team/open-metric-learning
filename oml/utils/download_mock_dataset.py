from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple, Union

import gdown
import pandas as pd

from oml.const import MOCK_DATASET_MD5, MOCK_DATASET_PATH, MOCK_DATASET_URL_GDRIVE
from oml.utils.io import check_exists_and_validate_md5
from oml.utils.remote_storage import download_folder_from_remote_storage


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=MOCK_DATASET_PATH)
    return parser


def download_mock_dataset(dataset_root: Union[str, Path], check_md5: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to download mock dataset which is already prepared in the required format.

    Args:
        dataset_root: Path to save the dataset
        check_md5: Set ``True`` to check md5sum

    Returns: Dataframes for the training and validation stages

    """
    dataset_root = Path(dataset_root)

    if not check_exists_and_validate_md5(dataset_root, MOCK_DATASET_MD5 if check_md5 else None):
        try:
            download_folder_from_remote_storage(remote_folder=MOCK_DATASET_PATH.name, local_folder=str(dataset_root))
        except Exception:
            gdown.download_folder(url=MOCK_DATASET_URL_GDRIVE, output=str(dataset_root))
    else:
        print(f"Mock dataset has been downloaded already to {dataset_root}")

    if not check_exists_and_validate_md5(dataset_root, MOCK_DATASET_MD5 if check_md5 else None):
        raise Exception("Downloaded mock dataset is invalid.")

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
