from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple, Union

import pandas as pd

MOCK_DATASET_URL = "https://drive.google.com/drive/folders/1plPnwyIkzg51-mLUXWTjREHgc1kgGrF4?usp=sharing"


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=Path)
    return parser


def download_mock_dataset(dataset_root: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # gdown.download_folder(url=MOCK_DATASET_URL, output=str(dataset_root))

    df = pd.read_csv(Path(dataset_root) / "df.csv")

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "validation"].reset_index(drop=True)

    return df_train, df_val


def main() -> None:
    args = get_argparser().parse_args()
    download_mock_dataset(dataset_root=args.dataset_root)


if __name__ == "__main__":
    main()
