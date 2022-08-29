import functools as ft
import io
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from oml.const import (
    IS_GALLERY_COLUMN,
    IS_QUERY_COLUMN,
    LABELS_COLUMN,
    PATH_COLUMN,
    SPLIT_COLUMN,
    X1_COLUMN,
    X2_COLUMN,
    Y1_COLUMN,
    Y2_COLUMN,
)
from oml.utils.dataframe_format import check_retrieval_dataframe_format


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=Path)
    return parser


def build_cub_df(dataset_root: Path) -> pd.DataFrame:
    dataset_root = Path(dataset_root)

    images_txt = dataset_root / "images.txt"
    train_test_split = dataset_root / "train_test_split.txt"
    bounding_boxes = dataset_root / "bounding_boxes.txt"
    image_class_labels = dataset_root / "image_class_labels.txt"

    for file in [images_txt, train_test_split, bounding_boxes, image_class_labels]:
        assert file.is_file(), f"File {file} does not exist."

    with open(images_txt, "r") as f:
        images = f.read()
        images = pd.read_csv(io.StringIO(images), delim_whitespace=True, header=None, names=["image_id", "image_name"])

    with open(train_test_split, "r") as f:
        split = f.read()
        split = pd.read_csv(
            io.StringIO(split), delim_whitespace=True, header=None, names=["image_id", "is_training_image"]
        )

    with open(bounding_boxes, "r") as f:
        bbs = f.read()
        bbs = pd.read_csv(
            io.StringIO(bbs), delim_whitespace=True, header=None, names=["image_id", "x", "y", "width", "height"]
        )

    with open(image_class_labels, "r") as f:
        class_labels = f.read()
        class_labels = pd.read_csv(
            io.StringIO(class_labels), delim_whitespace=True, header=None, names=["image_id", "class_id"]
        )

    df = ft.reduce(lambda left, right: pd.merge(left, right, on="image_id"), [images, bbs, class_labels, split])

    df[X1_COLUMN] = df["x"].apply(int)  # left
    df[X2_COLUMN] = (df["x"] + df["width"]).apply(int)  # right
    df[Y2_COLUMN] = (df["y"] + df["height"]).apply(int)  # bot
    df[Y1_COLUMN] = df["y"].apply(int)  # top
    df[PATH_COLUMN] = df["image_name"].apply(lambda x: dataset_root / "images" / x)

    df[SPLIT_COLUMN] = "train"
    df[SPLIT_COLUMN][df["is_training_image"] == 0] = "validation"

    df[IS_QUERY_COLUMN] = None
    df[IS_GALLERY_COLUMN] = None
    df[IS_QUERY_COLUMN][df[SPLIT_COLUMN] == "validation"] = True
    df[IS_GALLERY_COLUMN][df[SPLIT_COLUMN] == "validation"] = True

    df = df.rename(columns={"class_id": LABELS_COLUMN})
    df = df[
        [
            LABELS_COLUMN,
            PATH_COLUMN,
            SPLIT_COLUMN,
            IS_QUERY_COLUMN,
            IS_GALLERY_COLUMN,
            X1_COLUMN,
            X2_COLUMN,
            Y1_COLUMN,
            Y2_COLUMN,
        ]
    ]

    check_retrieval_dataframe_format(df, dataset_root=dataset_root)
    return df


def main() -> None:
    print("CUB200 2011 dataset preparation started...")
    args = get_argparser().parse_args()
    df = build_cub_df(args.dataset_root)
    df.to_csv(args.dataset_root / "df.csv", index=None)
    print("CUB200 2011 dataset preparation completed.")
    print(f"DataFrame saved in {args.dataset_root}\n")


if __name__ == "__main__":
    main()
