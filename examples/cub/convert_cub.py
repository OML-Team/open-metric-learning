import functools as ft
import io
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from oml.utils.dataframe_format import check_retrieval_dataframe_format


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=Path)
    return parser


def build_cub(dataset_root: Path) -> pd.DataFrame:
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

    df["x_1"] = df["x"].apply(int)  # left
    df["x_2"] = (df["x"] + df["width"]).apply(int)  # right
    df["y_2"] = (df["y"] + df["height"]).apply(int)  # bot
    df["y_1"] = df["y"].apply(int)  # top
    df["path"] = "images/" + df["image_name"]

    df["split"] = "train"
    df["split"][df["is_training_image"] == 0] = "validation"

    df["is_query"] = None
    df["is_gallery"] = None
    df["is_query"][df["split"] == "validation"] = True
    df["is_gallery"][df["split"] == "validation"] = True

    df = df.rename(columns={"class_id": "label"})
    df = df[["label", "path", "split", "is_query", "is_gallery", "x_1", "x_2", "y_1", "y_2"]]

    check_retrieval_dataframe_format(df, dataset_root=dataset_root)
    return df


def main() -> None:
    print("CUB200 2011 dataset preparation started...")
    args = get_argparser().parse_args()
    df = build_cub(args.dataset_root)
    df.to_csv(args.dataset_root / "df.csv", index=None)
    print("CUB200 2011 dataset preparation completed.")
    print(f"DataFrame saved in {args.dataset_root}\n")


if __name__ == "__main__":
    main()
