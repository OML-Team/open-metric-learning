from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import List

import pandas as pd

from oml.const import (
    CATEGORIES_COLUMN,
    IS_GALLERY_COLUMN,
    IS_QUERY_COLUMN,
    LABELS_COLUMN,
    PATHS_COLUMN,
    SPLIT_COLUMN,
    X1_COLUMN,
    X2_COLUMN,
    Y1_COLUMN,
    Y2_COLUMN,
)
from oml.utils.dataframe_format import check_retrieval_dataframe_format


def parse_file_row(row: pd.Series) -> List[str]:
    return list(filter(lambda x: x != "", row.replace("\n", "").split(" ")))


def txt_to_df(fpath: Path) -> pd.DataFrame:
    with open(fpath, "r") as f:
        data = f.readlines()

    data = data[1:]  # we drop 1st line, which indicates the total number of the lines in file

    cols = parse_file_row(data[0])

    content = defaultdict(list)

    for row in data[1:]:
        for col, val in zip(cols, parse_file_row(row)):
            content[col].append(val)

    df = pd.DataFrame(content)

    return df


def build_inshop_df(dataset_root: Path, no_bboxes: bool) -> pd.DataFrame:
    dataset_root = Path(dataset_root)

    list_eval_partition = dataset_root / "list_eval_partition.txt"
    list_bbox_inshop = dataset_root / "list_bbox_inshop.txt"

    for file in [list_eval_partition, list_bbox_inshop]:
        assert file.is_file(), f"File {file} does not exist."

    df = txt_to_df(list_eval_partition)
    df["path"] = df["image_name"].apply(lambda x: Path(dataset_root) / x)

    need_bboxes = not no_bboxes
    if need_bboxes:
        df_bbox = txt_to_df(list_bbox_inshop)
        for name in ["x_1", "y_1", "x_2", "y_2"]:
            df_bbox[name] = df_bbox[name].astype(int)

        df = df.merge(df_bbox, on="image_name", how="inner")
        df.reset_index(inplace=True, drop=True)

    df["label"] = df["item_id"].apply(lambda x: int(x[3:]))

    df["split"] = "validation"
    df["split"][df["evaluation_status"] == "train"] = "train"

    df["is_query"] = False
    df["is_gallery"] = False
    df["is_query"] = df["evaluation_status"] == "query"
    df["is_gallery"] = df["evaluation_status"] == "gallery"
    df["is_query"][df["split"] == "train"] = None
    df["is_gallery"][df["split"] == "train"] = None

    df["category"] = df["path"].apply(lambda x: x.parent.parent.name)

    cols_to_pick = ["label", "path", "split", "is_query", "is_gallery", "category"]
    if need_bboxes:
        cols_to_pick.extend(["x_1", "x_2", "y_1", "y_2"])
    df = df[cols_to_pick]

    # check stat
    assert df["path"].nunique() == len(df) == 52712
    assert df["label"].nunique() == 7982
    assert set(df["label"].astype(int).tolist()) == set(list(range(1, 7982 + 1)))

    # rm bad labels
    mask_non_single_images = df.groupby("label").label.transform("count") > 1
    df = df[mask_non_single_images]
    df.reset_index(drop=True, inplace=True)
    print(f"Dropped {len(mask_non_single_images) - mask_non_single_images.sum()} items with only 1 image.")

    check_retrieval_dataframe_format(df, dataset_root=dataset_root)

    df = df.rename(
        columns={
            "label": LABELS_COLUMN,
            "path": PATHS_COLUMN,
            "split": SPLIT_COLUMN,
            "is_query": IS_QUERY_COLUMN,
            "is_gallery": IS_GALLERY_COLUMN,
            "x_1": X1_COLUMN,
            "x_2": X2_COLUMN,
            "y_1": Y1_COLUMN,
            "y_2": Y2_COLUMN,
            "category": CATEGORIES_COLUMN,
        }
    )

    return df.reset_index(drop=True)


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=Path)
    parser.add_argument("--no_bboxes", action="store_true")
    return parser


def main() -> None:
    print("DeepFashion Inshop dataset preparation started...")
    args = get_argparser().parse_args()

    df = build_inshop_df(
        dataset_root=args.dataset_root,
        no_bboxes=args.no_bboxes,
    )

    fname = "df_no_bboxes" if args.no_bboxes else "df"
    df.to_csv(args.dataset_root / f"{fname}.csv", index=None)

    print("DeepFashion Inshop dataset preparation completed.")
    print(f"DataFrame saved in {args.dataset_root}\n")


if __name__ == "__main__":
    main()
