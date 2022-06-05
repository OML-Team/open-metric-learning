import os
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
from pandarallel import pandarallel

from oml.utils.dataframe_format import check_retrieval_dataframe_format
from oml.utils.images.images_resize import inverse_karesize_bboxes

BASE_SIZE_INSHOP = (256, 256)  # size of not high-res images in InShop


def parse_file_row(row: pd.Series) -> List[str]:
    return list(filter(lambda x: x != "", row.replace("\n", "").split(" ")))


def expand_squeezed_bboxes(df: pd.DataFrame, ratio_th: float, only_for_valid: bool) -> pd.DataFrame:
    df["ar"] = (df["y_2"] - df["y_1"]) / (df["x_2"] - df["x_1"])

    if only_for_valid:
        mask_bad = (df["ar"] > ratio_th) & (df["split"] == "validation")
    else:
        mask_bad = df["ar"] > ratio_th

    print(f"We will fix {mask_bad.sum()} bboxes")

    df["x_center"] = ((df["x_1"] + df["x_2"]) // 2).astype(int)
    df["half_h"] = (df["y_2"] - df["y_1"]).astype(int) // 2

    df["x_1"][mask_bad] = np.clip(df["x_center"][mask_bad] - df["half_h"], a_min=0, a_max=100_000)
    df["x_2"][mask_bad] = df["x_center"][mask_bad] + df["half_h"]

    return df


def convert_bbox(row: pd.Series) -> List[int]:
    bbox = int(row.x_1), int(row.y_1), int(row.x_2), int(row.y_2)
    img_hw = (row.h, row.w)
    bbox = inverse_karesize_bboxes(np.array(bbox)[np.newaxis, ...], BASE_SIZE_INSHOP, img_hw)[0, :]
    bbox[0] = np.clip(bbox[0], 0, img_hw[1])
    bbox[2] = np.clip(bbox[2], 0, img_hw[1])
    bbox[1] = np.clip(bbox[1], 0, img_hw[0])
    bbox[3] = np.clip(bbox[3], 0, img_hw[0])
    bbox = list(map(int, bbox))
    return bbox


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


def build_deepfashion_df(args: Namespace) -> pd.DataFrame:
    list_eval_partition = args.dataset_root / "list_eval_partition.txt"
    list_bbox_inshop = args.dataset_root / "list_bbox_inshop.txt"

    for file in [list_eval_partition, list_bbox_inshop]:
        assert file.is_file(), f"File {file} does not exist."

    pandarallel.initialize(nb_workers=os.cpu_count())

    df_part = txt_to_df(list_eval_partition)
    df_part["path"] = df_part["image_name"].apply(lambda x: args.dataset_root / x.replace("img/", "img_highres/"))

    df_part["cls"] = df_part["path"].apply(lambda x: x.parent.parent.name)

    df_bbox = txt_to_df(list_bbox_inshop)

    df = df_part.merge(df_bbox, on="image_name", how="inner")
    df.reset_index(inplace=True, drop=True)

    df["label"] = df["item_id"].apply(lambda x: int(x[3:]))

    df["hw"] = df["path"].parallel_apply(lambda x: cv2.imread(str(x)).shape[:2])
    df["h"] = df["hw"].apply(lambda x: x[0])
    df["w"] = df["hw"].apply(lambda x: x[1])
    del df["hw"]

    bboxes = list(zip(*df.apply(convert_bbox, axis=1)))
    for i, name in zip([0, 1, 2, 3], ["x_1", "y_1", "x_2", "y_2"]):
        df[name] = bboxes[i]

    thr_bbox_size = 10
    mask_bad_bboxes = df.apply(
        lambda row: (row["x_2"] - row["x_1"]) < thr_bbox_size or (row["y_2"] - row["y_1"]) < thr_bbox_size, axis=1
    )
    df = df[~mask_bad_bboxes]
    print(f"Dropped {mask_bad_bboxes.sum()} images with bad bboxes")
    df.reset_index(drop=True, inplace=True)

    mask_non_single_images = df.groupby("label").label.transform("count") > 1
    df = df[mask_non_single_images]
    print(f"Dropped {len(mask_non_single_images) - mask_non_single_images.sum()} items with only 1 image.")

    df["split"] = "validation"
    df["split"][df["evaluation_status"] == "train"] = "train"

    df["is_query"] = False
    df["is_gallery"] = False
    df["is_query"] = df["evaluation_status"] == "query"
    df["is_gallery"] = df["evaluation_status"] == "gallery"
    df["is_query"][df["split"] == "train"] = None
    df["is_gallery"][df["split"] == "train"] = None

    if args.fix_squeezed_bboxes:
        # todo: param
        df = expand_squeezed_bboxes(df, ratio_th=args.bboxes_aspect_ratio_to_fix, only_for_valid=True)

    df = df[["label", "path", "split", "is_query", "is_gallery", "x_1", "x_2", "y_1", "y_2"]]

    check_retrieval_dataframe_format(df, dataset_root=args.dataset_root)
    return df.reset_index(drop=True)


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=Path)
    parser.add_argument("--fix_squeezed_bboxes", action="store_true")
    parser.add_argument(
        "--bboxes_aspect_ratio_to_fix",
        type=float,
        default=2.5,
        help="will be used only when fix_squeezed_bboxes is True",
    )
    return parser


def main() -> None:
    print("DeepFashion Inshop dataset preparation started...")
    args = get_argparser().parse_args()
    df = build_deepfashion_df(args)
    # todo: apply only to valid
    df.to_csv(args.dataset_root / "df_fixed.csv", index=None)
    print("DeepFashion Inshop dataset preparation completed.")
    print(f"DataFrame saved in {args.dataset_root}\n")


if __name__ == "__main__":
    main()
