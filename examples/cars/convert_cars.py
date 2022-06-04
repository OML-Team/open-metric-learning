from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import io

from oml.utils.dataframe_format import check_retrieval_dataframe_format


def construct_df(annots: Dict[str, np.ndarray], meta: pd.DataFrame, path_to_imgs: Path) -> pd.DataFrame:
    annots = pd.DataFrame(annots["annotations"][0])
    annots["bbox_x1"] = annots["bbox_x1"].astype(int)
    annots["bbox_y1"] = annots["bbox_y1"].astype(int)
    annots["bbox_x2"] = annots["bbox_x2"].astype(int)
    annots["bbox_y2"] = annots["bbox_y2"].astype(int)
    annots["label"] = annots["class"].astype(int)
    annots["fname"] = annots["fname"].apply(lambda x: x[0])
    annots = annots.rename(columns={"bbox_x1": "x_1", "bbox_y1": "y_1", "bbox_x2": "x_2", "bbox_y2": "y_2"})
    annots = pd.merge(annots, meta, how="left", left_on="label", right_on="index").drop(["index"], axis=1)

    annots["path"] = annots["fname"].apply(lambda x: path_to_imgs / x)
    return annots


def build_cars196_df(dataset_root: Path) -> pd.DataFrame:
    cars_meta = dataset_root / "devkit" / "cars_meta.mat"
    cars_train_annos = dataset_root / "devkit" / "cars_train_annos.mat"
    cars_test_annos_withlabels = dataset_root / "devkit" / "cars_test_annos_withlabels.mat"

    for file in [cars_meta, cars_train_annos, cars_test_annos_withlabels]:
        assert file.is_file(), f"File {file} does not exist."

    meta = io.loadmat(str(cars_meta))
    meta = pd.DataFrame(meta["class_names"][0], columns=["class_names"]).reset_index()
    meta["class_names"] = meta["class_names"].apply(lambda x: x[0])

    train_annots = io.loadmat(str(cars_train_annos))
    train_annots = construct_df(train_annots, meta=meta, path_to_imgs=Path("cars_train"))
    test_annots = io.loadmat(str(cars_test_annos_withlabels))
    test_annots = construct_df(test_annots, meta=meta, path_to_imgs=Path("cars_test"))
    train_annots["is_query"] = None
    train_annots["is_gallery"] = None

    test_annots["is_query"] = True
    test_annots["is_gallery"] = True

    train_annots["split"] = "train"
    test_annots["split"] = "validation"

    df = pd.concat((train_annots, test_annots))
    df = df[["label", "path", "split", "is_query", "is_gallery", "x_1", "x_2", "y_1", "y_2"]]

    check_retrieval_dataframe_format(df, dataset_root=dataset_root)
    return df


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=Path)
    return parser


def main() -> None:
    print("Cars dataset preparation started...")
    args = get_argparser().parse_args()
    df = build_cars196_df(args.dataset_root)
    df.to_csv(args.dataset_root / "df.csv", index=None)
    print("Cars dataset preparation completed.")
    print(f"DataFrame saved in {args.dataset_root}\n")


if __name__ == "__main__":
    main()
