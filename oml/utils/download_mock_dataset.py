from pathlib import Path
from typing import Tuple, Union

import gdown
import pandas as pd

from oml.const import (
    CATEGORIES_COLUMN,
    IS_GALLERY_COLUMN,
    IS_QUERY_COLUMN,
    LABELS_COLUMN,
    MOCK_DATASET_MD5,
    MOCK_DATASET_PATH,
    MOCK_DATASET_URL_GDRIVE,
    SPLIT_COLUMN,
    TEXTS_COLUMN,
)
from oml.utils.io import check_exists_and_validate_md5
from oml.utils.remote_storage import download_folder_from_remote_storage


def get_mock_images_dataset(
    dataset_root: Union[str, Path] = MOCK_DATASET_PATH,
    check_md5: bool = True,
    df_name: str = "df.csv",
    global_paths: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to download mock dataset which is already prepared in the required format.

    Args:
        dataset_root: Path to save the dataset
        check_md5: Set ``True`` to check md5sum
        df_name: Name of csv file for which output DataFrames will be returned
        global_paths: Set ``True`` to cancat paths and ``dataset_root``

    Returns: Dataframes for the training and validation stages

    """
    dataset_root = Path(dataset_root)

    if not check_exists_and_validate_md5(dataset_root, MOCK_DATASET_MD5 if check_md5 else None):
        try:
            print("Downloading from oml.daloroserver.com")
            download_folder_from_remote_storage(remote_folder=MOCK_DATASET_PATH.name, local_folder=str(dataset_root))
            assert check_exists_and_validate_md5(dataset_root, MOCK_DATASET_MD5 if check_md5 else None)
        except Exception:
            print("We could not download from oml.daloroserver.com, let's try Google Drive.")
            gdown.download_folder(url=MOCK_DATASET_URL_GDRIVE, output=str(dataset_root))
    else:
        print(f"Mock dataset has been downloaded already to {dataset_root}")

    if not check_exists_and_validate_md5(dataset_root, MOCK_DATASET_MD5 if check_md5 else None):
        raise Exception("Downloaded mock dataset is invalid.")

    df = pd.read_csv(Path(dataset_root) / df_name)

    if global_paths:
        df["path"] = df["path"].apply(lambda x: str(Path(dataset_root) / x))

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "validation"].reset_index(drop=True)

    df_val["is_query"] = df_val["is_query"].astype(bool)
    df_val["is_gallery"] = df_val["is_gallery"].astype(bool)

    return df_train, df_val


def download_mock_dataset(
    dataset_root: Union[str, Path] = MOCK_DATASET_PATH,
    check_md5: bool = True,
    df_name: str = "df.csv",
    global_paths: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # for back compatibility

    return get_mock_images_dataset(
        dataset_root=dataset_root, check_md5=check_md5, df_name=df_name, global_paths=global_paths
    )


def get_mock_texts_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    bed = [
        "Luxura King Bed: Plush headboard and durable frame for ultimate comfort.",
        "EcoSleep Twin Bed: Sustainable materials for eco-conscious consumers.",
        "DreamRest Queen Bed: Ergonomic design for restful sleep.",
        "ClassicWood Full Bed: Sturdy wood construction with a sleek finish.",
    ]

    table = [
        "UrbanChic Dining Table: Glass top with metal legs seats six.",
        "RusticFarmhouse Coffee Table: Reclaimed wood for rustic charm.",
        "Minimalist Work Desk: Sleek, modern workspace.",
        "ArtisanCraft End Table: Handcrafted accent piece.",
        "Vintage Bistro Table: Classic design for cozy corners.",
        "Compact Folding Table: Versatile and easy to store.",
    ]

    tv = [
        'UltraHD Smart TV 55": Stunning visuals and smart features.',
        'Compact LED TV 32": Crisp picture in a space-saving design.',
        'Curved 4K TV 65": Panoramic viewing with brilliant colors.',
        'BudgetFriendly LCD TV 40": Clear picture and essential features.',
    ]

    chair = [
        "ErgoComfort Office Chair: Maximum comfort and support.",
        "ClassicWood Dining Chair: Timeless design with sturdy construction.",
        "RelaxLounge Recliner: Adjustable settings and plush cushioning.",
        "ModernAccent Chair: Contemporary style with vibrant color options.",
    ]

    phone = [
        "Galaxy X9: Sleek design with powerful performance.",
        "iPhone 12: Cutting-edge technology with a stylish look.",
        "Pixel 5: Pure Android experience with excellent camera.",
        "OnePlus 8T: High performance at a competitive price.",
        "Moto G Power: Long battery life for extended use.",
        "Sony Xperia 5: Superior camera and display quality.",
    ]

    audio = [
        "NoiseCancelling Headphones Pro: Superior sound and noise cancellation.",
        "Wireless Earbuds Sport: Great sound with a secure fit.",
        "StudioOverEar Headphones: Exceptional audio clarity.",
        "BudgetFriendly Wired Headphones: Good sound at an affordable price.",
        "Gaming Headset Pro: Surround sound for gaming.",
        "TravelNoiseCancelling Earbuds: Compact with excellent noise cancellation.",
    ]
    texts = [*bed, *table, *tv, *chair, *phone, *audio]

    labels = [0] * len(bed) + [1] * len(table) + [2] * len(tv) + [3] * len(chair) + [4] * len(phone) + [5] * len(audio)

    categories = (
        ["furniture"] * len(bed)
        + ["furniture"] * len(table)
        + ["electronic"] * len(tv)
        + ["furniture"] * len(chair)
        + ["electronic"] * len(phone)
        + ["electronic"] * len(audio)
    )

    n_train = len(bed) + len(table) + len(tv)
    n_val = len(chair) + len(phone) + len(audio)

    split = ["train"] * n_train + ["validation"] * n_val
    is_query = [None] * n_train + [True] * n_val  # type: ignore

    is_gallery = is_query

    data = {
        TEXTS_COLUMN: texts,
        LABELS_COLUMN: labels,
        CATEGORIES_COLUMN: categories,
        SPLIT_COLUMN: split,
        IS_QUERY_COLUMN: is_query,
        IS_GALLERY_COLUMN: is_gallery,
    }

    df = pd.DataFrame(data)

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "validation"].reset_index(drop=True)

    return df_train, df_val


__all__ = ["get_mock_images_dataset", "get_mock_texts_dataset"]
