from pathlib import Path
from typing import Tuple, Union

import gdown
import pandas as pd

from oml.const import (
    CATEGORIES_COLUMN,
    IS_GALLERY_COLUMN,
    IS_QUERY_COLUMN,
    LABELS_COLUMN,
    MOCK_AUDIO_DATASET_MD5,
    MOCK_AUDIO_DATASET_PATH,
    MOCK_AUDIO_DATASET_URL_GDRIVE,
    MOCK_DATASET_DEFAULT_CSV,
    MOCK_DATASET_MD5,
    MOCK_DATASET_PATH,
    MOCK_DATASET_URL_GDRIVE,
    SPLIT_COLUMN,
    TEXTS_COLUMN,
)
from oml.utils.io import check_exists_and_validate_md5
from oml.utils.remote_storage import download_folder_from_remote_storage


def _get_mock_dataset(
    dataset_local_folder: Union[str, Path],
    dataset_remote_folder: str,
    dataset_md5: str,
    dataset_gdrive_url: str,
    df_name: str,
    check_md5: bool = True,
    global_paths: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Downloads and prepares a mock dataset in the required format.

    Args:
        dataset_local_folder: The directory where the dataset will be saved.
        dataset_remote_folder: The remote directory on `oml.daloroserver.com` from which the dataset will be downloaded.
        dataset_md5: The MD5 checksum used to validate the dataset.
        dataset_gdrive_url: The Google Drive URL for the dataset download as a fallback option.
        df_name: The name of the CSV file from which the output DataFrames will be generated.
        check_md5: If ``True``, validates the dataset using an MD5 checksum.
        global_paths: If ``True``, concatenates the paths in the dataset with the dataset_local_folder.

    Returns:
        A tuple containing two DataFrames:
            - The first DataFrame is for the training stage.
            - The second DataFrame is for the validation stage.

    Raises:
        Exception: If the downloaded dataset is invalid.
    """
    dataset_local_folder = Path(dataset_local_folder)
    dataset_md5 = dataset_md5 if check_md5 else None

    if not check_exists_and_validate_md5(dataset_local_folder, dataset_md5):
        try:
            print("Downloading from oml.daloroserver.com")
            download_folder_from_remote_storage(
                remote_folder=dataset_remote_folder, local_folder=str(dataset_local_folder)
            )
        except Exception:
            print("We could not download from oml.daloroserver.com, let's try Google Drive.")
            gdown.download_folder(url=dataset_gdrive_url, output=str(dataset_local_folder))
    else:
        print(f"Mock dataset has been downloaded already to {dataset_local_folder}")

    if not check_exists_and_validate_md5(dataset_local_folder, dataset_md5):
        raise Exception("Downloaded mock dataset is invalid.")

    df = pd.read_csv(dataset_local_folder / df_name)

    if global_paths:
        df["path"] = df["path"].apply(lambda x: str(dataset_local_folder / x))

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "validation"].reset_index(drop=True)

    df_val = df_val.astype({"is_query": bool, "is_gallery": bool})

    return df_train, df_val


def get_mock_images_dataset(
    dataset_root: Union[str, Path] = MOCK_DATASET_PATH,
    df_name: str = MOCK_DATASET_DEFAULT_CSV,
    check_md5: bool = True,
    global_paths: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to download mock images dataset which is already prepared in the required format.

    Args:
        dataset_root: The directory where the dataset will be saved.
        df_name: The name of the CSV file from which the output DataFrames will be generated.
        check_md5: If ``True``, validates the dataset using an MD5 checksum.
        global_paths: If ``True``, concatenates the paths in the dataset with the dataset_local_folder.

    Returns:
        A tuple containing two DataFrames:
            - The first DataFrame is for the training stage.
            - The second DataFrame is for the validation stage.
    """
    return _get_mock_dataset(
        dataset_local_folder=dataset_root,
        dataset_remote_folder=MOCK_DATASET_PATH.name,
        dataset_md5=MOCK_DATASET_MD5,
        dataset_gdrive_url=MOCK_DATASET_URL_GDRIVE,
        df_name=df_name,
        check_md5=check_md5,
        global_paths=global_paths,
    )


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


def get_mock_audios_dataset(
    dataset_root: Union[str, Path] = MOCK_AUDIO_DATASET_PATH,
    df_name: str = MOCK_DATASET_DEFAULT_CSV,
    check_md5: bool = True,
    global_paths: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to download mock audios dataset which is already prepared in the required format.

    Args:
        dataset_root: The directory where the dataset will be saved.
        df_name: The name of the CSV file from which the output DataFrames will be generated.
        check_md5: If ``True``, validates the dataset using an MD5 checksum.
        global_paths: If ``True``, concatenates the paths in the dataset with the dataset_local_folder.

    Returns:
        A tuple containing two DataFrames:
            - The first DataFrame is for the training stage.
            - The second DataFrame is for the validation stage.
    """
    return _get_mock_dataset(
        dataset_local_folder=dataset_root,
        dataset_remote_folder=MOCK_AUDIO_DATASET_PATH.name,
        dataset_md5=MOCK_AUDIO_DATASET_MD5,
        dataset_gdrive_url=MOCK_AUDIO_DATASET_URL_GDRIVE,
        df_name=df_name,
        check_md5=check_md5,
        global_paths=global_paths,
    )


def get_mock_texts_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mock texts dataset useful for prototyping pipelines and understanding dataset structure.

    """

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

    potato = [
        "Yukon Gold Potato: Creamy and golden, perfect for roasting and mashing.",
        "Russet Potato: Classic and versatile, ideal for baking and frying.",
        "Red Potato: Smooth and firm, great for boiling and salads.",
        "Fingerling Potato: Rich and nutty, perfect for roasting or grilling.",
        "Purple Potato: Vibrant and sweet, adds color to any dish.",
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
    texts = [*bed, *table, *tv, *potato, *chair, *phone, *audio]

    labels = (
        [0] * len(bed)
        + [1] * len(table)
        + [2] * len(tv)
        + [3] * len(potato)
        + [4] * len(chair)
        + [5] * len(phone)
        + [6] * len(audio)
    )

    categories = (
        ["furniture"] * len(bed)
        + ["furniture"] * len(table)
        + ["electronic"] * len(tv)
        + ["food"] * len(potato)
        + ["furniture"] * len(chair)
        + ["electronic"] * len(phone)
        + ["electronic"] * len(audio)
    )

    n_train = len(bed) + len(table) + len(tv) + len(potato)
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


__all__ = ["get_mock_images_dataset", "get_mock_texts_dataset", "get_mock_audios_dataset"]
