import json
from pathlib import Path
from pprint import pprint
from typing import List

import albumentations as albu
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from oml.const import BBOXES_COLUMNS, PATHS_COLUMN, TCfg
from oml.datasets.list_dataset import ListDataset
from oml.exceptions import (
    InferenceConfigError,
    InvalidDataFrameColumnsException,
    InvalidImageException,
)
from oml.lightning.entrypoints.parser import parse_engine_params_from_config
from oml.registry.models import get_extractor_by_cfg
from oml.registry.transforms import get_transforms_by_cfg
from oml.utils.images.images import imread_cv2, verify_image_readable
from oml.utils.misc import dictconfig_to_dict


def inference(cfg: TCfg) -> None:
    """
    This is an entrypoint for the model inference in metric learning setup.

    The config can be specified as a dictionary or with hydra: https://hydra.cc/.
    For more details look at ``examples/README.md``

    """
    cfg = dictconfig_to_dict(cfg)
    pprint(cfg)

    engine_params = parse_engine_params_from_config(cfg)
    accelerator = engine_params["accelerator"]
    devices = engine_params["devices"]
    if accelerator == "gpu":
        device_idx = devices[0] if isinstance(devices, list) else 0
        device = f"cuda:{device_idx}"
    else:
        device = "cpu"

    images_folder = Path(cfg["images_folder"]) if cfg.get("images_folder") else None
    dataframe_name = cfg.get("dataframe_name")
    # Check if either images_folder present or dataframe_name but not both
    if images_folder is not None and dataframe_name is not None:
        raise InferenceConfigError("images_folder and dataframe_name cannot be provided both at the same time")
    if images_folder is None and dataframe_name is None:
        raise InferenceConfigError("Either images_folder or dataframe_name should be set")

    # Working with pure images if images folder is not None
    bboxes = None
    if images_folder is not None:
        extensions = (".jpeg", ".jpg", ".png")

        im_paths: List[Path] = []
        for path in sorted(images_folder.rglob("*")):
            if path.suffix.lower() in extensions and path.is_file():
                im_paths.append(path)

    # Working with dataframe if it's path is not None
    if dataframe_name is not None:
        delimiter = cfg.get("dataframe_separator")
        # read dataframe and get boxes. Use BaseDataset for inspiration
        df = pd.read_csv(dataframe_name, delimiter=delimiter)
        # begin check that columns are correct
        if PATHS_COLUMN not in df.columns:
            raise InvalidDataFrameColumnsException(
                f"Source CSV file {dataframe_name} is missing '{PATHS_COLUMN}' column."
            )

        if set(df.columns).intersection(set(BBOXES_COLUMNS)):
            if not all(col in df.columns for col in BBOXES_COLUMNS):
                raise InvalidDataFrameColumnsException(
                    f"Bounding boxes in datframe {dataframe_name} are invalid. If you want to use bounding "
                    f"boxes, you should specify them as the following columns: {BBOXES_COLUMNS}"
                )

            bboxes = []
            for row in df[BBOXES_COLUMNS].iterrows():
                x1, x2, y1, y2 = row[1]
                bboxes.append((x1, y1, x2, y2))
        # end columns check

        dataset_root = cfg["dataset_root"]
        if dataset_root is not None:
            dataset_root = Path(dataset_root)
            df[PATHS_COLUMN] = df[PATHS_COLUMN].apply(lambda x: dataset_root / x)
        else:
            df[PATHS_COLUMN] = df[PATHS_COLUMN].apply(lambda x: Path(x))
        im_paths = df[PATHS_COLUMN].tolist()

        # Check that files from dataframe exist
        for i, path in enumerate(im_paths):
            if not path.exists():
                raise FileNotFoundError(
                    f"Could not find image on line {i+1}: {str(path)} in dataframe {dataframe_name}"
                )

    # Check if images exist
    if not im_paths:
        raise FileNotFoundError("No images found in source directory or dataframe_name")

    # Check that files could be opened
    for path in im_paths:
        with path.open("rb") as fimage:
            if not verify_image_readable(fimage.read()):
                raise InvalidImageException(f"File can not be decoded as image: {path}")

    # Loading transformations
    trans_type = cfg.get("transforms")
    if trans_type:
        transform = get_transforms_by_cfg(cfg["transforms"])

        available_augs_types = (albu.Compose, torchvision.transforms.Compose)
        assert isinstance(transform, available_augs_types), f"Type of transforms must be in {available_augs_types}"
    else:
        raise InferenceConfigError("'transforms' field is missing in the config")

    dataset = ListDataset(filenames_list=im_paths, bboxes=bboxes, f_imread=imread_cv2, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])

    extractor = get_extractor_by_cfg(cfg["model"])
    extractor.to(device)
    features = []
    for batch in tqdm(loader):
        batch = batch[dataset.input_tensors_key]
        batch = batch.to(device)
        feats = extractor.extract(batch)
        features += [feat.squeeze().tolist() for feat in torch.split(feats, 1)]

    out_json_path = Path(cfg["features_file"])
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with out_json_path.open("w") as f:
        out_struct = {
            "images_folder": cfg.get("images_folder"),
            "dataframe_name": cfg.get("dataframe_name"),
            "model": cfg["model"],
            "transforms": cfg.get("transforms"),
            "filenames": list(map(str, im_paths)),
            "bboxes": bboxes,
            "features": features,
        }
        json.dump(out_struct, f)


__all__ = ["inference"]
