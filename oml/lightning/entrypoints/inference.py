import json
from pathlib import Path
from pprint import pprint

import albumentations as albu
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader

from oml.const import PATHS_COLUMN, X1_COLUMN, X2_COLUMN, Y1_COLUMN, Y2_COLUMN, TCfg
from oml.datasets.list_ import ListDataset
from oml.exceptions import InferenceConfigError
from oml.registry.models import get_extractor_by_cfg
from oml.registry.transforms import get_transforms_by_cfg
from oml.utils.misc import dictconfig_to_dict


def pl_infer(cfg: TCfg) -> None:
    """
    This is an entrypoint for the model validation in metric learning setup.

    The config can be specified as a dictionary or with hydra: https://hydra.cc/.
    For more details look at ``examples/README.md``

    """
    cfg = dictconfig_to_dict(cfg)

    pprint(cfg)

    images_folder = Path(cfg["images_folder"]) if cfg.get("images_folder") else None
    dataframe_name = cfg.get("dataframe_name")
    # Check if either images_folder present or dataframe_name but not both
    if images_folder is not None and dataframe_name is not None:
        raise InferenceConfigError("images_folder and dataframe_name cannot be provided both at the same time")
    if images_folder is None and dataframe_name is None:
        raise InferenceConfigError("Either images_folder or dataframe_name should be set")

    # Working with pure images if images folder is not None
    if images_folder is not None:
        im_paths = list(images_folder.rglob("*"))
        bboxes = None

    # Working with dataframe if it's path is not None
    if dataframe_name is not None:
        # read dataframe and get boxes. Use BaseDataset for inspiration
        df = pd.read_csv(dataframe_name)
        assert PATHS_COLUMN in df.columns

        bboxes_exist = all(coord in df.columns for coord in (X1_COLUMN, X2_COLUMN, Y1_COLUMN, Y2_COLUMN))
        if bboxes_exist:
            bboxes = []
            for row in df[[X1_COLUMN, Y1_COLUMN, X2_COLUMN, Y2_COLUMN]].iterrows():
                x1, y1, x2, y2 = row[1]
                bboxes.append((x1, y1, x2, y2))
        else:
            bboxes = None

        dataset_root = cfg["dataset_root"]
        if dataset_root is not None:
            dataset_root = Path(dataset_root)
            df[PATHS_COLUMN] = df[PATHS_COLUMN].apply(lambda x: str(dataset_root / x))
        else:
            df[PATHS_COLUMN] = df[PATHS_COLUMN].astype(str)
        im_paths = df[PATHS_COLUMN].tolist()

    kwargs = {}
    # Loading transformations
    trans_type = cfg.get("transforms")
    if trans_type:
        transform = get_transforms_by_cfg(cfg["transforms"])

        available_augs_types = (albu.Compose, torchvision.transforms.Compose)
        assert isinstance(transform, available_augs_types), f"Type of transforms must be in {available_augs_types}"
        kwargs["transform"] = transform

    dataset = ListDataset(filenames_list=im_paths, bboxes=bboxes, **kwargs)
    loader = DataLoader(dataset=dataset, batch_size=cfg["bs_val"], num_workers=cfg["num_workers"])

    extractor = get_extractor_by_cfg(cfg["model"])
    features = []
    for batch in loader:
        feats = extractor.extract(batch)
        features += torch.split(feats, 1)

    out_json_path = Path(cfg["features_file"])
    with out_json_path.open("w") as f:
        out_struct = {
            "images_folder": cfg["images_folder"],
            "dataframe_name": cfg["dataframe_name"],
            "model": cfg["model"],
            "transforms": cfg["transforms"],
            "filenames": list(map(str, im_paths)),
            "bboxes": bboxes,
            "features": features,
        }
        json.dump(out_struct, f)


__all__ = ["pl_infer"]
