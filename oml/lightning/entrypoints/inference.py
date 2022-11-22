from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd

from oml.const import TCfg
from oml.datasets.base import get_retrieval_datasets
from oml.lightning.callbacks.metric import MetricValCallback, MetricValCallbackDDP
from oml.lightning.entrypoints.parser import (
    check_is_config_for_ddp,
    parse_engine_params_from_config,
)
from oml.lightning.modules.retrieval import RetrievalModule, RetrievalModuleDDP
from oml.metrics.embeddings import EmbeddingMetrics, EmbeddingMetricsDDP
from oml.registry.models import get_extractor_by_cfg
from oml.registry.transforms import get_transforms_by_cfg
from oml.utils.misc import dictconfig_to_dict
from oml.datasets.list_ import ListDataset
from oml.const import (
    CATEGORIES_COLUMN,
    CATEGORIES_KEY,
    INPUT_TENSORS_KEY,
    IS_GALLERY_COLUMN,
    IS_GALLERY_KEY,
    IS_QUERY_COLUMN,
    IS_QUERY_KEY,
    LABELS_COLUMN,
    LABELS_KEY,
    PATHS_COLUMN,
    PATHS_KEY,
    SPLIT_COLUMN,
    X1_COLUMN,
    X1_KEY,
    X2_COLUMN,
    X2_KEY,
    Y1_COLUMN,
    Y1_KEY,
    Y2_COLUMN,
    Y2_KEY,
)


def pl_infer(cfg: TCfg):
    """
    This is an entrypoint for the model validation in metric learning setup.

    The config can be specified as a dictionary or with hydra: https://hydra.cc/.
    For more details look at ``examples/README.md``

    """
    cfg = dictconfig_to_dict(cfg)
    trainer_engine_params = parse_engine_params_from_config(cfg)
    is_ddp = check_is_config_for_ddp(trainer_engine_params)

    pprint(cfg)

    images_folder = Path(cfg["images_folder"])
    dataframe_name = cfg["dataframe_name"]
    if images_folder is not None and dataframe_name is not None:
        raise InferenceConfigError("images_folder and dataframe_name cannot be
        provided both at the same time")
    if images_folder is None and dataframe_name is None:
        raise InferenceConfigError("Either images_folder or dataframe_name
        should be set")

    if images_folder is not None:
        im_paths = list(images_folder.rglob("*"))
        bboxes = None

    if dataframe_name is not None:
        # read dataframe and get boxes. Use BaseDataset for inspiration
        df = pd.read_csv(dataframe_name)
        assert PATHS_COLUMN in df.columns

        bboxes_exist = all(coord in df.columns for coord in (X1_COLUMN, X2_COLUMN, Y1_COLUMN, Y2_COLUMN))
        if bboxes_exist:
            x1_key, x2_key, y1_key, y2_key = X1_COLUMN, X2_COLUMN, Y1_COLUMN, Y2_COLUMN
        else:
            bboxes = None

        sub_df = df[["path", "x_1", "y_1", "x_2", "y_2"]]
        sub_df["path"] = sub_df["path"].apply(lambda p: MOCK_DATASET_PATH / p)
        paths, bboxes = [], []
        for row in sub_df.iterrows():
            path, x1, y1, x2, y2 = row[1]
            paths.append(path)
            bboxes.append((x1, y1, x2, y2))


        if dataset_root is not None:
            dataset_root = Path(dataset_root)
            df[PATHS_COLUMN] = df[PATHS_COLUMN].apply(lambda x: str(dataset_root / x))
        else:
            df[PATHS_COLUMN] = df[PATHS_COLUMN].astype(str)

        self.df = df
        self.transform = transform if transform else get_transforms("norm_albu")
        self.f_imread = f_imread
        self.read_bytes_image_cached = lru_cache(maxsize=cache_size)(self._read_bytes_image)

        available_augs_types = (albu.Compose, torchvision.transforms.Compose)
        assert isinstance(self.transform, available_augs_types), f"Type of transforms must be in {available_augs_types}"
    # I copypasted till here

    # Get all images either from csv dataframe of from a folder
    if cfg["images_folder"] 
    dataset = ListDataset(
    loader_val = DataLoader(dataset=valid_dataset, batch_size=cfg["bs_val"], num_workers=cfg["num_workers"])

    extractor = get_extractor_by_cfg(cfg["model"])

    module_kwargs = {}
    if is_ddp:
        module_kwargs["loaders_val"] = loader_val
        module_constructor = RetrievalModuleDDP
    else:
        module_constructor = RetrievalModule  # type: ignore

    pl_model = module_constructor(
        model=extractor,
        criterion=None,
        optimizer=None,
        scheduler=None,
        input_tensors_key=valid_dataset.input_tensors_key,
        labels_key=valid_dataset.labels_key,
        **module_kwargs
    )

    metrics_constructor = EmbeddingMetricsDDP if is_ddp else EmbeddingMetrics
    metrics_calc = metrics_constructor(
        embeddings_key=pl_model.embeddings_key,
        categories_key=valid_dataset.categories_key,
        labels_key=valid_dataset.labels_key,
        is_query_key=valid_dataset.is_query_key,
        is_gallery_key=valid_dataset.is_gallery_key,
        extra_keys=(valid_dataset.paths_key, *valid_dataset.bboxes_keys),
        **cfg.get("metric_args", {})
    )
    metrics_clb_constructor = MetricValCallbackDDP if is_ddp else MetricValCallback
    clb_metric = metrics_clb_constructor(
        metric=metrics_calc,
        log_images=cfg.get("log_images", False),
    )

    trainer = pl.Trainer(callbacks=[clb_metric], precision=cfg.get("precision", 32), **trainer_engine_params)

    if is_ddp:
        logs = trainer.validate(verbose=True, model=pl_model)
    else:
        logs = trainer.validate(dataloaders=loader_val, verbose=True, model=pl_model)

    return trainer, logs


__all__ = ["pl_val"]
