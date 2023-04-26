from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from oml.const import TCfg
from oml.datasets.base import get_retrieval_datasets
from oml.lightning.callbacks.metric import MetricValCallback, MetricValCallbackDDP
from oml.lightning.modules.extractor import ExtractorModule, ExtractorModuleDDP
from oml.lightning.pipelines.parser import (
    check_is_config_for_ddp,
    parse_engine_params_from_config,
)
from oml.metrics.embeddings import EmbeddingMetrics, EmbeddingMetricsDDP
from oml.registry.models import get_extractor_by_cfg
from oml.registry.postprocessors import get_postprocessor_by_cfg
from oml.registry.transforms import get_transforms_by_cfg
from oml.utils.misc import dictconfig_to_dict


def extractor_validation_pipeline(cfg: TCfg) -> Tuple[pl.Trainer, Dict[str, Any]]:
    """
    This pipeline allows you to validate a feature extractor which
    represents images as feature vectors.

    The config can be specified as a dictionary or with hydra: https://hydra.cc/.
    For more details look at ``pipelines/features_extraction/README.md``

    """
    cfg = dictconfig_to_dict(cfg)
    trainer_engine_params = parse_engine_params_from_config(cfg)
    is_ddp = check_is_config_for_ddp(trainer_engine_params)

    pprint(cfg)

    _, valid_dataset = get_retrieval_datasets(
        dataset_root=Path(cfg["dataset_root"]),
        transforms_train=None,
        transforms_val=get_transforms_by_cfg(cfg["transforms_val"]),
        dataframe_name=cfg["dataframe_name"],
    )
    loader_val = DataLoader(dataset=valid_dataset, batch_size=cfg["bs_val"], num_workers=cfg["num_workers"])

    extractor = get_extractor_by_cfg(cfg["extractor"])

    module_kwargs = {}
    if is_ddp:
        module_kwargs["loaders_val"] = loader_val
        module_constructor = ExtractorModuleDDP
    else:
        module_constructor = ExtractorModule  # type: ignore

    pl_model = module_constructor(
        extractor=extractor,
        criterion=None,
        optimizer=None,
        scheduler=None,
        input_tensors_key=valid_dataset.input_tensors_key,
        labels_key=valid_dataset.labels_key,
        **module_kwargs
    )

    postprocessor = None if not cfg.get("postprocessor", None) else get_postprocessor_by_cfg(cfg["postprocessor"])

    # Note! We add the link to our extractor to a Lightning's Module, so it can recognize it and manipulate its devices
    pl_model.model_link_ = getattr(postprocessor, "extractor", None)

    metrics_constructor = EmbeddingMetricsDDP if is_ddp else EmbeddingMetrics
    metrics_calc = metrics_constructor(
        embeddings_key=pl_model.embeddings_key,
        categories_key=valid_dataset.categories_key,
        labels_key=valid_dataset.labels_key,
        is_query_key=valid_dataset.is_query_key,
        is_gallery_key=valid_dataset.is_gallery_key,
        extra_keys=(valid_dataset.paths_key, *valid_dataset.bboxes_keys),
        postprocessor=postprocessor,
        **cfg.get("metric_args", {})
    )
    metrics_clb_constructor = MetricValCallbackDDP if is_ddp else MetricValCallback
    clb_metric = metrics_clb_constructor(
        metric=metrics_calc,
        log_images=False,
    )

    trainer = pl.Trainer(callbacks=[clb_metric], precision=cfg.get("precision", 32), **trainer_engine_params)

    if is_ddp:
        logs = trainer.validate(verbose=True, model=pl_model)
    else:
        logs = trainer.validate(dataloaders=loader_val, verbose=True, model=pl_model)

    return trainer, logs


__all__ = ["extractor_validation_pipeline"]
