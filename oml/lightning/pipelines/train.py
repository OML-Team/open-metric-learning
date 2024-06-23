from pathlib import Path
from pprint import pprint
from typing import Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from oml.const import TCfg
from oml.datasets.images import get_retrieval_images_datasets
from oml.interfaces.datasets import ILabeledDataset, IQueryGalleryLabeledDataset
from oml.lightning.callbacks.metric import MetricValCallback
from oml.lightning.modules.extractor import ExtractorModule, ExtractorModuleDDP
from oml.lightning.pipelines.parser import (
    check_is_config_for_ddp,
    parse_ckpt_callback_from_config,
    parse_engine_params_from_config,
    parse_logger_from_config,
    parse_sampler_from_config,
    parse_scheduler_from_config,
)
from oml.metrics.embeddings import EmbeddingMetrics
from oml.registry.losses import get_criterion_by_cfg
from oml.registry.models import get_extractor_by_cfg
from oml.registry.optimizers import get_optimizer_by_cfg
from oml.registry.transforms import get_transforms_by_cfg
from oml.utils.misc import dictconfig_to_dict, set_global_seed


def get_retrieval_loaders(cfg: TCfg) -> Tuple[DataLoader, DataLoader, ILabeledDataset, IQueryGalleryLabeledDataset]:
    train_dataset, valid_dataset = get_retrieval_images_datasets(
        dataset_root=Path(cfg["dataset_root"]),
        transforms_train=get_transforms_by_cfg(cfg["transforms_train"]),
        transforms_val=get_transforms_by_cfg(cfg["transforms_val"]),
        dataframe_name=cfg["dataframe_name"],
        cache_size=cfg.get("cache_size", 0),
        verbose=cfg.get("show_dataset_warnings", True),
    )

    sampler = parse_sampler_from_config(cfg, train_dataset)

    if sampler is None:
        loader_train = DataLoader(
            dataset=train_dataset,
            num_workers=cfg["num_workers"],
            batch_size=cfg["bs_train"],
            drop_last=True,
            shuffle=True,
        )
    else:
        loader_train = DataLoader(
            dataset=train_dataset,
            batch_sampler=sampler,
            num_workers=cfg["num_workers"],
        )

    loader_val = DataLoader(dataset=valid_dataset, batch_size=cfg["bs_val"], num_workers=cfg["num_workers"])

    return loader_train, loader_val, train_dataset, valid_dataset


def extractor_training_pipeline(cfg: TCfg) -> None:
    """
    This pipeline allows you to train and validate a feature extractor which
    represents images as feature vectors.

    The config can be specified as a dictionary or with hydra: https://hydra.cc/.
    For more details look at ``pipelines/features_extraction/README.md``

    """
    set_global_seed(cfg["seed"])

    cfg = dictconfig_to_dict(cfg)
    pprint(cfg)

    logger = parse_logger_from_config(cfg)
    logger.log_pipeline_info(cfg)

    trainer_engine_params = parse_engine_params_from_config(cfg)
    is_ddp = check_is_config_for_ddp(trainer_engine_params)

    loader_train, loaders_val, dataset_train, dataset_val = get_retrieval_loaders(cfg)
    label2category = dataset_train.get_label2category()
    extractor = get_extractor_by_cfg(cfg["extractor"])
    criterion = get_criterion_by_cfg(cfg["criterion"], **{"label2category": label2category})  # type: ignore
    optimizable_parameters = [
        {"lr": cfg["optimizer"]["args"]["lr"], "params": extractor.parameters()},
        {"lr": cfg["optimizer"]["args"]["lr"], "params": criterion.parameters()},
    ]
    optimizer = get_optimizer_by_cfg(cfg["optimizer"], **{"params": optimizable_parameters})  # type: ignore

    module_kwargs = {}
    module_kwargs.update(parse_scheduler_from_config(cfg, optimizer=optimizer))
    if is_ddp:
        module_kwargs.update({"loaders_train": loader_train, "loaders_val": loaders_val})
        module_constructor = ExtractorModuleDDP
    else:
        module_constructor = ExtractorModule  # type: ignore

    pl_module = module_constructor(
        extractor=extractor,
        criterion=criterion,
        optimizer=optimizer,
        input_tensors_key=dataset_train.input_tensors_key,
        labels_key=dataset_train.labels_key,
        freeze_n_epochs=cfg.get("freeze_n_epochs", 0),
        **module_kwargs,
    )

    metric = EmbeddingMetrics(dataset=dataset_val, **cfg.get("metric_args", {}))
    metrics_clb = MetricValCallback(metric=metric, log_images=cfg.get("log_images", False))

    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        num_sanity_val_steps=0,
        check_val_every_n_epoch=cfg["valid_period"],
        default_root_dir=str(Path.cwd()),
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        callbacks=[metrics_clb, parse_ckpt_callback_from_config(cfg)],
        logger=logger,
        precision=cfg.get("precision", 32),
        **trainer_engine_params,
        **cfg.get("lightning_trainer_extra_args", {}),
    )

    if is_ddp:
        trainer.fit(model=pl_module)
    else:
        trainer.fit(model=pl_module, train_dataloaders=loader_train, val_dataloaders=loaders_val)


__all__ = ["extractor_training_pipeline"]
