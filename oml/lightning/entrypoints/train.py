from pathlib import Path
from pprint import pprint
from typing import Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from oml.const import TCfg
from oml.datasets.base import get_retrieval_datasets
from oml.lightning.callbacks.metric import MetricValCallback, MetricValCallbackDDP
from oml.lightning.entrypoints.parser import (
    check_is_config_for_ddp,
    initialize_logging,
    parse_ckpt_callback_from_config,
    parse_engine_params_from_config,
    parse_sampler_from_config,
    parse_scheduler_from_config,
)
from oml.lightning.modules.retrieval import RetrievalModule, RetrievalModuleDDP
from oml.metrics.embeddings import EmbeddingMetrics, EmbeddingMetricsDDP
from oml.registry.losses import get_criterion_by_cfg
from oml.registry.models import get_extractor_by_cfg
from oml.registry.optimizers import get_optimizer_by_cfg
from oml.registry.transforms import get_transforms_by_cfg
from oml.utils.misc import dictconfig_to_dict, load_dotenv, set_global_seed


def get_retrieval_loaders(cfg: TCfg) -> Tuple[DataLoader, DataLoader]:
    train_dataset, valid_dataset = get_retrieval_datasets(
        dataset_root=Path(cfg["dataset_root"]),
        transforms_train=get_transforms_by_cfg(cfg["transforms_train"]),
        transforms_val=get_transforms_by_cfg(cfg["transforms_val"]),
        dataframe_name=cfg["dataframe_name"],
        cache_size=cfg["cache_size"],
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

    return loader_train, loader_val


def pl_train(cfg: TCfg) -> None:
    """
    This is an entrypoint for the model training in metric learning setup.

    The config can be specified as a dictionary or with hydra: https://hydra.cc/.
    For more details look at ``examples/README.md``

    """
    # Here we try to load NEPTUNE_API_TOKEN from .env file
    # You can also set it up via `export NEPTUNE_API_TOKEN=...`
    load_dotenv()

    set_global_seed(cfg["seed"])

    cfg = dictconfig_to_dict(cfg)
    pprint(cfg)
    logger = initialize_logging(cfg)

    trainer_engine_params = parse_engine_params_from_config(cfg)
    is_ddp = check_is_config_for_ddp(trainer_engine_params)

    loader_train, loaders_val = get_retrieval_loaders(cfg)
    extractor = get_extractor_by_cfg(cfg["model"])
    criterion = get_criterion_by_cfg(cfg["criterion"], **{"label2category": loader_train.dataset.get_label2category()})
    optimizable_parameters = [
        {"lr": cfg["optimizer"]["args"]["lr"], "params": extractor.parameters()},
        {"lr": cfg["optimizer"]["args"]["lr"], "params": criterion.parameters()},
    ]
    optimizer = get_optimizer_by_cfg(cfg["optimizer"], **{"params": optimizable_parameters})  # type: ignore

    module_kwargs = {}
    module_kwargs.update(parse_scheduler_from_config(cfg, optimizer=optimizer))
    if is_ddp:
        module_kwargs.update({"loaders_train": loader_train, "loaders_val": loaders_val})
        module_constructor = RetrievalModuleDDP
    else:
        module_constructor = RetrievalModule  # type: ignore

    pl_module = module_constructor(
        model=extractor,
        criterion=criterion,
        optimizer=optimizer,
        input_tensors_key=loader_train.dataset.input_tensors_key,
        labels_key=loader_train.dataset.labels_key,
        freeze_n_epochs=cfg.get("freeze_n_epochs", 0),
        **module_kwargs,
    )

    metrics_constructor = EmbeddingMetricsDDP if is_ddp else EmbeddingMetrics
    metrics_calc = metrics_constructor(
        embeddings_key=pl_module.embeddings_key,
        categories_key=loaders_val.dataset.categories_key,
        labels_key=loaders_val.dataset.labels_key,
        is_query_key=loaders_val.dataset.is_query_key,
        is_gallery_key=loaders_val.dataset.is_gallery_key,
        extra_keys=(loaders_val.dataset.paths_key, *loaders_val.dataset.bboxes_keys),
        **cfg.get("metric_args", {}),
    )

    metrics_clb_constructor = MetricValCallbackDDP if is_ddp else MetricValCallback
    metrics_clb = metrics_clb_constructor(
        metric=metrics_calc,
        log_images=cfg.get("log_images", False),
    )

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
    )

    if is_ddp:
        trainer.fit(model=pl_module)
    else:
        trainer.fit(model=pl_module, train_dataloaders=loader_train, val_dataloaders=loaders_val)


__all__ = ["pl_train"]
