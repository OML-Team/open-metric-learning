import os
from pathlib import Path
from pprint import pprint

import albumentations as albu
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from torch.utils.data import DataLoader

from oml.const import (
    CATEGORIES_COLUMN,
    LABELS_COLUMN,
    OVERALL_CATEGORIES_KEY,
    PROJECT_ROOT,
    TCfg,
)
from oml.datasets.base import get_retrieval_datasets
from oml.interfaces.models import IExtractor
from oml.lightning.callbacks.metric import MetricValCallback, MetricValCallbackDDP
from oml.lightning.entrypoints.parser import (
    check_is_config_for_ddp,
    parse_engine_params_from_config,
)
from oml.lightning.modules.retrieval import RetrievalModule, RetrievalModuleDDP
from oml.metrics.embeddings import EmbeddingMetrics, EmbeddingMetricsDDP
from oml.registry.losses import get_criterion_by_cfg
from oml.registry.models import get_extractor_by_cfg
from oml.registry.optimizers import get_optimizer_by_cfg
from oml.registry.samplers import SAMPLERS_CATEGORIES_BASED, get_sampler_by_cfg
from oml.registry.schedulers import get_scheduler_by_cfg
from oml.registry.transforms import get_transforms_by_cfg
from oml.utils.misc import (
    dictconfig_to_dict,
    flatten_dict,
    load_dotenv,
    set_global_seed,
)


def pl_train(cfg: TCfg) -> None:
    """
    This is an entrypoint for the model training in metric learning setup.

    The config can be specified as a dictionary or with hydra: https://hydra.cc/.
    For more details look at ``examples/README.md``

    """
    cfg = dictconfig_to_dict(cfg)
    trainer_engine_params = parse_engine_params_from_config(cfg)
    is_ddp = check_is_config_for_ddp(trainer_engine_params)

    pprint(cfg)

    set_global_seed(cfg["seed"])

    cwd = Path.cwd()

    transforms_train = get_transforms_by_cfg(cfg["transforms_train"])
    transforms_val = get_transforms_by_cfg(cfg["transforms_val"])

    train_dataset, valid_dataset = get_retrieval_datasets(
        dataset_root=Path(cfg["dataset_root"]),
        transforms_train=transforms_train,
        transforms_val=transforms_val,
        dataframe_name=cfg["dataframe_name"],
        cache_size=cfg["cache_size"],
        verbose=cfg.get("show_dataset_warnings", True),
    )

    if isinstance(transforms_train, albu.Compose):
        augs_file = ".hydra/augs_cfg.yaml" if Path(".hydra").exists() else "augs_cfg.yaml"
        albu.save(filepath=augs_file, transform=transforms_train, data_format="yaml")
    else:
        augs_file = None

    if (not train_dataset.categories_key) and cfg["sampler"]["name"] in SAMPLERS_CATEGORIES_BASED.keys():
        raise ValueError(
            "NOTE! You are trying to use Sampler which works with the information related"
            "to categories, but there is no <categories_key> in your Dataset."
        )

    sampler_runtime_args = {"labels": train_dataset.get_labels()}
    label2category = None
    df = train_dataset.df
    if train_dataset.categories_key:
        label2category = dict(zip(df[LABELS_COLUMN], df[CATEGORIES_COLUMN]))
        sampler_runtime_args["label2category"] = label2category
    # note, we pass some runtime arguments to sampler here, but not all of the samplers use all of these arguments
    sampler = get_sampler_by_cfg(cfg["sampler"], **sampler_runtime_args) if cfg["sampler"] is not None else None

    extractor = get_extractor_by_cfg(cfg["model"])

    criterion = get_criterion_by_cfg(
        cfg["criterion"],
        label2category=label2category,
    )
    optimizable_parameters = [
        {"lr": cfg["optimizer"]["args"]["lr"], "params": extractor.parameters()},
        {"lr": cfg["optimizer"]["args"]["lr"], "params": criterion.parameters()},
    ]
    optimizer = get_optimizer_by_cfg(cfg["optimizer"], params=optimizable_parameters)  # type: ignore

    # unpack scheduler to the Lightning format
    if cfg.get("scheduling"):
        scheduler_kwargs = {
            "scheduler": get_scheduler_by_cfg(cfg["scheduling"]["scheduler"], optimizer=optimizer),
            "scheduler_interval": cfg["scheduling"]["scheduler_interval"],
            "scheduler_frequency": cfg["scheduling"]["scheduler_frequency"],
            "scheduler_monitor_metric": cfg["scheduling"].get("monitor_metric", None),
        }
    else:
        scheduler_kwargs = {"scheduler": None}

    assert isinstance(extractor, IExtractor), "You model must to be child of IExtractor"

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

    loaders_val = DataLoader(dataset=valid_dataset, batch_size=cfg["bs_val"], num_workers=cfg["num_workers"])

    module_kwargs = scheduler_kwargs
    if is_ddp:
        module_kwargs.update({"loaders_train": loader_train, "loaders_val": loaders_val})
        module_constructor = RetrievalModuleDDP
    else:
        module_constructor = RetrievalModule  # type: ignore

    pl_model = module_constructor(
        model=extractor,
        criterion=criterion,
        optimizer=optimizer,
        input_tensors_key=train_dataset.input_tensors_key,
        labels_key=train_dataset.labels_key,
        freeze_n_epochs=cfg.get("freeze_n_epochs", 0),
        **module_kwargs,
    )

    metrics_constructor = EmbeddingMetricsDDP if is_ddp else EmbeddingMetrics
    metrics_calc = metrics_constructor(
        embeddings_key=pl_model.embeddings_key,
        categories_key=valid_dataset.categories_key,
        labels_key=valid_dataset.labels_key,
        is_query_key=valid_dataset.is_query_key,
        is_gallery_key=valid_dataset.is_gallery_key,
        extra_keys=(valid_dataset.paths_key, *valid_dataset.bboxes_keys),
        **cfg.get("metric_args", {}),
    )

    metrics_clb_constructor = MetricValCallbackDDP if is_ddp else MetricValCallback
    metrics_clb = metrics_clb_constructor(
        metric=metrics_calc,
        log_images=cfg.get("log_images", False),
    )
    ckpt_clb = pl.callbacks.ModelCheckpoint(
        dirpath=Path.cwd() / "checkpoints",
        monitor=cfg.get("metric_for_checkpointing", f"{OVERALL_CATEGORIES_KEY}/cmc/1"),
        mode="max",
        save_top_k=1,
        verbose=True,
        filename="best",
    )

    # Here we try to load NEPTUNE_API_TOKEN from .env file
    # You can also set it up via `export NEPTUNE_API_TOKEN=...`
    load_dotenv()
    if ("NEPTUNE_API_TOKEN" in os.environ.keys()) and (cfg["neptune_project"] is not None):
        logger = NeptuneLogger(
            api_key=os.environ["NEPTUNE_API_TOKEN"],
            project=cfg["neptune_project"],
            tags=list(cfg["tags"]) + [cfg["postfix"]] + [cwd.name],
            log_model_checkpoints=False,
        )
        # log hyper params and files
        dict_to_log = {**dictconfig_to_dict(cfg), **{"dir": cwd}}
        logger.log_hyperparams(flatten_dict(dict_to_log, sep="|"))
        logger.run["dataset"].upload(str(Path(cfg["dataset_root"]) / cfg["dataframe_name"]))
        if augs_file is not None:
            logger.run["augs_cfg"].upload(augs_file)
        # log source code
        source_files = list(map(lambda x: str(x), PROJECT_ROOT.glob("**/*.py"))) + list(
            map(lambda x: str(x), PROJECT_ROOT.glob("**/*.yaml"))
        )
        logger.run["code"].upload_files(source_files)

    else:
        logger = True

    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        num_sanity_val_steps=0,
        check_val_every_n_epoch=cfg["valid_period"],
        default_root_dir=cwd,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        callbacks=[metrics_clb, ckpt_clb],
        logger=logger,
        precision=cfg.get("precision", 32),
        **trainer_engine_params,
    )

    if is_ddp:
        trainer.fit(model=pl_model)
    else:
        trainer.fit(model=pl_model, train_dataloaders=loader_train, val_dataloaders=loaders_val)


__all__ = ["pl_train"]
