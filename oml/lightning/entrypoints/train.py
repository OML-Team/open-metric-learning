import os
import warnings
from pathlib import Path

import albumentations as albu
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader

from oml.const import PROJECT_ROOT, TCfg
from oml.datasets.retrieval import get_retrieval_datasets
from oml.interfaces.criterions import ITripletLossWithMiner
from oml.interfaces.models import IExtractor
from oml.lightning.callbacks.metric import MetricValCallback
from oml.lightning.modules.retrieval import RetrievalModule
from oml.metrics.embeddings import EmbeddingMetrics
from oml.registry.losses import get_criterion_by_cfg
from oml.registry.models import get_extractor_by_cfg
from oml.registry.optimizers import get_optimizer_by_cfg
from oml.registry.samplers import SAMPLERS_CATEGORIES_BASED, get_sampler_by_cfg
from oml.registry.schedulers import get_scheduler_by_cfg
from oml.registry.transforms import get_augs
from oml.utils.misc import (
    dictconfig_to_dict,
    flatten_dict,
    load_dotenv,
    set_global_seed,
)


def pl_train(cfg: TCfg) -> None:
    cfg = dictconfig_to_dict(cfg)
    print(cfg)

    set_global_seed(cfg["seed"], cfg["num_workers"])

    cwd = Path.cwd()

    train_augs = get_augs(cfg["augs"]) if cfg["augs"] is not None else None
    train_dataset, valid_dataset = get_retrieval_datasets(
        dataset_root=Path(cfg["dataset_root"]),
        im_size_train=cfg["im_size_train"],
        im_size_val=cfg["im_size_val"],
        pad_ratio_train=cfg["im_pad_ratio_train"],
        pad_ratio_val=cfg["im_pad_ratio_val"],
        train_transform=train_augs,
        dataframe_name=cfg["dataframe_name"],
        cache_size=cfg["cache_size"],
    )
    df = train_dataset.df

    if isinstance(train_augs, albu.Compose):
        augs_file = ".hydra/augs_cfg.yaml" if Path(".hydra").exists() else "augs_cfg.yaml"
        albu.save(filepath=augs_file, transform=train_augs, data_format="yaml")
    else:
        augs_file = None

    if "category" not in df.columns:
        df["category"] = 0
        if cfg["sampler"]["name"] in SAMPLERS_CATEGORIES_BASED.keys():
            warnings.warn(
                "NOTE! You are trying to use Sampler which works with the information related"
                "to categories, but there is no <category> column in your DataFrame."
                "We will add this column filled with the trivial value."
            )

    # note, we pass some runtime arguments to sampler here, but not all of the samplers use all of these arguments
    runtime_args = {"labels": train_dataset.get_labels(), "label2category": dict(zip(df["label"], df["category"]))}
    sampler = get_sampler_by_cfg(cfg["sampler"], **runtime_args) if cfg["sampler"] is not None else None

    extractor = get_extractor_by_cfg(cfg["model"])
    criterion = get_criterion_by_cfg(cfg["criterion"])
    optimizer = get_optimizer_by_cfg(cfg["optimizer"], params=extractor.parameters())
    scheduler = get_scheduler_by_cfg(cfg["scheduler"], optimizer=optimizer) if cfg["scheduler"] is not None else None

    assert isinstance(extractor, IExtractor), "You model must to be child of IExtractor"
    assert isinstance(criterion, ITripletLossWithMiner), "You criterion must be child of ITripletLossWithMiner"

    loader_train = DataLoader(
        dataset=train_dataset,
        sampler=sampler,
        num_workers=cfg["num_workers"],
        batch_size=sampler.batch_size,
        drop_last=True,
    )

    loaders_val = DataLoader(dataset=valid_dataset, batch_size=cfg["bs_val"], num_workers=cfg["num_workers"])

    metrics_calc = EmbeddingMetrics(**cfg.get("metric_args", {}))
    metrics_clb = MetricValCallback(metric=metrics_calc)
    ckpt_clb = pl.callbacks.ModelCheckpoint(
        dirpath=Path.cwd() / "checkpoints",
        monitor="OVERALL/cmc/1",
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
        # log hyper params and augs config
        dict_to_log = {**dictconfig_to_dict(cfg), **{"dir": cwd}}
        logger.log_hyperparams(flatten_dict(dict_to_log, sep="|"))
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
        replace_sampler_ddp=False,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=cfg["valid_period"],
        default_root_dir=cwd,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        num_nodes=1,
        gpus=cfg["gpus"],
        strategy=DDPPlugin(find_unused_parameters=False) if cfg["gpus"] else None,
        callbacks=[metrics_clb, ckpt_clb],
        logger=logger,
    )

    pl_model = RetrievalModule(model=extractor, criterion=criterion, optimizer=optimizer, scheduler=scheduler)

    trainer.fit(model=pl_model, train_dataloaders=loader_train, val_dataloaders=loaders_val)


__all__ = ["pl_train"]
