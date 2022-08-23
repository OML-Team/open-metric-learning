from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from oml.const import TCfg
from oml.datasets.retrieval import get_retrieval_datasets
from oml.lightning.callbacks.metric import MetricValCallback
from oml.lightning.modules.retrieval import RetrievalModule
from oml.metrics.embeddings import EmbeddingMetrics
from oml.registry.models import get_extractor_by_cfg
from oml.utils.misc import dictconfig_to_dict


def pl_val(cfg: TCfg) -> Tuple[pl.Trainer, Dict[str, Any]]:
    """
    This is an entrypoint for the model validation in metric learning setup.

    The config can be specified as a dictionary or with hydra: https://hydra.cc/
    For more details look at examples/README.md

    """
    cfg = dictconfig_to_dict(cfg)
    pprint(cfg)

    _, valid_dataset = get_retrieval_datasets(
        dataset_root=Path(cfg["dataset_root"]),
        im_size_train=cfg["im_size"],
        im_size_val=cfg["im_size"],
        pad_ratio_train=0,
        pad_ratio_val=0,
        train_transform=None,
        dataframe_name=cfg["dataframe_name"],
    )
    loader_val = DataLoader(dataset=valid_dataset, batch_size=cfg["bs_val"], num_workers=cfg["num_workers"])

    extractor = get_extractor_by_cfg(cfg["model"])
    pl_model = RetrievalModule(model=extractor, criterion=None, loaders_val=loader_val, optimizer=None, scheduler=None)

    metrics_calc = EmbeddingMetrics(
        categories_key=valid_dataset.categories_key,
        extra_keys=("paths", "x1", "x2", "y1", "y2"),
        **cfg.get("metric_args", {})
    )
    clb_metric = MetricValCallback(metric=metrics_calc, log_only_main_category=cfg.get("log_only_main_category", True))

    trainer = pl.Trainer(
        gpus=cfg["gpus"],
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=False) if (cfg["gpus"] and len(cfg["gpus"]) > 1) else None,
        replace_sampler_ddp=False,
        callbacks=[clb_metric],
        precision=cfg.get("precision", 32),
    )

    logs = trainer.validate(model=pl_model, verbose=True)

    return trainer, logs


__all__ = ["pl_val"]
