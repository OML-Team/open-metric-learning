import hashlib
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import device as tdevice
from torch.utils.data import DataLoader

from oml.const import EMBEDDINGS_KEY, TCfg
from oml.datasets.base import ImageLabeledDataset, ImageQueryGalleryLabeledDataset
from oml.datasets.images import get_retrieval_images_datasets
from oml.inference import inference, inference_cached
from oml.interfaces.datasets import ILabeledDataset, IQueryGalleryLabeledDataset
from oml.interfaces.models import IPairwiseModel
from oml.lightning.callbacks.metric import MetricValCallback
from oml.lightning.modules.pairwise_postprocessing import (
    PairwiseModule,
    PairwiseModuleDDP,
)
from oml.lightning.pipelines.parser import (
    check_is_config_for_ddp,
    parse_ckpt_callback_from_config,
    parse_engine_params_from_config,
    parse_logger_from_config,
    parse_sampler_from_config,
    parse_scheduler_from_config,
)
from oml.metrics.embeddings import EmbeddingMetrics
from oml.miners.pairs import PairsMiner
from oml.registry.models import get_extractor_by_cfg
from oml.registry.optimizers import get_optimizer_by_cfg
from oml.registry.postprocessors import get_postprocessor_by_cfg
from oml.registry.transforms import get_transforms_by_cfg
from oml.retrieval.postprocessors.pairwise import PairwiseReranker
from oml.utils.misc import dictconfig_to_dict, flatten_dict, set_global_seed


def get_hash_of_extraction_stage_cfg(cfg: TCfg) -> str:
    def dict2str(dictionary: Dict[str, Any]) -> str:
        flatten_items = flatten_dict(dictionary).items()
        sorted(flatten_items, key=lambda x: x[0])
        return str(flatten_items)

    cfg_extraction_str = (
        dict2str(cfg["extractor"])
        + dict2str(cfg["transforms_extraction"])
        + str(cfg["dataframe_name"])
        + str(cfg.get("precision", 32))
    )

    md5sum = hashlib.md5(cfg_extraction_str.encode("utf-8")).hexdigest()
    return md5sum


def get_loaders_with_embeddings(
    cfg: TCfg,
) -> Tuple[DataLoader, DataLoader, ILabeledDataset, IQueryGalleryLabeledDataset]:
    device = tdevice("cuda:0") if parse_engine_params_from_config(cfg)["accelerator"] == "gpu" else tdevice("cpu")
    extractor = get_extractor_by_cfg(cfg["extractor"]).to(device)

    transforms_extraction = get_transforms_by_cfg(cfg["transforms_extraction"])

    train_extraction, val_extraction = get_retrieval_images_datasets(
        dataset_root=Path(cfg["dataset_root"]),
        dataframe_name=cfg["dataframe_name"],
        transforms_train=transforms_extraction,
        transforms_val=transforms_extraction,
    )

    args = {
        "model": extractor,
        "num_workers": cfg["num_workers"],
        "batch_size": cfg["batch_size_inference"],
        "use_fp16": int(cfg.get("precision", 32)) == 16,
    }

    if cfg["embeddings_cache_dir"] is not None:
        hash_ = get_hash_of_extraction_stage_cfg(cfg)[:5]
        dir_ = Path(cfg["embeddings_cache_dir"])
        emb_train = inference_cached(dataset=train_extraction, cache_path=str(dir_ / f"emb_train_{hash_}.pkl"), **args)
        emb_val = inference_cached(dataset=val_extraction, cache_path=str(dir_ / f"emb_val_{hash_}.pkl"), **args)
    else:
        emb_train = inference(dataset=train_extraction, **args)
        emb_val = inference(dataset=val_extraction, **args)

    train_dataset = ImageLabeledDataset(
        dataset_root=cfg["dataset_root"],
        cache_size=cfg.get("cache_size", 0),
        df=train_extraction.df,
        transform=get_transforms_by_cfg(cfg["transforms_train"]),
        extra_data={EMBEDDINGS_KEY: emb_train},
    )

    valid_dataset = ImageQueryGalleryLabeledDataset(
        dataset_root=cfg["dataset_root"],
        cache_size=cfg.get("cache_size", 0),
        df=val_extraction.df,
        transform=transforms_extraction,
        extra_data={EMBEDDINGS_KEY: emb_val},
    )

    sampler = parse_sampler_from_config(cfg, dataset=train_dataset)
    assert sampler is not None, "We will be training on pairs, so, having sampler is obligatory."

    loader_train = DataLoader(batch_sampler=sampler, dataset=train_dataset, num_workers=cfg["num_workers"])

    loader_val = DataLoader(
        dataset=valid_dataset, batch_size=cfg["batch_size_inference"], num_workers=cfg["num_workers"], shuffle=False
    )

    return loader_train, loader_val, train_dataset, valid_dataset


def postprocessor_training_pipeline(cfg: DictConfig) -> None:
    """
    This pipeline allows you to train and validate a pairwise postprocessor
    which fixes mistakes of a feature extractor in retrieval setup.

    The config can be specified as a dictionary or with hydra: https://hydra.cc/.
    For more details look at ``pipelines/postprocessing/pairwise_postprocessing/README.md``

    """
    set_global_seed(cfg["seed"])

    cfg = dictconfig_to_dict(cfg)
    pprint(cfg)

    logger = parse_logger_from_config(cfg)
    logger.log_pipeline_info(cfg)

    trainer_engine_params = parse_engine_params_from_config(cfg)
    is_ddp = check_is_config_for_ddp(trainer_engine_params)

    loader_train, loader_val, dataset_train, dataset_val = get_loaders_with_embeddings(cfg)

    postprocessor = None if not cfg.get("postprocessor", None) else get_postprocessor_by_cfg(cfg["postprocessor"])
    assert isinstance(postprocessor, PairwiseReranker), f"We only support {PairwiseReranker.__name__} at the moment."
    assert isinstance(postprocessor.model, IPairwiseModel), f"You model must be a child of {IPairwiseModel.__name__}"

    criterion = torch.nn.BCEWithLogitsLoss()
    pairs_miner = PairsMiner(hard_mining=cfg["hard_pairs_mining"])
    optimizer = get_optimizer_by_cfg(cfg["optimizer"], **{"params": postprocessor.model.parameters()})

    module_kwargs = {}
    module_kwargs.update(parse_scheduler_from_config(cfg, optimizer=optimizer))
    if is_ddp:
        module_kwargs.update({"loaders_train": loader_train, "loaders_val": loader_val})
        module_constructor = PairwiseModuleDDP
    else:
        module_constructor = PairwiseModule  # type: ignore

    pl_module = module_constructor(
        pairwise_model=postprocessor.model,
        pairs_miner=pairs_miner,
        criterion=criterion,
        optimizer=optimizer,
        input_tensors_key=dataset_train.input_tensors_key,
        labels_key=dataset_train.labels_key,
        embeddings_key=EMBEDDINGS_KEY,
        freeze_n_epochs=cfg.get("freeze_n_epochs", 0),
        **module_kwargs,
    )

    metric = EmbeddingMetrics(dataset=dataset_val, postprocessor=postprocessor, **cfg.get("metric_args", {}))
    metrics_clb = MetricValCallback(metric=metric, log_images=cfg.get("log_images", True))

    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        num_sanity_val_steps=0,
        check_val_every_n_epoch=cfg["valid_period"],
        default_root_dir=str(Path.cwd()),
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        precision=cfg.get("precision", 32),
        logger=logger,
        callbacks=[metrics_clb, parse_ckpt_callback_from_config(cfg)],
        **trainer_engine_params,
    )

    if is_ddp:
        trainer.fit(model=pl_module)
    else:
        trainer.fit(model=pl_module, train_dataloaders=loader_train, val_dataloaders=loader_val)


__all__ = ["postprocessor_training_pipeline"]
