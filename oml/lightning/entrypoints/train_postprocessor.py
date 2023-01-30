import hashlib
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import device as tdevice
from torch.utils.data import DataLoader

from oml.const import BBOXES_COLUMNS, EMBEDDINGS_KEY, TCfg
from oml.datasets.base import DatasetQueryGallery, DatasetWithLabels
from oml.inference.flat import inference_on_dataframe
from oml.lightning.callbacks.metric import MetricValCallback, MetricValCallbackDDP
from oml.lightning.entrypoints.parser import (
    check_is_config_for_ddp,
    initialize_logging,
    parse_ckpt_callback_from_config,
    parse_engine_params_from_config,
    parse_sampler_from_config,
    parse_scheduler_from_config,
)
from oml.lightning.modules.pairwise_postprocessing import (
    PairwiseModule,
    PairwiseModuleDDP,
)
from oml.metrics.embeddings import EmbeddingMetrics, EmbeddingMetricsDDP
from oml.miners.pairs import PairsMiner
from oml.registry.models import get_extractor_by_cfg, get_pairwise_model_by_cfg
from oml.registry.optimizers import get_optimizer_by_cfg
from oml.registry.transforms import get_transforms_by_cfg
from oml.retrieval.postprocessors.pairwise import PairwiseImagesPostprocessor
from oml.utils.misc import (
    dictconfig_to_dict,
    flatten_dict,
    load_dotenv,
    set_global_seed,
)


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


def get_loaders_with_embeddings(cfg: TCfg) -> Tuple[DataLoader, DataLoader]:
    # todo: support bounding bboxes
    df = pd.read_csv(Path(cfg["dataset_root"]) / cfg["dataframe_name"])
    assert not set(BBOXES_COLUMNS).intersection(
        df.columns
    ), "We've found bboxes in the dataframe, but they're not supported yet."

    device = tdevice("cuda:0") if parse_engine_params_from_config(cfg)["accelerator"] == "gpu" else tdevice("cpu")
    extractor = get_extractor_by_cfg(cfg["extractor"]).to(device)

    if cfg["embeddings_cache_dir"] is not None:
        cache_file = Path(cfg["embeddings_cache_dir"]) / f"embeddings_{get_hash_of_extraction_stage_cfg(cfg)[:5]}.pkl"
    else:
        cache_file = None

    emb_train, emb_val, df_train, df_val = inference_on_dataframe(
        extractor=extractor,
        dataset_root=cfg["dataset_root"],
        output_cache_path=cache_file,
        dataframe_name=cfg["dataframe_name"],
        transforms_extraction=get_transforms_by_cfg(cfg["transforms_extraction"]),
        num_workers=cfg["num_workers"],
        batch_size=cfg["bs_val"],
        use_fp16=int(cfg.get("precision", 32)) == 16,
    )

    train_dataset = DatasetWithLabels(
        df=df_train,
        transform=get_transforms_by_cfg(cfg["transforms_train"]),
        extra_data={EMBEDDINGS_KEY: emb_train},
    )

    valid_dataset = DatasetQueryGallery(
        df=df_val,
        transform=get_transforms_by_cfg(cfg["transforms_val"]),
        extra_data={EMBEDDINGS_KEY: emb_val},
    )

    sampler = parse_sampler_from_config(cfg, dataset=train_dataset)
    assert sampler is not None
    loader_train = DataLoader(batch_sampler=sampler, dataset=train_dataset, num_workers=cfg["num_workers"])

    loader_val = DataLoader(
        dataset=valid_dataset, batch_size=cfg["bs_val"], num_workers=cfg["num_workers"], shuffle=False
    )

    return loader_train, loader_val


def pl_train_postprocessor(cfg: DictConfig) -> None:
    load_dotenv()

    set_global_seed(cfg["seed"])

    cfg = dictconfig_to_dict(cfg)
    pprint(cfg)
    logger = initialize_logging(cfg)

    trainer_engine_params = parse_engine_params_from_config(cfg)
    is_ddp = check_is_config_for_ddp(trainer_engine_params)

    loader_train, loader_val = get_loaders_with_embeddings(cfg)

    siamese = get_pairwise_model_by_cfg(cfg["pairwise_model"])
    criterion = torch.nn.BCEWithLogitsLoss()
    pairs_miner = PairsMiner(hard_mining=cfg["hard_pairs_mining"])
    optimizer = get_optimizer_by_cfg(cfg["optimizer"], **{"params": siamese.parameters()})

    module_kwargs = {}
    module_kwargs.update(parse_scheduler_from_config(cfg, optimizer=optimizer))
    if is_ddp:
        module_kwargs.update({"loaders_train": loader_train, "loaders_val": loader_val})
        module_constructor = PairwiseModuleDDP
    else:
        module_constructor = PairwiseModule  # type: ignore

    pl_module = module_constructor(
        pairwise_model=siamese,
        pairs_miner=pairs_miner,
        criterion=criterion,
        optimizer=optimizer,
        input_tensors_key=loader_train.dataset.input_tensors_key,
        labels_key=loader_train.dataset.labels_key,
        embeddings_key=EMBEDDINGS_KEY,
        freeze_n_epochs=cfg.get("freeze_n_epochs", 0),
        **module_kwargs,
    )

    postprocessor = PairwiseImagesPostprocessor(
        pairwise_model=siamese,
        top_n=cfg["postprocessing_top_n"],
        transforms=get_transforms_by_cfg(cfg["transforms_val"]),
        batch_size=cfg["bs_val"],
        num_workers=cfg["num_workers"],
        use_fp16=int(cfg.get("precision", 32)) == 16,
    )

    metrics_constructor = EmbeddingMetricsDDP if is_ddp else EmbeddingMetrics
    metrics_calc = metrics_constructor(
        embeddings_key=pl_module.embeddings_key,
        categories_key=loader_val.dataset.categories_key,
        labels_key=loader_val.dataset.labels_key,
        is_query_key=loader_val.dataset.is_query_key,
        is_gallery_key=loader_val.dataset.is_gallery_key,
        extra_keys=(loader_val.dataset.paths_key, *loader_val.dataset.bboxes_keys),
        postprocessor=postprocessor,
        **cfg.get("metric_args", {}),
    )

    metrics_clb_constructor = MetricValCallbackDDP if is_ddp else MetricValCallback
    metrics_clb = metrics_clb_constructor(metric=metrics_calc, log_images=cfg.get("log_images", True))

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


__all__ = ["pl_train_postprocessor"]
