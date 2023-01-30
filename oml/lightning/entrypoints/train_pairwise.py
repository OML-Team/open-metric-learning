from pathlib import Path
from pprint import pprint
from typing import List, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pandas import DataFrame
from torch import Tensor
from torch import device as tdevice
from torch.utils.data import DataLoader

from oml.const import EMBEDDINGS_KEY, PATHS_COLUMN, TCfg
from oml.datasets.base import DatasetQueryGallery, DatasetWithLabels
from oml.inference.list_inference import inference_on_images
from oml.interfaces.miners import ITripletsMinerInBatch
from oml.interfaces.models import IExtractor
from oml.lightning.callbacks.metric import MetricValCallback, MetricValCallbackDDP
from oml.lightning.entrypoints.parser import (
    check_is_config_for_ddp,
    get_cfg_md5sum,
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
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.miners.inbatch_hard_tri import HardTripletsMiner
from oml.registry.models import get_extractor_by_cfg, get_pairwise_model_by_cfg
from oml.registry.optimizers import get_optimizer_by_cfg
from oml.registry.transforms import get_transforms_by_cfg
from oml.retrieval.postprocessors.pairwise import PairwiseImagesPostprocessor
from oml.transforms.images.utils import TTransforms
from oml.utils.misc import dictconfig_to_dict, load_dotenv, set_global_seed


class PairsMiner:
    miner: ITripletsMinerInBatch

    def __init__(self, mode: str):
        super().__init__()
        assert mode in ("hard", "all")

        self.mode = mode

        if self.mode == "all":
            self.miner = AllTripletsMiner()
        elif self.mode == "hard":
            self.miner = HardTripletsMiner()
        else:
            ValueError(f"Unexpected mining mode {self.mode}")

    def sample(self, features: Tensor, labels: List[int]) -> Tuple[Tensor, Tensor, Tensor]:
        ii_a, ii_p, ii_n = self.miner._sample(features, labels=labels)

        ii_a_1, ii_p = zip(*list(set(list(map(lambda x: tuple(sorted([x[0], x[1]])), zip(ii_a, ii_p))))))
        ii_a_2, ii_n = zip(*list(set(list(map(lambda x: tuple(sorted([x[0], x[1]])), zip(ii_a, ii_n))))))

        is_negative = torch.ones(len(ii_a_1) + len(ii_a_2)).bool()
        is_negative[: len(ii_a_1)] = 0

        return torch.tensor([*ii_a_1, *ii_a_2]).long(), torch.tensor([*ii_p, *ii_n]).long(), is_negative


def extract_embeddings(
    dataset_root: Path,
    dataframe_name: str,
    save_root: Path,
    save_file_postfix: str,
    extractor: IExtractor,
    transforms_extraction: TTransforms,
    num_workers: int,
    batch_size: int,
    cache_on_disk: int,
    use_fp16: bool,
) -> Tuple[Tensor, Tensor, DataFrame, DataFrame]:
    df = pd.read_csv(dataset_root / dataframe_name)

    # it has now affect if paths are global already
    df[PATHS_COLUMN] = df[PATHS_COLUMN].apply(lambda x: Path(dataset_root) / x)

    save_path = Path(save_root / f"embeddings_{save_file_postfix}.pkl")
    if save_path.is_file() and cache_on_disk:
        embeddings = torch.load(save_path, map_location="cpu")
        print("Embeddings have been loaded from the disk.")
    else:
        embeddings = inference_on_images(
            model=extractor,
            paths=df[PATHS_COLUMN],
            transform=transforms_extraction,
            num_workers=num_workers,
            batch_size=batch_size,
            verbose=True,
            use_fp16=use_fp16,
        ).cpu()
        if cache_on_disk:
            torch.save(embeddings, save_path)
            print("Embeddings have been saved to the disk.")

    train_mask = df["split"] == "train"

    emb_train = embeddings[train_mask]
    emb_val = embeddings[~train_mask]

    df_train = df[train_mask]
    df_train.reset_index(inplace=True, drop=True)

    df_val = df[~train_mask]
    df_val.reset_index(inplace=True, drop=True)

    return emb_train, emb_val, df_train, df_val


def get_loaders_with_embeddings(cfg: TCfg) -> Tuple[DataLoader, DataLoader]:
    device = tdevice("cuda:0") if parse_engine_params_from_config(cfg)["accelerator"] == "gpu" else tdevice("cpu")
    extractor = get_extractor_by_cfg(cfg["extractor"]).to(device)

    emb_train, emb_val, df_train, df_val = extract_embeddings(
        extractor=extractor,
        dataset_root=Path(cfg["dataset_root"]),
        save_root=Path(cfg["dataset_root"]),
        save_file_postfix=get_cfg_md5sum(cfg)[:5],
        dataframe_name=cfg["dataframe_name"],
        transforms_extraction=get_transforms_by_cfg(cfg["transforms_extraction"]),
        num_workers=cfg["num_workers"],
        batch_size=cfg["bs_val"],
        cache_on_disk=cfg["cache_extracted_embeddings_on_disk"],
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

    pairs_miner = PairsMiner(mode="hard")
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
    metrics_clb = metrics_clb_constructor(
        metric=metrics_calc,
        log_images=cfg.get("log_images", True),
    )

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
