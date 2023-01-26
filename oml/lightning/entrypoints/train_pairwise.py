# type: ignore
# flake8: noqa
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pandas import DataFrame
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader

from oml.const import (
    EMBEDDINGS_KEY,
    INPUT_TENSORS_KEY,
    LABELS_KEY,
    PAIR_1ST_KEY,
    PAIR_2ND_KEY,
    PATHS_COLUMN,
    TCfg,
)
from oml.datasets.base import DatasetQueryGallery, DatasetWithLabels
from oml.inference.list_inference import inference_on_images
from oml.interfaces.models import IExtractor, IFreezable, IPairwiseModel
from oml.lightning.callbacks.metric import MetricValCallback, MetricValCallbackDDP
from oml.lightning.entrypoints.parser import (
    check_is_config_for_ddp,
    initialize_logging,
    parse_ckpt_callback_from_config,
    parse_engine_params_from_config,
    parse_sampler_from_config,
    parse_scheduler_from_config,
)
from oml.lightning.modules.module_ddp import ModuleDDP
from oml.metrics.embeddings import EmbeddingMetrics, EmbeddingMetricsDDP
from oml.miners.inbatch_hard_tri import HardTripletsMiner
from oml.registry.models import get_extractor_by_cfg
from oml.registry.optimizers import get_optimizer_by_cfg
from oml.registry.transforms import get_transforms_by_cfg
from oml.retrieval.postprocessors.pairwise import PairwiseImagesPostprocessor
from oml.transforms.images.utils import TTransforms
from oml.utils.misc import dictconfig_to_dict, load_dotenv, set_global_seed
from oml.utils.misc_torch import elementwise_dist


class PairwiseModule(pl.LightningModule):
    def __init__(
        self,
        pairwise_model: IPairwiseModel,
        pairs_miner,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        scheduler_interval: str = "step",
        scheduler_frequency: int = 1,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        pair_1st_key: str = PAIR_1ST_KEY,
        pair_2nd_key: str = PAIR_2ND_KEY,
        labels_key: str = LABELS_KEY,
        embeddings_key: str = EMBEDDINGS_KEY,
        scheduler_monitor_metric: Optional[str] = None,
        freeze_n_epochs: int = 0,
    ):
        pl.LightningModule.__init__(self)

        self.model = pairwise_model
        self.miner = pairs_miner
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency
        self.input_tensors_key = input_tensors_key
        self.pair_1st_key = pair_1st_key
        self.pair_2nd_key = pair_2nd_key
        self.labels_key = labels_key
        self.embeddings_key = embeddings_key
        self.monitor_metric = scheduler_monitor_metric
        self.freeze_n_epochs = freeze_n_epochs

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        ids1, ids2, is_negative = self.miner.sample(features=batch[self.embeddings_key], labels=batch[self.labels_key])
        x1 = batch[self.input_tensors_key][ids1]
        x2 = batch[self.input_tensors_key][ids2]
        target = is_negative.float()

        predictions = self.model(x1=x1, x2=x2)

        loss = self.criterion(predictions, target.to(predictions.device))

        bs = len(batch[self.labels_key])

        self.log("loss", loss.item(), prog_bar=True, batch_size=bs, on_step=True, on_epoch=True)

        if self.scheduler is not None:
            print("xxx lr", self.scheduler.get_last_lr()[0])
            self.log("lr", self.scheduler.get_last_lr()[0], prog_bar=True, batch_size=bs, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, *_: Any) -> Dict[str, Any]:
        return batch

    def configure_optimizers(self) -> Any:
        if self.scheduler is None:
            return self.optimizer
        else:
            scheduler = {
                "scheduler": self.scheduler,
                "interval": self.scheduler_interval,
                "frequency": self.scheduler_frequency,
            }
            if isinstance(self.scheduler, ReduceLROnPlateau):
                scheduler["monitor"] = self.monitor_metric
            return [self.optimizer], [scheduler]

    def on_epoch_start(self) -> None:
        if self.freeze_n_epochs and isinstance(self.model, IFreezable):
            if self.current_epoch >= self.freeze_n_epochs:
                self.model.unfreeze()
            else:
                self.model.freeze()

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        # https://github.com/Lightning-AI/lightning/issues/1595
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict


class PairwiseModuleDDP(PairwiseModule, ModuleDDP):
    def __init__(
        self,
        loaders_train: Optional[TRAIN_DATALOADERS] = None,
        loaders_val: Optional[EVAL_DATALOADERS] = None,
        *args: Any,
        **kwargs: Any,
    ):
        ModuleDDP.__init__(self, loaders_train=loaders_train, loaders_val=loaders_val)
        PairwiseModule.__init__(self, *args, **kwargs)


class PairsMiner:
    def __init__(self):
        self.miner = HardTripletsMiner()  # todo
        # self.miner = AllTripletsMiner()

    def sample(self, features, labels):
        ii_a, ii_p, ii_n = self.miner._sample(features, labels=labels)

        ii_a_1, ii_p = zip(*list(set(list(map(lambda x: tuple(sorted([x[0], x[1]])), zip(ii_a, ii_p))))))
        ii_a_2, ii_n = zip(*list(set(list(map(lambda x: tuple(sorted([x[0], x[1]])), zip(ii_a, ii_n))))))

        is_negative = torch.ones(len(ii_a_1) + len(ii_a_2)).bool()
        is_negative[: len(ii_a_1)] = 0

        return torch.tensor([*ii_a_1, *ii_a_2]).long(), torch.tensor([*ii_p, *ii_n]).long(), is_negative


class ImagesSiamese(IPairwiseModel, IFreezable):
    def __init__(self, extractor: IExtractor) -> None:
        super(ImagesSiamese, self).__init__()
        self.extractor = extractor
        feat_dim = self.extractor.feat_dim

        # todo: parametrize
        self.head = nn.Sequential(
            *[
                nn.Linear(feat_dim, feat_dim // 2, bias=True),
                nn.Dropout(),
                nn.Sigmoid(),
                nn.Linear(feat_dim // 2, 1, bias=False),
            ]
        )

        self.train_backbone = True

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x = torch.concat([x1, x2], dim=2)

        with torch.set_grad_enabled(self.train_backbone):
            x = self.extractor(x)

        x = self.head(x)
        x = x.squeeze()

        return x

    def freeze(self) -> None:
        self.train_backbone = False

    def unfreeze(self) -> None:
        self.train_backbone = True


class ImagesSiameseTrivial(IPairwiseModel):
    def __init__(self, extractor: IExtractor) -> None:
        super(ImagesSiameseTrivial, self).__init__()
        self.extractor = extractor

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.extractor(x1)
        x2 = self.extractor(x2)
        return elementwise_dist(x1, x2, p=2)


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
        ).cpu()
        if cache_on_disk:
            torch.save(embeddings, save_path)
            print("Embeddings have been saved to the disk.")

    train_mask = df["split"] == "train"

    emb_train = embeddings[train_mask]
    emb_val = embeddings[~train_mask]

    df_train = df[train_mask]
    df_train.reset_index(inplace=True)

    df_val = df[~train_mask]
    df_val.reset_index(inplace=True)

    return emb_train, emb_val, df_train, df_val


def get_loaders_with_embeddings(cfg: TCfg) -> Tuple[DataLoader, DataLoader]:
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    extractor = get_extractor_by_cfg(cfg["extractor"]).to(device)

    emb_train, emb_val, df_train, df_val = extract_embeddings(
        extractor=extractor,
        dataset_root=Path(cfg["dataset_root"]),
        save_root=Path(cfg["dataset_root"]),
        save_file_postfix=cfg["extractor"]["args"].get("weights", "unknown_weights"),
        dataframe_name=cfg["dataframe_name"],
        transforms_extraction=get_transforms_by_cfg(cfg["transforms_extraction"]),
        num_workers=cfg["num_workers"],
        batch_size=cfg["bs_val"],
        cache_on_disk=cfg["cache_extracted_embeddings_on_disk"],
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


def pl_train_pairwise(cfg: DictConfig) -> None:
    load_dotenv()

    set_global_seed(cfg["seed"])

    cfg = dictconfig_to_dict(cfg)
    pprint(cfg)
    logger = initialize_logging(cfg)

    trainer_engine_params = parse_engine_params_from_config(cfg)
    is_ddp = check_is_config_for_ddp(trainer_engine_params)

    loader_train, loader_val = get_loaders_with_embeddings(cfg)

    extractor = get_extractor_by_cfg(cfg["extractor"])
    # siamese = ImagesSiamese(extractor=extractor)
    siamese = ImagesSiameseTrivial(extractor=extractor)
    criterion = torch.nn.BCEWithLogitsLoss()

    pairs_miner = PairsMiner()
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
        top_n=cfg["top_n"],
        transforms=get_transforms_by_cfg(cfg["transforms_val"]),
        batch_size=cfg["bs_val"],
        num_workers=cfg["num_workers"],
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


__all__ = ["pl_train_pairwise"]
