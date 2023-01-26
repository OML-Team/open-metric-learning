# type: ignore
# flake8: noqa
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Optional, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from oml.const import (
    EMBEDDINGS_KEY,
    INPUT_TENSORS_KEY,
    LABELS_KEY,
    PAIR_1ST_KEY,
    PAIR_2ND_KEY,
    PATHS_COLUMN,
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
from oml.transforms.images.utils import get_im_reader_for_transforms
from oml.utils.misc import dictconfig_to_dict, load_dotenv, set_global_seed
from oml.utils.misc_torch import elementwise_dist


class PairwiseModule(pl.LightningModule):
    def __init__(
        self,
        pairwise_model,
        pairs_miner,
        optimizer,
        scheduler,
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

        self.criterion = BCEWithLogitsLoss()

    def training_step(self, batch, batch_idx):
        ids1, ids2, is_negative = self.miner.sample(features=batch[self.embeddings_key], labels=batch[self.labels_key])
        x1 = batch[self.input_tensors_key][ids1]
        x2 = batch[self.input_tensors_key][ids2]
        target = is_negative.float()

        predictions = self.model(x1=x1, x2=x2)

        loss = self.criterion(predictions, target.to(predictions.device))

        bs = len(batch[self.labels_key])

        self.log("loss", loss.item(), prog_bar=True, batch_size=bs, on_step=True, on_epoch=True)

        if self.scheduler is not None:
            self.log("lr", self.scheduler.get_last_lr()[0], prog_bar=True, batch_size=bs, on_step=True, on_epoch=False)

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


def get_embeddings(
    dataset_root,
    dataframe_name,
    extractor,
    transforms,
    num_workers,
    batch_size,
):
    postfix = getattr(extractor, "weights")
    save_path = Path(dataset_root / f"embeddings_{postfix}.pkl")

    df = pd.read_csv(dataset_root / dataframe_name)
    df[PATHS_COLUMN] = df[PATHS_COLUMN].apply(lambda x: Path(dataset_root) / x)  # todo

    if save_path.is_file():
        embeddings = torch.load(save_path, map_location="cpu")
    else:
        embeddings = inference_on_images(
            model=extractor,
            paths=df[PATHS_COLUMN],
            transform=transforms,
            num_workers=num_workers,
            batch_size=batch_size,
            verbose=True,
        ).cpu()
        torch.save(embeddings, save_path)

    train_mask = df["split"] == "train"

    emb_train = embeddings[train_mask]
    emb_val = embeddings[~train_mask]

    df_train = df[train_mask]
    df_train.reset_index(inplace=True)

    df_val = df[~train_mask]
    df_val.reset_index(inplace=True)

    return emb_train, emb_val, df_train, df_val


def main(cfg: DictConfig) -> None:
    load_dotenv()

    cfg = dictconfig_to_dict(cfg)
    trainer_engine_params = parse_engine_params_from_config(cfg)
    is_ddp = check_is_config_for_ddp(trainer_engine_params)

    pprint(cfg)

    set_global_seed(cfg["seed"])

    cwd = Path.cwd()

    # Features extraction
    extractor = get_extractor_by_cfg(cfg["extractor"]).to("cuda:0")
    emb_train, emb_val, df_train, df_val = get_embeddings(
        extractor=extractor,
        dataset_root=Path(cfg["dataset_root"]),
        dataframe_name=cfg["dataframe_name"],
        transforms=get_transforms_by_cfg(cfg["transforms_extraction"]),
        num_workers=cfg["num_workers"],
        batch_size=cfg["bs_val"],
    )

    # Pairwise training
    # siamese = ImagesSiamese(extractor=extractor)
    siamese = ImagesSiameseTrivial(extractor=extractor)

    pairs_miner = PairsMiner()
    optimizer = get_optimizer_by_cfg(cfg["optimizer"], params=siamese.parameters())  # type: ignore

    transforms_train = get_transforms_by_cfg(cfg["transforms_train"])
    train_dataset = DatasetWithLabels(
        df=df_train,
        transform=transforms_train,
        f_imread=get_im_reader_for_transforms(transforms_train),
        extra_data={EMBEDDINGS_KEY: emb_train},
    )
    sampler = parse_sampler_from_config(cfg, dataset=train_dataset)
    loader_train = DataLoader(batch_sampler=sampler, dataset=train_dataset, num_workers=cfg["num_workers"])

    # Pairwise validation as postprocessor
    transforms_val = get_transforms_by_cfg(cfg["transforms_val"])
    valid_dataset = DatasetQueryGallery(
        df=df_val,
        transform=transforms_val,
        f_imread=get_im_reader_for_transforms(transforms_val),
        extra_data={EMBEDDINGS_KEY: emb_val},
    )
    loader_val = DataLoader(
        dataset=valid_dataset, batch_size=cfg["bs_val"], num_workers=cfg["num_workers"], shuffle=False
    )

    module_kwargs = {}
    module_kwargs.update(parse_scheduler_from_config(cfg, optimizer=optimizer))
    if is_ddp:
        module_kwargs["loaders_train"] = loader_train
        module_kwargs["loaders_val"] = loader_val
        module_constructor = PairwiseModuleDDP
    else:
        module_constructor = PairwiseModule  # type: ignore

    logger = initialize_logging(cfg)

    pl_module = module_constructor(
        pairwise_model=siamese,
        pairs_miner=pairs_miner,
        optimizer=optimizer,
        input_tensors_key=train_dataset.input_tensors_key,
        labels_key=train_dataset.labels_key,
        embeddings_key=EMBEDDINGS_KEY,
        freeze_n_epochs=cfg.get("freeze_n_epochs", 0),
        **module_kwargs,
    )

    metrics_constructor = EmbeddingMetricsDDP if is_ddp else EmbeddingMetrics

    postprocessor = PairwiseImagesPostprocessor(
        top_n=cfg["top_n"],
        pairwise_model=siamese,
        transforms=transforms_val,
        batch_size=cfg["bs_val"],
        num_workers=cfg["num_workers"],
    )

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
        default_root_dir=str(cwd),
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        precision=cfg.get("precision", 32),
        logger=logger,
        callbacks=[metrics_clb, parse_ckpt_callback_from_config(cfg)],
        **trainer_engine_params,
    )

    if is_ddp:
        # trainer.fit(model=pl_module)
        trainer.validate(verbose=True, model=pl_module)
    else:
        trainer.fit(model=pl_module, train_dataloaders=loader_train, val_dataloaders=loader_val)


__all__ = ["main"]
