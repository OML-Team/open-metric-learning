# type: ignore
# flake8: noqa
from pathlib import Path
from pprint import pprint

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from source import (
    ImagesSiamese,
    PairsMiner,
    PairwiseModule,
    PairwiseModuleDDP,
    get_embeddings,
)
from torch.utils.data import DataLoader

from oml.const import CATEGORIES_COLUMN, EMBEDDINGS_KEY, LABELS_COLUMN
from oml.datasets.base import DatasetQueryGallery, DatasetWithLabels
from oml.lightning.entrypoints.parser import (
    check_is_config_for_ddp,
    parse_engine_params_from_config,
)
from oml.registry.models import get_extractor_by_cfg
from oml.registry.optimizers import get_optimizer_by_cfg
from oml.registry.samplers import get_sampler_by_cfg
from oml.registry.schedulers import get_scheduler_by_cfg
from oml.registry.transforms import get_transforms_by_cfg
from oml.retrieval.postprocessors.pairwise import PairwiseImagesPostprocessor
from oml.transforms.images.utils import get_im_reader_for_transforms
from oml.utils.misc import dictconfig_to_dict, set_global_seed


def main(cfg: DictConfig) -> None:
    cfg = dictconfig_to_dict(cfg)
    trainer_engine_params = parse_engine_params_from_config(cfg)
    is_ddp = check_is_config_for_ddp(trainer_engine_params)

    pprint(cfg)

    set_global_seed(cfg["seed"])

    cwd = Path.cwd()

    # Features extraction
    extractor = get_extractor_by_cfg(cfg["extractor"])
    emb_train, emb_val, df_train, df_val = get_embeddings(
        extractor=extractor,
        dataset_root=Path(cfg["dataset_root"]),
        transforms=get_transforms_by_cfg(cfg["transforms_extraction"]),
        num_workers=cfg["num_workers"],
        batch_size=cfg["bs_val"],
    )

    # Pairwise training
    siamese = ImagesSiamese(extractor=extractor)

    pairs_miner = PairsMiner()
    optimizer = get_optimizer_by_cfg(cfg["optimizer"], params=siamese.parameters())  # type: ignore

    transforms_train = get_transforms_by_cfg(cfg["transforms_train"])
    dataset = DatasetWithLabels(
        df=df_train,
        transform=transforms_train,
        f_imread=get_im_reader_for_transforms(transforms_train),
        extra_data={EMBEDDINGS_KEY: emb_train},
    )

    sampler_runtime_args = {
        "labels": dataset.get_labels(),
        "label2category": dict(zip(df_train[LABELS_COLUMN], df_train[CATEGORIES_COLUMN])),
    }
    sampler = get_sampler_by_cfg(cfg["sampler"], **sampler_runtime_args) if cfg["sampler"] is not None else None

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

    loader_train = DataLoader(batch_sampler=sampler, dataset=dataset, num_workers=cfg["num_workers"])

    # Pairwise validation as postprocessor
    transforms_val = get_transforms_by_cfg(cfg["transforms_val"])
    valid_dataset = DatasetQueryGallery(
        df=df_val,
        transform=transforms_val,
        f_imread=get_im_reader_for_transforms(transforms_val),
        extra_data={EMBEDDINGS_KEY: emb_val},
    )
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=cfg["bs_val"], num_workers=cfg["num_workers"])

    postprocessor = PairwiseImagesPostprocessor(
        top_n=cfg["top_n"],
        pairwise_model=siamese,
        transforms=transforms_val,
        batch_size=cfg["bs_val"],
        num_workers=cfg["num_workers"],
    )

    module_kwargs = scheduler_kwargs
    if is_ddp:
        module_kwargs["loaders_train"] = loader_train
        module_kwargs["loaders_val"] = valid_loader
        module_constructor = PairwiseModuleDDP
    else:
        module_constructor = PairwiseModule  # type: ignore

    # todo: key from val dataset
    pl_model = module_constructor(
        pairwise_model=siamese,
        pairs_miner=pairs_miner,
        optimizer=optimizer,
        input_tensors_key=dataset.input_tensors_key,
        labels_key=dataset.labels_key,
        embeddings_key=EMBEDDINGS_KEY,
        freeze_n_epochs=cfg.get("freeze_n_epochs", 0),
        **module_kwargs
    )

    # Run
    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        num_sanity_val_steps=0,
        check_val_every_n_epoch=cfg["valid_period"],
        default_root_dir=str(cwd),
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        precision=cfg.get("precision", 32),
        **trainer_engine_params
    )

    if is_ddp:
        trainer.fit(model=pl_model)
    else:
        trainer.fit(model=pl_model, train_dataloaders=loader_train, val_dataloaders=valid_loader)


@hydra.main(config_path=".", config_name="train.yaml")
def main_hydra(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    main_hydra()
