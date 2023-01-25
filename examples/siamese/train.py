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

from oml.const import EMBEDDINGS_KEY
from oml.datasets.base import DatasetQueryGallery, DatasetWithLabels
from oml.lightning.entrypoints.parser import (
    check_is_config_for_ddp,
    initialize_logging,
    parse_engine_params_from_config,
    parse_sampler_from_config,
    parse_scheduler_from_config,
)
from oml.registry.models import get_extractor_by_cfg
from oml.registry.optimizers import get_optimizer_by_cfg
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
    valid_loader = DataLoader(
        dataset=valid_dataset, batch_size=cfg["bs_val"], num_workers=cfg["num_workers"], shuffle=False
    )

    postprocessor = PairwiseImagesPostprocessor(
        top_n=cfg["top_n"],
        pairwise_model=siamese,
        transforms=transforms_val,
        batch_size=cfg["bs_val"],
        num_workers=cfg["num_workers"],
    )

    module_kwargs = {}
    module_kwargs.update(parse_scheduler_from_config(cfg, optimizer=optimizer))
    if is_ddp:
        module_kwargs["loaders_train"] = loader_train
        module_kwargs["loaders_val"] = valid_loader
        module_constructor = PairwiseModuleDDP
    else:
        module_constructor = PairwiseModule  # type: ignore

    logger = initialize_logging(cfg)

    # todo: key from val dataset
    pl_model = module_constructor(
        pairwise_model=siamese,
        pairs_miner=pairs_miner,
        optimizer=optimizer,
        input_tensors_key=train_dataset.input_tensors_key,
        labels_key=train_dataset.labels_key,
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
        logger=logger,
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
