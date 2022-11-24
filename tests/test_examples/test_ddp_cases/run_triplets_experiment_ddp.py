from argparse import ArgumentParser, Namespace
from itertools import chain
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from oml.const import INPUT_TENSORS_KEY, MOCK_DATASET_PATH, TMP_PATH
from oml.datasets.triplet import TriDataset, TTriplet, tri_collate
from oml.ddp.utils import sync_dicts_ddp
from oml.lightning.entrypoints.parser import parse_engine_params_from_config
from oml.lightning.modules.module_ddp import (
    ModuleDDP,
    TTrainDataloaders,
    TValDataloaders,
)
from oml.losses.triplet import TripletLossPlain
from oml.miners.inbatch_all_tri import get_available_triplets
from oml.transforms.images.albumentations.transforms import (
    get_normalisation_resize_albu,
)
from oml.utils.download_mock_dataset import download_mock_dataset
from oml.utils.misc import set_global_seed


def get_triplets_from_retrieval_setup(df: pd.DataFrame) -> Tuple[List[TTriplet], List[TTriplet]]:
    outputs = []

    for split in ["train", "validation"]:
        df_split = df[df["split"] == split].reset_index(drop=True)
        ids_apn = get_available_triplets(list(df_split["label"]))
        triplets = list(zip(*map(lambda x: [df_split["path"][idx] for idx in x], ids_apn)))

        outputs.append(triplets)

    return outputs  # type: ignore


class DummyModule(ModuleDDP):
    save_path_ckpt_pattern = str(TMP_PATH / "ckpt_experiment_{experiment}.pth")
    save_path_train_ids_pattern = str(TMP_PATH / "train_ids_{experiment}_{epoch}.pth")
    save_path_val_ids_pattern = str(TMP_PATH / "val_ids_{experiment}_{epoch}.pth")

    def __init__(self, exp_num: int, loaders_val: TValDataloaders, loaders_train: TTrainDataloaders):
        super().__init__(loaders_val=loaders_val, loaders_train=loaders_train)
        self.exp_num = exp_num
        # We use a simple model, instead of ViT or ResNet, because they have Dropouts
        self.model = nn.Sequential(
            nn.AvgPool2d((32, 32)), nn.Flatten(), *([nn.Sigmoid(), nn.Linear(3, 3, bias=False)] * 3)
        )
        self.criterion = TripletLossPlain(margin=None)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, *_: Any) -> Dict[str, Any]:
        embeddings = self.model(batch[INPUT_TENSORS_KEY])
        return {**batch, **{"embeddings": embeddings}}

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        embeddings = self.model(batch[INPUT_TENSORS_KEY])
        loss = self.criterion(embeddings)
        batch["loss"] = loss
        return batch

    def check_and_save_ids(self, outputs: EPOCH_OUTPUT, mode: str) -> None:
        assert mode in ("train", "val")
        world_size = self.trainer.world_size

        ids_batches = [[int(idx_str[:-2]) for idx_str in batch_dict["tri_ids"][::3]] for batch_dict in outputs]
        ids_flatten = list(chain(*ids_batches))

        ids_flatten_synced = sync_dicts_ddp({"ids_flatten": ids_flatten}, world_size)["ids_flatten"]
        assert len(ids_flatten_synced) == len(ids_flatten) * world_size == len(set(ids_flatten_synced))

        ids_per_step = {step: ids for step, ids in enumerate(ids_batches)}
        ids_per_step_synced = sync_dicts_ddp(ids_per_step, world_size)  # type: ignore
        ids_per_step_synced = {step: sorted(synced_ids) for step, synced_ids in ids_per_step_synced.items()}

        pattern = self.save_path_train_ids_pattern if mode == "train" else self.save_path_val_ids_pattern
        torch.save(ids_per_step_synced, pattern.format(experiment=self.exp_num, epoch=self.trainer.current_epoch))

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.check_and_save_ids(outputs, "train")

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.check_and_save_ids(outputs, "val")

    def on_train_end(self) -> None:
        torch.save(self.model, self.save_path_ckpt_pattern.format(experiment=self.exp_num))

    def configure_optimizers(self) -> Any:
        return Adam(params=self.parameters(), lr=1e-3)


def experiment(args: Namespace) -> None:
    download_mock_dataset(MOCK_DATASET_PATH)

    devices = args.devices
    len_dataset = args.len_dataset
    batch_size = args.batch_size
    exp_num = args.exp_num
    max_epochs = args.max_epochs

    assert (
        len_dataset % (batch_size * devices) == 0
    ), "For this experiment, the parameters must be a multiple of each other"

    set_global_seed(1)

    df = pd.read_csv(MOCK_DATASET_PATH / "df.csv")
    triplets_train, triplets_val = get_triplets_from_retrieval_setup(df)

    assert len(triplets_train) > len_dataset
    assert len(triplets_val) > len_dataset
    triplets_train = triplets_train[:len_dataset]
    triplets_val = triplets_val[:len_dataset]

    transform = get_normalisation_resize_albu(im_size=32)
    train_dataset = TriDataset(triplets_train, transforms=transform, im_root=MOCK_DATASET_PATH, expand_ratio=0)
    val_dataset = TriDataset(triplets_val, transforms=transform, im_root=MOCK_DATASET_PATH, expand_ratio=0)

    train_dataloader = DataLoader(
        dataset=train_dataset, shuffle=True, drop_last=False, batch_size=batch_size, collate_fn=tri_collate
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, shuffle=False, drop_last=False, batch_size=batch_size, collate_fn=tri_collate
    )

    trainer_engine_params = parse_engine_params_from_config({"accelerator": "cpu", "devices": devices})
    pl_model = DummyModule(exp_num=exp_num, loaders_val=val_dataloader, loaders_train=train_dataloader)

    trainer = Trainer(
        max_epochs=max_epochs,
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        logger=False,
        **trainer_engine_params,
    )
    trainer.fit(model=pl_model)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--devices", type=int)
    parser.add_argument("--exp_num", type=int)
    parser.add_argument("--len_dataset", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_epochs", type=int)
    return parser


if __name__ == "__main__":
    experiment(get_parser().parse_args())
