from argparse import ArgumentParser, Namespace
from itertools import chain
from random import randint, random, sample
from typing import Any, Dict, List, Tuple

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from oml.const import INDEX_KEY, INPUT_TENSORS_KEY, LABELS_KEY, TMP_PATH
from oml.ddp.utils import sync_dicts_ddp
from oml.lightning.callbacks.metric import MetricValCallback
from oml.lightning.modules.ddp import ModuleDDP
from oml.lightning.pipelines.parser import parse_engine_params_from_config
from oml.losses.arcface import ArcFaceLoss
from oml.metrics.embeddings import EmbeddingMetrics
from oml.utils.misc import set_global_seed
from tests.test_integrations.utils import EmbeddingsQueryGalleryLabeledDataset


def create_pred_and_gt_labels(
    num_labels: int, min_max_instances: Tuple[int, int], err_prob: float
) -> Tuple[List[int], List[int]]:
    gt_labels = []
    pred_labels = []

    all_labels = set(range(num_labels))
    for label in range(num_labels):
        instances = [label] * randint(*min_max_instances)
        gt_labels += instances
        if random() < err_prob:
            instances = instances.copy()
            instances[0] = sample(list(all_labels - {label}), k=1)[0]
        pred_labels += instances

    return pred_labels, gt_labels


class DummyModule(ModuleDDP):
    save_path_ckpt_pattern = str(TMP_PATH / "ckpt_experiment_{experiment}.pth")
    save_path_train_ids_pattern = str(TMP_PATH / "train_ids_{experiment}_{epoch}.pth")
    save_path_val_ids_pattern = str(TMP_PATH / "val_ids_{experiment}_{epoch}.pth")

    def __init__(
        self,
        exp_num: int,
        in_features: int,
        num_classes: int,
        loaders_val: EVAL_DATALOADERS,
        loaders_train: TRAIN_DATALOADERS,
    ):
        super().__init__(loaders_val=loaders_val, loaders_train=loaders_train)
        self.exp_num = exp_num
        self.model = nn.Sequential(nn.AvgPool2d((10, 10)), nn.Flatten(), nn.Linear(3, 3, bias=False))
        self.criterion = ArcFaceLoss(in_features=in_features, num_classes=num_classes)

        self.training_step_outputs: List[Any] = []
        self.validation_step_outputs: List[Any] = []

        self.len_train = len(loaders_train.dataset)
        self.len_val = len(loaders_val.dataset)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> Dict[str, Any]:
        if batch_idx == 0:
            self.validation_step_outputs = []

        embeddings = self.model(batch[INPUT_TENSORS_KEY])

        self.validation_step_outputs.append(batch)
        return {**batch, **{"embeddings": embeddings}}

    def training_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> Dict[str, Any]:
        if batch_idx == 0:
            self.training_step_outputs = []

        embeddings = self.model(batch[INPUT_TENSORS_KEY])
        loss = self.criterion(embeddings, batch[LABELS_KEY])
        batch["loss"] = loss

        self.training_step_outputs.append(batch)
        return batch

    def on_train_epoch_end(self) -> None:
        self.check_outputs_of_epoch(self.training_step_outputs)
        self.check_and_save_ids(self.training_step_outputs, "train")

    def on_validation_epoch_end(self) -> None:
        self.check_outputs_of_epoch(self.validation_step_outputs)
        self.check_and_save_ids(self.validation_step_outputs, "val")

    def check_outputs_of_epoch(self, outputs: List[Any]) -> None:
        # Check point 1 of motivation
        world_size = self.trainer.world_size
        output_batches = [tuple(out[INDEX_KEY].tolist()) for out in outputs]
        output_batches_synced = sync_dicts_ddp({"batches": output_batches}, world_size)["batches"]

        assert len(output_batches_synced) == len(output_batches) * world_size
        max_num_not_unique_batches = world_size - 1

        n_batches_synced = len(output_batches_synced)
        n_batches_synced_unique = len(set(output_batches_synced))
        assert n_batches_synced - n_batches_synced_unique <= max_num_not_unique_batches, (
            n_batches_synced,
            n_batches_synced_unique,
            max_num_not_unique_batches,
        )

    def check_and_save_ids(self, outputs: List[Any], mode: str) -> None:
        assert mode in ("train", "val")
        world_size = self.trainer.world_size

        len_dataset = self.len_train if mode == "train" else self.len_val

        ids_batches = [batch_dict[INDEX_KEY].tolist() for batch_dict in outputs]
        ids_flatten = list(chain(*ids_batches))

        ids_flatten_synced = sync_dicts_ddp({"ids_flatten": ids_flatten}, world_size)["ids_flatten"]
        ids_flatten_synced = list(set(ids_flatten_synced))  # we drop duplicates appeared because of DDP padding

        n_ids = len(ids_flatten_synced)
        n_ids_unique = len(set(ids_flatten_synced))
        assert n_ids == n_ids_unique == len_dataset, (n_ids, n_ids_unique, len_dataset)

        ids_per_step = {step: ids for step, ids in enumerate(ids_batches)}
        ids_per_step_synced = sync_dicts_ddp(ids_per_step, world_size)  # type: ignore
        ids_per_step_synced = {step: sorted(synced_ids) for step, synced_ids in ids_per_step_synced.items()}

        pattern = self.save_path_train_ids_pattern if mode == "train" else self.save_path_val_ids_pattern
        torch.save(ids_per_step_synced, pattern.format(experiment=self.exp_num, epoch=self.trainer.current_epoch))

    def configure_optimizers(self) -> Any:
        return Adam(params=self.parameters(), lr=1e-3)

    def on_train_end(self) -> None:
        torch.save(self.model, self.save_path_ckpt_pattern.format(experiment=self.exp_num))


class MetricValCallbackWithSaving(MetricValCallback):
    """
    We add saving of metrics for later comparison
    """

    save_path_pattern = str(TMP_PATH / "devices_{devices}_batch_size_{batch_size}_num_labels_{num_labels}.pth")

    def __init__(self, devices: int, batch_size: int, num_labels: int, *args: Any, **kwargs: Any):
        super(MetricValCallbackWithSaving, self).__init__(*args, **kwargs)
        self.devices = devices
        self.batch_size = batch_size
        self.num_labels = num_labels

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        ret = super().on_validation_epoch_end(trainer, pl_module)
        save_path = self.save_path_pattern.format(
            devices=self.devices, batch_size=self.batch_size, num_labels=self.num_labels
        )
        torch.save(self.metric.metrics, save_path)  # type: ignore
        return ret


def experiment(args: Namespace) -> None:
    set_global_seed(1)

    devices = args.devices
    max_epochs = args.max_epochs
    num_labels = args.num_labels
    batch_size = args.batch_size
    exp_num = args.exp_num

    pred_labels, gt_labels = create_pred_and_gt_labels(num_labels=num_labels, min_max_instances=(3, 5), err_prob=0.3)

    pred_tensors = torch.stack([label * torch.ones((3, 10, 10)) for label in pred_labels]).float()
    is_query = torch.ones(len(gt_labels)).bool()
    is_gallery = torch.ones(len(gt_labels)).bool()
    gt_labels = torch.tensor(gt_labels).long()

    val_dataset = EmbeddingsQueryGalleryLabeledDataset(pred_tensors, gt_labels, is_query, is_gallery)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size)

    train_dataset = EmbeddingsQueryGalleryLabeledDataset(pred_tensors, gt_labels, is_query, is_gallery)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)

    emb_metrics = EmbeddingMetrics(dataset=val_dataset, cmc_top_k=(5, 10), precision_top_k=(5, 10), map_top_k=(5, 10))
    val_callback = MetricValCallbackWithSaving(
        metric=emb_metrics, devices=devices, num_labels=num_labels, batch_size=batch_size
    )

    trainer_engine_params = parse_engine_params_from_config({"accelerator": "cpu", "devices": devices})

    pl_model = DummyModule(
        in_features=3,
        num_classes=num_labels,
        exp_num=exp_num,
        loaders_val=val_dataloader,
        loaders_train=train_dataloader,
    )

    trainer = Trainer(
        callbacks=[val_callback],
        max_epochs=max_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        **trainer_engine_params,
    )
    trainer.fit(model=pl_model)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--devices", type=int)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--num_labels", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--exp_num", type=int, default=0)
    return parser


if __name__ == "__main__":
    experiment(get_parser().parse_args())
