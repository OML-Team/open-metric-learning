from argparse import Namespace, ArgumentParser
from pathlib import Path
from random import randint, random, sample
from typing import Any, Dict, Tuple, List

import torch
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS, EPOCH_OUTPUT
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from oml.const import TMP_PATH
from oml.lightning.callbacks.metric import MetricValCallback
from oml.lightning.entrypoints.utils import parse_runtime_params_from_config
from oml.lightning.modules.module_ddp import ModuleDDP
from oml.losses.triplet import TripletLossWithMiner
from oml.metrics.embeddings import EmbeddingMetrics
from oml.samplers.balance import BalanceSampler
from oml.utils.ddp import sync_dicts_ddp
from oml.utils.misc import set_global_seed


def create_pred_and_gt_labels(num_labels: int, min_max_instances: Tuple[int, int], err_prob: float) -> Tuple[List[int], List[int]]:
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


class DummyDataset(Dataset):
    input_name = "input_tensors"
    labels_name = "labels"
    item_name = "item"

    def __init__(self, pred_labels: List[int], gt_labels: List[int]):
        self.gt_labels = gt_labels
        self.pred_labels = pred_labels

    def __len__(self) -> int:
        return len(self.gt_labels)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return {
            self.input_name: self.pred_labels[item] * torch.ones((3, 10, 10)),
            self.labels_name: self.gt_labels[item],
            "is_query": True,
            "is_gallery": True,
            self.item_name: item
        }


class DummyModule(ModuleDDP):
    def __init__(self, loaders_val: EVAL_DATALOADERS, loaders_train: TRAIN_DATALOADERS):
        super().__init__(loaders_val=loaders_val, loaders_train=loaders_train)
        self.model = nn.Sequential(nn.AvgPool2d((10, 10)), nn.Flatten(), nn.Linear(3, 3, bias=False))
        self.criterion = TripletLossWithMiner(margin=None)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, *dataset_idx: int) -> Dict[str, Any]:
        embeddings = self.model(batch[DummyDataset.input_name])
        return {**batch, **{"embeddings": embeddings}}

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        embeddings = self.model(batch[DummyDataset.input_name])
        loss = self.criterion(embeddings, batch[DummyDataset.labels_name])
        batch['loss'] = loss
        return batch

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.check_outputs_of_epoch(outputs)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.check_outputs_of_epoch(outputs)

    def check_outputs_of_epoch(self, outputs: EPOCH_OUTPUT):
        world_size = self.trainer.world_size
        output_batches = [tuple(out[DummyDataset.item_name].tolist()) for out in outputs]
        output_batches_synced = sync_dicts_ddp({"batches": output_batches}, world_size)["batches"]

        assert len(output_batches_synced) == len(output_batches) * world_size
        max_num_not_unique_batches = world_size
        assert len(output_batches_synced) - len(set(output_batches_synced)) <= max_num_not_unique_batches

    def configure_optimizers(self) -> Any:
        return Adam(params=self.parameters(), lr=1e-7)


class MetricValCallbackWithSaving(MetricValCallback):
    save_path_pattern = str(TMP_PATH / 'devices_{devices}_batch_size_{batch_size}_num_labels_{num_labels}_epoch_{epoch}.pth')

    def __init__(self, devices: int, batch_size: int, num_labels: int, *args, **kwargs):
        super(MetricValCallbackWithSaving, self).__init__(*args, **kwargs)
        self.devices = devices
        self.batch_size = batch_size
        self.num_labels = num_labels

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        ret = super().on_validation_epoch_end(trainer, pl_module)
        save_path = self.save_path_pattern.format(devices=self.devices,
                                                  batch_size=self.batch_size,
                                                  num_labels=self.num_labels,
                                                  epoch=trainer.current_epoch)
        torch.save(self.metric.metrics, save_path)
        return ret



def experiment(args: Namespace) -> None:
    set_global_seed(1)

    devices = args.devices
    save_path = Path(args.save_path)
    num_labels = args.num_labels
    batch_size = args.batch_size

    pred_labels, gt_labels = create_pred_and_gt_labels(num_labels=num_labels, min_max_instances=(3, 5), err_prob=0.3)

    val_dataset = DummyDataset(gt_labels=gt_labels, pred_labels=pred_labels)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size)

    train_dataset = DummyDataset(gt_labels=gt_labels, pred_labels=gt_labels)
    batch_sampler = BalanceSampler(labels=gt_labels, n_labels=2, n_instances=2)
    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)

    emb_metrics = EmbeddingMetrics(cmc_top_k=(5, 10), precision_top_k=(5, 10), map_top_k=(5, 10))
    val_callback = MetricValCallback(metric=emb_metrics)

    trainer_engine_params = parse_runtime_params_from_config({"accelerator": "cpu", "devices": devices})

    pl_model = DummyModule(loaders_val=val_dataloader, loaders_train=train_dataloader)

    if save_path is not None:
        trainer = Trainer(callbacks=[val_callback], **trainer_engine_params)
        trainer.validate(model=pl_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(emb_metrics.metrics, save_path)
    else:
        trainer = Trainer(max_epochs=3, num_sanity_val_steps=0, **trainer_engine_params)
        trainer.fit(model=pl_model)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--devices", type=int)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--num_labels", type=int)
    parser.add_argument("--batch_size", type=int)
    return parser


if __name__ == "__main__":
    experiment(get_parser().parse_args())
