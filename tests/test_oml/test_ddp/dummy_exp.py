from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import randint, random, sample
from typing import Any, Dict, List

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch import nn
from torch.utils.data import DataLoader, Dataset

from oml.lightning.callbacks.metric import MetricValCallback
from oml.lightning.entrypoints.utils import parse_runtime_params_from_config
from oml.lightning.modules.module_ddp import ModuleDDP
from oml.metrics.embeddings import EmbeddingMetrics
from oml.utils.misc import set_global_seed


class DummyDataset(Dataset):
    def __init__(self, pred_labels: List[int], gt_labels: List[int]):
        self.gt_labels = gt_labels
        self.pred_labels = pred_labels

    def __len__(self) -> int:
        return len(self.gt_labels)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return {
            "input_tensors": self.pred_labels[item] + 0.5 * torch.rand((3, 10, 10)),
            "labels": self.gt_labels[item],
            "is_query": True,
            "is_gallery": True,
        }


class DummyModule(ModuleDDP):
    def __init__(self, loaders_val: EVAL_DATALOADERS):
        super().__init__(loaders_val=loaders_val)
        self.model = nn.Sequential(nn.AvgPool2d((10, 10)), nn.Flatten())

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, *dataset_idx: int) -> Dict[str, Any]:
        embeddings = self.model(batch["input_tensors"])
        return {**batch, **{"embeddings": embeddings}}


def experiment(args: Namespace) -> None:
    devices = args.devices
    save_path = Path(args.save_path)
    num_labels = args.num_labels
    batch_size = args.batch_size

    set_global_seed(1)
    gt_labels = []
    pred_labels = []

    all_labels = set(range(num_labels))
    err_prob = 0.3
    for label in range(num_labels):
        instances = [label] * randint(3, 5)
        gt_labels += instances
        if random() < err_prob:
            instances = instances.copy()
            instances[0] = sample(list(all_labels - {label}), k=1)[0]
        pred_labels += instances

    dataset = DummyDataset(gt_labels=gt_labels, pred_labels=pred_labels)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

    emb_metrics = EmbeddingMetrics(cmc_top_k=(5, 10), precision_top_k=(5, 10), map_top_k=(5, 10))
    val_callback = MetricValCallback(metric=emb_metrics)

    trainer_engine_params = parse_runtime_params_from_config({"accelerator": "cpu", "devices": devices})

    pl_model = DummyModule(loaders_val=dataloader)

    trainer = Trainer(num_nodes=1, callbacks=[val_callback], **trainer_engine_params)

    trainer.validate(model=pl_model, verbose=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb_metrics.metrics, save_path)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--devices", type=int)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--num_labels", type=int)
    parser.add_argument("--batch_size", type=int)
    return parser


if __name__ == "__main__":
    experiment(get_parser().parse_args())
