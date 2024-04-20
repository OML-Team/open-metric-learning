import sys
import tempfile
from functools import partial
from typing import Any, Dict, List

import pytest
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from oml.const import (
    EMBEDDINGS_KEY,
    INPUT_TENSORS_KEY,
    IS_GALLERY_KEY,
    IS_QUERY_KEY,
    LABELS_KEY,
)
from oml.lightning.callbacks.metric import MetricValCallback
from oml.losses.triplet import TripletLossWithMiner
from oml.metrics.embeddings import EmbeddingMetrics
from oml.samplers.balance import BalanceSampler


class DummyRetrievalDataset(Dataset):
    def __init__(self, labels: List[int], im_size: int):
        self.labels = labels
        self.im_size = im_size

    def __getitem__(self, item: int) -> Dict[str, Any]:
        input_tensors = torch.rand((3, self.im_size, self.im_size))
        label = torch.tensor(self.labels[item]).long()
        return {INPUT_TENSORS_KEY: input_tensors, LABELS_KEY: label, IS_QUERY_KEY: True, IS_GALLERY_KEY: True}

    def __len__(self) -> int:
        return len(self.labels)


class DummyCommonModule(pl.LightningModule):
    def __init__(self, im_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.AvgPool2d(kernel_size=(im_size, im_size)), nn.Flatten(start_dim=1), nn.Linear(3, 5), nn.Linear(5, 5)
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return Adam(self.model.parameters(), lr=1e-4)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, *_: Any) -> Dict[str, Any]:
        embeddings = self.model(batch[INPUT_TENSORS_KEY])
        return {**batch, **{EMBEDDINGS_KEY: embeddings.detach().cpu()}}


class DummyExtractorModule(DummyCommonModule):
    def __init__(self, im_size: int):
        super().__init__(im_size=im_size)
        self.criterion = TripletLossWithMiner(margin=None, need_logs=True)

    def training_step(self, batch_multidataloader: List[Dict[str, Any]], batch_idx: int) -> torch.Tensor:
        embeddings = torch.cat([self.model(batch[INPUT_TENSORS_KEY]) for batch in batch_multidataloader])
        labels = torch.cat([batch[LABELS_KEY] for batch in batch_multidataloader])
        loss = self.criterion(embeddings, labels)
        return loss


def create_retrieval_dataloader(
    num_samples: int, im_size: int, n_labels: int, n_instances: int, num_workers: int
) -> DataLoader:
    assert num_samples % (n_labels * n_instances) == 0

    labels = [idx // n_instances for idx in range(num_samples)]

    dataset = DummyRetrievalDataset(labels=labels, im_size=im_size)

    sampler_retrieval = BalanceSampler(labels=labels, n_labels=n_labels, n_instances=n_instances)
    train_retrieval_loader = DataLoader(
        dataset=dataset,
        batch_sampler=sampler_retrieval,
        num_workers=num_workers,
    )
    return train_retrieval_loader


def create_retrieval_callback(loader_idx: int, samples_in_getitem: int) -> MetricValCallback:
    metric = EmbeddingMetrics(
        embeddings_key=EMBEDDINGS_KEY,
        labels_key=LABELS_KEY,
        is_query_key=IS_QUERY_KEY,
        is_gallery_key=IS_GALLERY_KEY,
    )
    metric_callback = MetricValCallback(metric=metric, loader_idx=loader_idx, samples_in_getitem=samples_in_getitem)
    return metric_callback


@pytest.mark.skipif(sys.platform == "darwin", reason="Does not run on macOS")
@pytest.mark.parametrize(
    "samples_in_getitem, is_error_expected, pipeline",
    [
        (1, False, "retrieval"),
        (2, True, "retrieval"),
    ],
)
@pytest.mark.parametrize("num_dataloaders", [1, 2])
def test_lightning(
    samples_in_getitem: int, is_error_expected: bool, num_dataloaders: int, pipeline: str, num_workers: int
) -> None:
    num_samples = 12
    im_size = 6
    n_labels = 2
    n_instances = 3

    create_dataloader = partial(create_retrieval_dataloader, n_labels=n_labels, n_instances=n_instances)
    lightning_module = DummyExtractorModule(im_size=im_size)
    create_callback = create_retrieval_callback

    train_dataloaders = [
        create_dataloader(num_samples=num_samples, im_size=im_size, num_workers=num_workers)
        for _ in range(num_dataloaders)
    ]
    val_dataloaders = [
        create_dataloader(num_samples=num_samples, im_size=im_size, num_workers=num_workers)
        for _ in range(num_dataloaders)
    ]
    callbacks = [create_callback(loader_idx=k, samples_in_getitem=samples_in_getitem) for k in range(num_dataloaders)]

    trainer = pl.Trainer(
        default_root_dir=tempfile.gettempdir(),
        max_epochs=2,
        enable_progress_bar=False,
        num_nodes=1,
        use_distributed_sampler=False,
        callbacks=callbacks,
        num_sanity_val_steps=0,
    )

    if is_error_expected:
        with pytest.raises(ValueError, match=callbacks[0].metric.__class__.__name__):
            trainer.fit(model=lightning_module, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)
            trainer.validate(model=lightning_module, dataloaders=val_dataloaders)
    else:
        trainer.fit(model=lightning_module, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)
        trainer.validate(model=lightning_module, dataloaders=val_dataloaders)
