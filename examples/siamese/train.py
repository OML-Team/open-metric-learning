# type: ignore
# flake8: noqa
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from source import ImagesSiamese, PairsMiner, get_embeddings, validate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from oml.const import CATEGORIES_COLUMN, LABELS_COLUMN
from oml.datasets.base import DatasetWithLabels
from oml.registry.models import get_extractor_by_cfg
from oml.registry.optimizers import get_optimizer_by_cfg
from oml.registry.samplers import get_sampler_by_cfg
from oml.registry.transforms import get_transforms_by_cfg
from oml.retrieval.postprocessors.pairwise import PairwiseImagesPostprocessor
from oml.transforms.images.utils import get_im_reader_for_transforms
from oml.utils.misc import flatten_dict


def main(cfg: DictConfig) -> None:
    device = torch.device("cpu")

    transforms_extraction = get_transforms_by_cfg(cfg["transforms_extraction"])
    transforms_train = get_transforms_by_cfg(cfg["transforms_train"])
    transforms_val = get_transforms_by_cfg(cfg["transforms_val"])

    extractor = get_extractor_by_cfg(cfg["extractor"]).to(device)

    emb_train, emb_val, df_train, df_val = get_embeddings(
        dataset_root=Path(cfg["dataset_root"]),
        extractor=extractor,
        transforms=transforms_extraction,
        num_workers=cfg["num_workers"],
        batch_size=cfg["bs_val"],
    )

    dataset = DatasetWithLabels(
        df=df_train,
        transform=transforms_train,
        f_imread=get_im_reader_for_transforms(transforms_train),
    )

    sampler_runtime_args = {
        "labels": dataset.get_labels(),
        "label2category": dict(zip(df_train[LABELS_COLUMN], df_train[CATEGORIES_COLUMN])),
    }
    sampler = get_sampler_by_cfg(cfg["sampler"], **sampler_runtime_args) if cfg["sampler"] is not None else None

    loader_train = DataLoader(batch_sampler=sampler, dataset=dataset, num_workers=cfg["num_workers"])

    siamese = ImagesSiamese(extractor=extractor).to(device)
    optimizer = get_optimizer_by_cfg(cfg["optimizer"], params=siamese.parameters())  # type: ignore

    pairs_miner = PairsMiner()
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")

    postprocessor = PairwiseImagesPostprocessor(
        top_n=cfg["top_n"],
        pairwise_model=siamese,
        transforms=transforms_val,
        batch_size=cfg["bs_val"],
        num_workers=cfg["num_workers"],
    )

    writer = SummaryWriter()

    k = 0
    best_cmc = 0.0
    ckpt_metric = "OVERALL/cmc/1"

    for i_epoch in range(cfg["n_epochs"]):
        # tqdm_loader = tqdm([next(iter(loader_train))] * 2)
        tqdm_loader = tqdm(loader_train)

        if i_epoch < cfg["n_epoch_warm_up"]:
            siamese.frozen = True
            optimizer.param_groups[0]["lr"] = cfg["lr_warm_up"]
        else:
            siamese.frozen = False
            optimizer.param_groups[0]["lr"] = cfg["lr"]

        for batch in tqdm_loader:
            features = emb_train[batch["idx"]]
            ii1, ii2, gt_dist = pairs_miner.sample(features, batch["labels"])
            x1 = batch["input_tensors"][ii1]
            x2 = batch["input_tensors"][ii2]
            gt_dist = gt_dist.to(device)

            pred_dist = siamese(x1=x1.to(device), x2=x2.to(device))
            loss = criterion(pred_dist, gt_dist)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            accuracy = ((pred_dist > 0.5) == gt_dist).float().mean().item()
            writer.add_scalar("loss", loss.item(), global_step=k)
            writer.add_scalar("accuracy", accuracy, global_step=k)
            k += 1

        if (i_epoch + 1) % cfg["val_period"] == 0:
            metrics = validate(
                postprocessor=postprocessor,
                top_n=cfg["top_n"],
                df_val=df_val,
                emb_val=emb_val,
            )
            metrics = flatten_dict(metrics)
            print(metrics)

            for m, v in metrics.items():
                writer.add_scalar(m, v, global_step=i_epoch)

            if metrics[ckpt_metric] > best_cmc:
                siamese = siamese.cpu()
                torch.save(siamese.state_dict(), "best.pth")
                best_cmc = float(metrics[ckpt_metric])
                siamese = siamese.to(device)


@hydra.main(config_path=".", config_name="train.yaml")
def main_hydra(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    main_hydra()
