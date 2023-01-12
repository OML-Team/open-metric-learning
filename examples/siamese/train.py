from datetime import datetime
from pathlib import Path

import torch
import torchvision.transforms as t
from source import ImagesSiamese, PairsMiner, get_embeddings, validate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from oml.const import MEAN, STD
from oml.datasets.base import DatasetWithLabels
from oml.samplers.category_balance import CategoryBalanceSampler
from oml.transforms.images.utils import get_im_reader_for_transforms
from oml.utils.misc import flatten_dict

logs_root = Path("/nydl/logs/DeepFashion_InShop/2023")

dataset_root = Path("/nydl/data/DeepFashion_InShop/")
n_labels = 5
n_instances = 4
n_categories = 5
num_workers = 20

weights = "vits16_inshop"
normalize_features = False
n_epochs = 50
n_epoch_warm_up = 5
val_period = 1
lr_warm_up = 1e-3
lr = 1e-6
top_n = 5

train_transform = t.Compose(
    [
        t.RandomResizedCrop((224, 224), scale=(0.8, 1.0), interpolation=t.InterpolationMode.BICUBIC),
        t.RandomHorizontalFlip(),
        t.ToTensor(),
        t.Normalize(mean=MEAN, std=STD),
    ]
)

emb_train, emb_val, df_train, df_val = get_embeddings(dataset_root=dataset_root, weights=weights)

f_imread = get_im_reader_for_transforms(train_transform)
dataset = DatasetWithLabels(df_train, transform=train_transform, f_imread=f_imread, dataset_root=dataset_root)

loader = torch.utils.data.DataLoader(
    batch_sampler=CategoryBalanceSampler(
        labels=dataset.get_labels(),
        label2category=dict(zip(df_train["label"], df_train["category"])),
        n_labels=n_labels,
        n_instances=n_instances,
        n_categories=n_categories,
        resample_labels=True,
    ),
    dataset=dataset,
    num_workers=num_workers,
)

model = ImagesSiamese(weights=weights, normalise_features=normalize_features).cuda().train()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e10)

pairs_miner = PairsMiner()
criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")

writer = SummaryWriter(log_dir=str(logs_root / str(datetime.now()).replace(" ", "_").split(".")[0]))
k = 0

for i_epoch in range(n_epochs):
    tqdm_loader = tqdm([next(iter(loader))] * 80)
    # tqdm_loader = tqdm(loader)

    if i_epoch < n_epoch_warm_up:
        model.frozen = True
        optimizer.param_groups[0]["lr"] = lr_warm_up
    else:
        model.frozen = False
        optimizer.param_groups[0]["lr"] = lr

    for batch in tqdm_loader:
        features = emb_train[batch["idx"]]
        ii1, ii2, gt_dist = pairs_miner.sample(features, batch["labels"])
        x1, x2, gt_dist = batch["input_tensors"][ii1].cuda(), batch["input_tensors"][ii2].cuda(), gt_dist.cuda()

        pred_dist = model(x1=x1, x2=x2)
        loss = criterion(pred_dist, gt_dist)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        accuracy = ((pred_dist > 0.5) == gt_dist).float().mean().item()
        writer.add_scalar("loss", loss.item(), global_step=k)
        writer.add_scalar("accuracy", accuracy, global_step=k)
        k += 1

    if (i_epoch + 1) % val_period == 0:
        metrics = validate(model=model, top_n=top_n, df_val=df_val, emb_val=emb_val)
        for m, v in flatten_dict(metrics).items():
            writer.add_scalar(m, v, global_step=i_epoch)
