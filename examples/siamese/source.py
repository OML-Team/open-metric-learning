# type: ignore
# flake8: noqa

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from oml.datasets.base import BaseDataset
from oml.metrics.embeddings import EmbeddingMetrics
from oml.miners.inbatch_hard_tri import HardTripletsMiner
from oml.models.vit.vit import ViTExtractor
from oml.postprocessors.pairwise_images import PairwiseImagesPostprocessor
from oml.transforms.images.torchvision.transforms import get_normalisation_resize_hypvit
from oml.transforms.images.utils import get_im_reader_for_transforms


class PairsMiner:
    def __init__(self):
        self.miner = HardTripletsMiner()

    #         self.miner = AllTripletsMiner()

    def sample(self, features, labels):
        ii_a, ii_p, ii_n = self.miner._sample(features, labels=labels)

        ii_a_1, ii_p = zip(*list(set(list(map(lambda x: tuple(sorted([x[0], x[1]])), zip(ii_a, ii_p))))))
        ii_a_2, ii_n = zip(*list(set(list(map(lambda x: tuple(sorted([x[0], x[1]])), zip(ii_a, ii_n))))))

        gt_distance = torch.ones(len(ii_a_1) + len(ii_a_2))
        gt_distance[: len(ii_a_1)] = 0

        return torch.tensor([*ii_a_1, *ii_a_2]).long(), torch.tensor([*ii_p, *ii_n]).long(), gt_distance


class ImagesSiamese(torch.nn.Module):
    def __init__(self, weights, normalise_features) -> None:
        # todo: apply or don't apply sigmoid
        super(ImagesSiamese, self).__init__()
        self.model = ViTExtractor(weights=weights, arch="vits16", normalise_features=normalise_features)
        feat_dim = self.model.feat_dim

        self.head = nn.Sequential(
            *[
                nn.Linear(feat_dim, feat_dim // 2, bias=True),
                nn.Dropout(),
                nn.Sigmoid(),
                nn.Linear(feat_dim // 2, 1, bias=False),
            ]
        )

        self.frozen = False

    def forward(self, x1, x2):
        x = torch.concat([x1, x2], dim=2)

        if self.frozen:
            with torch.no_grad():
                x = self.model(x)
        else:
            x = self.model(x)

        x = self.head(x)
        x = x.squeeze()

        return x


def validate(model, top_n, df_val, emb_val):
    assert len(df_val) == len(emb_val)

    if model:
        processor = PairwiseImagesPostprocessor(
            model, top_n=top_n, image_transforms=get_normalisation_resize_hypvit(224, 224)
        )
    else:
        processor = None

    calculator = EmbeddingMetrics(
        cmc_top_k=(1, top_n), precision_top_k=(1, 3, top_n), postprocessor=processor, extra_keys=("paths",)
    )
    calculator.setup(len(df_val))
    calculator.update_data(
        {
            "embeddings": emb_val,
            "is_query": torch.tensor(df_val["is_query"]).bool(),
            "is_gallery": torch.tensor(df_val["is_gallery"]).bool(),
            "labels": torch.tensor(df_val["label"]).long(),
            "paths": df_val["path"].tolist(),
        }
    )
    metrics = calculator.compute_metrics()
    return metrics


def get_embeddings(dataset_root, weights):
    embeddings_path = Path(dataset_root / f"embeddings_{weights}.pkl")
    if not embeddings_path.is_file():
        extract_and_save_features(dataset_root, weights, save_path=embeddings_path)

    embeddings = torch.load(embeddings_path)
    df = pd.read_csv(dataset_root / "df.csv")
    train_mask = df["split"] == "train"

    emb_train = embeddings[train_mask]
    emb_val = embeddings[~train_mask]

    df_train = df[train_mask]
    df_train.reset_index(inplace=True)

    df_val = df[~train_mask]
    df_val.reset_index(inplace=True)

    return emb_train, emb_val, df_train, df_val


def extract_and_save_features(dataset_root, weights, save_path):
    batch_size = 1024

    df = pd.read_csv(dataset_root / "df.csv")

    transform = get_normalisation_resize_hypvit(im_size=224, crop_size=224)
    im_reader = get_im_reader_for_transforms(transform)

    dataset = BaseDataset(df=df, transform=transform, f_imread=im_reader)
    model = ViTExtractor(weights, arch=weights.split("_")[0], normalise_features=True).eval().cuda()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=20)

    embeddings = torch.zeros((len(df), model.feat_dim))

    with torch.no_grad():
        for i, batch in enumerate(tqdm(train_loader)):
            embs = model(batch["input_tensors"].cuda()).detach().cpu()
            ia = i * batch_size
            ib = min(len(embeddings), (i + 1) * batch_size)
            embeddings[ia:ib, :] = embs

    torch.save(embeddings, save_path)
