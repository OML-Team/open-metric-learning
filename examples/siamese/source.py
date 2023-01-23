# type: ignore
# flake8: noqa

from pathlib import Path

import pandas as pd
import torch
from torch import Tensor, nn

from oml.const import PATHS_COLUMN
from oml.inference.list_inference import inference_on_images
from oml.interfaces.models import IExtractor, IPairwiseModel
from oml.metrics.embeddings import EmbeddingMetrics
from oml.miners.inbatch_hard_tri import HardTripletsMiner
from oml.utils.misc_torch import elementwise_dist


class PairsMiner:
    def __init__(self):
        self.miner = HardTripletsMiner()
        # self.miner = AllTripletsMiner()

    def sample(self, features, labels):
        ii_a, ii_p, ii_n = self.miner._sample(features, labels=labels)

        ii_a_1, ii_p = zip(*list(set(list(map(lambda x: tuple(sorted([x[0], x[1]])), zip(ii_a, ii_p))))))
        ii_a_2, ii_n = zip(*list(set(list(map(lambda x: tuple(sorted([x[0], x[1]])), zip(ii_a, ii_n))))))

        gt_distance = torch.ones(len(ii_a_1) + len(ii_a_2))
        gt_distance[: len(ii_a_1)] = 0

        return torch.tensor([*ii_a_1, *ii_a_2]).long(), torch.tensor([*ii_p, *ii_n]).long(), gt_distance


class ImagesSiamese(IPairwiseModel):
    def __init__(self, extractor: IExtractor) -> None:
        super(ImagesSiamese, self).__init__()
        self.extractor = extractor
        feat_dim = self.extractor.feat_dim

        # todo: parametrize
        self.head = nn.Sequential(
            *[
                nn.Linear(feat_dim, feat_dim // 2, bias=True),
                nn.Dropout(),
                nn.Sigmoid(),
                nn.Linear(feat_dim // 2, 1, bias=False),
            ]
        )

        self.frozen = False

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x = torch.concat([x1, x2], dim=2)

        if self.frozen:
            with torch.no_grad():
                x = self.extractor(x)
        else:
            x = self.extractor(x)

        x = self.head(x)
        x = x.squeeze()

        return x


class ImagesSiameseTrivial(IPairwiseModel):
    def __init__(self, extractor: IExtractor) -> None:
        super(ImagesSiameseTrivial, self).__init__()
        self.extractor = extractor

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.extractor(x1)
        x2 = self.extractor(x2)
        return elementwise_dist(x1, x2, p=2)


def validate(postprocessor, top_n, df_val, emb_val):
    assert len(df_val) == len(emb_val)

    calculator = EmbeddingMetrics(
        cmc_top_k=(1, postprocessor.top_n),
        precision_top_k=(1, 3, top_n),
        postprocessor=postprocessor,
        extra_keys=("paths",),
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


def get_embeddings(
    dataset_root,
    extractor,
    transforms,
    num_workers,
    batch_size,
):
    postfix = getattr(extractor, "weights")
    save_path = Path(dataset_root / f"embeddings_{postfix}.pkl")

    df = pd.read_csv(dataset_root / "df.csv")
    df[PATHS_COLUMN] = df[PATHS_COLUMN].apply(lambda x: Path(dataset_root) / x)  # todo

    if save_path.is_file():
        embeddings = torch.load(save_path, map_location="cpu")
    else:
        embeddings = inference_on_images(
            model=extractor,
            paths=df[PATHS_COLUMN],
            transform=transforms,
            num_workers=num_workers,
            batch_size=batch_size,
            verbose=True,
        ).cpu()
        torch.save(embeddings, save_path)

    train_mask = df["split"] == "train"

    emb_train = embeddings[train_mask]
    emb_val = embeddings[~train_mask]

    df_train = df[train_mask]
    df_train.reset_index(inplace=True)

    df_val = df[~train_mask]
    df_val.reset_index(inplace=True)

    return emb_train, emb_val, df_train, df_val
