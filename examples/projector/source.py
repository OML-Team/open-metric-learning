# type: ignore
import numpy as np
import pandas as pd
import torch
import torchvision.ops

from oml.interfaces.datasets import IDatasetQueryGallery, IDatasetWithLabels
from oml.interfaces.models import IExtractor
from oml.registry.models import MODELS_REGISTRY


class TensorsWithLabels(IDatasetWithLabels):
    def __init__(self, df, embeddings, input_tensors_key="input_tensors", labels_key="label"):
        super(TensorsWithLabels, self).__init__()
        assert len(df) == len(embeddings)

        self.df = df
        self.embeddings = embeddings
        self.input_tensors_key = input_tensors_key
        self.labels_key = labels_key

    def __getitem__(self, idx):
        return {self.input_tensors_key: self.embeddings[idx], self.labels_key: self.df.iloc[idx]["label"]}

    def __len__(self):
        return len(self.df)

    def get_labels(self):
        return np.array(self.df["label"])


class TensorsQueryGallery(IDatasetQueryGallery):
    def __init__(
        self,
        df,
        embeddings,
        labels_key="label",
        categories_key="category",
        is_query_key="is_query",
        is_gallery_key="is_gallery",
        paths_key="path",
    ):
        super(TensorsQueryGallery, self).__init__()
        assert len(df) == len(embeddings)

        self.df = df
        self.embeddings = embeddings
        self.labels_key = labels_key
        self.categories_key = categories_key
        self.is_query_key = is_query_key
        self.is_gallery_key = is_gallery_key
        self.paths_key = paths_key

    def __getitem__(self, idx):
        return {
            "input_tensors": self.embeddings[idx],
            self.labels_key: self.df.iloc[idx]["label"],
            self.categories_key: self.df.iloc[idx]["category"],
            self.is_query_key: bool(self.df.iloc[idx]["is_query"]),
            self.is_gallery_key: bool(self.df.iloc[idx]["is_gallery"]),
            self.paths_key: self.df.iloc[idx]["path"],
        }

    def __len__(self):
        return len(self.df)


def get_datasets(dataset_root, dataframe_name, emb_name):
    embeddings = torch.load(dataset_root / emb_name)
    df = pd.read_csv(dataset_root / dataframe_name)

    train_mask = df["split"] == "train"
    train_mask_torch = torch.tensor(train_mask).bool()

    df_train = df.copy()[train_mask]
    df_train.reset_index(inplace=True)

    df_valid = df.copy()[~train_mask]
    df_valid.reset_index(inplace=True)

    train_dataset = TensorsWithLabels(df=df_train, embeddings=embeddings.clone()[train_mask_torch])
    val_dataset = TensorsQueryGallery(df=df_valid, embeddings=embeddings.clone()[~train_mask_torch])

    return train_dataset, val_dataset


class ProjExtractor(IExtractor):
    def __init__(self):
        super(ProjExtractor, self).__init__()
        self.mlp = torchvision.ops.MLP(in_channels=384, hidden_channels=[64])

    def forward(self, x):
        x = self.mlp(x)
        return x

    def feat_dim(self):
        return 64


MODELS_REGISTRY["mlp"] = ProjExtractor
