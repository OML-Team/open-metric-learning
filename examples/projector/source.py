# type: ignore
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from oml.functional.metrics import calc_gt_mask
from oml.interfaces.datasets import IDatasetQueryGallery, IDatasetWithLabels
from oml.interfaces.models import IExtractor
from oml.utils.misc_torch import pairwise_dist


class TensorsWithLabels(IDatasetWithLabels):
    def __init__(self, df, embeddings, input_tensors_key="input_tensors", labels_key="labels", category_key="category"):
        super(TensorsWithLabels, self).__init__()
        assert len(df) == len(embeddings)

        self.df = df
        self.embeddings = embeddings
        self.input_tensors_key = input_tensors_key
        self.labels_key = labels_key
        self.category_key = category_key

    def __getitem__(self, idx):
        return {self.input_tensors_key: self.embeddings[idx],
                self.labels_key: self.df.iloc[idx]["label"],
                self.category_key: self.df.iloc[idx]["category"]
                }

    def __len__(self):
        return len(self.df)

    def get_labels(self):
        return np.array(self.df["label"])


class TensorsQueryGallery(IDatasetQueryGallery):
    def __init__(
            self,
            df,
            embeddings,
            labels_key="labels",
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
    def __init__(self, normalise_features):
        super(ProjExtractor, self).__init__()
        self.normalise_features = normalise_features

        self.fc1 = torch.nn.Linear(in_features=384, out_features=384, bias=False)
        self.fc1.load_state_dict({"weight": torch.eye(384)})  # todo: add noise

        # self.fc2 = torch.nn.Linear(in_features=384, out_features=384, bias=False)
        # self.fc2.load_state_dict({"weight": torch.eye(384)})

    def forward(self, x):
        x = self.fc1(x)

        # x = torch.nn.Sigmoid()(x)
        # x = self.fc2(x)

        if self.normalise_features:
            xn = torch.linalg.norm(x, 2, dim=1).detach()
            x = x.div(xn.unsqueeze(1))

        return x

    def feat_dim(self):
        return 384




