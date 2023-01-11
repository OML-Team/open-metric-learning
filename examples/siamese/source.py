# type: ignore
# flake8: noqa

import numpy as np

from oml.interfaces.datasets import IDatasetQueryGallery, IDatasetWithLabels


class TensorsWithLabels(IDatasetWithLabels):
    def __init__(self, df, embeddings, input_tensors_key="input_tensors", labels_key="labels", category_key="category"):
        super(TensorsWithLabels, self).__init__()
        assert len(df) == len(embeddings)

        self.df = df
        self.embeddings = embeddings
        self.input_tensors_key = input_tensors_key
        self.labels_key = labels_key
        self.category_key = category_key

        self.labels = np.array(self.df["label"])

    def __getitem__(self, idx):
        return {
            self.input_tensors_key: self.embeddings[idx],
            self.labels_key: self.labels[idx],
            # self.category_key: self.df.iloc[idx]["category"],
        }

    def __len__(self):
        return len(self.df)

    def get_labels(self):
        return self.labels


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
