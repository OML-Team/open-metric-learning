from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from oml.datasets.retrieval import DatasetQueryGallery
from oml.metrics.embeddings import EmbeddingMetrics
from oml.models.vit.vit import ViTExtractor

model = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False, use_multi_scale=False, strict_load=True)
model.eval()

dataset_root = Path("/nydl/data/CUB_200_2011/")
df = pd.read_csv(dataset_root / "df.csv")[:10]

val_dataset = DatasetQueryGallery(df=df, im_size=224, pad_ratio=0.0, dataset_root=dataset_root)

val_loader = DataLoader(val_dataset, batch_size=8)
calculator = EmbeddingMetrics()
calculator.setup(num_samples=len(val_dataset))

with torch.no_grad():
    for batch in tqdm(val_loader):
        batch["embeddings"] = model(batch["input_tensors"])
        calculator.update_data(data_dict=batch)

metrics = calculator.compute_metrics()
