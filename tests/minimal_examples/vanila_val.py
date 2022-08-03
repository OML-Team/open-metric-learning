import torch
from tqdm import tqdm

from examples.cub.convert_cub import build_cub_df
from oml.datasets.retrieval import DatasetQueryGallery
from oml.metrics.embeddings import EmbeddingMetrics
from oml.models.vit.vit import ViTExtractor

model = ViTExtractor(
    "vits16_dino", arch="vits16", normalise_features=False, use_multi_scale=False, strict_load=True
).eval()

dataset_root = "/nydl/data/CUB_200_2011/"
# download dataset
df = build_cub_df(dataset_root)
df_val = df[df["split"] == "validation"].reset_index(drop=True)

val_dataset = DatasetQueryGallery(df=df, im_size=224, pad_ratio=0.0)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
calculator = EmbeddingMetrics()
calculator.setup(num_samples=len(val_dataset))

with torch.no_grad():
    for batch in tqdm(val_loader):
        batch["embeddings"] = model(batch["input_tensors"])
        calculator.update_data(data_dict=batch)

metrics = calculator.compute_metrics()
