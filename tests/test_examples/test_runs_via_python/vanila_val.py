import torch
from tqdm import tqdm

from oml.datasets.retrieval import DatasetQueryGallery
from oml.metrics.embeddings import EmbeddingMetrics
from oml.models.vit.vit import ViTExtractor
from oml.utils.download_mock_dataset import download_mock_dataset

model = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False).eval()

dataset_root = "/tmp/mock_dataset"
_, df_val = download_mock_dataset(dataset_root)

val_dataset = DatasetQueryGallery(df=df_val, im_size=32, pad_ratio=0.0, dataset_root=dataset_root)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4)
calculator = EmbeddingMetrics()
calculator.setup(num_samples=len(val_dataset))

with torch.no_grad():
    for batch in tqdm(val_loader):
        batch["embeddings"] = model(batch["input_tensors"])
        calculator.update_data(data_dict=batch)

metrics = calculator.compute_metrics()
