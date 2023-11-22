import torch
from tqdm import tqdm

from oml.const import GROUP_COLUMN, MOCK_DATASET_PATH
from oml.datasets.base import DatasetQueryGallery
from oml.metrics.embeddings import EmbeddingMetrics
from oml.models import ViTExtractor
from oml.utils.download_mock_dataset import download_mock_dataset
from oml.utils.misc import set_global_seed

_, df_val = download_mock_dataset(MOCK_DATASET_PATH)
df_val[GROUP_COLUMN] = list(range(len(df_val)))

# CASE 1

df_val["is_query"] = True
df_val["is_gallery"] = True

# CASE 2

# df1 = df_val.copy()
# df1["is_query"] = True
# df1['is_gallery'] = False
#
# df2 = df_val.copy()
# df2["is_query"] = False
# df2['is_gallery'] = True
#
# df_val = pd.concat([df1, df2])
# df_val = df_val.reset_index(drop=True)

# CASE 1 and CASE 2 metrics must be equal

# VALIDATION


set_global_seed(42)
extractor = ViTExtractor(None, "vits16", False, use_multi_scale=False).eval()

val_dataset = DatasetQueryGallery(df_val, dataset_root=MOCK_DATASET_PATH)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4)
calculator = EmbeddingMetrics(extra_keys=("paths",), groups_key=val_dataset.group_key, cmc_top_k=(1,))
calculator.setup(num_samples=len(val_dataset))

with torch.no_grad():
    for batch in tqdm(val_loader):
        batch["embeddings"] = extractor(batch["input_tensors"])
        calculator.update_data(batch)

metrics = calculator.compute_metrics()

print(metrics)
