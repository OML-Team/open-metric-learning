import torch
from tqdm import tqdm

from examples.cub.convert_cub import build_cub_df
from oml.datasets.retrieval import DatasetWithLabels
from oml.losses.triplet import TripletLossWithMiner
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.models.vit.vit import ViTExtractor
from oml.samplers.balance import BalanceBatchSampler
from oml.transforms.images.albumentations.default import get_default_albu

dataset_root = "/nydl/data/CUB_200_2011/"
# download dataset
df = build_cub_df(dataset_root)
df_train = df[df["split"] == "train"].reset_index(drop=True)

model = ViTExtractor(
    "vits16_dino", arch="vits16", normalise_features=False, use_multi_scale=False, strict_load=True
).train()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

train_dataset = DatasetWithLabels(df=df_train, im_size=224, pad_ratio=0.0, transform=get_default_albu())
criterion = TripletLossWithMiner(margin=0.1, miner=AllTripletsMiner())
sampler = BalanceBatchSampler(labels=train_dataset.get_labels(), n_labels=8, n_instances=8)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler)

for batch in tqdm(train_loader):
    embeddings = model(batch["input_tensors"])
    loss = criterion(embeddings, batch["labels"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
