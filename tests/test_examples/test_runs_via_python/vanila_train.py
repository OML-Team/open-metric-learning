import torch
from tqdm import tqdm

from oml.datasets.retrieval import DatasetWithLabels
from oml.losses.triplet import TripletLossWithMiner
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.models.vit.vit import ViTExtractor
from oml.samplers.balance import BalanceBatchSampler
from oml.utils.download_mock_dataset import download_mock_dataset

dataset_root = "/tmp/mock_dataset"
df_train, _ = download_mock_dataset(dataset_root)

model = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False).train()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

train_dataset = DatasetWithLabels(df=df_train, im_size=32, pad_ratio=0.0, dataset_root=dataset_root)
criterion = TripletLossWithMiner(margin=0.1, miner=AllTripletsMiner())
sampler = BalanceBatchSampler(labels=train_dataset.get_labels(), n_labels=2, n_instances=2)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler)

for batch in tqdm(train_loader):
    embeddings = model(batch["input_tensors"])
    loss = criterion(embeddings, batch["labels"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
