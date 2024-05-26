```python
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from oml.datasets import TextLabeledDataset
from oml.losses.triplet import TripletLossWithMiner
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.models.texts import HFWrapper
from oml.samplers.balance import BalanceSampler
from oml.utils import get_mock_texts_dataset

df_train, _ = get_mock_texts_dataset()
extractor = HFWrapper(AutoModel.from_pretrained("t5-small"), 768)
tokenizer = AutoTokenizer.from_pretrained("t5-small")
train = TextLabeledDataset(df_train, tokenizer=tokenizer)

optimizer = Adam(extractor.parameters(), lr=1e-4)
criterion = TripletLossWithMiner(0.1, AllTripletsMiner(), need_logs=True)
sampler = BalanceSampler(train.get_labels(), n_labels=2, n_instances=2)
train_loader = DataLoader(train, batch_sampler=sampler)

for batch in tqdm(train_loader):
    embeddings = extractor(batch["input_tensors"])
    loss = criterion(embeddings, batch["labels"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(criterion.last_logs)

# ===================================================

from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from oml.datasets import ImageLabeledDataset
from oml.losses.triplet import TripletLossWithMiner
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.models import ViTExtractor
from oml.registry.transforms import get_transforms_for_pretrained
from oml.samplers.balance import BalanceSampler
from oml.utils import get_mock_images_dataset

df_train, _ = get_mock_images_dataset()
extractor = ViTExtractor.from_pretrained("vits16_dino")
transforms, _ = get_transforms_for_pretrained("vits16_dino")
train = ImageLabeledDataset(df_train, transform=transforms)

optimizer = Adam(extractor.parameters(), lr=1e-4)
criterion = TripletLossWithMiner(0.1, AllTripletsMiner(), need_logs=True)
sampler = BalanceSampler(train.get_labels(), n_labels=2, n_instances=2)
train_loader = DataLoader(train, batch_sampler=sampler)

for batch in tqdm(train_loader):
    embeddings = extractor(batch["input_tensors"])
    loss = criterion(embeddings, batch["labels"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(criterion.last_logs)
```