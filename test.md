## Code Snippets Side-by-Side

<div style="display: flex; justify-content: space-between;">
  <div style="flex: 1; margin-right: 10px;">

```python
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

  </div>

  <div style="flex: 1; margin-left: 10px;">

```python
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from oml.datasets import TextLabeledDataset
from oml.models.texts import HFWrapper
from oml.losses.triplet import TripletLossWithMiner
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.samplers.balance import BalanceSampler
from oml.utils import get_mock_texts_dataset

df_train, _ = get_mock_texts_dataset()
extractor = HFWrapper(AutoModel.from_pretrained("bert-base-uncased"), 768)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
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
```

  </div>
</div>
