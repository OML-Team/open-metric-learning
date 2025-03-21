Here is an example of how to train, validate and post-process the model
on a tiny dataset of
[images](https://drive.google.com/drive/folders/1plPnwyIkzg51-mLUXWTjREHgc1kgGrF4),
[texts](https://github.com/OML-Team/open-metric-learning/blob/main/oml/utils/download_mock_dataset.py#L83),
or
[audios](https://drive.google.com/drive/folders/1NcKnyXqDyyYARrDETmhJcTTXegO3W0Ju).
See more details on dataset
[format](https://open-metric-learning.readthedocs.io/en/latest/oml/data.html).

SCROLL RIGHT FOR **IMAGES** > **TEXTS** > **AUDIOS**

<div style="overflow-x: auto;">

<table style="width: 100%; border-collapse: collapse; border-spacing: 0; margin: 0; padding: 0;">

<tr>
</tr>

<tr>
    <td style="text-align: left; padding: 0;"><b>IMAGES</b></td>
    <td style="text-align: left; padding: 0;"><b>TEXTS</b></td>
    <td style="text-align: left; padding: 0;"><b>AUDIOS</b></td>
</tr>

<tr>
</tr>

<tr>

<td>

[comment]:train-val-img-start
```python
from torch.optim import Adam
from torch.utils.data import DataLoader

from oml import datasets as d
from oml.inference import inference
from oml.losses import TripletLossWithMiner
from oml.metrics import calc_retrieval_metrics_rr
from oml.miners import AllTripletsMiner
from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained
from oml.retrieval import RetrievalResults, AdaptiveThresholding
from oml.samplers import BalanceSampler
from oml.utils import get_mock_images_dataset

model = ViTExtractor.from_pretrained("vits16_dino").to("cpu").train()
transform, _ = get_transforms_for_pretrained("vits16_dino")

df_train, df_val = get_mock_images_dataset(global_paths=True)
train = d.ImageLabeledDataset(df_train, transform=transform)
val = d.ImageQueryGalleryLabeledDataset(df_val, transform=transform)

optimizer = Adam(model.parameters(), lr=1e-4)
criterion = TripletLossWithMiner(0.1, AllTripletsMiner(), need_logs=True)
sampler = BalanceSampler(train.get_labels(), n_labels=2, n_instances=2)


def training():
    for batch in DataLoader(train, batch_sampler=sampler):
        embeddings = model(batch["input_tensors"])
        loss = criterion(embeddings, batch["labels"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(criterion.last_logs)


def validation():
    embeddings = inference(model, val, batch_size=4, num_workers=0)
    rr = RetrievalResults.from_embeddings(embeddings, val, n_items=3)
    rr = AdaptiveThresholding(n_std=2).process(rr)
    rr.visualize(query_ids=[2, 1], dataset=val, show=True)
    print(calc_retrieval_metrics_rr(rr, map_top_k=(3,), cmc_top_k=(1,)))


training()
validation()
```
[comment]:train-val-img-end

</td>

<td>

[comment]:train-val-txt-start
```python
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from oml import datasets as d
from oml.inference import inference
from oml.losses import TripletLossWithMiner
from oml.metrics import calc_retrieval_metrics_rr
from oml.miners import AllTripletsMiner
from oml.models import HFWrapper
from oml.retrieval import RetrievalResults, AdaptiveThresholding
from oml.samplers import BalanceSampler
from oml.utils import get_mock_texts_dataset

model = HFWrapper(AutoModel.from_pretrained("bert-base-uncased"), 768).to("cpu").train()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

df_train, df_val = get_mock_texts_dataset()
train = d.TextLabeledDataset(df_train, tokenizer=tokenizer)
val = d.TextQueryGalleryLabeledDataset(df_val, tokenizer=tokenizer)

optimizer = Adam(model.parameters(), lr=1e-4)
criterion = TripletLossWithMiner(0.1, AllTripletsMiner(), need_logs=True)
sampler = BalanceSampler(train.get_labels(), n_labels=2, n_instances=2)


def training():
    for batch in DataLoader(train, batch_sampler=sampler):
        embeddings = model(batch["input_tensors"])
        loss = criterion(embeddings, batch["labels"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(criterion.last_logs)


def validation():
    embeddings = inference(model, val, batch_size=4, num_workers=0)
    rr = RetrievalResults.from_embeddings(embeddings, val, n_items=3)
    rr = AdaptiveThresholding(n_std=2).process(rr)
    rr.visualize(query_ids=[2, 1], dataset=val, show=True)
    print(calc_retrieval_metrics_rr(rr, map_top_k=(3,), cmc_top_k=(1,)))


training()
validation()
```
[comment]:train-val-txt-end
</td>

<td>

[comment]:train-val-aud-start
```python
from torch.optim import Adam
from torch.utils.data import DataLoader

from oml import datasets as d
from oml.inference import inference
from oml.losses import TripletLossWithMiner
from oml.metrics import calc_retrieval_metrics_rr
from oml.miners import AllTripletsMiner
from oml.models import ECAPATDNNExtractor
from oml.retrieval import AdaptiveThresholding, RetrievalResults
from oml.samplers import BalanceSampler
from oml.utils import get_mock_audios_dataset

model = ECAPATDNNExtractor.from_pretrained("ecapa_tdnn_taoruijie").to("cpu").train()

df_train, df_val = get_mock_audios_dataset(global_paths=True)
train = d.AudioLabeledDataset(df_train)
val = d.AudioQueryGalleryLabeledDataset(df_val)

optimizer = Adam(model.parameters(), lr=1e-4)
criterion = TripletLossWithMiner(0.1, AllTripletsMiner(), need_logs=True)
sampler = BalanceSampler(train.get_labels(), n_labels=2, n_instances=2)


def training():
    for batch in DataLoader(train, batch_sampler=sampler):
        embeddings = model(batch["input_tensors"])
        loss = criterion(embeddings, batch["labels"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(criterion.last_logs)


def validation():
    embeddings = inference(model, val, batch_size=4, num_workers=0)
    rr = RetrievalResults.from_embeddings(embeddings, val, n_items=3)
    rr = AdaptiveThresholding(n_std=2).process(rr)
    rr.visualize(query_ids=[2, 1], dataset=val, show=True)
    print(calc_retrieval_metrics_rr(rr, map_top_k=(3,), cmc_top_k=(1,)))


training()
validation()
```
[comment]:train-val-aud-end
</td>

</tr>

<tr>
</tr>

<tr>

<td>

<details style="padding-bottom: 10px">
<summary>Output</summary>

```python
{'active_tri': 0.125, 'pos_dist': 82.5, 'neg_dist': 100.5}  # batch 1
{'active_tri': 0.0, 'pos_dist': 36.3, 'neg_dist': 56.9}     # batch 2

{'cmc': {1: 0.75}, 'precision': {5: 0.75}, 'map': {3: 0.8}}

```

<img src="https://i.ibb.co/MVxBf80/retrieval-img.png" height="200px">

</details>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Fr4HhDOqmjx1hCFS30G3MlYjeqBW5vDg?usp=sharing)

</td>

<td>

<details style="padding-bottom: 10px">
<summary>Output</summary>

```python
{'active_tri': 0.0, 'pos_dist': 8.5, 'neg_dist': 11.0}  # batch 1
{'active_tri': 0.25, 'pos_dist': 8.9, 'neg_dist': 9.8}  # batch 2

{'cmc': {1: 0.8}, 'precision': {5: 0.7}, 'map': {3: 0.9}}

```

<img src="https://i.ibb.co/HqfXdYd/text-retrieval.png" height="200px">

</details>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19o2Ox2VXZoOWOOXIns7mcs0aHJZgJWeO?usp=sharing)

</td>

<td>

<details style="padding-bottom: 10px">
<summary>Output</summary>

```python
{'active_tri': 0.25, 'pos_dist': 17.3, 'neg_dist': 18.4}  # batch 1
{'active_tri': 0.0, 'pos_dist': 17.1, 'neg_dist': 18.5}   # batch 2

{'cmc': {1: 1.0}, 'precision': {5: 1.0}, 'map': {3: 1.0}}

```

<img src="https://i.ibb.co/nN5ZF0Bw/audio-output.png" height="200px">

</details>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Sfz7xMdjXg634-3KmBPq8Zs6i_gbsWD1?usp=sharing)

</td>

</tr>

</table>

</div>

<br>

[Extra illustrations, explanations and tips](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/features_extraction#training)
for the code above.
