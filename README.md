**NOTE! The project is under construction. We will publish the 1st release soon.**


<div align="center">
<img src="https://i.ibb.co/wsmD5r4/photo-2022-06-06-17-40-52.jpg" width="400px">

<div align="left">

OML is a PyTorch-based framework to train and validate the models producing high-quality embeddings.


Specifically, our pipeline includes supervised training and a retrieval-like validation process.
* **Training part** implies using losses, well-established for metric learning, such as the angular losses
 (like ArcFace) or the combinations based losses (like TripletLoss or ContrastiveLoss).
 The latter benefits from effective mining schemas of triplets/pairs, so we pay great attention to it.
 Thus, during the training we:
   1. Use DataLoader + Sampler to form batches (for example, balanced in terms of labels or categories)
   2. [Only for losses based on combinations] Use Miner to form effective pairs or triplets, including
   those which utilize a memory bank.
   3. Compute loss.
* **Validation part** consists of several stages:
  1. Accumulating all of the embeddings.
  2. Calculating distances between them with respect to query/gallery split.
  3. Applying some specific retrieval techniques like query reranking or score normalisation.
  4. Calculating retrieval metrics like CMC@k, Recall@k or MeanAveragePrecision.

## FAQ

<details>
<summary>What is Metric Learning?</summary>
<p>

Metric Learning problem (also known as "extreme classification" problem) means a situation in which we
have thousands of ids of some entities, but only a few samples for every entity.
Often we assume that during the test stage (or production) we will deal with unseen entities
which makes it impossible to apply the vanilla classification pipeline directly. In many cases obtained embeddings
are used to perform search or matching procedures over them.

Here are a few examples of such tasks from the computer vision sphere:
* Person/Animal Re-Identification
* Face Recognition
* Landmark Recognition
* Searching engines for online shops
 and many others.
</p>
</details>


<details>
<summary>Glossary</summary>
<p>

* `embedding` - model's output (also known as `features vector` or `descriptor`).
* `query` - a sample which is used as a request in the retrieval procedure.
* `gallery set` - the set of entities to search items similar to `query` (also known
 as `reference` or `index`).
* `Sampler` - an argument for `DataLoader` which is used to form batches
* `Miner` - the object to form pairs or triplets after the batch was formed by `Sampler`. It's not necessary to form
the combinations of samples only inside the current batch, thus, the memory bank may be a part of `Miner`.
* `Samples`/`Labels`/`Instances` - as an example let's consider DeepFashion dataset. It includes thousands of
 fashion item ids (we name them `labels`) and several photos for each item id
 (we name the individual photo as `instance` or `sample`). All of the fashion item ids have their groups like
  "skirts", "jackets", "shorts" and so on (we name them `categories`).
  Note, we avoid using the term `class` to avoid misunderstanding.
</p>
</details>


<details>
<summary>What about Self-Supervised Learning?</summary>
<p>

Recent research in SSL definitely obtained great results. The problem is that these approaches
required an enormous amount of computing to train the model. But in our framework, we consider the most common case
when the average user has no more than a few GPUs.

At the same time, it would be unwise to ignore success in this sphere, so we still exploit it in two ways:
* As a source of checkpoints that would be great to start training with. From publications and our experience,
they are much better as initialisation than the default supervised model trained on ImageNet. Thus, we added the possibility
to initialise your models using these pretrained checkpoints only by passing an argument in the config or the constructor.
* As a source of inspiration. For example, we adapted the idea of a memory bank from MoCo for the
TripletLoss.
</p>
</details>


<details>
<summary>Do I need to know other frameworks to use OML?</summary>
<p>

No, you don't. OML is a framework-agnostic. Despite we use PyTorch Lightning as a loop
runner for the experiments, we also keep the possibility to run everything on pure PyTorch.
Thus, only the tiny part of OML is Lightning-specific and we keep this logic separately from
other code (see oml.lightning). Even when you use Lightning, you don't need to know it, since
we provide ready to use entry points with configs based API.

The possibility of using pure PyTorch and modular structure of the code leaves a room for utilizing
OML with your favourite framework after the implementation of the necessary wrappers.

Please, see "Minimal examples" section for more details.
</p>
</details>


<details>
<summary>Can I use OML without any knowledge in DataScience?</summary>
<p>

Yes. To run the experiment you only need to write a converter
 to our format (it means preparing the
table with 5 predefined columns). Then you adjust the config file and run the experiment.
That's it!

Probably we already have a suitable pre-trained model for your domain
in our models' zoo. In this case, you don't even need to train.
</p>
</details>


## Models zoo
|               model               |       dataset      |  cmc1 | download | end of hash |
|:---------------------------------:|:------------------:|:-----:|:--------:|:-----------:|
| VitExtractor("pretrained_inshop") | DeepFashion Inshop | 0.925 |   [link](https://drive.google.com/drive/folders/1vypEph09rSwKD7iydI4YYZqwZLrdVJPW?usp=sharing)   |    a69435   |


## Minimal examples
<details>
<summary>Using pure PyTorch</summary>
<p>

Training
```python
model = VitExtractor("pretrained_dino")
model.train()
optimizer = SGD(model.paremeters())
train_dataset = DatasetWithLabels(...)
criterion = TripletLossWithMiner(margin=0.1, miner=AllTripletsMiner())
sampler = BalanceBatchSampler(labels=dataset.get_labels(), p=4, k=4)
train_loader = DataLoader(train_dataset, batch_sampler=sampler)

for batch in train_loader:
    embeddings = model(batch["input_tensors"])
    loss = criterion(embeddings, batch["labels"])
    optmizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Validation
```python
model = VitExtractor("pretrained_dino")
model.eval()
val_dataset = DatasetQueryGallery(...)
val_loader = DataLoader(val_dataset)
caclulator = EmbeddingMetrics()
caclulator.setup()

with torch.no_grad():
    for batch in val_loader:
        batch["embeddings"] = model(batch["input_tensors"])
        calc.update_data(data_dict=batch)

metrics = calc.compute_metrics()
```
</p>
</details>

<details>
<summary>Using PyTorch Lightning</summary>
<p>

```python
model = ResnetExtractor("pretrained_moco")

# train
optimizer = SGD(model.paremeters())
train_dataset = DatasetWithLabels(...)
criterion = TripletLossWithMiner(margin=0.1, miner=AllTripletsMiner())
sampler = BalanceBatchSampler(labels=dataset.get_labels(), p=4, k=4)
train_loader = DataLoader(train_dataset, batch_sampler=sampler)

# val
val_dataset = DatasetQueryGallery(...)
val_loader = DataLoader(val_dataset)
metric_callback = MetricValCallback(EmbeddingMetrics())

# run
pl_model = RetrievalModule(model, criterion, optimizer)
trainer = pl.Trainer(callbacks=[metric_callback])
trainer.fit(pl_model, train_loader, val_loader)
```
</p>
</details>

## Acknowledgments
<a href="https://github.com/catalyst-team/catalyst" target="_blank"><img src="https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png" width="100"/></a>

The project was started in 2020 as a module for [Catalyst](https://github.com/catalyst-team/catalyst) library.
I want to thank people who worked with me on that module:
[Julia Shenshina](https://github.com/julia-shenshina),
[Nikita Balagansky](https://github.com/elephantmipt),
[Sergey Kolesnikov](https://github.com/Scitator)
and others.

I would like to thank people who continue working on this pipeline when it became a separe project:
[Julia Shenshina](https://github.com/julia-shenshina),
[Aleksei Tarasov](https://github.com/DaloroAT) and
[Verkhovtsev Leonid](https://github.com/leoromanovich).

<a href="https://www.newyorker.de/" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d8/New_Yorker.svg/1280px-New_Yorker.svg.png" width="100"/></a>

