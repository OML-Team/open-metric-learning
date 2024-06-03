<details open>
<summary>FAQ</summary>

<details>
<summary>Why do I need OML?</summary>
<p>

You may think *"If I need image embeddings I can simply train a vanilla classifier and take its penultimate layer"*.
Well, it makes sense as a starting point. But there are several possible drawbacks:

* If you want to use embeddings to perform searching you need to calculate some distance among them (for example, cosine or L2).
  Usually, **you don't directly optimize these distances during the training** in the classification setup. So, you can only hope that
  final embeddings will have the desired properties.

* **The second problem is the validation process**.
  In the searching setup, you usually care how related your top-N outputs are to the query.
  The natural way to evaluate the model is to simulate searching requests to the reference set
  and apply one of the retrieval metrics.
  So, there is no guarantee that classification accuracy will correlate with these metrics.

* Finally, you may want to implement a metric learning pipeline by yourself.
  **There is a lot of work**: to use triplet loss you need to form batches in a specific way,
  implement different kinds of triplets mining, tracking distances, etc. For the validation, you also need to
  implement retrieval metrics,
  which include effective embeddings accumulation during the epoch, covering corner cases, etc.
  It's even harder if you have several gpus and use DDP.
  You may also want to visualize your search requests by highlighting good and bad search results.
  Instead of doing it by yourself, you can simply use OML for your purposes.

</p>
</details>


<details>
<summary>What is the difference between Open Metric Learning and PyTorch Metric Learning?</summary>
<p>

[PML](https://github.com/KevinMusgrave/pytorch-metric-learning) is the popular library for Metric Learning,
and it includes a rich collection of losses, miners, distances, and reducers; that is why we provide straightforward
[examples](https://github.com/OML-Team/open-metric-learning#usage-with-pytorch-metric-learning) of using them with OML.
Initially, we tried to use PML, but in the end, we came up with our library, which is more pipeline / recipes oriented.
That is how OML differs from PML:

* OML has [Pipelines](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines)
  which allows training models by preparing a config and your data in the required format
  (it's like converting data into COCO format to train a detector from [mmdetection](https://github.com/open-mmlab/mmdetection)).

* OML focuses on end-to-end pipelines and practical use cases.
  It has config based examples on popular benchmarks close to real life (like photos of products of thousands ids).
  We found some good combinations of hyperparameters on these datasets, trained and published models and their configs.
  Thus, it makes OML more recipes oriented than PML, and its author
  [confirms](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/169#issuecomment-670814393)
  this saying that his library is a set of tools rather the recipes, moreover, the examples in PML are mostly for CIFAR and MNIST datasets.

* OML has the [Zoo](https://github.com/OML-Team/open-metric-learning#zoo) of pretrained models that can be easily accessed from
  the code in the same way as in `torchvision` (when you type `resnet50(pretrained=True)`).

* OML is integrated with [PyTorch Lightning](https://www.pytorchlightning.ai/), so, we can use the power of its
  [Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html).
  This is especially helpful when we work with DDP, so, you compare our
  [DDP example](https://open-metric-learning.readthedocs.io/en/latest/examples/python.html)
  and the
  [PMLs one](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/DistributedTripletMarginLossMNIST.ipynb).
  By the way, PML also has [Trainers](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/), but it's not
  widely used in the examples and custom `train` / `test` functions are used instead.

We believe that having Pipelines, laconic examples, and Zoo of pretrained models sets the entry threshold to a really low value.

</p>
</details>


<details>
<summary>What is Metric Learning?</summary>
<p>

Metric Learning problem (also known as *extreme classification* problem) means a situation in which we
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
<summary>Glossary (Naming convention) </summary>
<p>

* `embedding` - model's output (also known as `features vector` or `descriptor`).
* `query` - a sample which is used as a request in the retrieval procedure.
* `gallery set` - the set of entities to search items similar to `query` (also known as `reference` or `index`).
* `Sampler` - an argument for `DataLoader` which is used to form batches
* `Miner` - the object to form pairs or triplets after the batch was formed by `Sampler`. It's not necessary to form
  the combinations of samples only inside the current batch, thus, the memory bank may be a part of `Miner`.
* `Samples`/`Labels`/`Instances` - as an example let's consider DeepFashion dataset. It includes thousands of
  fashion item ids (we name them `labels`) and several photos for each item id
  (we name the individual photo as `instance` or `sample`). All of the fashion item ids have their groups like
  "skirts", "jackets", "shorts" and so on (we name them `categories`).
  Note, we avoid using the term `class` to avoid misunderstanding.
* `training epoch` - batch samplers which we use for combination-based losses usually have a length equal to
  `[number of labels in training dataset] / [numbers of labels in one batch]`. It means that we don't observe all of
  the available training samples in one epoch (as opposed to vanilla classification),
  instead, we observe all of the available labels.

</p>
</details>


<details>
<summary>How good may be a model trained with OML? </summary>
<p>

It may be comparable with the current (2022 year) [SotA](https://paperswithcode.com/task/metric-learning) methods,
for example, [Hyp-ViT](https://arxiv.org/pdf/2203.10833.pdf).
*(Few words about this approach: it's a ViT architecture trained with contrastive loss,
but the embeddings were projected into some hyperbolic space.
As the authors claimed, such a space is able to describe the nested structure of real-world data.
So, the paper requires some heavy math to adapt the usual operations for the hyperbolical space.)*

We trained the same architecture with triplet loss, fixing the rest of the parameters:
training and test transformations, image size, and optimizer. See configs in [Models Zoo](https://github.com/OML-Team/open-metric-learning#zoo).
The trick was in heuristics in our miner and sampler:

* [Category Balance Sampler](https://open-metric-learning.readthedocs.io/en/latest/contents/samplers.html#categorybalancesampler)
  forms the batches limiting the number of categories *C* in it.
  For instance, when *C = 1* it puts only jackets in one batch and only jeans into another one (just an example).
  It automatically makes the negative pairs harder: it's more meaningful for a model to realise why two jackets
  are different than to understand the same about a jacket and a t-shirt.

* [Hard Triplets Miner](https://open-metric-learning.readthedocs.io/en/latest/contents/miners.html#hardtripletsminer)
  makes the task even harder keeping only the hardest triplets (with maximal positive and minimal negative distances).

Here are *CMC@1* scores for 2 popular benchmarks.
SOP dataset: Hyp-ViT — 85.9, ours — 86.6. DeepFashion dataset: Hyp-ViT — 92.5, ours — 92.1.
Thus, utilising simple heuristics and avoiding heavy math we are able to perform on SotA level.

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
* As a source of inspiration. For example, we adapted the idea of a memory bank from *MoCo* for the *TripletLoss*.

</p>
</details>


<details>
<summary>Do I need to know other frameworks to use OML?</summary>
<p>

No, you don't. OML is a framework-agnostic. Despite we use PyTorch Lightning as a loop
runner for the experiments, we also keep the possibility to run everything on pure PyTorch.
Thus, only the tiny part of OML is Lightning-specific and we keep this logic separately from
other code (see `oml.lightning`). Even when you use Lightning, you don't need to know it, since
we provide ready to use [Pipelines](https://github.com/OML-Team/open-metric-learning/blob/main/pipelines/).

The possibility of using pure PyTorch and modular structure of the code leaves a room for utilizing
OML with your favourite framework after the implementation of the necessary wrappers.

</p>
</details>


<details>
<summary>Can I use OML without any knowledge in DataScience?</summary>
<p>

Yes. To run the experiment with [Pipelines](https://github.com/OML-Team/open-metric-learning/blob/main/pipelines/)
you only need to write a converter
to our format (it means preparing the
`.csv` table with a few predefined columns).
That's it!

Probably we already have a suitable pre-trained model for your domain
in our *Models Zoo*. In this case, you don't even need to train it.
</p>
</details>

</details>

<br>