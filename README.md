# Open Metric Learning

OML is a PyTorch-based framework to train the models to produce the high quality embeddings.
We focus on Supervised finetuning for your specific domain.
We consider the most common scenario when developer have no more than few GPUs, that is why
we don't work with Self-Supervised Learning (SSL) due to need of heavy compute, instead, we provide
a convinient way of using such models for the initialisation.

<details>
<summary>What is Metric Learning?</summary>
<p>

Metric Learning (or extreme classification problem) means a situation when we
have thousands ids of some entities, but only few samples for every entity.
Often we assume that during the test stage (or production) we will deal with unseen entities
which makes impossible applying the vanila classification pipeline directly. In many cases obtained embeddings
are used to perfrom search or matching procedure over them.

Here are few examples of such tasks from the computer vision sphere:
* Person/Animal Re-Identification
* Face Recognition
* Landmark Recognition
* Searching engines for online shops
 and many others.
</p>
</details>

<details>
<summary>Do I need to know other frameworks to use OML?</summary>
<p>
No, you don't. OML is a framework-agnostic. Despite we use PyTorch Lightning as a loop
runner for the experiments, we also keep a posibility to run everything on pure PyTorch.
Thus, only the tiny part of OML is Lightning-specific and we keep this logic separately from
other code (see oml.lightning). Even when you use Lightning, you don't really need to know it, since
we provide ready to use entrypoints with config based API.

The possibility of using pure PyTorch and modular structure of the code leaves a room for utilizing
OML with you favorite framework after the implementing of the necessary wrappers.

</p>
</details>

<details>
<summary>Can I use OML without any knowledge in DataScience?</summary>
<p>
Yes. But you likele need GPU :) To run our experiment you only need to write a converter
 to our format (specifically, it means preparing
.csv file with 5 predifined columns). Than you adjust config file and run experiment. That's it!

If your domain is not very specific, probably we already have a suitable pretrained model for you
in our models zoo. In this case everything is even easier.

</p>
</details>



<details>
<summary>Naming agreements</summary>
<p>

**Samples, Labels, Categories**

We use the word `embedding` as a synonim to `features vector` or `discriptor`.

As an example let's consider DeepFashion dataset.
It includes thousands of fashion item ids (we name them `labels` and several photos for each item id
 (we name this individual photos as `samples`).
All of the fashion item ids have their groups like "skirts", "jackets", "shorts" and so on (we name them `categories`).

Note, we avoid using the term `classes` to avoid misunderstanding.

**Miner, Sampler**
* `Sampler` - uses to form batches and passes to `DataLoader`
* `Miner` - uses to form pairs or triplets, usually after batch was formed by `Sampler`

</p>
</details>




## Acknowledgments
<a href="https://www.newyorker.de/" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d8/New_Yorker.svg/1280px-New_Yorker.svg.png" width="100"/></a>

<a href="https://github.com/catalyst-team/catalyst" target="_blank"><img src="https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png" width="100"/></a>
