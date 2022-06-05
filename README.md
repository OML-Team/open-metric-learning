# Open Metric Learning

<details>
<summary>Naming agreements</summary>
<p>

**Samples, Labels, Categories**

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


# Acknowledgments
<a href="https://www.newyorker.de/" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d8/New_Yorker.svg/1280px-New_Yorker.svg.png" width="100"/></a>

<a href="https://github.com/catalyst-team/catalyst" target="_blank"><img src="https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png" width="100"/></a>
