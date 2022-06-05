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
