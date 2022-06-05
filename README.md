# Open Metric Learning

<details>
<summary>##Naming agreements</summary>
<p>

### Samples, Labels, Categories

As an example let's consider DeepFashion dataset.
It includes thousands of fashion item ids (we name them `labels` and several photos for each item id
 (we name this individual photos as `samples`).
All of the fashion item ids have their groups like "skirts", "jackets", "shorts" and so on (we name them `categories`).

Note, we avoid using the term `classes` to avoid misunderstanding.


### Miner, Sampler
* `Sampler` - uses to form batches and passes to `DataLoader`
* `Miner` - uses to form pairs or triplets, usually after batch was formed by `Sampler`

</p>
</details>





### Retrieval DataFrame Format
Expecting columns: `label`, `path`, `split`, `is_query`, `is_gallery` and
optional `x_1`, `x_2`, `y_1`, `y_2`.

* `split` must be on of 2 values: `train` or `validation`
* `is_query` and `is_gallery` have to be `None` where `split == train` and `True`
or `False` where `split == validation`. Note, that both values may be equal `True` in
the same time.
* `x_1`, `x_2`, `y_1`, `y_2` are in the following format `left`, `right`, `top`, `bot` (`y_1` must be less than `y_2`)
