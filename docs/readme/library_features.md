## OML features

<table style="width: 100%; border-collapse: collapse; border-spacing: 0; margin: 0; padding: 0;">

<tr>
</tr>

<tr>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/contents/losses.html"> Losses</a> |
<a href="https://open-metric-learning.readthedocs.io/en/latest/contents/miners.html"> Miners</a>

```python
miner = AllTripletsMiner()
miner = NHardTripletsMiner()
miner = MinerWithBank()
...
criterion = TripletLossWithMiner(0.1, miner)
criterion = ArcFaceLoss()
criterion = SurrogatePrecision()
...
```

</td>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/contents/samplers.html"> Samplers</a>

```python
sampler = BalanceSampler()
sampler = CategoryBalanceSampler()
sampler = DistinctCategoryBalanceSampler()
```

</td>
</tr>

<tr>
</tr>

<tr>
<td style="text-align: left;">
<a href="https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/">Using configs</a>

```yaml
sampler:
  name: balance
  args:
    n_labels: 2
    n_instances: 2

max_epochs: 10
```

</td>
<td style="text-align: left;">
<a href="https://github.com/OML-Team/open-metric-learning/tree/docs?tab=readme-ov-file#zoo">Models Zoo</a>

```python
txt_model = HFWrapper(AutoModel.from_pretrained("roberta-base"))
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

img_model = ViTExtractor.from_pretrained("vits16_dino")
transforms, reader = get_transforms_for_pretrained("vits16_dino")
```

</td>
</tr>

<tr>
</tr>

<tr>
<td style="text-align: left;"><a href="https://open-metric-learning.readthedocs.io/en/latest/postprocessing/postprocessing/postprocessing_home.html#algorithmic-postprocessing">Post-processing</a>

```python
emb = inference(extractor, dataset)
rr = RetrievalResults.from_embeddings(emb, dataset)
# todo
postprocessor = SmartThresholding()
rr_upd = postprocessor.process(rr, dataset)
```

</td>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/postprocessing/python_examples.html">Post-processing by NN</a> |
<a href="https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/postprocessing/pairwise_postprocessing">Paper</a>

```python
embeddings = inference(extractor, dataset)
rr = RetrievalResults.from_embeddings(embeddings, dataset)

postprocessor = PairwiseReranker(top_n=3, pairwise_model=ConcatSiamese())
rr_upd = postprocessor.process(rr, dataset)
```

</td>
</tr>

<tr>
</tr>

<tr>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/oml/logging.html#">Logging</a><br>

```python
logger = TensorBoardPipelineLogger()
logger = NeptunePipelineLogger()
logger = WandBPipelineLogger()
logger = MLFlowPipelineLogger()
logger = ClearMLPipelineLogger()
```

</td>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html#usage-with-pytorch-metric-learning">PyTorch Metric Learning</a><br>

```python
from pytorch_metric_learning import losses

criterion = losses.TripletMarginLoss(0.2, "all")
pred = ViTExtractor()(data)
criterion(pred, gts)
```

</td>
</tr>

<tr>
</tr>

<tr>
<td style="text-align: left;"><a href="https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html#handling-categories">Handling categories</a>

```python
# train
loader = DataLoader(CategoryBalanceSampler())

# validation
rr = RetrievalResults.from_embeddings()
m.calc_retrieval_metrics_rr(rr, query_categories)
```

</td>
<td style="text-align: left;"><a href="https://open-metric-learning.readthedocs.io/en/latest/contents/metrics.html">Misc metrics</a>

```python
embeddigs = inference(model, dataset)
rr = RetrievalResults.from_embeddings(embeddings, dataset)

m.calc_retrieval_metrics_rr(rr, cmc_top_k=(3,5), map_top_k=(5,), ...)
m.calc_fnmr_at_fmr_rr(rr, fmr_vals=(0.1,))
m.calc_topological_metrics(embeddings, pcf_variance=(0.5,))
```

</td>
</tr>

<tr>
</tr>

<tr>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html#usage-with-pytorch-lightning">Lightning</a><br>

```python
clb = MetricValCallback(EmbeddingMetrics(dataset))
module = ExtractorModule(model, criterion, optimizer)

trainer = pl.Trainer(max_epochs=3, callbacks=[clb])
trainer.fit(module, train_loader, val_loader)
```

</td>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html#usage-with-pytorch-lightning">Lightning DDP</a><br>

```python
clb = MetricValCallback(metric=EmbeddingMetrics(val_dataset))
module = ExtractorModuleDDP(model, criterion, optimizer, train_loader, val_loader)

ddp_args = {"devices": 2, "strategy": DDPStrategy(), "use_distributed_sampler": False}
trainer = pl.Trainer(max_epochs=3, callbacks=[clb], **ddp_args)
trainer.fit(module)
```

</td>
</tr>

</table>

