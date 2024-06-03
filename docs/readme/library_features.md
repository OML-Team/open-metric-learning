## OML features

<div style="overflow-x: auto;">

<table style="width: 100%; border-collapse: collapse; border-spacing: 0; margin: 0; padding: 0;">

<tr>
</tr>

<tr>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/contents/losses.html"> <b>Losses</b></a> |
<a href="https://open-metric-learning.readthedocs.io/en/latest/contents/miners.html"> <b>Miners</b></a>

```python
miner = AllTripletsMiner()
miner = NHardTripletsMiner()
miner = MinerWithBank()
...
criterion = TripletLossWithMiner(0.1, miner)
criterion = ArcFaceLoss()
criterion = SurrogatePrecision()
```

</td>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/contents/samplers.html"> <b>Samplers</b></a>

```python
labels = train.get_labels()
l2c = train.get_label2category()


sampler = BalanceSampler(labels)
sampler = CategoryBalanceSampler(labels, l2c)
sampler = DistinctCategoryBalanceSampler(labels, l2c)
```

</td>
</tr>

<tr>
</tr>

<tr>
<td style="text-align: left;">
<a href="https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/"><b>Configs support</b></a>

```yaml
max_epochs: 10
sampler:
  name: balance
  args:
    n_labels: 2
    n_instances: 2
```

</td>
<td style="text-align: left;">
<a href="https://github.com/OML-Team/open-metric-learning/tree/docs?tab=readme-ov-file#zoo"><b>Pre-trained models</b></a>

```python
model_hf = AutoModel.from_pretrained("roberta-base")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
extractor_txt = HFWrapper(model_hf)

extractor_img = ViTExtractor.from_pretrained("vits16_dino")
transforms, _ = get_transforms_for_pretrained("vits16_dino")
```

</td>
</tr>

<tr>
</tr>

<tr>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/postprocessing/postprocessing/postprocessing_home.html#algorithmic-postprocessing"><b>Post-processing</b></a>

```python
emb = inference(extractor, dataset)
rr = RetrievalResults.from_embeddings(emb, dataset)
# todo
postprocessor = SmartThresholding()
rr_upd = postprocessor.process(rr, dataset)
```

</td>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/postprocessing/python_examples.html"><b>Post-processing by NN</b></a> |
<a href="https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/postprocessing/pairwise_postprocessing"><b>Paper</b></a>

```python
embeddings = inference(extractor, dataset)
rr = RetrievalResults.from_embeddings(embeddings, dataset)

postprocessor = PairwiseReranker(ConcatSiamese(), top_n=3)
rr_upd = postprocessor.process(rr, dataset)
```

</td>
</tr>

<tr>
</tr>

<tr>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/oml/logging.html#"><b>Logging</b></a><br>

```python
logger = TensorBoardPipelineLogger()
logger = NeptunePipelineLogger()
logger = WandBPipelineLogger()
logger = MLFlowPipelineLogger()
logger = ClearMLPipelineLogger()
```

</td>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html#usage-with-pytorch-metric-learning"><b>PML</b></a><br>

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
<td style="text-align: left;"><a href="https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html#handling-categories"><b>Categories support</b></a>

```python
# train
loader = DataLoader(CategoryBalanceSampler())

# validation
rr = RetrievalResults.from_embeddings()
m.calc_retrieval_metrics_rr(rr, query_categories)
```

</td>
<td style="text-align: left;"><a href="https://open-metric-learning.readthedocs.io/en/latest/contents/metrics.html"><b>Misc metrics</b></a>

```python
embeddigs = inference(model, dataset)
rr = RetrievalResults.from_embeddings(embeddings, dataset)

m.calc_retrieval_metrics_rr(rr, precision_top_k=(5,))
m.calc_fnmr_at_fmr_rr(rr, fmr_vals=(0.1,))
m.calc_topological_metrics(embeddings, pcf_variance=(0.5,))
```

</td>
</tr>

<tr>
</tr>

<tr>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html#usage-with-pytorch-lightning"><b>Lightning</b></a><br>

```python
import pytorch_lightning as pl

model = ViTExtractor.from_pretrained("vits16_dino")
clb = MetricValCallback(EmbeddingMetrics(dataset))
module = ExtractorModule(model, criterion, optimizer)

trainer = pl.Trainer(max_epochs=3, callbacks=[clb])
trainer.fit(module, train_loader, val_loader)
```

</td>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html#usage-with-pytorch-lightning"><b>Lightning DDP</b></a><br>

```python
clb = MetricValCallback(EmbeddingMetrics(val))
module = ExtractorModuleDDP(
    model, criterion, optimizer, train, val
)

ddp = {"devices": 2, "strategy": DDPStrategy()}
trainer = pl.Trainer(max_epochs=3, callbacks=[clb], **ddp)
trainer.fit(module)
```

</td>
</tr>

</table>

</div>
