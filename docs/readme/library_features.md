## OML features

<table style="width: 100%; border-collapse: collapse; border-spacing: 0; margin: 0; padding: 0;">
<tr>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/contents/losses.html"> Losses</a> |
<a href="https://open-metric-learning.readthedocs.io/en/latest/contents/miners.html"> Miners</a>

```python
from oml import losses as l
from oml import miners as m

miner = m.AllTripletsMiner()
miner = m.NHardTripletsMiner()
miner = m.MinerWithBank()
...

criterion = l.TripletLossWithMiner(0.1, miner)
criterion = l.ArcFaceLoss()
criterion = l.SurrogatePrecision()
...
```

</td>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/contents/samplers.html"> Samplers</a>

```python
from oml import samplers as s

sampler = s.BalanceSampler()
sampler = s.CategoryBalanceSampler()
sampler = s.DistinctCategoryBalanceSampler()
```

</td>
</tr>
<tr>
<td style="text-align: left;">
<a href="https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/">Using configs</a>

```yaml
sampler:
  name: category_balance
  args:
    n_labels: 30
    n_instances: 4
    n_categories: 5

max_epochs: 10
...
```

</td>
<td style="text-align: left;">
<a href="https://github.com/OML-Team/open-metric-learning/tree/docs?tab=readme-ov-file#zoo">Models Zoo</a>

```python
from oml.models import HFWrapper, ViTExtractor
from transformers import AutoModel, AutoTokenizer
from oml.registry.transforms import get_transforms_for_pretrained

txt_model = HFWrapper(AutoModel.from_pretrained('bert-base-uncased'), 768)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

img_model = ViTExtractor.from_pretrained("vits16_dino")
transforms, reader = get_transforms_for_pretrained("vits16_dino")
```

</td>
</tr>
<tr>
<td style="text-align: left;"><a href="https://open-metric-learning.readthedocs.io/en/latest/postprocessing/postprocessing/postprocessing_home.html#algorithmic-postprocessing">Post-processing</a>

```python
from oml.retrieval import RetrievalResults, SmartThresholding

...
embeddings = inference(extractor, dataset)
rr = RetrievalResults.from_embeddings(embeddings, dataset)

postprocessor = SmartThresholding()  # todo
rr_upd = postprocessor.process(rr, dataset=dataset)

```

</td>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/postprocessing/python_examples.html">Post-processing by NN</a> |
<a href="https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/postprocessing/pairwise_postprocessing">Paper</a>

```python
from oml.models import ConcatSiamese
from oml.retrieval import PairwiseReranker
from oml.retrieval import RetrievalResults

...
embeddings = inference(extractor, dataset)
rr = RetrievalResults.from_embeddings(embeddings, dataset)

siamese = ConcatSiamese(extractor=extractor, mlp_hidden_dims=[100])
postprocessor = PairwiseReranker(top_n=3, pairwise_model=siamese)
rr_upd = postprocessor.process(rr, dataset=dataset)

```

</td>
</tr>
<tr>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/oml/logging.html#">Logging</a><br>

```python
from oml.lightning.pipelines import logging

logger = logging.TensorBoardPipelineLogger()
logger = logging.NeptunePipelineLogger()
logger = logging.WandBPipelineLogger()
logger = logging.MLFlowPipelineLogger()
logger = logging.ClearMLPipelineLogger()
```

</td>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html#usage-with-pytorch-metric-learning">PyTorch Metric Learning</a><br>

```python
from oml.models import ViTExtractor
from pytorch_metric_learning import losses


extractor = ViTExtractor()
criterion = losses.TripletMarginLoss(0.2, "all")
...
pred = extractor(data)
criterion(pred, gts)
```

</td>
</tr>
<tr>
<td style="text-align: left;"><a href="https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html#handling-categories">Handling categories</a>

```python
from oml import metrics as m
from oml.retrieval import RetrievalResults
from oml.samplers import CategoryBalanceSampler

# train
loader = DataLoader(batch_sampler=CategoryBalanceSampler())
...

# validation
rr = RetrievalResults.from_embeddings()
m.calc_retrieval_metrics_rr(rr, query_categories)
...
```

</td>
<td style="text-align: left;"><a href="https://open-metric-learning.readthedocs.io/en/latest/contents/metrics.html">Misc metrics</a>

```python
from oml import metrics as m
from oml.retrieval import RetrievalResults

...
embeddigs = inference(model, dataset)
rr = RetrievalResults.from_embeddings(embeddings, dataset)

m.calc_topological_metrics(embeddings, pcf_variance=(0.5,))
m.calc_fnmr_at_fmr_rr(rr, fmr_vals=(0.1,))
m.calc_retrieval_metrics_rr(
    rr, cmc_top_k=(3,5), precision_top_k=(5,), map_top_k=(5,)
)

```

</td>
</tr>
<tr>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html#usage-with-pytorch-lightning">Lightning</a><br>

```python
import pytorch_lightning as pl
from oml.lightning import MetricValCallback
from oml.lightning import ExtractorModule
from oml.metrics import EmbeddingMetrics
from oml.models import ViTExtractor


extractor = ViTExtractor.from_pretrained("vits16_dino")
criterion = ArcFaceLoss(extractor.feat_dim, n_classes)
clb = MetricValCallback(EmbeddingMetrics(val_dataset))
pl_model = ExtractorModule(extractor, criterion, optimizer)

...
trainer = pl.Trainer(max_epochs=3, callbacks=[clb])
trainer.fit(pl_model, train_loader, val_loader)
```

</td>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html#usage-with-pytorch-lightning">Lightning DDP</a><br>

```python
import pytorch_lightning as pl
from oml.lightning import MetricValCallback
from oml.lightning import ExtractorModuleDDP
from oml.metrics import EmbeddingMetrics
from oml.models import ViTExtractor


extractor = ViTExtractor.from_pretrained("vits16_dino")
criterion = ArcFaceLoss(extractor.feat_dim, n_classes)
metric_callback = MetricValCallback(metric=EmbeddingMetrics(val_dataset))
pl_model = ExtractorModuleDDP(
    extractor, criterion, optimizer, train_loader, val_loader
)

...
ddp_args = {"devices": 2, "strategy": DDPStrategy(), "use_distributed_sampler": False}
trainer = pl.Trainer(max_epochs=3, callbacks=[metric_callback], **ddp_args)
trainer.fit(pl_model)
```

</td>
</tr>
</table>

