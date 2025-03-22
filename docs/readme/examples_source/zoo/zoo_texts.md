Here is a lightweight integration with [HuggingFace Transformers](https://github.com/huggingface/transformers) models.
You can replace it with other arbitrary models inherited from [IExtractor](https://open-metric-learning.readthedocs.io/en/latest/contents/interfaces.html#iextractor).

```shell
pip install open-metric-learning[nlp]
```

<details style="padding-bottom: 15px">
<summary><b>See how to use models</b></summary>
<p>

[comment]:zoo-text-start
```python
from transformers import AutoModel, AutoTokenizer

from oml.models import HFWrapper

model = AutoModel.from_pretrained('bert-base-uncased').eval()
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
extractor = HFWrapper(model=model, feat_dim=768)

inp = tokenizer(text="Hello world", return_tensors="pt", add_special_tokens=True)
embeddings = extractor(inp)
```
[comment]:zoo-text-end

</p>
</details>

Note, we don't have our own text models zoo at the moment.
