### How to use text models?

Here is a lightweight integration with [HuggingFace Transformers](https://github.com/huggingface/transformers) models.
You can replace it with other arbitrary models inherited from [IExtractor](https://open-metric-learning.readthedocs.io/en/latest/contents/interfaces.html#iextractor).

Note, we don't have our own text models zoo at the moment.

<details>
<summary>See example</summary>
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
<br>

### How to use image models?

You can use an image model from our Zoo or
use other arbitrary models after you inherited it from [IExtractor](https://open-metric-learning.readthedocs.io/en/latest/contents/interfaces.html#iextractor).

<details>
<summary>See example</summary>
<p>

[comment]:zoo-image-start
```python
from oml.const import CKPT_SAVE_ROOT as CKPT_DIR, MOCK_DATASET_PATH as DATA_DIR
from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained

model = ViTExtractor.from_pretrained("vits16_dino").eval()
transforms, im_reader = get_transforms_for_pretrained("vits16_dino")

img = im_reader(DATA_DIR / "images" / "circle_1.jpg")  # put path to your image here
img_tensor = transforms(img)
# img_tensor = transforms(image=img)["image"]  # for transforms from Albumentations

features = model(img_tensor.unsqueeze(0))

# Check other available models:
print(list(ViTExtractor.pretrained_models.keys()))

# Load checkpoint saved on a disk:
model_ = ViTExtractor(weights=CKPT_DIR / "vits16_dino.ckpt", arch="vits16", normalise_features=False)
```
[comment]:zoo-image-end

</p>
</details>
<br>

### Image models

Models, trained by us.
The metrics below are for **224 x 224** images:

|                      model                      | cmc1  |         dataset          |                                              weights                                              |                                                    experiment                                                     |
|:-----------------------------------------------:|:-----:|:------------------------:|:-------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------:|
| `ViTExtractor.from_pretrained("vits16_inshop")` | 0.921 |    DeepFashion Inshop    |    [link](https://drive.google.com/file/d/1niX-TC8cj6j369t7iU2baHQSVN3MVJbW/view?usp=sharing)     | [link](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/features_extraction/extractor_inshop) |
|  `ViTExtractor.from_pretrained("vits16_sop")`   | 0.866 | Stanford Online Products |   [link](https://drive.google.com/file/d/1zuGRHvF2KHd59aw7i7367OH_tQNOGz7A/view?usp=sharing)      |  [link](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/features_extraction/extractor_sop)   |
| `ViTExtractor.from_pretrained("vits16_cars")`   | 0.907 |         CARS 196         |   [link](https://drive.google.com/drive/folders/17a4_fg94dox2sfkXmw-KCtiLBlx-ut-1?usp=sharing)    |  [link](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/features_extraction/extractor_cars)  |
|  `ViTExtractor.from_pretrained("vits16_cub")`   | 0.837 |       CUB 200 2011       |   [link](https://drive.google.com/drive/folders/1TPCN-eZFLqoq4JBgnIfliJoEK48x9ozb?usp=sharing)    |  [link](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/features_extraction/extractor_cub)   |

Models, trained by other researchers.
Note, that some metrics on particular benchmarks are so high because they were part of the training dataset (for example `unicom`).
The metrics below are for 224 x 224 images:

|                            model                             | Stanford Online Products | DeepFashion InShop | CUB 200 2011 | CARS 196 |
|:------------------------------------------------------------:|:------------------------:|:------------------:|:------------:|:--------:|
|    `ViTUnicomExtractor.from_pretrained("vitb16_unicom")`     |          0.700           |       0.734        |    0.847     |  0.916   |
|    `ViTUnicomExtractor.from_pretrained("vitb32_unicom")`     |          0.690           |       0.722        |    0.796     |  0.893   |
|    `ViTUnicomExtractor.from_pretrained("vitl14_unicom")`     |          0.726           |       0.790        |    0.868     |  0.922   |
| `ViTUnicomExtractor.from_pretrained("vitl14_336px_unicom")`  |          0.745           |       0.810        |    0.875     |  0.924   |
|    `ViTCLIPExtractor.from_pretrained("sber_vitb32_224")`     |          0.547           |       0.514        |    0.448     |  0.618   |
|    `ViTCLIPExtractor.from_pretrained("sber_vitb16_224")`     |          0.565           |       0.565        |    0.524     |  0.648   |
|    `ViTCLIPExtractor.from_pretrained("sber_vitl14_224")`     |          0.512           |       0.555        |    0.606     |  0.707   |
|   `ViTCLIPExtractor.from_pretrained("openai_vitb32_224")`    |          0.612           |       0.491        |    0.560     |  0.693   |
|   `ViTCLIPExtractor.from_pretrained("openai_vitb16_224")`    |          0.648           |       0.606        |    0.665     |  0.767   |
|   `ViTCLIPExtractor.from_pretrained("openai_vitl14_224")`    |          0.670           |       0.675        |    0.745     |  0.844   |
|        `ViTExtractor.from_pretrained("vits16_dino")`         |          0.648           |       0.509        |    0.627     |  0.265   |
|         `ViTExtractor.from_pretrained("vits8_dino")`         |          0.651           |       0.524        |    0.661     |  0.315   |
|        `ViTExtractor.from_pretrained("vitb16_dino")`         |          0.658           |       0.514        |    0.541     |  0.288   |
|         `ViTExtractor.from_pretrained("vitb8_dino")`         |          0.689           |       0.599        |    0.506     |  0.313   |
|       `ViTExtractor.from_pretrained("vits14_dinov2")`        |          0.566           |       0.334        |    0.797     |  0.503   |
|     `ViTExtractor.from_pretrained("vits14_reg_dinov2")`      |          0.566           |       0.332        |    0.795     |  0.740   |
|       `ViTExtractor.from_pretrained("vitb14_dinov2")`        |          0.565           |       0.342        |    0.842     |  0.644   |
|     `ViTExtractor.from_pretrained("vitb14_reg_dinov2")`      |          0.557           |       0.324        |    0.833     |  0.828   |
|       `ViTExtractor.from_pretrained("vitl14_dinov2")`        |          0.576           |       0.352        |    0.844     |  0.692   |
|     `ViTExtractor.from_pretrained("vitl14_reg_dinov2")`      |          0.571           |       0.340        |    0.840     |  0.871   |
|    `ResnetExtractor.from_pretrained("resnet50_moco_v2")`     |          0.493           |       0.267        |    0.264     |  0.149   |
| `ResnetExtractor.from_pretrained("resnet50_imagenet1k_v1")`  |          0.515           |       0.284        |    0.455     |  0.247   |

*The metrics may be different from the ones reported by papers,
because the version of train/val split and usage of bounding boxes may differ.*
