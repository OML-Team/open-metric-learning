Below are the models trained with OML on 4 public datasets.
For more details about the training process and configs, please, visit *examples* submodule and it's
[Readme](https://github.com/OML-Team/open-metric-learning/blob/main/examples/).
All metrics below were obtained on the images with the sizes of **224 x 224**:

|                            model                            | cmc1  |         dataset          |                                              weights                                              |                                           configs                                            |
|:-----------------------------------------------------------:|:-----:|:------------------------:|:-------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|
| `ViTExtractor(weights="vits16_inshop", arch="vits16", ...)` | 0.921 |    DeepFashion Inshop    |    [link](https://drive.google.com/file/d/1niX-TC8cj6j369t7iU2baHQSVN3MVJbW/view?usp=sharing)     | [link](https://github.com/OML-Team/open-metric-learning/tree/main/examples/extractor_inshop) |
|  `ViTExtractor(weights="vits16_sop", arch="vits16", ...)`   | 0.866 | Stanford Online Products |   [link](https://drive.google.com/file/d/1zuGRHvF2KHd59aw7i7367OH_tQNOGz7A/view?usp=sharing)      |  [link](https://github.com/OML-Team/open-metric-learning/tree/main/examples/extractor_sop)   |
|  `ViTExtractor(weights="vits16_cars", arch="vits16", ...)`  | 0.907 |         CARS 196         |   [link](https://drive.google.com/drive/folders/17a4_fg94dox2sfkXmw-KCtiLBlx-ut-1?usp=sharing)    |  [link](https://github.com/OML-Team/open-metric-learning/tree/main/examples/extractor_cars)  |
|  `ViTExtractor(weights="vits16_cub", arch="vits16", ...)`   | 0.837 |       CUB 200 2011       |   [link](https://drive.google.com/drive/folders/1TPCN-eZFLqoq4JBgnIfliJoEK48x9ozb?usp=sharing)    |  [link](https://github.com/OML-Team/open-metric-learning/tree/main/examples/extractor_cub)   |

We also provide an integration with the models pretrained by other researchers.
All metrics below were obtained on the images with the sizes of **224 x 224**:

|                           model                            | Stanford Online Products | DeepFashion InShop | CUB 200 2011 | CARS 196 |
|:----------------------------------------------------------:|:------------------------:|:------------------:|:------------:|:--------:|
|  `ViTCLIPExtractor("sber_vitb32_224", "vitb32_224")` todo  |          0.547           |       0.514        |    0.448     |  0.618   |
|  `ViTCLIPExtractor("sber_vitb16_224", "vitb16_224")` todo  |          0.565           |       0.565        |    0.524     |  0.648   |
|  `ViTCLIPExtractor("sber_vitl14_224", "vitl14_224")` todo  |          0.512           |       0.555        |    0.606     |  0.707   |
| `ViTCLIPExtractor("openai_vitb32_224", "vitb32_224")` todo |          0.612           |       0.491        |    0.560     |  0.693   |
| `ViTCLIPExtractor("openai_vitb16_224", "vitb16_224")` todo |          0.648           |       0.606        |    0.665     |  0.767   |
| `ViTCLIPExtractor("openai_vitl14_224", "vitl14_224")` todo |          0.670           |       0.675        |    0.745     |  0.844   |
|      `ViTExtractor("vits16_dino", "vits16")`               |          0.648           |       0.509        |    0.627     |  0.265   |
|           `ViTExtractor("vits8_dino", "vits8")`            |          0.651           |       0.524        |    0.661     |  0.315   |
|          `ViTExtractor("vitb16_dino", "vitb16")`           |          0.658           |       0.514        |    0.541     |  0.288   |
|           `ViTExtractor("vitb8_dino", "vitb8")`            |          0.689           |       0.599        |    0.506     |  0.313   |
|     `ResnetExtractor("resnet50_moco_v2", "resnet50")`      |          0.527           |       0.303        |    0.244     |  0.141   |


You can specify the desired weights and architecture to automatically download pretrained checkpoint (by the analogue with torchvision.models):

[comment]:checkpoint-start
```python
import oml
from oml.models.vit.vit import ViTExtractor

# We are downloading vits16 pretrained on CARS dataset:
model = ViTExtractor(weights="vits16_cars", arch="vits16", normalise_features=False)

# You can also check other available pretrained models:
print(list(ViTExtractor.pretrained_models.keys()))

# To load checkpoint saved on a disk:
model_from_disk = ViTExtractor(weights=oml.const.CKPT_SAVE_ROOT / "vits16_cars.ckpt", arch="vits16", normalise_features=False)
```
[comment]:checkpoint-end
