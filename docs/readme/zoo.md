Below are the models trained with OML on 4 public datasets.
For more details about the training process, please, visit *examples* submodule and it's
[Readme](https://github.com/OML-Team/open-metric-learning/blob/main/examples/).

|                            model                            | cmc1  |         dataset          |                                           weights                                            |                                           configs                                            | hash (the beginning) |
|:-----------------------------------------------------------:|:-----:|:------------------------:|:--------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|:--------------------:|
| `ViTExtractor(weights="vits16_inshop", arch="vits16", ...)` | 0.903 |    DeepFashion Inshop    | [link](https://drive.google.com/file/d/1wjjwBC6VomVZQF-JeXepEMk9CtV0Nste/view?usp=sharing)   | [link](https://github.com/OML-Team/open-metric-learning/tree/main/examples/inshop/configs)   |        e1017d        |
|  `ViTExtractor(weights="vits16_sop", arch="vits16", ...)`   | 0.830 | Stanford Online Products | [link](https://drive.google.com/drive/folders/1WfPqCKbZ2KjRRQURGOOwrlQ87EUb7Zra?usp=sharing) | [link](https://github.com/OML-Team/open-metric-learning/tree/main/examples/sop/configs)      |        85cfa5        |
|  `ViTExtractor(weights="vits16_cars", arch="vits16", ...)`  | 0.907 |         CARS 196         | [link](https://drive.google.com/drive/folders/17a4_fg94dox2sfkXmw-KCtiLBlx-ut-1?usp=sharing) | [link](https://github.com/OML-Team/open-metric-learning/tree/main/examples/cars/configs)     |        9f1e59        |
|  `ViTExtractor(weights="vits16_cub", arch="vits16", ...)`   | 0.837 |       CUB 200 2011       | [link](https://drive.google.com/drive/folders/1TPCN-eZFLqoq4JBgnIfliJoEK48x9ozb?usp=sharing) | [link](https://github.com/OML-Team/open-metric-learning/tree/main/examples/cub/configs)      |        e82633        |

We also provide an integration with the models pretrained by other researchers (todo: finish the table):

|                        model                              |   Stanford Online Products |   DeepFashion InShop |   CUB 200 2011 |   CARS 196 |
|:----------------------------------------------------------|:--------------------------:|:--------------------:|:--------------:|:----------:|
| Sber ViT-CLIP Base, patch 32                              |                      0.539 |                0.499 |          0.452 |      0.616 |
| Sber ViT-CLIP Base, patch 16                              |                      0.560 |                0.557 |          0.528 |      0.642 |
| Sber ViT-CLIP Large, patch 14                             |                      0.508 |                0.549 |          0.610 |      0.696 |
| OpenAI ViT-CLIP Base, patch 32                            |                      0.594 |                0.472 |          0.562 |      0.679 |
| OpenAI ViT-CLIP Base, patch 16                            |                      0.640 |                0.597 |          0.664 |      0.760 |
| OpenAI ViT-CLIP Large, patch 14                           |                      0.661 |                0.667 |          0.744 |      0.839 |
| `ViTExtractor(weights="vitb16_dino", arch="vitb16", ...)` |                      0.636 |                0.464 |          0.626 |      0.340 |
| `ViTExtractor(weights="vitb8_dino", arch="vitb8", ...)`   |                      0.xxx |                0.xxx |          0.xxx |      0.xxx |
| `ViTExtractor(weights="vits16_dino", arch="vits16", ...)` |                      0.xxx |                0.xxx |          0.xxx |      0.xxx |
| `ViTExtractor(weights="vits8_dino", arch="vits8", ...)`   |                      0.xxx |                0.xxx |          0.xxx |      0.xxx |
| MoCo, Resnet50                                            |                      0.xxx |                0.xxx |          0.xxx |      0.xxx |

*All figures above were obtained on the images with the sizes of 224 x 224.
Also note, that the models above expect the crop of the region of interest rather than the whole picture.*


You can specify the desired weights and architecture to automatically download pretrained checkpoint (by the analogue with `torchvision.models`):

[comment]:checkpoint-start
```python
import oml
from oml.models.vit.vit import ViTExtractor

# We are downloading vits16 pretrained on CARS dataset:
model = ViTExtractor(weights="vits16_cars", arch="vits16", normalise_features=False)

# You can also check other available pretrained models...
print(list(ViTExtractor.pretrained_models.keys()))

# ...or check other available types of architectures
print(oml.registry.models.MODELS_REGISTRY)

# It's also possible to use `weights` argument to directly pass the path to the checkpoint:
model_from_disk = ViTExtractor(weights=oml.const.CKPT_SAVE_ROOT / "vits16_cars.ckpt", arch="vits16", normalise_features=False)
```
[comment]:checkpoint-end