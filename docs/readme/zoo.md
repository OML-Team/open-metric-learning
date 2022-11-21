Below are the models trained with OML on 4 public datasets.
For more details about the training process and configs, please, visit *examples* submodule and it's
[Readme](https://github.com/OML-Team/open-metric-learning/blob/main/examples/).
All figures below were obtained on the images with the sizes of 224 x 224:

|                            model                            | cmc1  |         dataset          |                                              weights                                              |                                           configs                                            |                                                                               Transforms                                                                               | hash (the beginning) |
|:-----------------------------------------------------------:|:-----:|:------------------------:|:-------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------:|
| `ViTExtractor(weights="vits16_inshop", arch="vits16", ...)` | 0.921 |    DeepFashion Inshop    |    [link](https://drive.google.com/file/d/1niX-TC8cj6j369t7iU2baHQSVN3MVJbW/view?usp=sharing)     | [link](https://github.com/OML-Team/open-metric-learning/tree/main/examples/inshop/configs)   |       [link](https://github.com/OML-Team/open-metric-learning/blob/5105622fa11d8109e4d33657628ed8479edbef99/oml/transforms/images/torchvision/transforms.py#L32)       |        e1017d        |
|  `ViTExtractor(weights="vits16_sop", arch="vits16", ...)`   | 0.866 | Stanford Online Products |   [link](https://drive.google.com/file/d/1zuGRHvF2KHd59aw7i7367OH_tQNOGz7A/view?usp=sharing)      | [link](https://github.com/OML-Team/open-metric-learning/tree/main/examples/sop/configs)      |       [link](https://github.com/OML-Team/open-metric-learning/blob/5105622fa11d8109e4d33657628ed8479edbef99/oml/transforms/images/torchvision/transforms.py#L32)       |        85cfa5        |
|  `ViTExtractor(weights="vits16_cars", arch="vits16", ...)`  | 0.907 |         CARS 196         |   [link](https://drive.google.com/drive/folders/17a4_fg94dox2sfkXmw-KCtiLBlx-ut-1?usp=sharing)    | [link](https://github.com/OML-Team/open-metric-learning/tree/main/examples/cars/configs)     |     [link](https://github.com/OML-Team/open-metric-learning/blob/5105622fa11d8109e4d33657628ed8479edbef99/oml/transforms/images/albumentations/transforms.py#L128)     |        9f1e59        |
|  `ViTExtractor(weights="vits16_cub", arch="vits16", ...)`   | 0.837 |       CUB 200 2011       |   [link](https://drive.google.com/drive/folders/1TPCN-eZFLqoq4JBgnIfliJoEK48x9ozb?usp=sharing)    | [link](https://github.com/OML-Team/open-metric-learning/tree/main/examples/cub/configs)      |         [link](https://github.com/OML-Team/open-metric-learning/blob/5105622fa11d8109e4d33657628ed8479edbef99/oml/transforms/images/albumentations/transforms.py#L128) |        e82633        |

We also provide an integration with the models pretrained by other researchers.
All figures below were obtained on the images with the sizes of 224 x 224:

|                        model                              |   Stanford Online Products |   DeepFashion InShop |   CUB 200 2011 |   CARS 196 |                                                                           Transforms                                                                           |
|:---------------------------------------------------------:|:--------------------------:|:--------------------:|:--------------:|:----------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `ViTCLIPExtractor("sber_vitb32_224", "vitb32_224")`       |                      0.547 |                0.514 |          0.448 |      0.618 | [link](https://github.com/OML-Team/open-metric-learning/blob/5105622fa11d8109e4d33657628ed8479edbef99/oml/transforms/images/albumentations/transforms.py#L139) |
| `ViTCLIPExtractor("sber_vitb16_224", "vitb16_224")`       |                      0.565 |                0.565 |          0.524 |      0.648 | [link](https://github.com/OML-Team/open-metric-learning/blob/5105622fa11d8109e4d33657628ed8479edbef99/oml/transforms/images/albumentations/transforms.py#L139) |
| `ViTCLIPExtractor("sber_vitl14_224", "vitl14_224")`       |                      0.512 |                0.555 |          0.606 |      0.707 | [link](https://github.com/OML-Team/open-metric-learning/blob/5105622fa11d8109e4d33657628ed8479edbef99/oml/transforms/images/albumentations/transforms.py#L139) |
| `ViTCLIPExtractor("openai_vitb32_224", "vitb32_224")`     |                      0.612 |                0.491 |          0.560 |      0.693 | [link](https://github.com/OML-Team/open-metric-learning/blob/5105622fa11d8109e4d33657628ed8479edbef99/oml/transforms/images/albumentations/transforms.py#L139) |
| `ViTCLIPExtractor("openai_vitb16_224", "vitb16_224")`     |                      0.648 |                0.606 |          0.665 |      0.767 | [link](https://github.com/OML-Team/open-metric-learning/blob/5105622fa11d8109e4d33657628ed8479edbef99/oml/transforms/images/albumentations/transforms.py#L139) |
| `ViTCLIPExtractor("openai_vitl14_224", "vitl14_224")`     |                      0.670 |                0.675 |          0.745 |      0.844 | [link](https://github.com/OML-Team/open-metric-learning/blob/5105622fa11d8109e4d33657628ed8479edbef99/oml/transforms/images/albumentations/transforms.py#L139) |
| `ViTExtractor("vits16_dino", "vits16")`                   |                      0.629 |                0.456 |          0.693 |      0.313 | [link](https://github.com/OML-Team/open-metric-learning/blob/5105622fa11d8109e4d33657628ed8479edbef99/oml/transforms/images/albumentations/transforms.py#L128) |
| `ViTExtractor("vits8_dino", "vits8")`                     |                      0.637 |                0.478 |          0.703 |      0.344 | [link](https://github.com/OML-Team/open-metric-learning/blob/5105622fa11d8109e4d33657628ed8479edbef99/oml/transforms/images/albumentations/transforms.py#L128) |
| `ViTExtractor("vitb16_dino", "vitb16")`                   |                      0.636 |                0.464 |          0.626 |      0.340 | [link](https://github.com/OML-Team/open-metric-learning/blob/5105622fa11d8109e4d33657628ed8479edbef99/oml/transforms/images/albumentations/transforms.py#L128) |
| `ViTExtractor("vitb8_dino", "vitb8")`                     |                      0.673 |                0.548 |          0.546 |      0.342 | [link](https://github.com/OML-Team/open-metric-learning/blob/5105622fa11d8109e4d33657628ed8479edbef99/oml/transforms/images/albumentations/transforms.py#L128) |
| `ResnetExtractor("resnet50_moco_v2", "resnet50")`         |                      0.491 |                0.310 |          0.244 |      0.155 | [link](https://github.com/OML-Team/open-metric-learning/blob/5105622fa11d8109e4d33657628ed8479edbef99/oml/transforms/images/albumentations/transforms.py#L128) |


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
