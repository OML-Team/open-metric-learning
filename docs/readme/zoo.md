|                            model                            | cmc1  |         dataset          |                                           weights                                            |                                           configs                                            | hash (the beginning) |
|:-----------------------------------------------------------:|:-----:|:------------------------:|:--------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|:--------------------:|
| `ViTExtractor(weights="vits16_inshop", arch="vits16", ...)` | 0.903 |    DeepFashion Inshop    | [link](https://drive.google.com/file/d/1wjjwBC6VomVZQF-JeXepEMk9CtV0Nste/view?usp=sharing)   | [link](https://github.com/OML-Team/open-metric-learning/tree/main/examples/inshop/configs)   |        e1017d        |
|  `ViTExtractor(weights="vits16_sop", arch="vits16", ...)`   | 0.830 | Stanford Online Products | [link](https://drive.google.com/drive/folders/1WfPqCKbZ2KjRRQURGOOwrlQ87EUb7Zra?usp=sharing) | [link](https://github.com/OML-Team/open-metric-learning/tree/main/examples/sop/configs)      |        85cfa5        |
|  `ViTExtractor(weights="vits16_cars", arch="vits16", ...)`  | 0.907 |         CARS 196         | [link](https://drive.google.com/drive/folders/17a4_fg94dox2sfkXmw-KCtiLBlx-ut-1?usp=sharing) | [link](https://github.com/OML-Team/open-metric-learning/tree/main/examples/cars/configs)     |        9f1e59        |
|  `ViTExtractor(weights="vits16_cub", arch="vits16", ...)`   | 0.837 |       CUB 200 2011       | [link](https://drive.google.com/drive/folders/1TPCN-eZFLqoq4JBgnIfliJoEK48x9ozb?usp=sharing) | [link](https://github.com/OML-Team/open-metric-learning/tree/main/examples/cub/configs)      |        e82633        |

Note, that the models above expect the crop of the region of interest rather than the whole picture.

You can specify the desired weights and architecture and automatically download pretrained checkpoint (by the analogue with `torchvision.models`).
However, you may also do it manually by the link in `weights` column.

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

For more details about the training process, please, visit *examples* submodule and it's
[Readme](https://github.com/OML-Team/open-metric-learning/blob/main/examples/).
