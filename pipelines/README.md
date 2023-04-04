## What are Pipelines?
Pipelines are predefined collection of scripts/recipes that provide a way to run metric learning
experiments via changing only the config file.
You only need to prepare the `.csv` table which describes your dataset
(the format is explained [here](https://open-metric-learning.readthedocs.io/en/latest/oml/data.html)).

Pipelines may help you if:
* You have a dataset which format can be aligned with one required by a pipeline
* You need reproducibility for you experiments
* Prefer changing config rather diving into the code
It will not work if:
* You need to implement some specific logic

At this moment OML has the following pipelines:
* training + validation for feature extractor
* training + validation for pairwise postprocessor for feature extractor
* standalone validation for feature extractor (possibly joined together with postprocessor)

todo: add links

## Minimal example of Pipeline
Each Pipeline is built around 3 components:
* Config file
* Registry of classes & functions (it maps confi)
* Code which implements logic of a run

Let's consider oversimplified example.

`config.yaml`:

[comment]:config-start
```.yaml
model:
  name: resnet50
  args:
    weights: IMAGENET1K_V1

device: cpu
```
[comment]:config-end

`registry.py`:

[comment]:registry-start
```python
from torchvision.models import resnet18, resnet50

MODELS_REGISTRY = {"resnet18": resnet18, "resnet50": resnet50}

def get_model(config):
  return MODELS_REGISTRY[config["name"]](**config["args"])


```
[comment]:registry-end

`run.py`

[comment]:script-start
```python
import hydra
import torch
from registry import get_model

@hydra.main(config_name="config.yaml")
def main(config):
  model = get_model(config["model"]).eval()
  inp = torch.ones((1, 3, 32, 32)).float()
  output = model(inp.to(config["device"]))

main()
```
[comment]:script-end

Shell command:

[comment]:shell-start
```shell
python run.py model.args.weights=null

```
[comment]:shell-end


## How to work with a config?
We use [Hydra](https://hydra.cc/docs/intro/) as a parser for `.yaml` configs.
So, you can change whatever you want directly in the config file or override some parameters
using command line interface:
```
python train_cars.py optimizer.args.lr=0.000001 bs_val=128
```

## How to use my own implementation of loss, model, augmentations, etc?
You should put your python object inside the corresponding registry by some key.
It allows you to access this object in the config file by that key.

You may change the following blocks and to work correctly some of them have to inherit our interfaces:
* `Transforms`, `Sampler`, `Optimizer`, `Scheduler` - follow the standard PyTorch interfaces.
* `Model` - have to be successor of `IExtractor` (see `oml.interfaces.models`)
* `Criterion` - have to be successor of `ITripletLossWithMiner` (see `oml.interfaces.criterions`)
  * You may want to change only `Miner` inside the criterion. It has to be a successor of `ITripletsMiner` (see `oml.interfaces.miners`).


Let's consider an example of using custom augmentations & model.
Your `config.yaml` and `train.py` may look like this:
```yaml
...

transforms_train:
  name: custom_augmentations
  args: {}

model:
  name: custom_model
  args:
    pretrained: True

...
```

```python
import hydra
import torchvision.transforms as t
from omegaconf import DictConfig
from torchvision.models import resnet18

from oml.interfaces.models import IExtractor
from oml.lightning.pipelines.train import extractor_training_pipeline
from oml.registry.models import EXTRACTORS_REGISTRY
from oml.registry.transforms import TRANSFORMS_REGISTRY


class CustomModel(IExtractor):

  def __init__(self, pretrained):
    super().__init__()
    self.resnet = resnet18(pretrained=pretrained)

  def forward(self, x):
    return self.resnet(x)

  @property
  def feat_dim(self):
    return self.resnet.fc.out_features


def get_custom_augs() -> t.Compose:
  return t.Compose([
    t.RandomHorizontalFlip(),
    t.RandomGrayscale(),
    t.ToTensor(),
    t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])


TRANSFORMS_REGISTRY["custom_augmentations"] = get_custom_augs
EXTRACTORS_REGISTRY["custom_model"] = CustomModel


@hydra.main(config_path="configs", config_name="train.yaml")
def main_hydra(cfg: DictConfig) -> None:
  extractor_training_pipeline(cfg)


if __name__ == "__main__":
  main_hydra()
```

The same logic works for models, samplers, losses, etc.
