This page is a good place to start and get an overview of Pipelines approach in general.
For the details of exact Pipeline, please, visit the corresponding page:
* [features_extraction](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/features_extraction)
* [postprocessing](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/postprocessing) (for feature extractor)

## What are Pipelines?
Pipelines are predefined collection of scripts/recipes that provide a way to run metric learning
experiments via changing only the config.
You need to prepare the `.csv` file which describes your dataset
(the format is explained [here](https://open-metric-learning.readthedocs.io/en/latest/oml/data.html)).

Pipelines may help you if:
* You have a dataset which format can be aligned with a one required by a pipeline
* You need reproducibility for you experiments
* You prefer changing config over diving into the code

They will not work if:
* You deal with a corner case and flexibility of an existing Pipeline isn't enough

## Minimal example of Pipeline
Each Pipeline is built around 3 components:
* Config file
* Registry of classes & functions (it takes config of some entity and returns a python object)
* Script which implements logic of a pipeline

Let's consider an oversimplified example: we create a model and apply it to a tensor of ones
using the predefined device:

`config.yaml`:

[comment]:config-start
```yaml
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

`run.py`:

[comment]:script-start
```python
import hydra
import torch
from registry import get_model

@hydra.main(config_name="config.yaml")
def toy_pipeline(config):
  model = get_model(config["model"]).eval()
  inp = torch.ones((1, 3, 32, 32)).float()
  output = model(inp.to(config["device"]))

toy_pipeline()
```
[comment]:script-end

Shell command:

[comment]:shell-start
```shell

python run.py model.args.weights=null

```
[comment]:shell-end

Note, we use [Hydra](https://hydra.cc/docs/intro/) as a config parser. One of its abilities
is to change part of the config from a command line, as showed above.

## Building blocks of Pipelines
Like every python program Pipelines consist of functions and objects, that sum up in the desired logic.
Some of them, like model or optimizer, may be completely replaced via config.
Others, like trainer or metrics calculator will stay there anyway, but you can change their behaviour
via config as well.

Let's say, you work with one of the Pipelines, and it assumes that a model must be successor of
[IExtractor](https://open-metric-learning.readthedocs.io/en/latest/contents/interfaces.html#iextractor)
interface. You have two options if you want to use another model:
* You can check the existed successors of `IExtractor` in the library and pick one of them;
* Or you can implement your own successor, see the section below for details.

To see what exact parts of each config can be modified, please, visit their own subpages.

## How to use my own implementation of loss, model, etc.?

You should put a constructor of your python object inside the corresponding registry by some key.
It allows you to access this object in the config file by that key.

Let's consider an example of using custom augmentations & model to train your own feature extractor.

Your `train.py` and `config.yaml` may look like this:
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

  # this property is obligatory for IExtractor
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


# Put model & transforms constructors to the registries
TRANSFORMS_REGISTRY["custom_augmentations"] = get_custom_augs
EXTRACTORS_REGISTRY["custom_model"] = CustomModel


@hydra.main(config_path="configs", config_name="train.yaml")
def main_hydra(cfg: DictConfig) -> None:
  extractor_training_pipeline(cfg)


if __name__ == "__main__":
  main_hydra()
```

```yaml
...

transforms_train:
  name: custom_augmentations  # this name is a key for transforms registry we set above
  args: {}  # our augmentations have no obligatory initial arguments

model:
  name: custom_model  # this name is a key for models registry we set above
  args:
    pretrained: True  # our model has one argument that has to be set

...
```

The same logic works for optimisers, samplers, losses, etc., depending on the exact Pipeline
and its building blocks.
