This page is a good place to start and get an overview of Pipelines approach in general.
For the details of the exact Pipeline, please, visit the corresponding page:
* [features_extraction](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/features_extraction)
* [postprocessing](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/postprocessing) (to re-rank vector search by Siamese model)

## What are Pipelines?

Pipelines are a predefined collection of scripts/recipes that provide a way to run metric learning
experiments by changing only the config.
Pipelines require your data to be prepared in a special `.csv` file, described in details
[here](https://open-metric-learning.readthedocs.io/en/latest/oml/data.html).

Pipelines may help you if:
* You have a dataset which format can be aligned with one required by a pipeline
* You need reproducibility for your experiments
* You prefer changing config over diving into the code

They will not work if:
* You deal with a corner case and the flexibility of an existing Pipeline isn't enough

## How to work with Pipelines?

The recommended way is the following:
1. Install OML: `pip install open-metric-learning`
2. Prepare your dataset in the required [format](https://open-metric-learning.readthedocs.io/en/latest/oml/data.html). (There are [converters](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/datasets_converters) for 4 popular datasets).
3. Go to Pipeline's folder and copy `.py` script and its `.yaml` to your workdir. Modify the config if needed.
4. Run the script via the command line.

## Minimal example of a Pipeline

Each Pipeline is built around 3 components:
* Config file
* Registry of classes & functions
* Entrypoint function for a Pipeline + script to run it

Let's consider an oversimplified example: we create a model and "validate" it via
applying it to a tensor of ones using the predefined device:

`pipeline.py` (implements the logic of a Pipeline, **part of OML package**):

[comment]:pipeline-start
```python
import torch
from registry import get_model

def toy_validation(config):
  model = get_model(config["model"]).eval()
  inp = torch.ones((1, 3, 32, 32)).float()
  output = model(inp.to(config["device"]))
```
[comment]:pipeline-end

`registry.py` (maps entity's config to a Python constructor, **part of OML package**):

[comment]:registry-start
```python
from torchvision.models import resnet18, resnet50

MODELS_REGISTRY = {"resnet18": resnet18, "resnet50": resnet50}

def get_model(config):
  return MODELS_REGISTRY[config["name"]](**config["args"])
```
[comment]:registry-end

`config.yaml` (describes the whole run, **your local file**):

[comment]:config-start
```yaml
model:
  name: resnet50
  args:
    weights: IMAGENET1K_V1

device: cpu
```
[comment]:config-end

`validate.py` (script which simply runs pipeline, **your local file**):

[comment]:script-start
```python
import hydra
from pipeline import toy_validation

@hydra.main(config_name="config.yaml")
def main_hydra(cfg):
    toy_validation(cfg)

if __name__ == "__main__":
    main_hydra()
```
[comment]:script-end

Shell command:

[comment]:shell-start
```shell
python validate.py model.args.weights=null
```
[comment]:shell-end

Note, we use [Hydra](https://hydra.cc/docs/intro/) as a config parser. One of its abilities
is to change part of the config from a command line, as showed above.

## Building blocks of Pipelines

Like every Python program Pipelines consist of functions and objects, that sum up in the desired logic.
Some of them, like extractor or optimizer, may be completely replaced via config.
Others, like a trainer or a metrics calculator will stay there anyway, but you can change their behaviour
via config as well.

Let's say, you work with one of the Pipelines, and it assumes that an extractor must be a successor of
[IExtractor](https://open-metric-learning.readthedocs.io/en/latest/contents/interfaces.html#iextractor)
interface. You have two options if you want to use another extractor:
* You can check the existing successors of `IExtractor` in the library and pick one of them;
* Or you can implement your successor, see the section below for details.

To see what exact parts of each config can be modified, please, visit their subpages.

## How to use my own implementation of loss, extractor, etc.?

You should put a constructor of your Python object inside the corresponding registry by some key.
It allows you to access this object in the config file by that key.

Let's consider an example of using custom augmentations & extractor to train your feature extractor.

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


class CustomExtractor(IExtractor):

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


# Put extractor & transforms constructors to the registries
TRANSFORMS_REGISTRY["custom_augmentations"] = get_custom_augs
EXTRACTORS_REGISTRY["custom_extractor"] = CustomExtractor


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

extractor:
  name: custom_extractor  # this name is a key for models registry we set above
  args:
    pretrained: True  # our model has one argument that has to be set

...
```

The same logic works for optimisers, samplers, losses, etc., depending on the exact Pipeline
and its building blocks.

## Configuration via config is not flexible enough in my case

Let's say you want to change the implementation of `Dataset`, which is not configurable
in a pipeline of your interest. In other words, you can only change its initial arguments,
but cannot replace the corresponding class.

In this case, you can copy the source code of the main pipeline
entrypoint function and modify it as you want.
For example, if you want to train your feature extractor with your own implementation of `Dataset`,
you need to copy & modify
[extractor_training_pipeline](https://github.com/OML-Team/open-metric-learning/blob/d3ff382afa89d2c36faa307c4369c0fd4f3c2362/oml/lightning/pipelines/train.py#L60).
To find an entrypoint function for other pipelines simply check what is used inside the desired `*.py` file.
