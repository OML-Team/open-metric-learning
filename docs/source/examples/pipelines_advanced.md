

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
from oml.lightning.entrypoints.train import extractor_training_pipeline
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
