## Usage with custom dataset
The only difference from the public dataset case is that you have to implement the converter by yourself.
We expect the `.csv` file in the following format:

Required columns:
* `label` - integer value indicates the label.
* `path` - path to sample.
* `split` - must be one of 2 values: `train` or `validation`.
* `is_query`, `is_gallery` - have to be `None` where `split == train` and `True`
  or `False` where `split == validation`. Note, that both values can be `True` at
  the same time. Then we will validate every item
  in the validation set using the "1 vs rest" approach (datasets of this kind are `CARS196` or `CUB`).

Optional columns:
* `category` - category which groups sets of similar labels (like `dresses`, or `furniture`).
* `x_1`, `x_2`, `y_1`, `y_2` - integers, the format is `left`, `right`, `top`, `bot` (`y_1` must be less than `y_2`).
  If only part of your images has bounding boxes, just fill the corresponding row with empty values.

[Here](https://drive.google.com/drive/folders/12QmUbDrKk7UaYGHreQdz5_nPfXG3klNc?usp=sharing)
are the tables examples for the public datasets. You can also use helper to check if your dataset
is in the right format:
```python
import pandas as pd
from oml.utils.dataframe_format import check_retrieval_dataframe_format

check_retrieval_dataframe_format(df=pd.read_csv("/path/to/your/table.csv"), dataset_root="/path/to/your/dataset_converters/root/")
```


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

augs:
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
from oml.lightning.entrypoints.train import pl_train
from oml.registry.models import EXTRACTORS_REGISTRY
from oml.registry.transforms import TRANSFORMS_REGISTRY


class CustomModel(IExtractor):

    def __init__(self, pretrained):
        super().__init__()
        self.resnet = resnet18(pretrained=pretrained)

    def forward(self, x):
        self.resnet(x)

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
    pl_train(cfg)


if __name__ == "__main__":
    main_hydra()
```

The same logic works for models, samplers, losses, etc.
