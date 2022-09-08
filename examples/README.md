# Config API

Config API is a way to run metric learning experiments via changing only the config.
It allows you to train the model without knowing Machine Learning.
All you need is to prepare the `.csv` table which describes your dataset, so the converter
can be implemented in any programming language.


## Usage with public datasets

We've prepared the examples on 4 popular benchmarks used by researchers to evaluate metric learning models,
[link](https://paperswithcode.com/task/metric-learning).
After downloading a dataset you can train or validate your model by the following commands:
```shell script
cd <example>
python convert_<example>.py --dataset_root=/path/to/example/dataset
python train_<example>.py
python validate_<example>.py
```

Note, you can find our pretrained checkpoints for these datasets in the `Models zoo` section of the main readme.

<details>
<summary>CARS 196</summary>
<p>

[Dataset page.](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

The dataset contains 16,185 images of 196 labels of cars.
The data is split into 8,144 training images and 8,041 testing images,
where each label has been split roughly in a 50-50 split.

```
└── CARS196
    ├── cars_test_annos_withlabels.mat
    ├── devkit
    │   ├── cars_meta.mat
    │   ├── cars_train_annos.mat
    │   └── ...
    ├── cars_train
    │   ├── 00001.jpg
    │   └── ...
    └── cars_test
        ├── 00001.jpg
        └── ...
```
</p>
</details>


<details>
<summary>CUB 200 2011</summary>
<p>

[Dataset page.](https://deepai.org/dataset/cub-200-2011)

The dataset contains 11,788 images of 200 labels belonging to birds,
5,994 for training and 5,794 for testing.

```
└── CUB_200_2011
    ├── images.txt
    ├── train_test_split.txt
    ├── bounding_boxes.txt
    ├── image_class_labels.txt
    └── images
        ├── 001.Black_footed_Albatross
        │   ├── Black_Footed_Albatross_0001_796111.jpg
        │   └── ...
        ├── 002.Laysan_Albatross
        │   ├── Laysan_Albatross_0001_545.jpg
        │   └── ...
        └── ...
```
</p>
</details>


<details>
<summary>InShop (DeepFashion)</summary>
<p>

[Dataset page](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html).
[Download from Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pVDZFQXRsMDZCX1E?resourcekey=0-4R4v6zl4CWhHTsUGOsTstw).

The dataset contains 52,712 images for 7,982 of clothing items.

```
└── DeepFashion_InShop
    ├── list_eval_partition.txt
    ├── list_bbox_inshop.txt
    └── img_highres
        ├── MEN
        │   └── ...
        └── WOMEN
            └── ...
```
</p>
</details>


<details>
<summary>SOP (Stanford Online Products)</summary>
<p>

[Dataset page](https://cvgl.stanford.edu/projects/lifted_struct/).
[Download from Google Drive.](https://drive.google.com/uc?export=download&id=1TclrpQOF_ullUP99wk_gjGN8pKvtErG8)

The dataset has 22,634 labels with 120,053 product images. The first 11,318 labels (59,551 images)
are split for training and the other 11,316 (60,502 images) labels are used for testing.

```
└── Stanford_Online_Products
    ├── Ebay_train.txt
    ├── Ebay_test.txt
    ├── bicycle_final
    │   ├── 111085122871_0.JPG
    │   └── ...
    └── cabinet_final
        ├── 110715681235_0.JPG
        └── ...
```
</p>
</details>


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

check_retrieval_dataframe_format(df=pd.read_csv("/path/to/your/table.csv"), dataset_root="/path/to/your/datasets/root/")
```


## How to work with a config?
We use [Hydra](https://hydra.cc/docs/intro/) as a parser for `.yaml` configs.
So, you can change whatever you want directly in the config file or override some parameters
using command line interface:
```
python train_cars.py optimizer.args.lr=0.000001 bs_val=128
```

You can change not only the training parameters in the config file, but also
the model, loss, optimizer, etc. Please, check our registry for the possible options.
You can manually inspect `oml.registry` or `oml.configs` or use the function:
```python
from oml.registry import show_registry
show_registry()
```

## How to use my own implementation of loss, model, augmentations, etc?
You should put your python object inside the corresponding registry by some key.
It allows you to access this object in the config file by that key.

You may change the following blocks and to work correctly some of them have to inherit our interfaces:
* `Transforms`, `Sampler`, `Optimizer`, `Scheduler` - follow the standard PyTorch interfaces.
* `Model` - have to be successor of `IExtractor` (see `oml.interfaces.models`)
* `Criterion` - have to be successor of `ITripletLossWithMiner` (see `oml.interfaces.criterions`)
    * You may want to change only `Miner` inside the criterion. It has to be a successor of `ITripletsMiner`
      (see `oml.interfaces.miners`).


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
from oml.registry.models import MODELS_REGISTRY
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
MODELS_REGISTRY["custom_model"] = CustomModel


@hydra.main(config_path="configs", config_name="train.yaml")
def main_hydra(cfg: DictConfig) -> None:
    pl_train(cfg)


if __name__ == "__main__":
    main_hydra()
```

The same logic works for models, samplers, losses, etc.
