## Config API
Config API is a way to run metric learning experiments via changing only the config file.
All you need is to prepare the `.csv` table which describes your dataset, so the converter
can be implemented in any programming language.


## Usage with public datasets

We've prepared the [examples](https://github.com/OML-Team/open-metric-learning/tree/main/examples)
on 4 popular benchmarks used by researchers to evaluate metric learning models,
see [metric learning leaderboard](https://paperswithcode.com/task/metric-learning).
After downloading a dataset you can train or validate your model by the following commands:

```shell
cd <example>
python convert_<example>.py --dataset_root=/path/to/example/dataset
python train_<example>.py
python validate_<example>.py
```

Note, you can find our pretrained checkpoints for these datasets in the *Models zoo* section of the main readme.

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

You can find more instructions in the
[documentation](https://open-metric-learning.readthedocs.io/en/latest/examples/config.html#usage-with-custom-dataset).
