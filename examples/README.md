# Config API

With our Config API you can run experiments on your own datasets without writing any ML code.
All you need is to prepare the `.csv` table which describes your dataset.

<details>
<summary>Table Format</summary>
<p>

Expected columns: `label`, `path`, `split`, `is_query`, `is_gallery` and
optional `category`, `category_name`, `x_1`, `x_2`, `y_1`, `y_2`.

* `split` must be one of 2 values: `train` or `validation`
* `is_query` and `is_gallery` have to be `None` where `split == train` and `True`
or `False` where `split == validation`. Note, that both values can be `True` at
the same time. In this case, we perform a validation procedure for every item
 in the validation set using the "1 vs rest" approach (datasets of this kind are `CARS196` or `CUB`).
* `x_1`, `x_2`, `y_1`, `y_2` are in the following format `left`, `right`, `top`, `bot` (`y_1` must be less than `y_2`)
</p>
</details>

todo

### CARS 196
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

### CUB 200 2011
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

### INSHOP (DEEPFASHION)
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

### SOP (STANFORD ONLINE PRODUCTS)
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
