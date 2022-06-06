### Retrieval DataFrame Format
Expecting columns: `label`, `path`, `split`, `is_query`, `is_gallery` and
optional `x_1`, `x_2`, `y_1`, `y_2`.

* `split` must be on of 2 values: `train` or `validation`
* `is_query` and `is_gallery` have to be `None` where `split == train` and `True`
or `False` where `split == validation`. Note, that both values may be equal `True` in
the same time.
* `x_1`, `x_2`, `y_1`, `y_2` are in the following format `left`, `right`, `top`, `bot` (`y_1` must be less than `y_2`)


### CARS 196
[Dataset page.](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
```
└── CARS196
    ├── devkit
    │   ├── cars_meta.mat
    │   ├── cars_train_annos.mat
    │   └── cars_test_annos_withlabels.mat
    ├── cars_train
    │   ├── 00001.jpg
    │   └── ...
    └── cars_test
        ├── 00001.jpg
        └── ...
```

### CUB 200 2011
[Dataset page.](https://deepai.org/dataset/cub-200-2011)
```
└── CUB_200_2011
    ├── images.txt
    ├── train_test_split.txt
    ├── bounding_boxes.txt
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